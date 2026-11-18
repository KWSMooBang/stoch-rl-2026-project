"""E03 (T018): consolidate the E02 buffer into probe-ready arrays.

Reads /workspace/stochrl_t017 (features fp16, episodes, privileged LABEL-ONLY files) and writes
/workspace/stochrl_t018/cache/:
  feat_<key>.npy   [8000, D] fp16   probe inputs (frozen VLA features + raw baselines)
  labels.npz       probe targets (privileged sim labels, outcomes) + split / episode / decision
  cand_feat_<key>.npy, cand_labels.npz  candidate-level ae_cand features + branch outcomes (397x16)
Privileged arrays are written ONLY to labels.npz / cand_labels.npz (never into feat_*).
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

SRC = "/workspace/stochrl_t017"
DST = "/workspace/stochrl_t018/cache"
GAMMA = 0.98          # per env step, for the MC value label
K_SUCC_STEPS = 15     # k-step success horizon = 3 decisions
PHASE4 = ["approach", "grasp_transport", "post_grasp_lost", "placed"]  # grasp_contact merged into transport


def feature_keys(npz):
    """Flatten E02 feature arrays into named probe inputs [16, D]."""
    out = {}
    for k in npz.files:
        a = npz[k]
        if k.startswith("vlm") or k.startswith("sig"):
            out[k] = a.reshape(a.shape[0], -1)
        elif k == "ae_ctx_mean":                       # [16,4,1024] layers 4,9,13,17
            for li, L in enumerate([4, 9, 13, 17]):
                out[f"aectx{L}_mean"] = a[:, li]
        elif k == "ae_ctx_tok_last":                   # [16,8,1024]
            out["aectx17_tok"] = a.reshape(a.shape[0], -1)
        elif k == "ae_cand_tok_last":
            out["aecand17_tok"] = a.reshape(a.shape[0], -1)
    return out


def ae_cand_split(a):
    """ae_cand_mean [N, 4, 4, 1024]: E02 stores (denoise step, layer) (e02_common._pool). Use the LAST denoise step per layer."""
    return {f"aecand{L}_mean": a[:, -1, li] for li, L in enumerate([4, 9, 13, 17])}


def load_episode(eid_str):
    f = np.load(f"{SRC}/features/{eid_str}.npz")
    feats = feature_keys(f)
    feats.update(ae_cand_split(f["ae_cand_mean"]))
    e = np.load(f"{SRC}/episodes/{eid_str}.npz")
    rgb = torch.from_numpy(e["rgb"][:16]).permute(0, 3, 1, 2).float() / 255.0
    feats["raw_rgb32"] = F.interpolate(rgb, size=(24, 32), mode="area").flatten(1).numpy()
    feats["raw_rgb64"] = F.interpolate(rgb, size=(48, 64), mode="area").flatten(1).numpy()
    st = e["state"][:16]
    feats["raw_proprio"] = st
    feats["raw_time"] = np.arange(16, dtype=np.float32)[:, None]
    feats["raw_proprio_time"] = np.concatenate([st, feats["raw_time"]], 1)
    feats = {k: v.astype(np.float16) for k, v in feats.items()}

    p = np.load(f"{SRC}/privileged/{eid_str}.npz")
    ss = e["step_success"]
    tfs = int(np.argmax(ss)) if ss.any() else -1
    t_env = np.arange(16) * 5
    lab = {}
    lab["success_once"] = np.full(16, bool(ss.any()))
    lab["success_at_end"] = np.full(16, bool(ss[-1]))
    lab["success_before"] = np.array([tfs >= 0 and tfs < t for t in t_env])
    if tfs >= 0:
        lab["value"] = np.where(t_env <= tfs, GAMMA ** np.maximum(tfs - t_env, 0), 1.0).astype(np.float32)
        lab["succ_k15"] = np.array([tfs >= 0 and (t <= tfs < t + K_SUCC_STEPS) for t in t_env])
    else:
        lab["value"] = np.zeros(16, np.float32)
        lab["succ_k15"] = np.zeros(16, bool)
    d = {k: p[k][:16] for k in p.files if k.startswith("dec_")}
    gp_rel = d["dec_pos_plate"] - d["dec_pos_gripper"]
    lab["geo"] = np.concatenate([
        d["dec_gripper_carrot_rel"], d["dec_carrot_plate_rel"], gp_rel,
        d["dec_gripper_carrot_dist"][:, None], d["dec_carrot_plate_dist"][:, None],
        d["dec_gripper_plate_dist"][:, None]], 1).astype(np.float32)          # 12
    lab["abs_pos"] = np.concatenate([d["dec_pos_gripper"], d["dec_pos_carrot"], d["dec_pos_plate"]], 1).astype(np.float32)
    lab["grasping_now"] = d["dec_grasping_now"]
    lab["contact_any_finger"] = d["dec_contact_any_finger"]
    lab["ever_grasped"] = d["dec_ever_grasped"]
    ph = d["dec_phase"].astype(str)
    lab["phase5"] = ph
    lab["phase4"] = np.array([PHASE4.index("grasp_transport") if x in ("grasp_contact", "transport") else PHASE4.index(x) for x in ph])
    lab["obj_id"] = d["dec_carrot_id"]
    lab["overlay_id"] = d["dec_overlay_id"]
    lab["decision"] = np.arange(16)
    return eid_str, feats, lab


def cand_states():
    rows = []
    for f in sorted(glob.glob(f"{SRC}/branch/branch_results_*.jsonl")):
        for line in open(f):
            rows.append(json.loads(line))
    by = {}
    for r in rows:
        by.setdefault((r["state_id"], r["cand"]), []).append(r)
    states = sorted({r["state_id"] for r in rows})
    return states, by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    os.makedirs(DST, exist_ok=True)
    torch.set_num_threads(1)
    idx = [l.strip().split(",") for l in open(f"{SRC}/feature_index.csv")][1:]
    eps = []
    split_of = {}
    for r in idx:
        if r[0] not in split_of:
            eps.append(r[0])
            split_of[r[0]] = r[2]
    # cross-check with split manifest
    man = json.load(open(f"{SRC}/split_manifest.json"))
    mspl = {f"ep{e['eid']:06d}": e["split"] for e in man["episodes"]}
    assert all(mspl[e] == split_of[e] for e in eps) and len(eps) == 500
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(load_episode, eps, chunksize=4))
    feats = {k: np.concatenate([r[1][k] for r in res]) for k in res[0][1]}
    labs = {k: np.concatenate([r[2][k] for r in res]) for k in res[0][2]}
    labs["episode"] = np.repeat(np.array(eps), 16)
    labs["split"] = np.repeat(np.array([split_of[e] for e in eps]), 16)
    # cross-check success flags with decisions jsonl
    dj = {}
    for f in glob.glob(f"{SRC}/decisions_*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            dj[(r["episode_id"], r["decision"])] = r
    for i in range(len(labs["episode"])):
        r = dj[(labs["episode"][i], int(labs["decision"][i]))]
        assert r["success_once"] == bool(labs["success_once"][i]) and r["success_before"] == bool(labs["success_before"][i]), i
        assert r["split"] == labs["split"][i]
    for k, v in feats.items():
        np.save(f"{DST}/feat_{k}.npy", v)
    np.savez(f"{DST}/labels.npz", **labs)
    print("decision feats:", {k: v.shape for k, v in feats.items()})

    # ---- candidate-level (branch) set ----
    states, by = cand_states()
    cf, cl = {}, {"state_id": [], "episode": [], "decision": [], "split": [], "cand": [], "succ_mean": [], "n_cont": []}
    for s in states:
        ep, dd = s.split("_d")
        c = np.load(f"{SRC}/branch/candidates/{s}.npz")
        fs = ae_cand_split(c["ae_cand_mean"])
        fs["aecand17_tok"] = c["ae_cand_tok_last"].reshape(16, -1)
        for k, v in fs.items():
            cf.setdefault(k, []).append(v.astype(np.float16))
        for ci in range(16):
            rr = by.get((s, ci), [])
            cl["state_id"].append(s); cl["episode"].append(ep); cl["decision"].append(int(dd))
            cl["split"].append(split_of[ep]); cl["cand"].append(ci)
            cl["succ_mean"].append(np.mean([r["success_once"] for r in rr]) if rr else np.nan)
            cl["n_cont"].append(len(rr))
    for k, v in cf.items():
        np.save(f"{DST}/cand_feat_{k}.npy", np.concatenate(v))
    np.savez(f"{DST}/cand_labels.npz", **{k: np.array(v) for k, v in cl.items()})
    print("cand states", len(states), "rows", len(cl["cand"]))


if __name__ == "__main__":
    main()

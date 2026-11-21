"""T040 data assembly: T017 buffer transitions (IQL) + T026/T031 branch Monte-Carlo labels -> one npz.

z  = VLM layer-17 image-token mean, standardized with vla_residual/feat_stats_vlm17_img_mean.npz, clip +-5
     (identical to ResidualVLAEnv.feat).
A  = executed 5x7 env actions mapped to pi0.5 model-normalized space (per-dim affine, fitted on T026 exec_env/exec_model).
  python -m qvgm.data --out /workspace/stochrl_t040/data.npz
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
T017 = Path("/workspace/stochrl_t017")
T026C, T031C = Path("/workspace/stochrl_t026/candidates"), Path("/workspace/stochrl_t031/candidates")
T026P, T031P = "/outputs/StochRL/t026/branch_rollouts.parquet", "/outputs/StochRL/t031/branch_rollouts.parquet"
GAMMA, K = 0.99, 5


def action_affine():
    """env = s * model + b per dim (exact up to fp32)."""
    fs = sorted(glob.glob(str(T026C / "*.npz")))
    E = np.concatenate([np.load(f)["exec_env"].reshape(-1, 7) for f in fs]).astype(np.float64)
    M = np.concatenate([np.load(f)["exec_model"].reshape(-1, 7) for f in fs]).astype(np.float64)
    s, b, err = np.zeros(7), np.zeros(7), np.zeros(7)
    for j in range(7):
        X = np.c_[M[:, j], np.ones(len(M))]
        c = np.linalg.lstsq(X, E[:, j], rcond=None)[0]
        s[j], b[j] = c
        err[j] = np.abs(X @ c - E[:, j]).max()
    return s, b, err


def to_model(a_env, s, b):
    return ((a_env - b) / s).astype(np.float32)


class Feats:
    def __init__(self):
        st = np.load(REPO / "vla_residual/feat_stats_vlm17_img_mean.npz")
        self.mean, self.std = st["mean"].astype(np.float32), st["std"].astype(np.float32)
        self.cache = {}

    def ep(self, eid):
        if eid not in self.cache:
            f = np.load(T017 / f"features/ep{eid:06d}.npz")["vlm17_img_mean"].astype(np.float32)
            self.cache[eid] = np.clip((f - self.mean) / self.std, -5, 5).astype(np.float16)
        return self.cache[eid]


def buffer(F, s, b):
    rows = [json.loads(l) for fn in ("decisions_0.jsonl", "decisions_250.jsonl") for l in open(T017 / fn)]
    df = pd.DataFrame(rows).sort_values(["eid", "decision"]).reset_index(drop=True)
    out = dict(z=[], zn=[], A=[], r=[], term=[], eid=[], dec=[], split=[], ep_succ=[], valid_next=[])
    n_drop_trunc = 0
    for eid, g in df.groupby("eid"):
        ex = np.load(T017 / f"episodes/ep{eid:06d}.npz")["exec_env"]
        z = F.ep(eid)
        for _, r in g.iterrows():
            d = int(r.decision)
            if r.success_before:
                break  # after first success: episode terminated (train semantics)
            succ = bool(r.success_in_chunk)
            has_next = d + 1 < z.shape[0]
            if not succ and not has_next:
                n_drop_trunc += 1  # truncation at decision 15 without cached z' -> excluded (prereg 1.3)
                continue
            out["z"].append(z[d]); out["zn"].append(z[d + 1] if has_next else np.zeros_like(z[d]))
            out["A"].append(to_model(ex[d], s, b).reshape(-1)); out["r"].append(float(succ)); out["term"].append(succ)
            out["eid"].append(eid); out["dec"].append(d); out["split"].append(r.split)
            out["ep_succ"].append(bool(r.success_once)); out["valid_next"].append(has_next)
    o = {f"buf_{k}": np.asarray(v) for k, v in out.items()}
    return o, n_drop_trunc


def branch(F, s, b):
    recs = []
    for src, P, CD in (("t026", T026P, T026C), ("t031", T031P, T031C)):
        x = pd.read_parquet(P)
        if src == "t031":
            x = x[x.cand_name != "E3"]
            name = x.cand_name
        else:
            name = x.kind + "_" + x.cand.astype(str)
        x = x.assign(name=name.values)
        for sid, g in x.groupby("state_id"):
            c = np.load(CD / f"{sid}.npz")["exec_env"]
            eid, d, t = int(g.eid.iloc[0]), int(g.decision.iloc[0]), int(g.t.iloc[0])
            for (cand, nm), h in g.groupby(["cand", "name"]):
                h = h.sort_values("rep")
                succ = h.success_once.values.astype(bool)
                k = np.where(succ, (h.first_success_t.values - t) // K, 0)
                assert (k[succ] >= 0).all()
                G = np.where(succ, GAMMA ** k, 0.0)
                recs.append(dict(src=src, sid=sid, eid=eid, dec=d, cand=int(cand), name=nm,
                                 is_base=int(cand) == 0, A=to_model(c[int(cand)], s, b).reshape(-1),
                                 z=F.ep(eid)[d], succ=succ, end=h.success_at_end.values.astype(bool), G=G.mean(),
                                 reps=h.rep.values))
    o = {"br_src": np.array([r["src"] for r in recs]), "br_sid": np.array([r["sid"] for r in recs]),
         "br_eid": np.array([r["eid"] for r in recs]), "br_dec": np.array([r["dec"] for r in recs]),
         "br_cand": np.array([r["cand"] for r in recs]), "br_name": np.array([r["name"] for r in recs]),
         "br_is_base": np.array([r["is_base"] for r in recs]), "br_A": np.stack([r["A"] for r in recs]),
         "br_z": np.stack([r["z"] for r in recs]), "br_succ": np.stack([r["succ"] for r in recs]),
         "br_end": np.stack([r["end"] for r in recs]), "br_G": np.array([r["G"] for r in recs])}
    assert all((r["reps"] == np.arange(4)).all() for r in recs)
    return o


def folds(br_sid, br_eid, br_src, n=5, seed=0):
    """Episode-level 5-fold over branch states, stratified by source."""
    rng = np.random.default_rng(seed)
    fold_of = {}
    for src in ("t026", "t031"):
        eids = np.unique(br_eid[br_src == src])
        rng.shuffle(eids)
        for i, e in enumerate(eids):
            fold_of[int(e)] = i % n
    return np.array([fold_of[int(e)] for e in br_eid])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/stochrl_t040/data.npz")
    a = ap.parse_args()
    s, b, err = action_affine()
    print("affine scale", s.round(5), "bias", b.round(5), "max err", err.max())
    F = Feats()
    bo, ndrop = buffer(F, s, b)
    br = branch(F, s, b)
    br["br_fold"] = folds(br["br_sid"], br["br_eid"], br["br_src"])
    # every branch state is a T017 train-split decision; check & map fold to buffer episodes
    splits = dict(zip(bo["buf_eid"].tolist(), bo["buf_split"].tolist()))
    assert all(splits.get(int(e), "train") == "train" for e in np.unique(br["br_eid"]))
    fold_of_eid = {int(e): int(f) for e, f in zip(br["br_eid"], br["br_fold"])}
    bo["buf_fold"] = np.array([fold_of_eid.get(int(e), -1) for e in bo["buf_eid"]])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, act_scale=s, act_bias=b, **bo, **br)
    meta = dict(n_buf=len(bo["buf_r"]), n_buf_trainval=int((bo["buf_split"] != "test").sum()),
                n_buf_succ=int(bo["buf_r"].sum()), n_drop_trunc=ndrop, n_br_rows=len(br["br_G"]),
                n_br_states=int(len(np.unique(br["br_sid"]))),
                fold_states={int(f): int(len(np.unique(br["br_sid"][br["br_fold"] == f]))) for f in range(5)},
                affine_max_err=float(err.max()), gamma=GAMMA)
    print(json.dumps(meta, indent=1))
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

"""E02 dataset card + feature index + leakage checks for /workspace/stochrl_t017.

usage: python -m repcritic.e02_card   -> dataset_card_stats.json, feature_index.csv, features_schema.json
"""
from __future__ import annotations

import collections
import csv
import json
import os
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("RC_E02_OUT", "/workspace/stochrl_t017"))
PHASES = ["approach", "grasp_contact", "transport", "post_grasp_lost", "placed"]

FEATURE_DOC = {
    "vlm{L}_img_mean": ("VLM prefix (gemma_2b) layer L, mean over 256 image tokens", "context z(o,l)"),
    "vlm{L}_img_grid4": ("VLM prefix layer L, image tokens 16x16 -> 4x4 avg pool (row-major)", "context z(o,l)"),
    "vlm{L}_txt_mean": ("VLM prefix layer L, mean over valid prompt tokens", "context z(o,l)"),
    "vlm{L}_txt_last": ("VLM prefix layer L, last valid prompt token", "context z(o,l)"),
    "sig{L}_mean": ("SigLIP So400m layer L, mean over 256 patches (valid camera)", "context z(o)"),
    "sig{L}_grid4": ("SigLIP layer L, 16x16 -> 4x4 avg pool", "context z(o)"),
    "ae_ctx_mean": ("Action expert (gemma_300m) layers [4,9,13,17], extra suffix pass with x_t=0,t=1; mean over 8 action tokens", "context z(o,l) (candidate-independent)"),
    "ae_ctx_tok_last": ("Action expert layer 17, same x_t=0 pass, all 8 action tokens", "context z(o,l) (candidate-independent)"),
    "ae_cand_mean": ("Action expert layers [4,9,13,17] at the 4 Euler denoise steps of the SAMPLED/executed chunk; mean over 8 tokens", "candidate-dependent z(o,l,a)"),
    "ae_cand_tok_last": ("Action expert layer 17 at the last denoise step, all 8 tokens", "candidate-dependent z(o,l,a)"),
}


def doc_for(k):
    import re
    kk = re.sub(r"(vlm|sig)\d+", lambda m: m.group(1) + "{L}", k)
    return FEATURE_DOC.get(kk, ("?", "?"))


def main():
    rows = [json.loads(l) for f in sorted(OUT.glob("decisions_*.jsonl")) for l in open(f)]
    man = json.loads((OUT / "split_manifest.json").read_text())
    split_of = {e["eid"]: e["split"] for e in man["episodes"]}
    eps = collections.OrderedDict()
    for r in rows:
        eps.setdefault(r["eid"], []).append(r)
    st = dict(n_episodes=len(eps), n_decisions=len(rows), split_seed=man["split_seed"], ep_seed=man["ep_seed"])
    # integrity
    assert all(len(v) == 16 for v in eps.values()), "each episode must have 16 decisions"
    assert all(sorted(r["decision"] for r in v) == list(range(16)) for v in eps.values())
    assert all(split_of[e] == v[0]["split"] for e, v in eps.items())
    # leakage: episode ids disjoint across splits
    by_split = collections.defaultdict(set)
    for e, v in eps.items():
        for r in v:
            by_split[r["split"]].add(e)
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    st["leak_episode_overlap"] = {f"{a}&{b}": len(by_split[a] & by_split[b]) for a, b in pairs}
    st["leak_rows_split_consistent"] = all(len({r["split"] for r in v}) == 1 for v in eps.values())
    # per split stats
    st["splits"] = {}
    phase_counts = {}
    scene_ids = collections.defaultdict(lambda: collections.defaultdict(set))
    rew_check = []
    for sp in ("train", "val", "test"):
        E = [e for e in eps if eps[e][0]["split"] == sp]
        so = np.array([eps[e][0]["success_once"] for e in E]); se = np.array([eps[e][0]["success_at_end"] for e in E])
        dec = [r for e in E for r in eps[e]]
        pc = collections.Counter()
        pc_pre = collections.Counter()
        for e in E:
            z = np.load(OUT / "privileged" / f"ep{e:06d}.npz")
            ph = z["dec_phase"][:16]
            pc.update(ph.tolist())
            sb = [r["success_before"] for r in sorted(eps[e], key=lambda r: r["decision"])]
            pc_pre.update([p for p, b in zip(ph.tolist(), sb) if not b])
            scene_ids["carrot"][sp].add(int(z["dec_carrot_id"][0]))
            scene_ids["overlay"][sp].add(int(z["dec_overlay_id"][0]))
            scene_ids["pose"][sp].add(int(e % (784 * 4)))  # pos*quat index (episode_id % (l1*l2))
        st["splits"][sp] = dict(
            episodes=len(E), decisions=len(dec), success_once=float(so.mean()), success_at_end=float(se.mean()),
            n_success_episodes=int(so.sum()), n_fail_episodes=int((~so).sum()),
            decisions_in_success_episodes=int(sum(r["success_once"] for r in dec)),
            decisions_in_fail_episodes=int(sum(not r["success_once"] for r in dec)),
            decisions_pre_success=int(sum(not r["success_before"] for r in dec)),
            chunks_with_success_step=int(sum(r["success_in_chunk"] for r in dec)),
            phase=dict(pc), phase_pre_success=dict(pc_pre))
        phase_counts[sp] = pc
    st["scene_factor_overlap"] = {f: {f"{a}&{b}": len(scene_ids[f][a] & scene_ids[f][b]) for a, b in pairs}
                                  | {sp: len(scene_ids[f][sp]) for sp in ("train", "val", "test")}
                                  for f in scene_ids}
    allso = np.array([v[0]["success_once"] for v in eps.values()])
    st["all"] = dict(success_once=float(allso.mean()), success_at_end=float(np.mean([v[0]["success_at_end"] for v in eps.values()])))
    # transition consistency on a sample: per-step arrays vs rows
    e0 = next(iter(eps))
    z = np.load(OUT / "episodes" / f"ep{e0:06d}.npz")
    st["array_shapes"] = {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}
    for e in list(eps)[:50]:
        z = np.load(OUT / "episodes" / f"ep{e:06d}.npz")
        rs = sorted(eps[e], key=lambda r: r["decision"])
        rew_check.append(np.allclose([r["reward_sum"] for r in rs], z["step_reward"].reshape(16, 5).sum(1), atol=1e-5)
                         and bool(z["step_success"].any()) == rs[0]["success_once"]
                         and np.allclose(z["exec_env"], np.clip(z["chunk_env"][:, :5], [-1, -1, -1, -np.pi/2, -np.pi/2, -np.pi/2, -1], [1, 1, 1, np.pi/2, np.pi/2, np.pi/2, 1]), atol=1e-6))
    st["transition_consistency_50ep"] = bool(all(rew_check))
    zp = np.load(OUT / "privileged" / f"ep{e0:06d}.npz")
    st["privileged_keys"] = {k: [list(zp[k].shape), str(zp[k].dtype)] for k in zp.files}
    # features
    zf = np.load(OUT / "features" / f"ep{e0:06d}.npz")
    schema = {}
    tot = 0
    for k in zf.files:
        d, dep = doc_for(k)
        nbytes = int(np.prod(zf[k].shape[1:])) * zf[k].dtype.itemsize
        tot += nbytes
        schema[k] = dict(shape_per_decision=list(zf[k].shape[1:]), dtype=str(zf[k].dtype), bytes_per_decision=nbytes,
                         description=d, dependence=dep)
    (OUT / "features_schema.json").write_text(json.dumps(schema, indent=1))
    st["feature_bytes_per_decision"] = tot
    st["feature_keys"] = len(schema)
    # feature index
    with open(OUT / "feature_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_id", "eid", "split", "decision", "t_env", "feature_file", "row", "obs_file", "obs_row",
                    "privileged_file", "snapshot_file"])
        for e, v in eps.items():
            for r in sorted(v, key=lambda r: r["decision"]):
                n = f"ep{e:06d}"
                w.writerow([n, e, r["split"], r["decision"], r["t_env"], f"features/{n}.npz", r["decision"],
                            f"episodes/{n}.npz", r["decision"], f"privileged/{n}.npz", f"snapshots/{n}.pt"])
    # split manifests
    for sp in ("train", "val", "test"):
        (OUT / f"split_{sp}.txt").write_text("\n".join(f"ep{e:06d}" for e in eps if eps[e][0]["split"] == sp) + "\n")
    # disk
    st["disk_GB"] = {d: round(sum(p.stat().st_size for p in (OUT / d).glob("*")) / 1e9, 2)
                     for d in ("episodes", "features", "privileged", "snapshots")}
    # branch label splits
    st["branch"] = {}
    for name in ("state_select", "state_select_ext"):
        p = OUT / "branch" / f"{name}.json"
        if p.exists():
            S = json.loads(p.read_text())["states"]
            st["branch"][name] = dict(n_states=len(S), n_episodes=len({s["eid"] for s in S}),
                                      by_split=dict(collections.Counter(s["split"] for s in S)),
                                      split_consistent=all(split_of[s["eid"]] == s["split"] for s in S),
                                      n_oracle=sum(s["oracle"] for s in S))
    (OUT / "dataset_card_stats.json").write_text(json.dumps(st, indent=1, default=str))
    print(json.dumps(st, indent=1, default=str))


if __name__ == "__main__":
    main()

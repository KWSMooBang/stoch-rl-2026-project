"""T026 diagnostic: outcome agreement with base (cand 0) under the same CRN replicate, and after-K gripper displacement
relative to base. If perturbed candidates move the gripper as much as VLA samples (or more) but agree with base as
often, the frozen policy's continuation absorbs (recovers from) the difference.
usage: .venv/bin/python -m repcritic.t026_recovery --out /outputs/StochRL/t026
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from repcritic.t026_branch import NC_VLA, OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/outputs/StochRL/t026")
    a = ap.parse_args()
    df = pd.DataFrame([json.loads(l) for l in open(OUT / "branch_results_main.jsonl")])
    df["gp"] = df["afterK"].map(lambda d: np.array(d["pos_gripper"]))
    df["gdist"] = df["afterK"].map(lambda d: d["gripper_carrot_dist"])
    df["grasp"] = df["afterK"].map(lambda d: d["grasping_now"])
    base = df[df.cand == 0].set_index(["state_id", "rep"])
    rows = []
    for _, r in df[df.cand > 0].iterrows():
        b = base.loc[(r.state_id, r.rep)]
        rows.append(dict(state_id=r.state_id, stratum=r.stratum, cand=r.cand, kind=r.kind,
                         agree=r.success_once == b.success_once, disp=float(np.linalg.norm(r.gp - b.gp)),
                         d_gdist=r.gdist - b.gdist, grasp_diff=bool(r.grasp != b.grasp)))
    R = pd.DataFrame(rows)
    # control: candidate-0 outcome variability across replicates (how often two reps of base agree)
    agree_rep = []
    for sid, g in base.groupby(level=0):
        y = g.success_once.values.astype(int)
        agree_rep.append(np.mean([y[i] == y[j] for i in range(len(y)) for j in range(i + 1, len(y))]))
    out = dict(
        overall={k: dict(agree_with_base_same_rep=float(g.agree.mean()), afterK_disp_from_base_m=float(g.disp.median()),
                         afterK_grasp_state_differs=float(g.grasp_diff.mean()), n=len(g))
                 for k, g in R.groupby("kind")},
        base_rep_vs_rep_agreement=float(np.mean(agree_rep)),
        by_stratum={s: {k: dict(agree=float(gg.agree.mean()), disp=float(gg.disp.median())) for k, gg in g.groupby("kind")}
                    for s, g in R.groupby("stratum")},
    )
    Path(a.out, "recovery_diag.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

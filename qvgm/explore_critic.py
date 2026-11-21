"""T040 exploratory (not pre-registered): where does the critic's same-state ranking edge come from?
Pairwise accuracy (once) split by candidate size class, out-of-fold scores as in analyze_critic.
  python -m qvgm.explore_critic --out /outputs/StochRL/t040
"""
import argparse
import json
from pathlib import Path

import numpy as np

from qvgm.analyze_critic import build_states, load

LARGE = {"T1_+u", "T1_-u", "T1_+v", "T1_-v", "grip_flip", "E1"}


def acc(D, S, score, filt, nb=2000):
    per = []
    for s in S:
        nm, q, p = D["br_name"][s["idx"]], score[s["idx"]], s["p"]
        c = n = 0.0
        for i in range(len(q)):
            for j in range(i + 1, len(q)):
                if p[i] == p[j] or not filt(nm[i], nm[j]):
                    continue
                dq = q[i] - q[j]
                c += 0.5 if dq == 0 else float(np.sign(dq) == np.sign(p[i] - p[j])); n += 1
        per.append((c, n))
    per = np.array(per)
    rng = np.random.default_rng(0)
    bs = [per[ix, 0].sum() / max(per[ix, 1].sum(), 1) for ix in rng.integers(0, len(S), (nb, len(S)))]
    return dict(acc=float(per[:, 0].sum() / max(per[:, 1].sum(), 1)), n_pairs=int(per[:, 1].sum()),
                ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/outputs/StochRL/t040")
    a = ap.parse_args()
    D, P, _, _ = load()
    S = build_states(D, P)
    groups = {
        "pair_has_one_large_change(T1 10cm/grip_flip/E1)": lambda x, y: (x in LARGE) != (y in LARGE),
        "both_small(T026 all, T031 base/T05/yaw)": lambda x, y: x not in LARGE and y not in LARGE,
        "T026_vla_vs_vla": lambda x, y: x.startswith("vla") and y.startswith("vla"),
        "T026_pert2.5cm_vs_base": lambda x, y: {x, y} & {"vla_0"} and (x.startswith("pert") or y.startswith("pert")),
    }
    res = {}
    for g, f in groups.items():
        r = {}
        pf = pa = None
        for k, key in (("full", "Qf"), ("action_only", "Qa")):
            r[k], per = acc(D, S, P[key], f)
            if k == "full":
                pf = per
            else:
                pa = per
        rng = np.random.default_rng(0)
        bs = []
        for ix in rng.integers(0, len(S), (2000, len(S))):
            den = max(pf[ix, 1].sum(), 1)
            bs.append((pf[ix, 0].sum() - pa[ix, 0].sum()) / den)
        r["diff_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        res[g] = r
        print(g, json.dumps(r))
    (Path(a.out) / "critic_explore.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

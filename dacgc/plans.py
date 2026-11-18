"""Named experiment plans (lists of training configs).  python -m dacgc.plans <name> <out.json>"""
from __future__ import annotations

import json
import sys

GATE_SEEDS = [0, 1, 2]
MAIN_SEEDS = [0, 1, 2, 3, 4]
VAL_SEEDS = [100, 101]

# shared by every method / condition (same env-step budget, same network)
BASE = dict(total_steps=3_000_000, eval_every=300_000, eval_eps_curve=20, eval_eps_final=100)

# Phase 1 conditions, in the pre-registered order (design 4.3)
CONDITIONS = {
    "balanced": dict(skew=0.5),
    "skew0.8": dict(skew=0.8),
    "skew0.95": dict(skew=0.95),
    "skew0.95+biasedBC": dict(skew=0.95, bc_init=True),
}

# Phase 3 validation grids (validation seeds only; selection rule in aggregate.select).
# Keys: first key must be lam (tie-break = smaller lambda).
VAL_GRID = {
    "B4": [dict(lam=l, tau=t, K=k) for l in (0.3, 1.0, 3.0) for t in (0.005, 0.02) for k in (10, 50)],
    "B3": [dict(lam=l, K=k) for l in (0.3, 1.0, 3.0) for k in (10, 50)],
    "B6": [dict(lam=l) for l in (0.3, 1.0, 3.0)],
    "B2": [dict(lam=l) for l in (0.3, 1.0, 3.0)],
}


def pilot():
    runs = []
    for sk in [0.5, 0.8, 0.95]:
        for m in ["B0", "B1"]:
            for seed in [0, 1]:
                runs.append(dict(method=m, skew=sk, seed=seed, total_steps=300_000, tag=f"{m}-sk{sk}-s{seed}"))
    return runs


def smoke():
    base = dict(total_steps=12_000, learn_start=2_000, eval_every=6_000, eval_eps_final=10, eval_eps_curve=5,
                bc_demos=64, bc_updates=50, skew=0.95)
    runs = []
    for m in ["B0", "B1", "B2", "B3", "B4", "B6"]:
        runs.append(dict(base, method=m, tag=f"smoke-{m}"))
    runs.append(dict(base, method="B4", mu="learned", tag="smoke-B4-learned"))
    runs.append(dict(base, method="B4", enc="film", tag="smoke-B4-film"))
    runs.append(dict(base, method="B0", bc_init=True, tag="smoke-B0-bc"))
    return runs


PLANS = {"pilot": pilot, "smoke": smoke}

if __name__ == "__main__":
    runs = PLANS[sys.argv[1]]()
    json.dump(dict(name=sys.argv[1], runs=runs), open(sys.argv[2], "w"), indent=0)
    print(len(runs), "runs")

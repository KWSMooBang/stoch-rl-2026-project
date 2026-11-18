"""Entry point for harness runs (CPU only).  Writes $RUN_DIR/metrics.json (+ full.json).

  python -m dacgc.harness_main gate  [--procs P]            # Phase 0-1
  python -m dacgc.harness_main main --cond <condition> [--procs P]   # Phase 3 (validation -> main)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from .plans import BASE, CONDITIONS, GATE_SEEDS, MAIN_SEEDS, VAL_SEEDS, VAL_GRID
from .runner import run_plan
from . import aggregate as AG


if os.environ.get("DACGC_SMOKE"):
    BASE = dict(BASE, total_steps=6_000, learn_start=1_000, eval_every=3_000, eval_eps_curve=3, eval_eps_final=5,
                bc_demos=64, bc_updates=20, cache_states=32)
    GATE_SEEDS, MAIN_SEEDS, VAL_SEEDS = GATE_SEEDS[:2], MAIN_SEEDS[:2], VAL_SEEDS


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def write(run_dir, name, obj):
    with open(os.path.join(run_dir, name), "w") as f:
        json.dump(obj, f, indent=1)


def phase0_checks():
    """Unit tests: clone/restore, dynamics, oracle == brute-force BFS, leakage assert."""
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], capture_output=True, text=True)
    return dict(pytest_rc=r.returncode, pytest_tail=r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-300:])


def gate(args):
    runs = []
    for cond in AG.GATE_ORDER:
        for s in GATE_SEEDS:
            runs.append(dict(BASE, **CONDITIONS[cond], method="B0", seed=s, tag=f"{cond}|B0|s{s}"))
    # diagnostics only (NOT used for the gate decision), T013b: oracle-supervised B6 with FiLM (does the
    # held-out failure turn from timeout into wrong-target?) and the concat-encoding ablation.
    diags = [("balanced", "B6", {}, "B6"), ("skew0.95", "B6", {}, "B6"),
             ("balanced", "B6", dict(enc="concat"), "B6concat"), ("balanced", "B0", dict(enc="concat"), "B0concat")]
    for cond, m, extra, name in diags:
        for s in GATE_SEEDS:
            runs.append(dict(BASE, **CONDITIONS[cond], **extra, method=m, seed=s, tag=f"diag:{cond}:{name}|{m}|s{s}"))
    log(f"gate: {len(runs)} runs")
    res, wall = run_plan(runs, args.procs, log)
    agg = AG.group(res, lambda c: c["tag"].split("|")[0])
    dec = AG.gate_decision(agg)
    out = dict(phase="0-1", wall_s=wall, checks=phase0_checks(), decision=dec,
               conditions={k: {m: v for m, v in a.items() if m != "curve"} for k, a in agg.items()},
               curves={k: a["curve"] for k, a in agg.items()},
               errors=[r["error"] for r in res if "error" in r])
    write(args.run_dir, "metrics.json", out)
    write(args.run_dir, "full.json", dict(results=res))
    log(json.dumps(dec, indent=1))


def label_of(c):
    return c["tag"].split("|")[0]


def main(args):
    cond = CONDITIONS[args.cond]
    t0 = time.time()
    # ---- stage V: validation (seeds VAL_SEEDS), never touches held-out goals ----
    vruns = []
    for method, grid in VAL_GRID.items():
        for hp in grid:
            key = ",".join(f"{k}={v}" for k, v in hp.items())
            for s in VAL_SEEDS:
                vruns.append(dict(BASE, **cond, method=method, seed=s, **hp, tag=f"{method}:{key}|s{s}",
                                  do_switch=False, do_rank=False))
    log(f"validation: {len(vruns)} runs")
    vres, vwall = run_plan(vruns, args.procs, log)
    vagg = AG.group(vres, label_of)
    selected, sel_table = {}, {}
    for method, grid in VAL_GRID.items():
        sub = {}
        for hp in grid:
            key = ",".join(f"{k}={v}" for k, v in hp.items())
            a = vagg.get(f"{method}:{key}")
            if a is not None:
                sub[tuple(hp.values())] = a
        best, table = AG.select(sub)
        selected[method] = dict(zip(grid[0].keys(), best))
        sel_table[method] = [dict(hp=dict(zip(grid[0].keys(), k)), seen_rare_sr=r, SR_seen=s) for k, r, s in table]
    log(f"selected: {selected}")
    # ---- stage M: main comparison, seeds MAIN_SEEDS ----
    B4 = selected["B4"]
    variants = {
        "B0": dict(method="B0"),
        "B1_HER": dict(method="B1"),
        "B2_naivesep": dict(method="B2", **selected["B2"]),
        "B3_ungated_rank": dict(method="B3", **selected["B3"]),
        "B4_DA-CGC": dict(method="B4", **B4),
        "B6_Qstar_reg": dict(method="B6", **selected["B6"]),
        "B4_learned_mu": dict(method="B4", mu="learned", **B4),
        "B0_concat": dict(method="B0", enc="concat"),
        "B4_concat": dict(method="B4", enc="concat", **B4),
        "B6_concat": dict(method="B6", enc="concat", **selected["B6"]),
    }
    mruns = []
    for name, v in variants.items():
        for s in MAIN_SEEDS:
            mruns.append(dict(BASE, **cond, **v, seed=s, tag=f"{name}|s{s}"))
    log(f"main: {len(mruns)} runs")
    mres, mwall = run_plan(mruns, args.procs, log)
    magg = AG.group(mres, label_of)
    out = dict(phase="3", condition=args.cond, selected=selected, selection_table=sel_table,
               wall_validation_s=vwall, wall_main_s=mwall, wall_total_s=time.time() - t0,
               validation={k: {m: v for m, v in a.items() if m != "curve"} for k, a in vagg.items()},
               main={k: {m: v for m, v in a.items() if m != "curve"} for k, a in magg.items()},
               curves={k: a["curve"] for k, a in magg.items()},
               errors=[r["error"] for r in vres + mres if "error" in r])
    write(args.run_dir, "metrics.json", out)
    write(args.run_dir, "full.json", dict(validation=vres, main=mres))
    log("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["gate", "main"])
    ap.add_argument("--cond", default=None)
    ap.add_argument("--procs", type=int, default=64)
    ap.add_argument("--run_dir", default=os.environ.get("RUN_DIR", "."))
    a = ap.parse_args()
    os.makedirs(a.run_dir, exist_ok=True)
    gate(a) if a.phase == "gate" else main(a)

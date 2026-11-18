"""Run a list of configs (each = one training run) in a process pool; collect into one JSON.

usage: python -m dacgc.runner <plan.json> <out.json> [n_procs]
plan.json: {"name": ..., "runs": [ {cfg dict}, ... ]}
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
import multiprocessing as mp

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _one(cfg):
    from dacgc.train import run_cfg
    try:
        return run_cfg(cfg)
    except Exception:
        return dict(cfg=cfg, error=traceback.format_exc())


def run_plan(runs, n_procs, log=print):
    t0 = time.time()
    out = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_procs, maxtasksperchild=1) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, runs)):
            out.append(r)
            c = r["cfg"]
            if "error" in r:
                log(f"[{i+1}/{len(runs)}] ERROR {c.get('tag')} {r['error'][-500:]}")
            else:
                S = r["summary"]
                log(f"[{i+1}/{len(runs)}] {c.get('tag')} SRseen={S['SR_seen']:.2f} SRunseen={S['SR_unseen']:.2f} "
                    f"SER={S['SER_unseen']:.2f} (freq {S['SER_unseen_freq']:.2f}) t={r['wall_total_s']:.0f}s "
                    f"elapsed={time.time()-t0:.0f}s")
            sys.stdout.flush()
    return out, time.time() - t0


if __name__ == "__main__":
    plan = json.load(open(sys.argv[1]))
    n = int(sys.argv[3]) if len(sys.argv) > 3 else min(len(plan["runs"]), 64)
    res, wall = run_plan(plan["runs"], n)
    json.dump(dict(name=plan.get("name"), wall_s=wall, results=res), open(sys.argv[2], "w"))

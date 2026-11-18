"""Harness entry: runs a whole phase (all configs x seeds) in one process pool on CPU.

  python -m aapr.run_phase --phase p1 --out $RUN_DIR --workers 40
  python -m aapr.run_phase --phase p3 --out $RUN_DIR --workers 80

Per-job results go to OUT/jobs/<name>.json (a finished job is skipped on restart);
OUT/metrics.json holds the aggregate table (rewritten after every job).

Phase 3 lambda selection (fixed before seeing main results; same budget for every method):
  validation setting = persistent wind (ar1), training seeds 100 and 101 (disjoint from main
  seeds 0-4), lambda grid {0.3, 1, 3}, VAL_FRAC of the main step budget; pick the lambda with the
  highest mean AUC (ties -> the smaller |log lambda|). Fixed-K ablations k4/k8 reuse mh's lambda.
  sparse has no shaping term (lambda irrelevant).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from multiprocessing import Pool

import numpy as np

from .train import train

P1_CONDS = ["iid", "ar1", "bias"]
P1_METHODS = ["sparse", "dist", "prog", "pbrs"]
P3_CONDS = ["ar1", "bias", "iid", "ar1_hi"]
P3_METHODS = ["sparse", "dist", "prog", "pbrs", "aapr1", "mh", "k4", "k8", "mh_ind", "aug"]
VAL_METHODS = ["dist", "prog", "pbrs", "aapr1", "mh", "mh_ind", "aug"]
LAM_GRID = [0.3, 1.0, 3.0]
VAL_SEEDS = [100, 101]
VAL_FRAC = 0.5


def job_name(j):
    return f"{j['tag']}_{j['cond']}_{j['method']}_lam{j['lam']}_s{j['seed']}"


def _run(j):
    out_dir = j.pop("out_dir")
    path = f"{out_dir}/jobs/{job_name(j)}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    tag = j.pop("tag")
    logf = open(f"{out_dir}/logs/{tag}_{j['cond']}_{j['method']}_lam{j['lam']}_s{j['seed']}.log", "w")
    try:
        r = train(**j, log=lambda s: (logf.write(s + "\n"), logf.flush()))
    except Exception:
        logf.write(traceback.format_exc()); logf.close()
        return dict(tag=tag, error=traceback.format_exc(), **j)
    logf.close()
    r["tag"] = tag
    with open(path, "w") as f:
        json.dump(r, f, default=float)
    return r


def aggregate(results):
    rows = {}
    for r in results:
        if "error" in r:
            continue
        key = (r["tag"], r["cond"], r["method"], r["lam"])
        rows.setdefault(key, []).append(r)
    table = []
    for (tag, cond, method, lam), rs in sorted(rows.items()):
        f = lambda g: [g(r) for r in rs]
        sr = f(lambda r: r["final"]["sr"]); auc = f(lambda r: r["auc"])
        hp = [r["wind_helps"]["sr"] for r in rs if r.get("wind_helps")]
        att = [r.get("attribution", {}) for r in rs]
        mean_att = {k: float(np.nanmean([a[k] for a in att if k in a])) for k in (att[0] if att else {})}
        se = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
        table.append(dict(tag=tag, cond=cond, method=method, lam=lam, n_seeds=len(rs),
                          sr_mean=float(np.mean(sr)), sr_se=se(sr), auc_mean=float(np.mean(auc)), auc_se=se(auc),
                          wind_helps_sr=float(np.mean(hp)) if hp else None,
                          sim_transitions=float(np.mean(f(lambda r: r["sim_transitions"]))),
                          wall_clock_train_s=float(np.mean(f(lambda r: r["wall_clock_train_s"]))),
                          mean_abs_shaping=float(np.mean(f(lambda r: r["mean_abs_shaping"]))),
                          final_alpha=float(np.mean(f(lambda r: r["final_alpha"]))),
                          effort=float(np.mean(f(lambda r: r["final"]["effort"]))),
                          steps_to_goal=float(np.mean(f(lambda r: r["final"]["steps"]))),
                          seeds_sr=sr, seeds_auc=auc,
                          curve_sr=np.mean([[c["sr"] for c in r["curve"]] for r in rs], 0).tolist(),
                          curve_steps=[c["step"] for c in rs[0]["curve"]],
                          attribution=mean_att))
    return table


def write_metrics(out, results, extra):
    m = dict(table=aggregate(results), n_jobs_done=sum("error" not in r for r in results),
             n_jobs_failed=sum("error" in r for r in results),
             errors=[r["error"][-500:] for r in results if "error" in r][:5], **extra)
    with open(f"{out}/metrics.json.tmp", "w") as f:
        json.dump(m, f, indent=1, default=float)
    os.replace(f"{out}/metrics.json.tmp", f"{out}/metrics.json")


def run_pool(jobs, workers, out, results, extra):
    with Pool(workers, maxtasksperchild=1) as p:
        for r in p.imap_unordered(_run, jobs):
            results.append(r)
            write_metrics(out, results, extra)
            print(time.strftime("%H:%M:%S"), "done", r.get("tag"), r.get("cond"), r.get("method"),
                  r.get("lam"), r.get("seed"), "SR", r.get("final", {}).get("sr"), flush=True)


def select_lambda(results):
    sel = {}
    for m in VAL_METHODS:
        best = None
        for lam in LAM_GRID:
            aucs = [r["auc"] for r in results if r.get("tag") == "val" and r.get("method") == m
                    and r.get("lam") == lam and "error" not in r]
            if not aucs:
                continue
            key = (np.mean(aucs), -abs(np.log(lam)))
            if best is None or key > best[0]:
                best = (key, lam)
        sel[m] = best[1] if best else 1.0
    sel["k4"] = sel["k8"] = sel["mh"]
    sel["sparse"] = 1.0
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["p1", "p3", "smoke"])
    ap.add_argument("--out", default=os.environ.get("RUN_DIR", "out"))
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()
    out = args.out
    os.makedirs(f"{out}/jobs", exist_ok=True); os.makedirs(f"{out}/logs", exist_ok=True)
    t0 = time.time()
    base = dict(out_dir=out, total_steps=args.steps)
    results = []
    extra = dict(phase=args.phase, steps=args.steps, started=t0)
    if args.phase == "smoke":
        jobs = [dict(base, tag="smoke", cond=c, method=m, lam=1.0, seed=0, total_steps=1500, eval_every=500,
                     eval_eps=5, final_eval_eps=5, learning_starts=500)
                for c in ["ar1"] for m in P3_METHODS]
        run_pool(jobs, args.workers, out, results, extra)
    elif args.phase == "p1":
        seeds = range(args.seeds or 3)
        jobs = [dict(base, tag="p1", cond=c, method=m, lam=1.0, seed=s)
                for s in seeds for c in P1_CONDS for m in P1_METHODS]
        run_pool(jobs, args.workers, out, results, extra)
    else:
        vsteps = int(args.steps * VAL_FRAC)
        vjobs = [dict(base, tag="val", cond="ar1", method=m, lam=lam, seed=s, total_steps=vsteps,
                      eval_every=max(vsteps // 6, 1000), attribution=False)
                 for s in VAL_SEEDS for m in VAL_METHODS for lam in LAM_GRID]
        run_pool(vjobs, args.workers, out, results, extra)
        sel = select_lambda(results)
        extra["selected_lambda"] = sel
        extra["val_wall_clock_s"] = time.time() - t0
        print("selected lambda", sel, flush=True)
        seeds = range(args.seeds or 5)
        # put the slow counterfactual methods first so the pool tail is short
        order = {m: i for i, m in enumerate(["mh", "mh_ind", "k8", "aug", "k4", "aapr1"])}
        mjobs = [dict(base, tag="main", cond=c, method=m, lam=sel[m], seed=s)
                 for s in seeds for c in P3_CONDS for m in P3_METHODS]
        mjobs.sort(key=lambda j: order.get(j["method"], 99))
        run_pool(mjobs, args.workers, out, results, extra)
    extra["wall_clock_total_s"] = time.time() - t0
    write_metrics(out, results, extra)


if __name__ == "__main__":
    main()

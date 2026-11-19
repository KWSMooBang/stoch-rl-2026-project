"""T022: $RUN_DIR/metrics.json from the env episode logs (eval_episodes.jsonl / train_episodes.jsonl) + tensorboard.

- held-out eval: 128 fixed episode ids, same pi0.5 noise per (id, decision) as the B0 validation -> paired vs B0.
- curves: x = env steps (sim steps = decisions x 5) and wall-clock seconds since training start.
- peak VRAM: whole-GPU memory.used samples (5 s) and minus the pre-run baseline (external processes may add).
"""
import argparse
import json
import os

import numpy as np

B0_JSON = "/outputs/StochRL/t022/val/b0.json"


def read(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def jl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(x) for x in open(path) if x.strip()]


def paired(a, b, n_boot=2000, seed=0):
    """mean(a-b) with bootstrap CI over episodes."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return dict(diff=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                n=int(len(d)), wins=int((d > 0).sum()), losses=int((d < 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--algo", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--rc", type=int, default=0)
    a = ap.parse_args()
    t0 = float(read(os.path.join(a.run_dir, "t_start"), "0"))
    t1 = float(read(os.path.join(a.run_dir, "t_end"), "0"))

    tr = jl(os.path.join(a.run_dir, "train_episodes.jsonl"))
    ev = jl(os.path.join(a.run_dir, "eval_episodes.jsonl"))
    # train curve: cumulative env steps from finished train episodes (decisions x 5), success over windows of 200 eps
    train_curve, steps = [], 0
    win = []
    for i, r in enumerate(tr):
        steps += 5 * r["dec"]
        win.append(r)
        if len(win) == 200 or i == len(tr) - 1:
            pt = dict(episodes=i + 1, env_steps=steps, wall_s=round(r["wall"] - t0, 1),
                      success_once=float(np.mean([w["success_once"] for w in win])))
            if "success_at_end" in r:  # T029 hold reward: train episodes run all 16 decisions
                pt["success_at_end"] = float(np.mean([w["success_at_end"] for w in win]))
                pt["return"] = float(np.mean([w["ret"] for w in win]))
                pt["lost_per_episode"] = float(np.mean([w["n_lost"] for w in win]))
            train_curve.append(pt)
            win = []
    total_train_steps = steps
    # eval curve: one point per evaluation (128 held-out eps); env steps = train steps finished before it ended
    b0 = {}
    if os.path.exists(B0_JSON):
        b0 = {e["eid"]: e for e in json.load(open(B0_JSON))["episodes"]}
    tr_wall = np.array([r["wall"] for r in tr]) if tr else np.zeros(0)
    tr_steps = np.cumsum([5 * r["dec"] for r in tr]) if tr else np.zeros(0)
    curve = [dict(eval=-1, env_steps=0, wall_s=0.0, success_once=float(np.mean([e["success_once"] for e in b0.values()])),
                  success_at_end=float(np.mean([e["success_at_end"] for e in b0.values()])), n=len(b0),
                  note="B0 = residual 0 (zero-initialised actor mean => identical to the untrained deterministic policy)")] if b0 else []
    final = None
    for k in sorted({r["eval"] for r in ev}):
        rows = {r["eid"]: r for r in ev if r["eval"] == k}
        wend = max(r["wall"] for r in rows.values())
        n_done = int((tr_wall <= wend).sum()) if len(tr_wall) else 0
        pt = dict(eval=k, env_steps=int(tr_steps[n_done - 1]) if n_done else 0, wall_s=round(wend - t0, 1),
                  success_once=float(np.mean([r["success_once"] for r in rows.values()])),
                  success_at_end=float(np.mean([r["success_at_end"] for r in rows.values()])), n=len(rows))
        common = sorted(set(rows) & set(b0))
        if common:
            pt["paired_vs_b0_once"] = paired([rows[e]["success_once"] for e in common], [b0[e]["success_once"] for e in common])
            pt["paired_vs_b0_end"] = paired([rows[e]["success_at_end"] for e in common], [b0[e]["success_at_end"] for e in common])
        curve.append(pt)
        final = dict(pt, episodes=[dict(eid=e, success_once=r["success_once"], success_at_end=r["success_at_end"])
                                   for e, r in sorted(rows.items())])

    base_mib = 0
    b = read(os.path.join(a.run_dir, "gpu_baseline.txt"), "")
    if b:
        try:
            base_mib = int(b.split(",")[1].split()[0])
        except (IndexError, ValueError):
            pass
    vt = [int(x) for x in (read(os.path.join(a.run_dir, "vram_trace.txt"), "") or "").split() if x.isdigit()]
    # T024/T029: AULC of held-out success curves (trapezoid over env steps incl. the B0 point at 0, / final env steps)
    aulc = aulc_end = None
    if len(curve) >= 2 and curve[-1]["env_steps"] > 0:
        xs = np.array([c["env_steps"] for c in curve], float)
        aulc = float(np.trapz([c["success_once"] for c in curve], xs) / xs[-1])
        aulc_end = float(np.trapz([c["success_at_end"] for c in curve], xs) / xs[-1])
    hold = None
    if tr and "ret" in tr[0]:
        thirds = np.array_split(np.arange(len(tr)), 3)
        hold = dict(n_episodes=len(tr),
                    return_eq_success_at_end=bool(all(abs(r["ret"] - float(r["success_at_end"])) < 1e-6 for r in tr)),
                    train_success_once_by_third=[float(np.mean([tr[i]["success_once"] for i in ix])) for ix in thirds if len(ix)],
                    train_success_at_end_by_third=[float(np.mean([tr[i]["success_at_end"] for i in ix])) for ix in thirds if len(ix)],
                    train_lost_per_episode_by_third=[float(np.mean([tr[i]["n_lost"] for i in ix])) for ix in thirds if len(ix)])
    m = {
        "task": os.path.basename(os.path.normpath(a.run_dir)).split("-")[0].upper(),
        "aulc_once": aulc, "aulc_end": aulc_end, "hold": hold, "algo": a.algo, "seed": a.seed, "return_code": a.rc, "config": os.path.basename(a.cfg),
        "config_overrides": (read(os.path.join(a.run_dir, "t022_overrides.txt"), "") or "").splitlines()[3:],
        "heldout_success_once": final["success_once"] if final else None,
        "heldout_success_at_end": final["success_at_end"] if final else None,
        "heldout_n": final["n"] if final else 0,
        "paired_vs_b0_once": final.get("paired_vs_b0_once") if final else None,
        "paired_vs_b0_end": final.get("paired_vs_b0_end") if final else None,
        "best_heldout_success_once": max((c["success_once"] for c in curve if c["eval"] >= 0), default=None),
        "b0_success_once": curve[0]["success_once"] if b0 else None,
        "train_env_steps": int(total_train_steps), "train_episodes": len(tr),
        "train_time_s": round(t1 - t0, 1) if t1 > 0 else None,
        "peak_vram_mib_gpu_total": max(vt) if vt else None,
        "peak_vram_mib_minus_baseline": (max(vt) - base_mib) if vt else None,
        "eval_curve": [{k: v for k, v in c.items()} for c in curve],
        "train_curve": train_curve,
        "final_eval_episodes": final["episodes"] if final else [],
    }
    with open(os.path.join(a.run_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=1)
    print(json.dumps({k: v for k, v in m.items() if not k.endswith(("curve", "episodes"))}, indent=1))


if __name__ == "__main__":
    main()

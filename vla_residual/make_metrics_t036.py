"""T036: add a `gate` block to $RUN_DIR/metrics.json (written by make_metrics.py) from the env logs.

  train_episodes.jsonl   per finished train episode: g_mean, v_mean, s_abs (= mean |tanh u|), gs_abs (= mean |g tanh u|)
  gate_decisions.jsonl   per held-out eval decision: V_hat, g, success labels at decision time (privileged, analysis)
  checkpoint (g2)        learned gate coefficients (w_v, w_t, b) and log-std of the gate logit
"""
import argparse
import glob
import json
import os

import numpy as np


def jl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(x) for x in open(path) if x.strip()]


def st(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return dict(n=0)
    return dict(n=int(x.size), mean=float(x.mean()), median=float(np.median(x)), frac_lt_0p1=float((x < 0.1).mean()),
                frac_gt_0p9=float((x > 0.9).mean()), q10=float(np.percentile(x, 10)), q90=float(np.percentile(x, 90)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--cond", required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--temp", type=float, required=True)
    ap.add_argument("--scale", type=float, default=100.0)
    a = ap.parse_args()
    mp = os.path.join(a.run_dir, "metrics.json")
    m = json.load(open(mp)) if os.path.exists(mp) else {}
    tr = jl(os.path.join(a.run_dir, "train_episodes.jsonl"))
    gd = jl(os.path.join(a.run_dir, "gate_decisions.jsonl"))
    gate = dict(cond=a.cond, tau=a.tau, temp=a.temp,
                mode={"g1": "fixed", "g2": "learned", "one": "one", "e2": "none"}.get(a.cond, a.cond))
    if tr:
        thirds = [ix for ix in np.array_split(np.arange(len(tr)), 3) if len(ix)]
        gate["train_success"] = float(np.mean([r["success_once"] for r in tr]))
        gate["train_success_by_third"] = [float(np.mean([tr[i]["success_once"] for i in ix])) for ix in thirds]
        gate["train_success_first2_updates"] = float(np.mean([r["success_once"] for r in tr[:128]]))
        for k in ("g_mean", "v_mean", "s_abs", "gs_abs"):
            if k in tr[0]:
                gate[f"train_{k}"] = float(np.mean([r[k] for r in tr]))
                gate[f"train_{k}_by_third"] = [float(np.mean([tr[i][k] for i in ix])) for ix in thirds]
        if "g_mean" in tr[0]:
            gate["train_episode_g_mean_dist"] = st([r["g_mean"] for r in tr])
            for lab, sel in (("success", True), ("fail", False)):
                gate[f"train_g_mean_{lab}_eps"] = float(np.mean([r["g_mean"] for r in tr if r["success_once"] == sel] or [np.nan]))
    evals = {}
    for k in sorted({r["eval"] for r in gd}):
        rows = [r for r in gd if r["eval"] == k]
        g = np.array([r["g"] for r in rows]); ever = np.array([r["ever"] for r in rows]); now = np.array([r["now"] for r in rows])
        na = np.array([r["now_after"] for r in rows]); sa = np.array([r["s_abs"] for r in rows])
        v = np.array([r["v"] for r in rows])
        evals[str(k)] = dict(
            g_all=st(g), g_pre_success=st(g[~ever]), g_success_flag_on=st(g[now]), g_success_lost=st(g[ever & ~now]),
            vhat_pre_success=st(v[~ever]), vhat_success_flag_on=st(v[now]),
            loss_per_success_decision=float((now & ~na).sum() / max(int(now.sum()), 1)), n_success_decisions=int(now.sum()),
            s_abs_mean=float(sa.mean()), gs_abs_mean=float((g * sa).mean()),
            collapsed_to_0=bool((g < 0.05).mean() > 0.95), collapsed_to_1=bool((g > 0.95).mean() > 0.95))
    gate["eval"] = evals
    if a.cond == "g2":
        try:
            import torch

            ck = sorted(glob.glob(os.path.join(a.run_dir, "rlinf_logs", "*", "checkpoints", "global_step_*")),
                        key=lambda p: int(p.rsplit("_", 1)[-1]))
            sd = torch.load(os.path.join(ck[-1], "actor/model_state_dict/full_weights.pt"), map_location="cpu")
            w = (sd["actor_mean.p"].float() * a.scale).tolist()
            gate["learned"] = dict(w_v=w[0], w_t=w[1], b=w[2], init=dict(w_v=-1 / a.temp, w_t=0.0, b=a.tau / a.temp),
                                   scale=a.scale, logstd=[float(x) for x in sd["actor_logstd"].flatten()],
                                   tau_equiv_at_t0=-w[2] / w[0] if w[0] else None, ckpt=ck[-1])
        except Exception as e:  # noqa: BLE001
            gate["learned_error"] = repr(e)
    m["task"] = "T036"
    m["gate"] = gate
    with open(mp, "w") as f:
        json.dump(m, f, indent=1)
    print(json.dumps({k: v for k, v in gate.items() if k != "eval"}, indent=1))
    for k, e in evals.items():
        print("eval", k, json.dumps({q: e[q] for q in ("loss_per_success_decision", "gs_abs_mean", "collapsed_to_0", "collapsed_to_1")}),
              "g pre/on", e["g_pre_success"].get("mean"), e["g_success_flag_on"].get("mean"))


if __name__ == "__main__":
    main()

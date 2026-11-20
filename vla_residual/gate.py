"""T036: competence gate g(V_hat) for the residual,  a = a_base + alpha * g * tanh(u).

V_hat = frozen T023 success predictor (vla, Platt) on the VLM layer-17 image feature of the decision state, the same
pi0.5 forward that produced the base chunk (no extra forward, no critic).

Modes (cfg.vla_residual.gate.mode):
  none     gate code path off (E2, bit-identical to before)
  one      g = 1 through the gate code path (bitwise check against `none`)
  fixed    G1: g = sigmoid((tau - V_hat) / temp)
  learned  G2: g = sigmoid(z), z = 8th policy output. Its mean is the linear gate S * (p_v V_hat + p_t t/16 + p_b)
           (vla_residual.rlinf_ext gate head, initialised to G1); train rollouts sample z, evaluation uses the mean.

`python -m vla_residual.gate` picks (tau, temp) from the T017 TRAIN split only (pre-registered rule, see select()).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

VHAT_ALL = Path("/outputs/StochRL/t023/vhat_all.npz")
LABELS = Path("/workspace/stochrl_t018/cache/labels.npz")
OUT = Path("/outputs/StochRL/t036")
ND = 16


def g_fixed(v, tau, temp):
    """numpy or torch."""
    if isinstance(v, np.ndarray):
        return 1.0 / (1.0 + np.exp(-(tau - v) / temp))
    import torch

    return torch.sigmoid((tau - v) / temp)


def linear_init(tau, temp):
    """(w_v, w_t, b) of z = w_v V_hat + w_t t/16 + b with sigmoid(z) == G1."""
    return -1.0 / temp, 0.0, tau / temp


def pav(x, y):
    """Isotonic (non-decreasing) least-squares fit of y on x; returns fitted values in the order of x."""
    o = np.argsort(x, kind="stable")
    ys = y[o].astype(float)
    val, wt, cnt = [], [], []
    for t in ys:
        val.append(t); wt.append(1.0); cnt.append(1)
        while len(val) > 1 and val[-2] > val[-1]:
            w = wt[-2] + wt[-1]
            val[-2] = (val[-2] * wt[-2] + val[-1] * wt[-1]) / w
            wt[-2] = w; cnt[-2] += cnt[-1]
            val.pop(); wt.pop(); cnt.pop()
    fit = np.repeat(val, cnt)
    out = np.empty_like(fit)
    out[o] = fit
    return out


def load_train():
    d = np.load(VHAT_ALL, allow_pickle=True)
    lab = np.load(LABELS, allow_pickle=True)
    assert (lab["episode"].reshape(500, ND)[:, 0] == d["episodes"]).all()
    V = d["vla@platt"].astype(np.float64)
    y = lab["success_once"].reshape(500, ND)[:, 0].astype(bool)
    post = lab["success_before"].reshape(500, ND).astype(bool)
    phase = lab["phase5"].reshape(500, ND).astype(str)
    return V, y, post, phase, d["split"].astype(str)


VHAT_END = OUT / "vhat_end_all.npz"
PRED_ROOT = {"t023": Path("/outputs/StochRL/t023/predictors"), "end": OUT / "predictors"}


def load_train_end():
    """V_hat_end (vla_residual/fit_vhat_end.py): label success_at_end, all decisions. Same return layout."""
    d = np.load(VHAT_END, allow_pickle=True)
    V = d["vhat_end"].astype(np.float64)
    lab = np.load(LABELS, allow_pickle=True)
    ep = lab["episode"].reshape(500, ND)[:, 0]
    idx = {e: i for i, e in enumerate(ep.tolist())}
    rows = [idx[e] for e in d["episodes"].tolist()]
    phase = lab["phase5"].reshape(500, ND).astype(str)[rows]
    return V, d["y_end"].astype(bool), d["post"].astype(bool), phase, d["split"].astype(str)


def select(source="t023", p_lo=0.1, p_hi=0.9, g_open=0.9, g_closed=0.1):
    """Rule (train split, pre-success decisions of the frozen-base buffer only):
    isotonic fit P(base success | V_hat);  v_lo = largest V_hat with P <= p_lo,  v_hi = smallest V_hat with P >= p_hi
    (both rounded to 0.01);  g(v_lo) = g_open, g(v_hi) = g_closed  ->  tau = (v_lo + v_hi) / 2,
    temp = (v_hi - v_lo) / (2 ln(g_open / (1 - g_open)))."""
    if source == "end":  # same rule; population = every train decision, label = success_at_end (the predictor's own)
        V, y, post, phase, split = load_train_end()
        tr = split == "train"
        pre = ~post
        fitm = np.ones_like(pre[tr])
    else:
        V, y, post, phase, split = load_train()
        tr = split == "train"
        pre = ~post
        fitm = pre[tr]
    v_fit = V[tr][fitm]
    y_fit = np.repeat(y[tr][:, None], ND, 1)[fitm].astype(float)
    p_fit = pav(v_fit, y_fit)
    v = V[tr][pre[tr]]
    yy = np.repeat(y[tr][:, None], ND, 1)[pre[tr]].astype(float)
    v_lo = round(float(v_fit[p_fit <= p_lo].max()), 2)
    v_hi = round(float(v_fit[p_fit >= p_hi].min()), 2)
    assert abs(g_closed - (1 - g_open)) < 1e-12
    tau = (v_lo + v_hi) / 2
    temp = (v_hi - v_lo) / (2 * math.log(g_open / (1 - g_open)))
    g = g_fixed(v, tau, temp)
    Vt = V[tr]
    G = g_fixed(Vt, tau, temp)
    yt, pret, pht = y[tr], pre[tr], phase[tr]
    q = [5, 25, 50, 75, 95]

    def st(x):
        x = np.asarray(x, float)
        return dict(n=int(x.size), mean=float(x.mean()), frac_lt_0p1=float((x < 0.1).mean()),
                    frac_gt_0p9=float((x > 0.9).mean()), q5_25_50_75_95=[float(t) for t in np.percentile(x, q)])

    bins = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001]
    out = dict(
        source=source,
        rule=dict(p_lo=p_lo, p_hi=p_hi, g_open=g_open, g_closed=g_closed, data="T017 buffer, split=train (350 ep); " + (
            "all decisions, label success_at_end; V_hat_end (vhat_end_all.npz)" if source == "end" else
            "pre-success decisions, label success_once; V_hat = T023 vla@platt (vhat_all.npz)")),
        n_train_episodes=int(tr.sum()), n_decisions=int(v_fit.size), train_success_rate=float(yt.mean()),
        v_lo=v_lo, v_hi=v_hi, tau=tau, temp=temp,
        frac_states_v_ge_hi=float((v_fit >= v_hi).mean()), frac_states_v_le_lo=float((v_fit <= v_lo).mean()),
        success_rate_v_ge_hi=float(y_fit[v_fit >= v_hi].mean()), success_rate_v_le_lo=float(y_fit[v_fit <= v_lo].mean()),
        reliability=[dict(lo=a, hi=min(b, 1.0), n=int(((v_fit >= a) & (v_fit < b)).sum()),
                          success=float(y_fit[(v_fit >= a) & (v_fit < b)].mean()) if ((v_fit >= a) & (v_fit < b)).any() else None)
                     for a, b in zip(bins[:-1], bins[1:])],
        g_train_pre_all=st(g), g_train_pre_success_eps=st(g[yy == 1]), g_train_pre_fail_eps=st(g[yy == 0]),
        g_train_post_success_states=st(G[~pret]), vhat_train_post_success_states=st(Vt[~pret]),
        vhat_train_pre_success_eps=st(v[yy == 1]), vhat_train_pre_fail_eps=st(v[yy == 0]),
        g_by_decision=[dict(dec=k, success_eps_pre=float(G[yt, k][pret[yt, k]].mean()) if pret[yt, k].any() else None,
                            fail_eps=float(G[~yt, k].mean())) for k in range(ND)],
        g_by_phase5={ph: st(G[pht == ph]) for ph in sorted(set(pht.reshape(-1)))},
    )
    return out


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "t023"
    r = select(src)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("gate_select.json" if src == "t023" else f"gate_select_{src}.json")).write_text(json.dumps(r, indent=1))
    keep = ("v_lo", "v_hi", "tau", "temp", "frac_states_v_ge_hi", "frac_states_v_le_lo", "success_rate_v_ge_hi",
            "success_rate_v_le_lo")
    print(json.dumps({k: r[k] for k in keep}))
    for k in ("g_train_pre_all", "g_train_pre_success_eps", "g_train_pre_fail_eps", "g_train_post_success_states"):
        print(k, {a: round(b, 3) for a, b in r[k].items() if a != "q5_25_50_75_95"})
    print({k: round(v["mean"], 3) for k, v in r["g_by_phase5"].items()})

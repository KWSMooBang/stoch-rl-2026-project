"""T026 analysis: oracle gap for VLA-only vs VLA+perturbation candidate sets (pre-registered, t026_prereg.md).

Sets (same state, 4 CRN replicates per candidate):
  VLA4 = cand 0-3, VLA8 = cand 0-7, MIX8 = cand 0-3 + perturbations 8-11 (equal N and replicates as VLA8).
Per set & state: naive = max p_hat - mean p_hat; null = same after permuting the set's outcomes within the state;
excess = naive - null; split-half = choose argmax on reps {0,1}, score on {2,3} (and swap) minus set mean on the
scoring half; s_b^2 = var_i(p_hat) - mean p_hat(1-p_hat)/(m-1); beta-binomial E[max p]-E[p] with the pooled ICC.
Primary: delta excess (MIX8 - VLA8), paired state bootstrap.

usage: .venv/bin/python -m repcritic.t026_analyze --out /outputs/StochRL/t026
       .venv/bin/python -m repcritic.t026_analyze --restore_test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from repcritic.t026_branch import NC, NC_VLA, NREP, OUT, PERTS

SETS = {"VLA4": [0, 1, 2, 3], "VLA8": list(range(8)), "MIX8": [0, 1, 2, 3, 8, 9, 10, 11]}
STRATA = ["A0_init", "A1_far", "A2_near", "C_contact", "L_lost", "T_transport"]


def load(metric="success_once", path=None):
    rows = [json.loads(l) for l in open(path or OUT / "branch_results_main.jsonl")]
    sel = json.loads((OUT / "state_select.json").read_text())["states"]
    ids = [s["state_id"] for s in sel]
    Y = np.full((len(ids), NC, NREP), np.nan)
    pos = {s: i for i, s in enumerate(ids)}
    for r in rows:
        Y[pos[r["state_id"]], r["cand"], r["rep"]] = float(r[metric])
    keep = ~np.isnan(Y).reshape(len(ids), -1).any(1)
    return rows, [s for s, k in zip(sel, keep) if k], Y[keep]


def per_state(Y, rng, nperm=1000, reps=None):
    """Y [S, n, m] -> dict of per-state arrays."""
    if reps is not None:
        Y = Y[:, :, reps]
    S, n, m = Y.shape
    ph = Y.mean(2)
    naive = ph.max(1) - ph.mean(1)
    flat = Y.reshape(S, -1)
    null = np.zeros(S)
    for _ in range(nperm):
        perm = np.argsort(rng.rand(S, n * m), axis=1)
        Yp = np.take_along_axis(flat, perm, 1).reshape(S, n, m).mean(2)
        null += (Yp.max(1) - Yp.mean(1)) / nperm
    # CRN-aware null (sensitivity, added after seeing CRN correlation): permute candidate labels WITHIN each
    # replicate column, which keeps the shared-continuation (replicate) effect.
    null_rep = np.zeros(S)
    for _ in range(nperm):
        perm = np.argsort(rng.rand(S, n, m), axis=1)
        Yp = np.take_along_axis(Y, perm, 1).mean(2)
        null_rep += (Yp.max(1) - Yp.mean(1)) / nperm
    out = dict(p_mean=ph.mean(1), naive=naive, null=null, excess=naive - null, max_p=ph.max(1),
               null_rep=null_rep, excess_rep=naive - null_rep)
    if m > 1:  # two-way (candidate x replicate) variance component of candidates
        resid = Y - Y.mean(2, keepdims=True) - Y.mean(1, keepdims=True) + Y.mean((1, 2), keepdims=True)
        ms_e = (resid ** 2).sum((1, 2)) / ((n - 1) * (m - 1))
        ms_c = m * ph.var(1, ddof=1)
        out["var_between_2way"] = (ms_c - ms_e) / m
    if m >= 2:
        h = m // 2
        A, B = list(range(h)), list(range(h, 2 * h))

        def sh(a, b):
            pa = Y[:, :, a].mean(2) + rng.rand(S, n) * 1e-6
            i = pa.argmax(1)
            pb = Y[:, :, b].mean(2)
            return pb[np.arange(S), i] - pb.mean(1), pb[np.arange(S), i]
        g1, v1 = sh(A, B)
        g2, v2 = sh(B, A)
        out["split_gain"] = 0.5 * (g1 + g2)
        out["split_value"] = 0.5 * (v1 + v2)
    out["var_between"] = ph.var(1, ddof=1) - (ph * (1 - ph)).mean(1) / (m - 1) if m > 1 else np.zeros(S)
    return out


def icc(ps):
    pb = ps["p_mean"]
    tv = pb * (1 - pb)
    mx = tv > 0
    return float(ps["var_between"][mx].sum() / tv[mx].sum()) if mx.any() else float("nan")


def G_betabinom(pbar, rho, N, rng, ndraw=4000):
    g = []
    for p in pbar:
        if p <= 0 or p >= 1 or not rho > 0:
            g.append(0.0)
            continue
        k = 1 / rho - 1
        x = rng.beta(p * k, (1 - p) * k, size=(ndraw, N))
        g.append(float((x.max(1) - x.mean(1)).mean()))
    return float(np.mean(g))


def ci(v, rng, nboot=2000, idx_list=None):
    S = len(v)
    bs = [v[rng.randint(0, S, S)].mean() for _ in range(nboot)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def summarize(Y, rng, reps=None, nboot=2000, with_bb=True):
    res = {}
    P = {k: per_state(Y[:, c], rng, reps=reps) for k, c in SETS.items()}
    for k, ps in P.items():
        d = {m: float(ps[m].mean()) for m in ps}
        d["ci"] = {m: ci(ps[m], rng, nboot) for m in ("excess", "excess_rep", "naive", "split_gain", "split_value", "max_p", "p_mean")
                   if m in ps}
        d["icc"] = icc(ps)
        if "var_between_2way" in ps:
            d["icc_2way"] = icc(dict(ps, var_between=ps["var_between_2way"]))
        if with_bb:
            d["G_betabinom"] = G_betabinom(ps["p_mean"], d["icc"], len(SETS[k]), rng, 2000)
            d["G_betabinom_2way"] = G_betabinom(ps["p_mean"], d["icc_2way"], len(SETS[k]), rng, 2000)
            # bootstrap CI of G (resample states, recompute pooled ICC)
            gs = []
            for _ in range(100):
                idx = rng.randint(0, len(ps["p_mean"]), len(ps["p_mean"]))
                sub = {q: v[idx] for q, v in ps.items()}
                gs.append(G_betabinom(sub["p_mean"], icc(sub), len(SETS[k]), rng, 500))
            d["ci"]["G_betabinom"] = [float(np.percentile(gs, 2.5)), float(np.percentile(gs, 97.5))]
        res[k] = d
    for a, b in (("MIX8", "VLA8"), ("MIX8", "VLA4"), ("VLA8", "VLA4")):
        dd = {}
        for m in ("excess", "excess_rep", "naive", "split_gain", "split_value", "var_between", "var_between_2way", "max_p", "p_mean"):
            if m in P[a]:
                v = P[a][m] - P[b][m]
                dd[m] = dict(mean=float(v.mean()), ci=ci(v, rng, nboot))
        res[f"{a}-{b}"] = dd
    return res, P


def pert_vs_base(Y, states, rng):
    """p_hat(pert) - p_hat(base cand 0) and - mean VLA8, per direction, overall and per stratum."""
    ph = Y.mean(2)
    base = ph[:, 0]
    vla = ph[:, :NC_VLA].mean(1)
    out = {}
    strata = np.array([s["stratum"] for s in states])
    for j, name in enumerate(PERTS):
        c = NC_VLA + j
        d0 = ph[:, c] - base
        dv = ph[:, c] - vla
        e = dict(p=float(ph[:, c].mean()), minus_base=dict(mean=float(d0.mean()), ci=ci(d0, rng)),
                 minus_vla8=dict(mean=float(dv.mean()), ci=ci(dv, rng)),
                 n_clear_better=int(((Y[:, c].sum(1) == NREP) & (Y[:, 0].sum(1) == 0)).sum()),
                 n_clear_worse=int(((Y[:, c].sum(1) == 0) & (Y[:, 0].sum(1) == NREP)).sum()),
                 by_stratum={k: float(d0[strata == k].mean()) for k in STRATA if (strata == k).any()})
        out[name] = e
    # best perturbation per state vs base (selection over 4 perts: biased up; reported with split-half)
    S = len(Y)
    A, B = [0, 1], [2, 3]
    gains = []
    for a, b in ((A, B), (B, A)):
        pa = Y[:, NC_VLA:, :][:, :, a].mean(2) + rng.rand(S, len(PERTS)) * 1e-6
        i = pa.argmax(1)
        pb = Y[:, NC_VLA:, :][:, :, b].mean(2)
        gains.append(pb[np.arange(S), i] - Y[:, 0, b].mean(1))
    g = 0.5 * (gains[0] + gains[1])
    out["best_pert_split_minus_base"] = dict(mean=float(g.mean()), ci=ci(g, rng),
                                              by_stratum={k: float(g[strata == k].mean()) for k in STRATA if (strata == k).any()})
    # VLA analogue: best of cand 1..7 (split) vs base
    gains = []
    for a, b in ((A, B), (B, A)):
        pa = Y[:, 1:NC_VLA, :][:, :, a].mean(2) + rng.rand(S, NC_VLA - 1) * 1e-6
        i = pa.argmax(1)
        pb = Y[:, 1:NC_VLA, :][:, :, b].mean(2)
        gains.append(pb[np.arange(S), i] - Y[:, 0, b].mean(1))
    g = 0.5 * (gains[0] + gains[1])
    out["best_vla7_split_minus_base"] = dict(mean=float(g.mean()), ci=ci(g, rng))
    return out


def restore_test():
    rows = [json.loads(l) for l in open(OUT / "branch_results_restore_test.jsonl")]
    by = {}
    for r in rows:
        by.setdefault((r["state_id"], r["rep"]), []).append(r["success_once"])
    agree = []
    for k, v in by.items():
        v = np.array(v, float)
        agree.append(max(v.mean(), 1 - v.mean()))
    cands = {}
    for sid in sorted({k[0] for k in by}):
        z = np.load(OUT / "candidates" / f"{sid}.npz")
        cands[sid] = dict(cand0_recompute_maxdiff=float(z["cand0_recompute_maxdiff"]),
                          restored_img_meanabsdiff=float(z["restored_img_meanabsdiff"]),
                          realized_shift=z["realized_shift"].round(4).tolist(), dir_fallback=bool(z["dir_fallback"]),
                          per_rep_success={str(r): float(np.mean(by[(sid, r)])) for r in range(NREP) if (sid, r) in by})
    out = dict(n_groups=len(by), copies_per_group=len(next(iter(by.values()))), mean_majority_agreement=float(np.mean(agree)),
               min_majority_agreement=float(np.min(agree)), states=cands)
    (OUT / "restore_test.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/outputs/StochRL/t026")
    ap.add_argument("--restore_test", action="store_true")
    args = ap.parse_args()
    if args.restore_test:
        restore_test()
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    rows, states, Y = load()
    S = len(states)
    res = dict(n_states=S, n_branches=len(rows))
    res["main"], P = summarize(Y, rng)
    res["two_rep_subset"], _ = summarize(Y, rng, reps=[0, 1], with_bb=False)
    res["success_at_end"], _ = summarize(load("success_at_end")[2], rng, with_bb=False)
    res["pert_vs_base"] = pert_vs_base(Y, states, rng)
    strata = np.array([s["stratum"] for s in states])
    res["by_stratum"] = {}
    for k in STRATA + ["approach_all", "non_approach"]:
        m = (strata == k) if k in STRATA else (np.isin(strata, ["A0_init", "A1_far", "A2_near"]) if k == "approach_all"
                                                else ~np.isin(strata, ["A0_init", "A1_far", "A2_near"]))
        if m.sum() == 0:
            continue
        e = dict(S=int(m.sum()))
        for name in SETS:
            e[name] = {q: float(P[name][q][m].mean()) for q in ("p_mean", "excess", "excess_rep", "split_gain", "split_value", "max_p", "var_between_2way")}
        dv = P["MIX8"]["excess"][m] - P["VLA8"]["excess"][m]
        e["d_excess_MIX8_VLA8"] = dict(mean=float(dv.mean()), ci=ci(dv, rng, 1000))
        ds = P["MIX8"]["split_value"][m] - P["VLA8"]["split_value"][m]
        e["d_split_value_MIX8_VLA8"] = dict(mean=float(ds.mean()), ci=ci(ds, rng, 1000))
        res["by_stratum"][k] = e
    # CRN check: within-rep correlation of outcomes across candidates vs across reps
    ph = Y.mean(1)  # [S, m] rep mean over candidates
    res["crn"] = dict(var_rep_means_within_state=float(ph.var(1, ddof=1).mean()),
                      expected_if_independent=float((Y.mean((1, 2)) * (1 - Y.mean((1, 2)))).mean() / NC))
    # pre-registered decision
    d = res["main"]["MIX8-VLA8"]
    go = d["excess"]["ci"][0] > 0 and d["split_value"]["mean"] > 0
    res["decision"] = dict(primary_delta_excess=d["excess"], delta_split_value=d["split_value"],
                           perturbation_widens_gap=bool(go))
    (out / "analysis.json").write_text(json.dumps(res, indent=1))

    # ---- tables ----
    import pandas as pd
    df = pd.DataFrame(rows)
    df = df.drop(columns=["afterK"]).join(pd.json_normalize(df["afterK"]).add_prefix("afterK_"))
    for c in df.columns:
        if df[c].map(lambda x: isinstance(x, list)).any():
            df[c] = df[c].map(lambda x: json.dumps(x) if isinstance(x, list) else x)
    df.to_parquet(out / "branch_rollouts.parquet", index=False)
    phat = Y.mean(2)
    rk = []
    for i, s in enumerate(states):
        e = dict(state_id=s["state_id"], stratum=s["stratum"], phase=s["phase"], t=s["t"],
                 gripper_obj_dist=round(s["gripper_obj_dist"], 4), episode_success_once=s["episode_success_once"])
        for name in SETS:
            e[f"{name}_pmean"] = round(float(P[name]["p_mean"][i]), 3)
            e[f"{name}_excess"] = round(float(P[name]["excess"][i]), 3)
        e["dExcess_MIX8_VLA8"] = round(float(P["MIX8"]["excess"][i] - P["VLA8"]["excess"][i]), 3)
        for c in range(NC):
            e[f"p_c{c}" if c < NC_VLA else f"p_{PERTS[c - NC_VLA]}"] = float(phat[i, c])
        e["best_cand"] = int(phat[i].argmax())
        rk.append(e)
    pd.DataFrame(rk).sort_values("MIX8_excess", ascending=False).to_csv(out / "ranking_by_gap.csv", index=False)
    plot(res, P, strata, Y, out / "oracle_headroom.png")
    show = {k: res[k] for k in ("n_states", "n_branches", "decision", "crn")}
    show["sets"] = {k: {q: round(res["main"][k][q], 4) for q in ("p_mean", "naive", "null", "excess", "null_rep", "excess_rep", "split_gain",
                                                                  "split_value", "max_p", "icc", "icc_2way", "G_betabinom", "G_betabinom_2way")}
                    for k in SETS}
    print(json.dumps(show, indent=1))


def plot(res, P, strata, Y, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"VLA4": "#9aa5b1", "VLA8": "#3b6ea8", "MIX8": "#d9822b"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    groups = ["all", "approach_all", "non_approach"] + STRATA
    x = np.arange(len(groups))
    w = 0.26
    for j, name in enumerate(SETS):
        vals, lo, hi = [], [], []
        for g in groups:
            if g == "all":
                v = res["main"][name]["excess"]
                c = res["main"][name]["ci"]["excess"]
            else:
                e = res["by_stratum"].get(g)
                v = e[name]["excess"] if e else np.nan
                c = [v, v]
            vals.append(v); lo.append(v - c[0]); hi.append(c[1] - v)
        ax[0].bar(x + (j - 1) * w, vals, w, color=col[name], label=name, ecolor="k", yerr=[[l if i == 0 else 0 for i, l in enumerate(lo)], [h if i == 0 else 0 for i, h in enumerate(hi)]],
                  capsize=2, error_kw=dict(lw=0.8))
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"{g}\n(n={res['n_states'] if g == 'all' else res['by_stratum'].get(g, {}).get('S', 0)})"
                           for g in groups], fontsize=7, rotation=35, ha="right")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_title("noise-corrected oracle gap (excess = naive − perm null)\nCI only for 'all'", fontsize=9)
    ax[0].legend(fontsize=8, frameon=False)
    # per-perturbation p - base
    pv = res["pert_vs_base"]
    names = list(PERTS) + ["best pert (split)", "best VLA 1-7 (split)"]
    m = [pv[n]["minus_base"]["mean"] for n in PERTS] + [pv["best_pert_split_minus_base"]["mean"],
                                                         pv["best_vla7_split_minus_base"]["mean"]]
    c = [pv[n]["minus_base"]["ci"] for n in PERTS] + [pv["best_pert_split_minus_base"]["ci"],
                                                       pv["best_vla7_split_minus_base"]["ci"]]
    xx = np.arange(len(names))
    ax[1].bar(xx, m, color=["#d9822b"] * 5 + ["#3b6ea8"], yerr=[[a - b[0] for a, b in zip(m, c)],
                                                                    [b[1] - a for a, b in zip(m, c)]], capsize=3)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xticks(xx)
    ax[1].set_xticklabels(names, fontsize=8, rotation=20)
    ax[1].set_title("p(candidate) − p(base cand 0), 95% CI", fontsize=9)
    # per-state scatter VLA8 excess vs MIX8 excess
    ax[2].scatter(P["VLA8"]["excess"], P["MIX8"]["excess"], s=14, c=[STRATA.index(s) for s in strata], cmap="tab10",
                  vmin=0, vmax=9)
    lim = [min(P["VLA8"]["excess"].min(), P["MIX8"]["excess"].min()) - 0.02,
           max(P["VLA8"]["excess"].max(), P["MIX8"]["excess"].max()) + 0.02]
    ax[2].plot(lim, lim, "k--", lw=0.6)
    ax[2].set_xlabel("VLA8 excess (per state)")
    ax[2].set_ylabel("MIX8 excess (per state)")
    ax[2].set_title("per state (colour = stratum)", fontsize=9)
    for i, s in enumerate(STRATA):
        ax[2].scatter([], [], color=plt.cm.tab10(i / 9 if False else i), label=s, s=14)
    ax[2].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()

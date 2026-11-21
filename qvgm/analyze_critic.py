"""T040 critic evaluation (prereg 1.5): out-of-fold ranking / gradient alignment / calibration vs action-only and V(z).

  python -m qvgm.analyze_critic --out /outputs/StochRL/t040
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def roc_auc_score(y, s):
    y = np.asarray(y, bool); r = rankdata(s)
    npos, nneg = y.sum(), (~y).sum()
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))

ROOT = Path("/workspace/stochrl_t040")
NB = 2000


def load(tag=""):
    D = dict(np.load(ROOT / "data.npz"))
    n = len(D["br_G"])
    P = {k: np.zeros(n) for k in ("Qf", "Qa", "V", "Qf_std")}
    P["gf"], P["ga"] = np.zeros((n, 35)), np.zeros((n, 35))
    tV, tQ, cnt = 0, 0, 0
    for f in range(5):
        m = D["br_fold"] == f
        seeds = [np.load(ROOT / f"critic{tag}/oof_{f}_s{s}.npz") for s in (0, 1)]
        for k in P:
            P[k][m] = np.mean([o[k][m] for o in seeds], 0)
        for o in seeds:
            tV, tQ, cnt = tV + o["test_V"], tQ + o["test_Q"], cnt + 1
    return D, P, tV / cnt, tQ / cnt


def build_states(D, P):
    S = []
    for sid in np.unique(D["br_sid"]):
        idx = np.where(D["br_sid"] == sid)[0]
        names = D["br_name"][idx]
        keep = idx[names != "u0_ind"]
        base = idx[D["br_is_base"][idx]][0]
        S.append(dict(sid=sid, src=D["br_src"][base], idx=keep, base=base,
                      p=D["br_succ"][keep].mean(1), succ=D["br_succ"][keep], pe=D["br_end"][keep].mean(1)))
    return S


def pair_stats(S, score, use="p"):
    """Per state: (n_correct, n_pairs) over candidate pairs with different empirical success."""
    out = []
    for s in S:
        q, p = score[s["idx"]], s[use]
        i, j = np.triu_indices(len(q), 1)
        m = p[i] != p[j]
        dq, dp = q[i][m] - q[j][m], p[i][m] - p[j][m]
        c = np.where(dq == 0, 0.5, (np.sign(dq) == np.sign(dp)).astype(float))
        out.append((c.sum(), m.sum()))
    return np.array(out, float)


def auc_stats(S, score):
    """Within-state AUROC over (cand, rep) outcomes: (correct pos>neg pairs, pos*neg)."""
    out = []
    for s in S:
        q = np.repeat(score[s["idx"]], 4)
        y = s["succ"].reshape(-1)
        pos, neg = q[y], q[~y]
        if len(pos) == 0 or len(neg) == 0:
            out.append((0.0, 0.0)); continue
        d = pos[:, None] - neg[None]
        out.append(((d > 0).sum() + 0.5 * (d == 0).sum(), d.size))
    return np.array(out, float)


def split_half(S):
    """Oracle-ish ceiling: p from reps (0,1) ranks pairs by p from reps (2,3), and vice versa."""
    out = []
    for s in S:
        a, b = s["succ"][:, :2].mean(1), s["succ"][:, 2:].mean(1)
        cs, ns = 0.0, 0.0
        for x, y in ((a, b), (b, a)):
            i, j = np.triu_indices(len(x), 1)
            m = y[i] != y[j]
            dx, dy = x[i][m] - x[j][m], y[i][m] - y[j][m]
            cs += np.where(dx == 0, 0.5, (np.sign(dx) == np.sign(dy)).astype(float)).sum(); ns += m.sum()
        out.append((cs / 2, ns / 2))
    return np.array(out, float)


def align_rows(D, P, S, g_key):
    """Per (state, cand != base): s_c = g(A_base).(A_c - A_base), Delta_c = paired CRN success diff, FD dQ."""
    rows = []
    for k, s in enumerate(S):
        b = s["base"]
        g = P[g_key][b]
        qk = "Qf" if g_key == "gf" else "Qa"
        for r in s["idx"]:
            if r == b:
                continue
            rows.append((k, float(g @ (D["br_A"][r] - D["br_A"][b])),
                         float((D["br_succ"][r].astype(float) - D["br_succ"][b]).mean()),
                         float(P[qk][r] - P[qk][b]),
                         float((D["br_end"][r].astype(float) - D["br_end"][b]).mean())))
    return np.array(rows)


def spear(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(spearmanr(x, y).correlation)


def boot(S, fn, nb=NB, seed=0):
    rng = np.random.default_rng(seed)
    n = len(S)
    return np.array([fn(rng.integers(0, n, n)) for _ in range(nb)])


def ci(v):
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/outputs/StochRL/t040")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D, P, tV, tQ = load(a.tag)
    S = build_states(D, P)
    n = len(S)
    rng0 = np.random.default_rng(1)
    rand_score = rng0.normal(size=len(D["br_G"]))
    scores = {"full": P["Qf"], "action_only": P["Qa"], "state_only_V": P["V"], "random": rand_score}
    PA = {k: pair_stats(S, v) for k, v in scores.items()}
    PAe = {k: pair_stats(S, v, "pe") for k, v in scores.items()}
    AU = {k: auc_stats(S, v) for k, v in scores.items()}
    SH = split_half(S)
    AL = {"full": align_rows(D, P, S, "gf"), "action_only": align_rows(D, P, S, "ga")}
    ratio = lambda X, ix: X[ix, 0].sum() / max(X[ix, 1].sum(), 1e-9)  # noqa: E731

    def al_fn(key, col, ix):
        R = AL[key]
        sel = np.concatenate([np.where(R[:, 0] == i)[0] for i in ix])
        return spear(R[sel, col], R[sel, 2])

    def al_end(key, ix):
        R = AL[key]
        sel = np.concatenate([np.where(R[:, 0] == i)[0] for i in ix])
        return spear(R[sel, 1], R[sel, 4])

    full_ix = np.arange(n)
    rows = []

    def add(metric, model, fn, diff_with=None):
        est = fn(full_ix)
        bs = boot(S, fn)
        r = dict(metric=metric, model=model, estimate=est, ci_lo=ci(bs)[0], ci_hi=ci(bs)[1], n_states=n)
        rows.append(r)
        return bs

    bsPA = {k: add("pairwise_acc_once", k, lambda ix, k=k: ratio(PA[k], ix)) for k in scores}
    bsPAe = {k: add("pairwise_acc_end", k, lambda ix, k=k: ratio(PAe[k], ix)) for k in scores}
    add("pairwise_acc_once", "split_half_oracle", lambda ix: ratio(SH, ix))
    bsAU = {k: add("within_state_auroc", k, lambda ix, k=k: ratio(AU[k], ix)) for k in scores}
    bsAL = {k: add("grad_align_spearman", k, lambda ix, k=k: al_fn(k, 1, ix)) for k in AL}
    bsFD = {k: add("fd_dQ_spearman", k, lambda ix, k=k: al_fn(k, 3, ix)) for k in AL}
    add("grad_align_spearman_end", "full", lambda ix: al_end("full", ix))
    add("grad_align_spearman_end", "action_only", lambda ix: al_end("action_only", ix))
    # paired differences (same bootstrap seed -> same resamples)
    for met, bsd, fn in (("pairwise_acc_once", bsPA, lambda k, ix: ratio(PA[k], ix)),
                         ("pairwise_acc_end", bsPAe, lambda k, ix: ratio(PAe[k], ix)),
                         ("within_state_auroc", bsAU, lambda k, ix: ratio(AU[k], ix)),
                         ("grad_align_spearman", bsAL, lambda k, ix: al_fn(k, 1, ix)),
                         ("fd_dQ_spearman", bsFD, lambda k, ix: al_fn(k, 3, ix))):
        d = bsd["full"] - bsd["action_only"]
        rows.append(dict(metric=met, model="full_minus_action_only",
                         estimate=fn("full", full_ix) - fn("action_only", full_ix), ci_lo=ci(d)[0], ci_hi=ci(d)[1],
                         n_states=n))
    # calibration
    G = D["br_G"]
    allsucc = D["br_succ"]
    for nm, sc in (("full", P["Qf"]), ("action_only", P["Qa"]), ("state_only_V", P["V"])):
        bins = np.clip((sc * 10).astype(int), 0, 9)
        ece = sum(abs(sc[bins == k].mean() - G[bins == k].mean()) * (bins == k).mean() for k in range(10)
                  if (bins == k).any())
        auc = roc_auc_score(allsucc.reshape(-1), np.repeat(sc, 4))
        rows.append(dict(metric="calib_ece_vs_G", model=nm, estimate=float(ece), n_states=n))
        rows.append(dict(metric="calib_mse_vs_G", model=nm, estimate=float(((sc - G) ** 2).mean()), n_states=n))
        rows.append(dict(metric="cross_state_auroc_branch", model=nm, estimate=float(auc), n_states=n))
        rows.append(dict(metric="mean_pred_minus_mean_G", model=nm, estimate=float(sc.mean() - G.mean()), n_states=n))
    # T017 test episodes (never trained on): V(z), Q(z, A_logged) vs episode success, within decision index
    tm = D["buf_split"] == "test"
    y = D["buf_ep_succ"][tm]; dd = D["buf_dec"][tm]
    for nm, sc in (("state_only_V", tV), ("full_logged", tQ)):
        num = den = 0.0
        for d in np.unique(dd):
            m = dd == d
            if y[m].all() or (~y[m]).all():
                continue
            w = y[m].sum() * (~y[m]).sum()
            num += roc_auc_score(y[m], sc[m]) * w; den += w
        rows.append(dict(metric="t017_test_auroc_same_dec", model=nm, estimate=num / den, n_states=int(tm.sum())))
    # scale diagnostics
    within = np.mean([np.std(P["Qf"][s["idx"]]) for s in S]); across = np.std([P["Qf"][s["base"]] for s in S])
    rows.append(dict(metric="Qf_within_state_sd", model="full", estimate=float(within)))
    rows.append(dict(metric="Qf_across_state_sd_base", model="full", estimate=float(across)))
    rows.append(dict(metric="Qa_within_state_sd", model="action_only",
                     estimate=float(np.mean([np.std(P["Qa"][s["idx"]]) for s in S]))))
    rows.append(dict(metric="ensemble_sd_mean", model="full", estimate=float(P["Qf_std"].mean())))
    rows.append(dict(metric="n_pairs_once", model="-", estimate=float(PA["full"][:, 1].sum())))
    rows.append(dict(metric="n_align_rows", model="-", estimate=float(len(AL["full"]))))
    # by source (exploratory)
    for src in ("t026", "t031"):
        ix = np.array([i for i, s in enumerate(S) if s["src"] == src])
        for k in ("full", "action_only"):
            rows.append(dict(metric=f"pairwise_acc_once[{src}]", model=k, estimate=ratio(PA[k], ix), n_states=len(ix)))
            rows.append(dict(metric=f"grad_align_spearman[{src}]", model=k, estimate=al_fn(k, 1, ix), n_states=len(ix)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "critic_eval.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df.round(4).to_string())
    # gate summary
    g = df.set_index(["metric", "model"])
    c1 = any(g.loc[(m, "full_minus_action_only"), "ci_lo"] > 0 for m in ("pairwise_acc_once", "grad_align_spearman"))
    (out / "critic_gate.json").write_text(json.dumps(dict(condition1=bool(c1)), indent=1))
    print("condition 1:", c1)


if __name__ == "__main__":
    main()

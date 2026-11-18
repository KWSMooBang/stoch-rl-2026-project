"""E02 analysis: multi-continuation Oracle (E[max p_i] - E[p_i]) + candidate-label stats.

Oracle states: 8 candidates x 4 continuations. p_hat_i = mean over continuations.
Estimators of the selection headroom  G = E_s[ max_i p_i - mean_i p_i ]:
  naive      : plug-in max_i p_hat_i - mean_i p_hat_i (biased UP by binomial noise of p_hat).
  null       : same statistic after permuting the 32 outcomes across cells within the state
               (H0: all candidates share one p). naive - null = noise-corrected excess.
  split-half : pick argmax_i on continuations {0,1}, score it on {2,3} (and swap); minus the mean of
               {2,3}. Unbiased value of a *realisable* noisy selector (lower bound on G).
  varcomp    : unbiased between-candidate variance  s_b^2 = var_i(p_hat) - mean_i p_hat(1-p_hat)/(m-1);
               G_gauss = E[max of 8 N(0,1)] * sqrt(s_b^2)  (Gaussian approx) ; ICC = s_b^2 / pbar(1-pbar).
CIs: bootstrap over states (one oracle state per episode).

usage: python -m repcritic.e02_analyze [out_json]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("RC_E02_OUT", "/workspace/stochrl_t017"))
EMAX8 = 1.4236  # E[max of 8 iid N(0,1)]


def load():
    states = []
    for name in ("state_select", "state_select_ext"):
        p = OUT / "branch" / f"{name}.json"
        if p.exists():
            for s in json.loads(p.read_text())["states"]:
                states.append(dict(s, sel=name))
    rows = [json.loads(l) for f in sorted((OUT / "branch").glob("branch_results_*shard*.jsonl")) for l in open(f)]
    return {"states": states}, rows


def phase_of(eid, d):
    z = np.load(OUT / "privileged" / f"ep{eid:06d}.npz")
    return str(z["dec_phase"][d])


def oracle_stats(Y, rng, nperm=2000, nboot=2000):
    """Y: [S, 8, 4] binary."""
    S, n, m = Y.shape
    ph = Y.mean(2)
    naive = ph.max(1) - ph.mean(1)
    # permutation null per state
    flat = Y.reshape(S, -1)
    null = np.zeros(S)
    tot_null = np.zeros(nperm)
    for b in range(nperm):
        perm = np.argsort(rng.rand(S, n * m), axis=1)
        Yp = np.take_along_axis(flat, perm, 1).reshape(S, n, m).mean(2)
        g = Yp.max(1) - Yp.mean(1)
        null += g / nperm
        tot_null[b] = g.mean()
    p_perm = float((tot_null >= naive.mean()).mean())
    # split half
    def sh(a, b_):
        pa = Y[:, :, a].mean(2)
        noise = rng.rand(*pa.shape) * 1e-6
        i = (pa + noise).argmax(1)
        pb = Y[:, :, b_].mean(2)
        return pb[np.arange(S), i] - pb.mean(1)
    split = 0.5 * (sh([0, 1], [2, 3]) + sh([2, 3], [0, 1]))
    # variance components
    vb = ph.var(1, ddof=1) - (ph * (1 - ph)).mean(1) / (m - 1)
    pbar = ph.mean(1)
    tv = pbar * (1 - pbar)
    mixed = tv > 0
    out = dict(S=int(S), p_mean=float(pbar.mean()),
               naive=float(naive.mean()), null=float(null.mean()), excess=float((naive - null).mean()),
               perm_p_value=p_perm, split_half=float(split.mean()),
               var_between_mean=float(vb.mean()), G_gauss=float(EMAX8 * np.sqrt(max(vb.mean(), 0))),
               icc_pooled=float(vb[mixed].sum() / tv[mixed].sum()) if mixed.any() else float("nan"),
               n_states_mixed_outcomes=int(mixed.sum()),
               frac_states_all0=float((Y.reshape(S, -1).max(1) == 0).mean()),
               frac_states_all1=float((Y.reshape(S, -1).min(1) == 1).mean()),
               max_p_mean=float(ph.max(1).mean()))
    # bootstrap CIs over states
    bs = {k: [] for k in ("naive", "excess", "split_half", "G_gauss", "p_mean")}
    for _ in range(nboot):
        idx = rng.randint(0, S, S)
        bs["naive"].append(naive[idx].mean())
        bs["excess"].append((naive - null)[idx].mean())
        bs["split_half"].append(split[idx].mean())
        bs["G_gauss"].append(EMAX8 * np.sqrt(max(vb[idx].mean(), 0)))
        bs["p_mean"].append(pbar[idx].mean())
    out["ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
    out["per_state"] = dict(naive=naive.tolist(), null=null.tolist(), split=split.tolist(), var_between=vb.tolist(),
                            p_hat=ph.tolist())
    return out


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "branch" / "analysis.json"
    sel, rows = load()
    states = {s["state_id"]: s for s in sel["states"]}
    rng = np.random.RandomState(0)
    res = dict(n_rows=len(rows))
    # ---- oracle ----
    cells = {}
    for r in rows:
        if r["oracle"] and r["cand"] < 8:
            cells.setdefault(r["state_id"], {})[(r["cand"], r["cont"])] = r
    ids = sorted(k for k, v in cells.items() if len(v) == 32)
    res["oracle_states_complete"] = len(ids)
    for metric in ("success_once", "success_at_end"):
        Y = np.array([[[cells[s][(i, j)][metric] for j in range(4)] for i in range(8)] for s in ids], float)
        o = oracle_stats(Y, rng)
        o["state_ids"] = ids
        res[f"oracle_{metric}"] = o
        # subsets
        ph = {s: phase_of(states[s]["eid"], states[s]["decision"]) for s in ids}
        subsets = {
            "d0": [s for s in ids if states[s]["decision"] == 0],
            "d>0": [s for s in ids if states[s]["decision"] > 0],
            "episode_failed": [s for s in ids if not states[s]["episode_success_once"]],
            "episode_succeeded": [s for s in ids if states[s]["episode_success_once"]],
            "mixed_outcome": [s for s in ids if 0 < Y[ids.index(s)].mean() < 1],
        }
        for p in sorted(set(ph.values())):
            subsets[f"phase_{p}"] = [s for s in ids if ph[s] == p]
        res[f"oracle_{metric}_subsets"] = {}
        for name, ss in subsets.items():
            if len(ss) < 3:
                res[f"oracle_{metric}_subsets"][name] = dict(S=len(ss))
                continue
            Ys = Y[[ids.index(s) for s in ss]]
            o2 = oracle_stats(Ys, rng, nperm=500, nboot=500)
            o2.pop("per_state")
            res[f"oracle_{metric}_subsets"][name] = o2
    # ---- candidate labels (all states, first continuation of each candidate) ----
    lab = {}
    for r in rows:
        if r["cont"] == 0:
            lab.setdefault(r["state_id"], {})[r["cand"]] = r
    full = sorted(k for k, v in lab.items() if len(v) == 16)
    Y16 = np.array([[lab[s][c]["success_once"] for c in range(16)] for s in full], float)
    def passN(y, N):
        from math import comb
        n = len(y); c = int(y.sum())
        return 1.0 if n - c < N else 1 - comb(n - c, N) / comb(n, N)
    res["labels"] = dict(
        n_states=len(full), n_branches=int(len(rows)),
        by_sel={sl: int(sum(states[s]["sel"] == sl for s in full)) for sl in ("state_select", "state_select_ext")},
        p1=float(Y16.mean()), pass4=float(np.mean([passN(y, 4) for y in Y16])),
        pass16=float(np.mean([passN(y, 16) for y in Y16])),
        frac_mixed=float(np.mean([(0 < y.mean() < 1) for y in Y16])),
        n_informative_pairs=int(sum(int(y.sum()) * int(16 - y.sum()) for y in Y16)),
        by_split={sp: dict(n_states=int(sum(states[s]["split"] == sp for s in full)),
                           p1=float(np.mean([Y16[i].mean() for i, s in enumerate(full) if states[s]["split"] == sp])),
                           frac_mixed=float(np.mean([(0 < Y16[i].mean() < 1) for i, s in enumerate(full)
                                                     if states[s]["split"] == sp])))
                  for sp in ("train", "val", "test")},
        cand0_vs_logged=None,
    )
    # cand0 (logged chunk) branch outcome vs logged episode outcome (sim-noise / continuation agreement)
    agree = [lab[s][0]["success_once"] == states[s]["episode_success_once"] for s in full]
    res["labels"]["cand0_branch_vs_logged_episode_agree"] = float(np.mean(agree))
    # restore fidelity
    diffs = []
    for s in full:
        z = np.load(OUT / "branch" / "candidates" / f"{s}.npz")
        diffs.append((float(z["restored_img_meanabsdiff"]), float(z["cand0_recompute_maxdiff"])))
    diffs = np.array(diffs)
    res["restore_fidelity"] = dict(img_meanabsdiff_mean=float(diffs[:, 0].mean()), img_meanabsdiff_max=float(diffs[:, 0].max()),
                                   cand0_recompute_maxdiff_median=float(np.median(diffs[:, 1])),
                                   cand0_recompute_maxdiff_max=float(diffs[:, 1].max()))
    out_path.write_text(json.dumps(res, indent=1))
    show = {k: v for k, v in res.items() if not k.startswith("oracle_success")}
    for m in ("success_once", "success_at_end"):
        if f"oracle_{m}" in res:
            show[f"oracle_{m}"] = {k: v for k, v in res[f"oracle_{m}"].items() if k not in ("per_state", "state_ids")}
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    main()

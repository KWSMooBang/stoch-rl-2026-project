"""E04 (T019) report: aggregate runs, bootstrap CIs, prereg selection + gate, noise ceiling, figures.

  python -m repcritic.e04_report --out /outputs/StochRL/t019
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd

OUT = "/workspace/stochrl_t019"
E07_OK = lambda inp: inp != "AECand"
SCAL = ["conc_1c", "conc_or", "pick_gain_1c", "pick_gain_or", "auroc_br", "auroc_buf", "auroc_buf_within_t",
        "within_var_frac", "ece_br", "brier_br", "ece_buf", "brier_buf",
        "ood_gauss0.5_win", "ood_gauss0.5_z", "ood_gauss1.0_win", "ood_gauss1.0_z", "ood_gauss2.0_win", "ood_gauss2.0_z",
        "ood_uniform_win", "ood_uniform_z", "ood_otherstate_win", "ood_otherstate_z",
        "grad_finite_frac", "grad_norm_med", "grad_norm_p90", "grad_share_trans", "grad_share_rot", "grad_share_grip",
        "fd_relerr_med", "grad_cos_local", "grad_align_1c", "grad_align_or", "ascent_dz", "ascent_win",
        "ascent_clip_frac"]


def load_runs():
    rows = []
    for f in glob.glob(f"{OUT}/runs/*.jsonl"):
        for line in open(f):
            rows.append(json.loads(line))
    return rows


def boot_conc(parts_list, n=2000, seed=0):
    """parts_list: per seed dict state->(num,den). Seed-averaged concordance + state-bootstrap 95% CI."""
    states = sorted(parts_list[0].keys())
    N = np.array([[p[s][0] for s in states] for p in parts_list])  # [seeds, states]
    Dn = np.array([[p[s][1] for s in states] for p in parts_list])
    point = float(np.mean(N.sum(1) / Dn.sum(1)))
    rs = np.random.RandomState(seed)
    bs = []
    for _ in range(n):
        ii = rs.randint(0, len(states), len(states))
        bs.append(np.mean(N[:, ii].sum(1) / np.maximum(Dn[:, ii].sum(1), 1)))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return point, float(lo), float(hi)


def ensemble_conc(rs, D, sp, n=2000):
    """z-score each seed's scores within split, average -> concordance + CI (single-cont and oracle)."""
    m = D["b_split"] == sp
    st, y, nc = D["b_state"][m], D["b_y"][m], D["b_ncont"][m]
    S = np.mean([(np.array(r[f"{sp}_scores_br"]) - np.mean(r[f"{sp}_scores_br"])) / (np.std(r[f"{sp}_scores_br"]) + 1e-9)
                 for r in rs], 0)
    from repcritic.e04_critic import conc_parts
    out = {}
    for tag, mm in (("1c", np.ones(len(st), bool)), ("or", nc >= 4)):
        p = conc_parts(S[mm], y[mm], st[mm])
        out[tag] = boot_conc([p], n)
    gains = [y[st == g][np.argmax(S[st == g])] - y[st == g].mean() for g in np.unique(st)]
    out["pick_gain_1c"] = float(np.mean(gains))
    return out


def noise_ceiling(D, sp="test", icc=0.21, n_sim=300, seed=0):
    """Concordance with SINGLE-continuation labels of a critic that knows the true p_i exactly (beta-binomial, ICC from
    T017 oracle). State means from the 16 single-cont labels (shrunk toward the global mean)."""
    m = D["b_split"] == sp
    st, y = D["b_state"][m], D["b_y"][m]
    rs = np.random.RandomState(seed)
    k = 1 / icc - 1
    gm = y.mean()
    res, gains = [], []
    states = np.unique(st)
    mus = np.array([(y[st == g].sum() + 2 * gm) / (16 + 2) for g in states])
    for _ in range(n_sim):
        num = den = 0.0
        gsum = []
        for mu in mus:
            p = rs.beta(max(mu * k, 1e-3), max((1 - mu) * k, 1e-3), 16)
            yy = (rs.rand(16) < p).astype(float)
            ds = p[:, None] - p[None, :]
            pm = (yy[:, None] - yy[None, :]) > 0
            num += (ds[pm] > 0).sum() + 0.5 * (ds[pm] == 0).sum()
            den += pm.sum()
            gsum.append(yy[np.argmax(p)] - yy.mean())
        res.append(num / den)
        gains.append(np.mean(gsum))
    return float(np.mean(res)), float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5)), float(np.mean(gains))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/outputs/StochRL/t019")
    args = ap.parse_args()
    os.makedirs(f"{args.out}/fig", exist_ok=True)
    D = dict(np.load(f"{OUT}/data.npz"))
    runs = load_runs()
    seeds = pd.DataFrame([{**{k: r[k] for k in ("name", "input", "obj", "regime", "nce", "seed", "frac", "train_s",
                                                  "peak_mb", "best_step", "n_train_buf", "n_train_br")},
                           **{f"{sp}_{k}": r.get(f"{sp}_{k}", np.nan) for sp in ("val", "test") for k in SCAL}}
                          for r in runs])
    seeds.to_csv(f"{args.out}/critic_eval_seeds.csv", index=False)
    by = {}
    for r in runs:
        by.setdefault(r["name"], []).append(r)
    agg = []
    for name, rs in by.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        r0 = rs[0]
        row = dict(name=name, input=r0["input"], obj=r0["obj"] + ("-k1" if r0["obj"] == "nce" and r0["nce"] == "k1" else ""),
                   regime=r0["regime"], frac=r0["frac"], n_seeds=len(rs), n_train_buf=r0["n_train_buf"],
                   n_train_br=r0["n_train_br"], train_s=np.mean([r["train_s"] for r in rs]),
                   peak_mb=np.max([r["peak_mb"] for r in rs]))
        for sp in ("val", "test"):
            for k in SCAL:
                v = [r.get(f"{sp}_{k}", np.nan) for r in rs]
                row[f"{sp}_{k}"] = float(np.nanmean(v)) if not all(x is None or np.isnan(x) for x in v) else np.nan
                if k in ("conc_1c", "conc_or", "auroc_buf_within_t"):
                    row[f"{sp}_{k}_sd"] = float(np.nanstd(v))
            for tag in ("1c", "or"):
                p, lo, hi = boot_conc([r[f"{sp}_conc_{tag}_parts"] for r in rs])
                row[f"{sp}_conc_{tag}_lo"], row[f"{sp}_conc_{tag}_hi"] = lo, hi
            row[f"{sp}_conc_1c_min_seed"] = float(np.min([r[f"{sp}_conc_1c"] for r in rs]))
            if r0["frac"] == 1.0:
                e = ensemble_conc(rs, D, sp)
                row[f"{sp}_ens_conc_1c"], row[f"{sp}_ens_conc_1c_lo"], row[f"{sp}_ens_conc_1c_hi"] = e["1c"]
                row[f"{sp}_ens_conc_or"], row[f"{sp}_ens_conc_or_lo"], row[f"{sp}_ens_conc_or_hi"] = e["or"]
                row[f"{sp}_ens_pick_gain_1c"] = e["pick_gain_1c"]
        agg.append(row)
    A = pd.DataFrame(agg).sort_values(["frac", "regime", "obj", "input"])
    A.to_csv(f"{args.out}/critic_eval.csv", index=False)

    # ---------------- prereg selection (val only) + gate (test)
    full = A[(A.frac == 1.0)]
    cand = full[full.input.map(E07_OK)].sort_values(["val_conc_1c", "val_conc_or"], ascending=False)
    best = cand.iloc[0]
    gate = dict(best=best["name"], val_conc_1c=best.val_conc_1c, test_conc_1c=best.test_conc_1c,
                test_ci=[best.test_conc_1c_lo, best.test_conc_1c_hi], test_conc_or=best.test_conc_or,
                test_or_ci=[best.test_conc_or_lo, best.test_conc_or_hi], test_min_seed=best.test_conc_1c_min_seed,
                passed=bool(best.test_conc_1c_lo > 0.5))
    per_obj = {o: full[(full.obj == o) & full.input.map(E07_OK)].sort_values("val_conc_1c", ascending=False).iloc[0]["name"]
               for o in sorted(full.obj.unique())}
    per_inp = {i: full[full.input == i].sort_values("val_conc_1c", ascending=False).iloc[0]["name"]
               for i in sorted(full.input.unique())}
    nc_test = noise_ceiling(D, "test")
    nc_val = noise_ceiling(D, "val")
    sel = dict(gate=gate, best_per_objective=per_obj, best_per_input=per_inp,
               noise_ceiling_test=dict(conc_1c=nc_test[0], ci=nc_test[1:3], pick_gain_1c=nc_test[3]),
               noise_ceiling_val=dict(conc_1c=nc_val[0], ci=nc_val[1:3], pick_gain_1c=nc_val[3]),
               top10_val=cand.head(10)[["name", "val_conc_1c", "val_conc_or", "test_conc_1c", "test_conc_1c_lo",
                                        "test_conc_1c_hi"]].to_dict("records"))
    json.dump(sel, open(f"{args.out}/selection.json", "w"), indent=1, default=float)
    # checkpoint of the selected critic (+ best per objective)
    for nm in {best["name"], *per_obj.values()}:
        dst = f"{args.out}/critic_ckpt/{nm}"
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(f"{OUT}/ckpt/{nm}/*.pt"):
            shutil.copy(f, dst)
    print(json.dumps(sel, indent=1, default=float))
    figures(A, args.out)


def figures(A, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # small-data curves (buf+br), test conc_1c and test AUROC within-t
    inputs = ["Raw", "Proprio", "Full", "Fusion", "FusionLC", "RandomProj"]
    cols = {"Raw": "#7f7f7f", "Proprio": "#bcbd22", "Full": "#1f77b4", "Fusion": "#d62728", "FusionLC": "#ff9896",
            "RandomProj": "#2ca02c"}
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for j, obj in enumerate(("bce", "q", "nce")):
        for i, met in enumerate(("test_conc_1c", "test_auroc_buf_within_t")):
            ax = axs[i, j]
            for inp in inputs:
                d = A[(A.obj == obj) & (A.regime == "buf+br") & ((A.input == inp) |
                                                                 ((inp == "FusionLC") & (A.input == "Fusion") & (A.frac == 1.0)))]
                d = d.sort_values("frac")
                if len(d):
                    ax.plot(d.frac * 100, d[met], "o-", color=cols[inp], label=inp)
            ax.set_xscale("log")
            ax.set_xticks([10, 25, 50, 100])
            ax.set_xticklabels(["10", "25", "50", "100"])
            if met == "test_conc_1c":
                ax.axhline(0.5, color="k", lw=0.8, ls=":")
            ax.set_title(f"{obj} · {met.replace('test_', '')}")
            ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel("train episodes (%)")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("E04 sample efficiency (test, 3-seed mean; buf+br)")
    fig.tight_layout()
    fig.savefig(f"{out}/fig/sample_efficiency.png", dpi=120)
    plt.close(fig)
    # ranking with CIs at 100%
    d = A[(A.frac == 1.0) & (A.regime == "buf+br")].copy()
    d = d.sort_values("val_conc_1c")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.22 * len(d))))
    y = np.arange(len(d))
    ax.errorbar(d.test_conc_1c, y, xerr=[d.test_conc_1c - d.test_conc_1c_lo, d.test_conc_1c_hi - d.test_conc_1c],
                fmt="o", ms=3, color="#1f77b4", label="test conc (1-cont)")
    ax.plot(d.val_conc_1c, y, "x", color="#d62728", label="val conc (selection)")
    ax.axvline(0.5, color="k", lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(d.name.str.replace("_buf+br_f1.0", "", regex=False), fontsize=7)
    ax.set_xlabel("same-state candidate ranking concordance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/fig/ranking_ci.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()

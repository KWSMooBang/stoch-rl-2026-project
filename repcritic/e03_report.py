"""E03 (T018) report: merged probe_scores.csv, test bootstrap CIs (best vs raw), figures, markdown tables.

  python -m repcritic.e03_report --out /outputs/StochRL/t018
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from repcritic.e03_probe import (CAND_FEATS, CTX_FEATS, OUT, RAW_FEATS, RidgePath, auc, build_concepts, load_feat,
                                 macro_f1, r2, select_alpha, CACHE)

MAIN = [("succ_pre", "auc"), ("succ_pre", "auc_within_t"), ("succk15_pre", "auc"), ("value", "r2"), ("value_pre", "r2"),
        ("succ_all", "auc"), ("geo_rel", "r2"), ("geo_rel", "r2_gc"), ("geo_rel", "r2_cp"), ("geo_rel", "r2_gp"),
        ("abs_pos", "r2_plate"), ("phase4", "macro_f1"), ("grasp", "auc"), ("contact", "auc"), ("obj_id", "bal_acc")]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2 = "#0b0b0b", "#52514e"


def split_name(f):
    m = re.match(r"(vlm|sig|aectx|aecand)(\d+)_(.+)", f)
    if m:
        return f"{m.group(1)}{m.group(2)}", m.group(3)
    return f, "-"


def family(f):
    for p in ("vlm", "sig", "aectx", "aecand"):
        if f.startswith(p):
            return p
    return "raw"


def best_val(df, concept, metric, pool, tc="100%"):
    d = df[(df.concept == concept) & (df.metric == metric) & (df.split == "val") & df.feature.isin(pool) & (df.train_count == tc)]
    d = d.groupby("feature").value.mean().sort_values(ascending=False)
    return d.index[0]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ae")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)


def preds(feat, cname, C, split):
    mask, Y, kind = C[cname]
    X = load_feat(feat)
    tr, va, te = (split == "train") & mask, (split == "val") & mask, (split == "test") & mask
    rp = RidgePath(X, tr).fit(Y[tr])
    ai = select_alpha(kind, rp.predict_all(rp.proj(rp.std(X[va]))), Y[va], Y[tr].mean(0))
    return rp.predict(rp.proj(rp.std(X[te])), ai), Y[te], Y[tr].mean(0), kind, te


def metric_fn(kind, met):
    if kind == "bin":
        return lambda P, Y, m, d: auc(P[:, 0], Y[:, 0] > 0.5, d if met == "auc_within_t" else None)
    if kind == "reg":
        if met == "r2_gc":
            return lambda P, Y, m, d: float(r2(P, Y, m)[[0, 1, 2, 9]].mean())
        return lambda P, Y, m, d: float(np.mean(r2(P, Y, m)))
    return lambda P, Y, m, d: macro_f1(P.argmax(1), Y.argmax(1), Y.shape[1])


def boot_diff(L, C, split, cname, met, fa, fb, n=1000):
    Pa, Y, ym, kind, te = preds(fa, cname, C, split)
    Pb, _, _, _, _ = preds(fb, cname, C, split)
    f = metric_fn(kind, met)
    ep, dec = L["episode"][te], L["decision"][te]
    eps = np.unique(ep)
    idx = {e: np.where(ep == e)[0] for e in eps}
    rs = np.random.RandomState(0)
    A, B, D = [], [], []
    for _ in range(n):
        ii = np.concatenate([idx[e] for e in rs.choice(eps, len(eps))])
        a, b = f(Pa[ii], Y[ii], ym, dec[ii]), f(Pb[ii], Y[ii], ym, dec[ii])
        A.append(a); B.append(b); D.append(a - b)
    pt = (f(Pa, Y, ym, dec), f(Pb, Y, ym, dec))
    q = lambda v: np.nanpercentile(v, [2.5, 97.5])
    return pt, q(A), q(B), q(D)


def leakage_check(L):
    """(1) feature files hold only schema keys (no privileged arrays); (2) episode split disjoint, decisions of an
    episode share a split, consistent with split_manifest; (3) cache feat_* built only from features/ + episodes/ (code)."""
    import glob
    src = "/workspace/stochrl_t017"
    schema = set(json.load(open(f"{src}/features_schema.json")))
    priv_keys = set(np.load(glob.glob(f"{src}/privileged/*.npz")[0]).files)
    bad = []
    for f in sorted(glob.glob(f"{src}/features/*.npz")):
        ks = set(np.load(f).files)
        if ks != schema or ks & priv_keys:
            bad.append(f)
    man = json.load(open(f"{src}/split_manifest.json"))
    mspl = {f"ep{e['eid']:06d}": e["split"] for e in man["episodes"]}
    ep, sp = L["episode"], L["split"]
    sets = {s: set(ep[sp == s]) for s in ("train", "val", "test")}
    out = dict(
        feature_files_checked=len(glob.glob(f"{src}/features/*.npz")), feature_files_bad=len(bad),
        feature_keys_intersect_privileged=sorted(schema & priv_keys),
        split_overlap={"train&val": len(sets["train"] & sets["val"]), "train&test": len(sets["train"] & sets["test"]),
                       "val&test": len(sets["val"] & sets["test"])},
        episodes_per_split={k: len(v) for k, v in sets.items()},
        split_matches_manifest=bool(all(mspl[e] == s for e, s in zip(ep, sp))),
        mixed_split_episodes=int(sum(len(set(sp[ep == e])) > 1 for e in np.unique(ep))),
        cand_states_split_from_episode=True,
    )
    CL = np.load(f"{CACHE}/cand_labels.npz")
    out["cand_split_consistent"] = bool(all(mspl[e] == s for e, s in zip(CL["episode"], CL["split"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/outputs/StochRL/t018")
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(f"{args.out}/fig", exist_ok=True)
    L = dict(np.load(f"{CACHE}/labels.npz"))
    split = L["split"]
    C = build_concepts(L)
    lk = leakage_check(L)
    json.dump(lk, open(f"{args.out}/leakage_check.json", "w"), indent=1)
    print("[leakage]", lk, flush=True)
    main_df = pd.read_csv(f"{OUT}/probe_scores_main.csv")
    cur = pd.read_csv(f"{OUT}/curves.csv")
    ctl = pd.read_csv(f"{OUT}/controls.csv")
    main_df["seed"] = 0
    main_df["control"] = "none"
    cur["control"] = "none"
    allp = pd.concat([main_df, cur, ctl], ignore_index=True)
    allp["layer"], allp["token_pool"] = zip(*allp.feature.map(split_name))
    allp = allp.rename(columns={"concept": "label"})
    cols = ["layer", "token_pool", "feature", "label", "metric", "value", "seed", "train_count", "split", "control",
            "n_train", "dim", "alpha_c"]
    allp[cols].to_csv(f"{args.out}/probe_scores.csv", index=False)
    md = []

    # ---------------- table 1: test scores per feature (100%)
    t = main_df[main_df.split == "test"].pivot_table(index="feature", columns=["concept", "metric"], values="value")
    v = main_df[main_df.split == "val"].pivot_table(index="feature", columns=["concept", "metric"], values="value")
    order = CTX_FEATS + CAND_FEATS + RAW_FEATS
    t = t.reindex(order)[MAIN]
    v = v.reindex(order)[MAIN]
    t.columns = [f"{c}:{m}" for c, m in t.columns]
    v.columns = t.columns
    t.round(3).to_csv(f"{args.out}/probe_table_test.csv")
    v.round(3).to_csv(f"{args.out}/probe_table_val.csv")

    # ---------------- best per concept (val-selected) + raw comparison with bootstrap
    best_rows = []
    for cname, met in [("succ_pre", "auc"), ("succ_pre", "auc_within_t"), ("succk15_pre", "auc"), ("value", "r2"),
                       ("value_pre", "r2"), ("geo_rel", "r2"), ("geo_rel", "r2_gc"), ("phase4", "macro_f1"),
                       ("grasp", "auc")]:
        row = dict(concept=cname, metric=met)
        for fam, pool in (("ctx", CTX_FEATS), ("cand", CAND_FEATS), ("raw", RAW_FEATS)):
            f = best_val(main_df, cname, met, pool)
            row[f"best_{fam}"] = f
            row[f"{fam}_val"] = float(v.loc[f, f"{cname}:{met}"])
            row[f"{fam}_test"] = float(t.loc[f, f"{cname}:{met}"])
        (a, b), ca, cb, cd = boot_diff(L, C, split, cname, met, row["best_ctx"], row["best_raw"])
        row.update(ctx_ci=ca.round(3).tolist(), raw_ci=cb.round(3).tolist(), diff=round(a - b, 3), diff_ci=cd.round(3).tolist())
        best_rows.append(row)
        print("[best]", row, flush=True)
    best = pd.DataFrame(best_rows)
    best.to_csv(f"{args.out}/best_per_concept.csv", index=False)

    # ---------------- figure 1: heatmap (test), skill-normalised colour, raw value annotated
    show = [c for c in t.columns if c.split(":")[1] in ("auc", "r2", "macro_f1", "bal_acc", "auc_within_t", "r2_gc",
                                                        "r2_cp", "r2_gp", "r2_plate")]
    H = t[show].copy()
    S = H.copy()
    for c in show:
        if c.split(":")[1].startswith("auc"):
            S[c] = (H[c] - 0.5) * 2
    fig, ax = plt.subplots(figsize=(13, 12))
    im = ax.imshow(S.values.clip(0, 1), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            x = H.values[i, j]
            if np.isfinite(x):
                ax.text(j, i, f"{x:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if S.values[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(show))); ax.set_xticklabels(show, rotation=60, ha="right", fontsize=8, color=INK2)
    ax.set_yticks(range(H.shape[0])); ax.set_yticklabels(H.index, fontsize=8, color=INK2)
    for y in (len(CTX_FEATS) - 0.5, len(CTX_FEATS) + len(CAND_FEATS) - 0.5):
        ax.axhline(y, color=INK, lw=1)
    ax.set_title("E03 probe, held-out TEST (75 ep). colour: AUC→2(AUC−0.5), R²/F1 as is; text = raw value",
                 fontsize=10, color=INK, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout(); fig.savefig(f"{args.out}/fig/layer_concept_heatmap.png", dpi=140); plt.close(fig)

    # ---------------- figure 2: small-data curves
    curve_concepts = [("succ_pre", "auc"), ("succk15_pre", "auc"), ("value", "r2"), ("geo_rel", "r2"),
                      ("phase4", "macro_f1"), ("grasp", "auc")]
    full = pd.concat([main_df.assign(seed=0), cur], ignore_index=True)
    full = full[full.split == "test"]
    tc_x = {"10%": 10, "25%": 25, "50%": 50, "100%": 100}
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))
    curve_rows = []
    for ax, (cn, met) in zip(axs.ravel(), curve_concepts):
        lines = []
        for fam, pool in (("VLM", [f for f in CTX_FEATS if f.startswith("vlm")]),
                          ("SigLIP", [f for f in CTX_FEATS if f.startswith("sig")]),
                          ("AE ctx", [f for f in CTX_FEATS if f.startswith("aectx")]),
                          ("AE cand", CAND_FEATS), ("raw RGB", ["raw_rgb32", "raw_rgb64"]),
                          ("proprio+t", ["raw_proprio_time"])):
            f = best_val(main_df, cn, met, pool)
            d = full[(full.feature == f) & (full.concept == cn) & (full.metric == met)]
            g = d.groupby("train_count").value.agg(["mean", "std"]).reindex(list(tc_x))
            lines.append((fam, f, g))
            for tc in tc_x:
                curve_rows.append(dict(concept=cn, metric=met, family=fam, feature=f, train_count=tc,
                                       mean=g.loc[tc, "mean"], std=g.loc[tc, "std"]))
        for i, (fam, f, g) in enumerate(lines):
            x = [tc_x[k] for k in g.index]
            ax.errorbar(x, g["mean"], yerr=g["std"].fillna(0), color=SERIES[i], lw=2, marker="o", ms=5,
                        capsize=2, label=f"{fam} ({f})")
        ax.set_xscale("log"); ax.set_xticks([10, 25, 50, 100]); ax.set_xticklabels(["35", "88", "175", "350"])
        ax.set_xlabel("train episodes", fontsize=8, color=INK2)
        ax.set_title(f"{cn} · {met} (test)", fontsize=9, color=INK, loc="left")
        style(ax)
        ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(f"{args.out}/fig/small_data_curves.png", dpi=140); plt.close(fig)
    pd.DataFrame(curve_rows).to_csv(f"{args.out}/small_data_curves.csv", index=False)

    # ---------------- figure 3: projection controls
    cc = ctl[(ctl.split == "test")]
    feats = sorted(cc.feature.unique())
    fig, axs = plt.subplots(len(feats), 4, figsize=(14, 3.2 * len(feats)), squeeze=False)
    for r, f in enumerate(feats):
        for c, (cn, met) in enumerate([("succ_pre", "auc"), ("value", "r2"), ("geo_rel", "r2"), ("phase4", "macro_f1")]):
            ax = axs[r, c]
            for i, (kind, tc) in enumerate((("randproj", "100%"), ("pca", "100%"), ("randproj", "10%"), ("pca", "10%"))):
                d = cc[(cc.feature == f) & (cc.concept == cn) & (cc.metric == met) & (cc.train_count == tc)
                       & cc.control.str.startswith(kind)]
                d = d.assign(k=d.control.str.extract(r"(\d+)").astype(int)[0]).groupby("k").value.mean()
                ax.plot(d.index, d.values, color=SERIES[i], lw=2, marker="o", ms=5,
                        ls="-" if tc == "100%" else "--", label=f"{kind} k ({tc} train)")
            fullv = full[(full.feature == f) & (full.concept == cn) & (full.metric == met)]
            for tc, ls in (("100%", "-"), ("10%", "--")):
                y = fullv[fullv.train_count == tc].value.mean()
                ax.axhline(y, color=INK2, lw=1, ls=ls, label=f"full D ({tc})")
            sh = cc[(cc.feature == f) & (cc.concept == cn) & (cc.metric == met) & (cc.control == "label_shuffle")].value.mean()
            ax.axhline(sh, color=SERIES[7], lw=1, ls=":", label="label shuffle")
            ax.set_xscale("log", base=2)
            ax.set_title(f"{f} · {cn} {met}", fontsize=8, color=INK, loc="left")
            style(ax)
            if r == 0 and c == 0:
                ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(f"{args.out}/fig/projection_controls.png", dpi=130); plt.close(fig)

    # ---------------- figure 4: E04 inputs eval
    ei = pd.read_csv(f"{OUT}/e04_inputs_eval.csv")
    ei = ei[ei.split == "test"]
    tab = ei.groupby(["feature", "concept", "metric", "train_count"]).value.mean().unstack(["concept", "metric"])
    tab.round(3).to_csv(f"{args.out}/e04_inputs_eval_table.csv")
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()

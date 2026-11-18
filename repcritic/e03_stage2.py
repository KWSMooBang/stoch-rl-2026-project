"""E03 (T018) stage 2: controls (random/PCA projection, label shuffle, logistic), candidate-level ranking,
E04 input selection (TRAIN fit + VAL choice only) and evaluation of the selected inputs.

  python -m repcritic.e03_stage2 --stage {controls,cand,select,eval_inputs}
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from repcritic.e03_probe import (ALPHA_C, CACHE, CAND_FEATS, CTX_FEATS, DT, GEO_NAMES, OUT, RidgePath, auc,
                                 build_concepts, episode_subsample, load_feat, logreg_check, r2, run_probe,
                                 select_alpha)


def concordance_within(score, y, groups):
    """Pairwise concordance within group for pairs with different y (score ties = 0.5)."""
    num = den = 0.0
    for g in np.unique(groups):
        m = groups == g
        s, t = score[m], y[m]
        ds = s[:, None] - s[None, :]
        pm = (t[:, None] - t[None, :]) > 0
        num += (ds[pm] > 0).sum() + 0.5 * (ds[pm] == 0).sum()
        den += pm.sum()
    return num / den if den else np.nan


def orth_random(D, k, seed):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(D, k, generator=g, dtype=DT))
    return Q


def best_by_val(df, concept, metric, pool):
    d = df[(df.concept == concept) & (df.metric == metric) & (df.split == "val") & (df.feature.isin(pool))]
    return d.sort_values("value", ascending=False).feature.tolist()


def std_all(X, tr):
    Xt = torch.from_numpy(X[tr]).to(DT)
    mu, sd = Xt.mean(0), Xt.std(0).clamp_min(1e-6)
    return mu, sd, (torch.from_numpy(X).to(DT) - mu) / sd


def proj(Xs, P):
    return (Xs @ P).numpy().astype(np.float32)


def shuffle_index(split, seed):
    """Permute 16-decision label blocks across TRAIN episodes; val/test rows map to themselves."""
    rs = np.random.RandomState(seed)
    tr_idx = np.where(split == "train")[0].reshape(-1, 16)
    idx = np.arange(len(split))
    idx[tr_idx.ravel()] = tr_idx[rs.permutation(len(tr_idx))].ravel()
    return idx


# ---------------------------------------------------------------- controls
def stage_controls(L, C, split, df, rows):
    tops = set()
    for cn, met in (("succ_pre", "auc"), ("geo_rel", "r2"), ("phase4", "macro_f1"), ("value", "r2")):
        tops.update(best_by_val(df, cn, met, CTX_FEATS)[:1])
    tops = sorted(tops)
    ctl = {k: C[k] for k in ("succ_pre", "succk15_pre", "value", "geo_rel", "phase4", "grasp")}
    tr = split == "train"
    for k in tops:
        X = load_feat(k)
        _, _, Xs = std_all(X, tr)
        _, _, Vh = torch.linalg.svd(Xs[torch.from_numpy(tr)], full_matrices=False)
        for kd in (8, 32, 128, 512):
            for seed in range(3):
                Z = proj(Xs, orth_random(X.shape[1], kd, seed))
                for tc, trm in (("100%", tr), ("10%", episode_subsample(L, 0.1, seed))):
                    run_probe(Z, L, ctl, split, trm, k, tc, rows, extra={"control": f"randproj{kd}", "seed": seed})
            Z = proj(Xs, Vh[:kd].T)
            for seed in range(3):
                for tc, trm in ((("100%", tr),) if seed == 0 else ()) + (("10%", episode_subsample(L, 0.1, seed)),):
                    run_probe(Z, L, ctl, split, trm, k, tc, rows, extra={"control": f"pca{kd}", "seed": seed})
        for seed in range(3):
            idx = shuffle_index(split, 500 + seed)
            Csh = {cn: (mask[idx], Y[idx], kind) for cn, (mask, Y, kind) in ctl.items()}
            run_probe(X, L, Csh, split, tr, k, "100%", rows, extra={"control": "label_shuffle", "seed": seed})
        print("[controls]", k, flush=True)
    lr_rows = []
    for cn, met in (("succ_pre", "auc"), ("succk15_pre", "auc"), ("grasp", "auc"), ("phase4", "macro_f1")):
        mask, Y, kind = C[cn]
        feats = (best_by_val(df, cn, met, CTX_FEATS)[:3] + best_by_val(df, cn, met, CAND_FEATS)[:1]
                 + ["raw_proprio_time", "raw_rgb32"])
        y_int = Y.argmax(1) if kind == "cls" else (Y[:, 0] > 0.5).astype(int)
        for k in feats:
            X = load_feat(k)
            v, lam, te = logreg_check(X, y_int, (split == "train") & mask, (split == "val") & mask,
                                      (split == "test") & mask, Y.shape[1] if kind == "cls" else 2)
            lr_rows.append(dict(feature=k, concept=cn, metric=met + "_logreg", val=v, test=te, l2=lam))
            print("[logreg]", cn, k, round(v, 3), round(te, 3), flush=True)
    pd.DataFrame(lr_rows).to_csv(f"{OUT}/logreg_check.csv", index=False)


# ---------------------------------------------------------------- candidate-level
def stage_cand(L, C, split, rows):
    CL = dict(np.load(f"{CACHE}/cand_labels.npz"))
    csplit, cstate, y = CL["split"], CL["state_id"], CL["succ_mean"].astype(float)
    ok = ~np.isnan(y)
    oracle = CL["n_cont"] >= 4
    tr = split == "train"
    for k in CAND_FEATS:
        X = load_feat(k)
        Xc = np.load(f"{CACHE}/cand_feat_{k}.npy").astype(np.float32)
        for cn in ("succ_all", "succ_pre", "succk15_pre", "value"):
            mask, Y, kind = C[cn]
            rp = RidgePath(X, tr & mask).fit(Y[tr & mask])
            va = (split == "val") & mask
            ai = select_alpha(kind, rp.predict_all(rp.proj(rp.std(X[va]))), Y[va], Y[tr & mask].mean(0))
            sc = rp.predict(rp.proj(rp.std(Xc)), ai)[:, 0]
            for sp in ("train", "val", "test"):
                m = ok & (csplit == sp)
                rows.append(dict(feature=k, mode=f"transfer_{cn}", split=sp, label="branch_1cont",
                                 value=concordance_within(sc[m], y[m], cstate[m]), n_states=len(np.unique(cstate[m]))))
            m = ok & oracle
            rows.append(dict(feature=k, mode=f"transfer_{cn}", split="oracle_all", label="oracle_p4",
                             value=concordance_within(sc[m], y[m], cstate[m]), n_states=len(np.unique(cstate[m]))))
        trc, vac = ok & (csplit == "train"), ok & (csplit == "val")
        rp = RidgePath(Xc, trc).fit(y[trc][:, None])
        Pv = rp.predict_all(rp.proj(rp.std(Xc[vac])))
        a = int(np.nanargmax([concordance_within(Pv[i][:, 0], y[vac], cstate[vac]) for i in range(len(rp.alphas))]))
        sc = rp.predict(rp.proj(rp.std(Xc)), [a])[:, 0]
        for sp in ("val", "test"):
            m = ok & (csplit == sp)
            rows.append(dict(feature=k, mode="direct_branch", split=sp, label="branch_1cont",
                             value=concordance_within(sc[m], y[m], cstate[m]), n_states=len(np.unique(cstate[m]))))
        m = ok & oracle & (csplit != "train")
        rows.append(dict(feature=k, mode="direct_branch", split="oracle_valtest", label="oracle_p4",
                         value=concordance_within(sc[m], y[m], cstate[m]), n_states=len(np.unique(cstate[m]))))
        print("[cand]", k, flush=True)


# ---------------------------------------------------------------- E04 selection
def fit_subspace(X, split, targets, kinds, rowmask=None, episodes=None, nfold=5):
    """Ridge W [D,T] on TRAIN rows; per-target alpha chosen on VAL (AUC for bin, R2 for reg).
    Also returns Z_cf: TRAIN rows replaced by episode-grouped out-of-fold predictions (same alphas), so that the
    critic never sees in-sample (label-memorising) probe outputs on its own training rows."""
    rm = np.ones(len(split), bool) if rowmask is None else rowmask
    tr, va = (split == "train") & rm, (split == "val") & rm
    rp = RidgePath(X, tr).fit(targets[tr])
    Pv = rp.predict_all(rp.proj(rp.std(X[va])))
    ai = []
    for j in range(targets.shape[1]):
        if kinds[j] == "bin":
            sc = [auc(Pv[a][:, j], targets[va][:, j] > 0.5) for a in range(len(rp.alphas))]
        else:
            sc = [r2(Pv[a][:, j:j + 1], targets[va][:, j:j + 1], targets[tr][:, j].mean())[0] for a in range(len(rp.alphas))]
        ai.append(int(np.nanargmax(sc)))
    W, b = rp.coef(np.array(ai))
    Z = (rp.std(X).numpy() @ W + b).astype(np.float32)
    Q, _ = np.linalg.qr(W)
    Zcf = Z.copy()
    tr_eps = np.unique(episodes[split == "train"])
    folds = np.array_split(np.random.RandomState(0).permutation(tr_eps), nfold)
    for fe in folds:
        infold = np.isin(episodes, fe)
        rpf = RidgePath(X, tr & ~infold).fit(targets[tr & ~infold])
        Zcf[infold] = rpf.predict(rpf.proj(rpf.std(X[infold])), np.array(ai)).astype(np.float32)
    info = dict(mu=rp.mu.numpy(), sd=rp.sd.numpy(), W=W, b=b, Q=Q, alpha_c=ALPHA_C[np.array(ai)], z_crossfit=Zcf)
    return info, Z


def stage_select(L, C, split, df):
    import yaml
    od = f"{OUT}/e04_inputs"
    os.makedirs(od, exist_ok=True)
    tr = split == "train"
    sone = L["success_once"].astype(float)
    succ_by15 = (L["success_before"] | L["succ_k15"]).astype(float)
    sel = {"goal": best_by_val(df, "succ_pre", "auc", CTX_FEATS)[0],
           "geo": best_by_val(df, "geo_rel", "r2", CTX_FEATS)[0],
           "phase": best_by_val(df, "phase4", "macro_f1", CTX_FEATS)[0]}
    v = df[(df.split == "val") & df.feature.isin(CTX_FEATS)]
    piv = v.pivot_table(index="feature", columns=["concept", "metric"], values="value")
    score = (piv[("succ_pre", "auc")] - 0.5) * 2 + piv[("geo_rel", "r2")] + piv[("phase4", "macro_f1")]
    sel["full"] = score.idxmax()
    sel["full_score_val_top8"] = {k: round(float(x), 4) for k, x in score.sort_values(ascending=False).head(8).items()}
    sel["cand_dependent_goal_best"] = best_by_val(df, "succ_pre", "auc", CAND_FEATS)[0]
    targets = {
        "Goal": (np.stack([sone, L["value"], succ_by15], 1), ["bin", "reg", "bin"],
                 ["success_once", "value_g0.98", "success_by_t+15"]),
        "Geo": (C["geo_rel"][1], ["reg"] * 12, GEO_NAMES),
        "Phase": (np.concatenate([C["phase4"][1], C["grasp"][1], C["contact"][1]], 1), ["bin"] * 6,
                  ["ph_approach", "ph_grasp_transport", "ph_post_grasp_lost", "ph_placed", "grasping_now",
                   "contact_any_finger"]),
    }
    man = {"task": "T018 E03 -> E04 critic inputs",
           "protocol": "standardizer + ridge fit on TRAIN (350 ep); feature key and per-target alpha chosen on VAL (75 ep); "
                       "TEST never used for any choice",
           "data": "/workspace/stochrl_t018/cache/feat_<feature_key>.npy rows = labels.npz order (episode, decision), 8000 x D fp16",
           "usage": "z = ((h - mu) / sd) @ W + b  (concept probes);  z = ((h - mu) / sd) @ Q  (RandomProj / PCA). "
                    "z arrays precomputed for all 8000 decisions. For critic TRAINING use `z_crossfit` (train rows = "
                    "5-fold episode-grouped out-of-fold probe outputs; val/test rows identical to z). In-sample `z` on "
                    "train rows memorises labels (e.g. success_once) and would leak.",
           "candidate_dependence": "all inputs below are candidate-INDEPENDENT z(o,l); action enters only via the critic's action encoder",
           "inputs": {}}
    Zs = {}
    for c in ("Goal", "Geo", "Phase"):
        Y, kinds, names = targets[c]
        key = sel[c.lower()]
        info, Z = fit_subspace(load_feat(key), split, Y.astype(float), kinds,
                               rowmask=C["geo_rel"][0] if c == "Geo" else None, episodes=L["episode"])
        np.savez(f"{od}/C-{c}.npz", feature_key=key, target_names=np.array(names), z=Z, **info)
        Zs[c] = info["z_crossfit"]
        man["inputs"][f"C-{c}"] = dict(feature_key=key, dim=int(Z.shape[1]), targets=names, file=f"C-{c}.npz")
    Zf = np.concatenate([Zs["Goal"], Zs["Geo"], Zs["Phase"]], 1)
    kf = int(Zf.shape[1])
    np.savez(f"{od}/C-Fusion.npz", z_crossfit=Zf, parts=np.array(["C-Goal", "C-Geo", "C-Phase"]),
             z=np.concatenate([np.load(f"{od}/C-{c}.npz")["z"] for c in ("Goal", "Geo", "Phase")], 1))
    man["inputs"]["C-Fusion"] = dict(feature_key=[sel["goal"], sel["geo"], sel["phase"]], dim=kf, file="C-Fusion.npz",
                                     note="concat z of C-Goal|C-Geo|C-Phase; variant C-Fusion+Full = concat with C-Full")
    X = load_feat(sel["full"])
    mu, sd, Xs = std_all(X, tr)
    np.savez(f"{od}/C-Full.npz", feature_key=sel["full"], mu=mu.numpy(), sd=sd.numpy())
    man["inputs"]["C-Full"] = dict(feature_key=sel["full"], dim=int(X.shape[1]), file="C-Full.npz",
                                   note="standardize with mu/sd; critic learns its own projection")
    R = orth_random(X.shape[1], kf, 0)
    np.savez(f"{od}/C-RandomProj.npz", feature_key=sel["full"], mu=mu.numpy(), sd=sd.numpy(), Q=R.numpy(), z=proj(Xs, R))
    man["inputs"]["C-RandomProj"] = dict(feature_key=sel["full"], dim=kf, file="C-RandomProj.npz",
                                         note="random orthonormal Q (torch.Generator seed 0, QR of N(0,1)), dim = C-Fusion")
    _, _, Vh = torch.linalg.svd(Xs[torch.from_numpy(tr)], full_matrices=False)
    P = Vh[:kf].T
    np.savez(f"{od}/C-PCA.npz", feature_key=sel["full"], mu=mu.numpy(), sd=sd.numpy(), Q=P.numpy(), z=proj(Xs, P))
    man["inputs"]["C-PCA"] = dict(feature_key=sel["full"], dim=kf, file="C-PCA.npz", note="extra control: TRAIN PCA, dim = C-Fusion")
    Yall = np.concatenate([targets[c][0] for c in ("Goal", "Geo", "Phase")], 1).astype(float)
    kinds_all = sum([targets[c][1] for c in ("Goal", "Geo", "Phase")], [])
    shi = shuffle_index(split, 77)
    info, Zsh = fit_subspace(X, split, Yall[shi], kinds_all, rowmask=C["geo_rel"][0][shi], episodes=L["episode"])
    np.savez(f"{od}/C-Shuffle.npz", feature_key=sel["full"], z=Zsh, **info)
    man["inputs"]["C-Shuffle"] = dict(feature_key=sel["full"], dim=kf, file="C-Shuffle.npz",
                                      note="extra control: same 21 ridge targets, TRAIN labels shuffled across episodes (seed 77)")
    man["selection_val"] = sel
    with open(f"{od}/manifest.yaml", "w") as f:
        yaml.safe_dump(man, f, sort_keys=False, allow_unicode=True)
    json.dump(sel, open(f"{OUT}/selection.json", "w"), indent=1)
    return sel


def stage_eval_inputs(L, C, split, rows):
    """Probe the E04 inputs themselves. NOTE: concept subspaces already consumed 100% TRAIN labels."""
    od = f"{OUT}/e04_inputs"
    ev = {k: C[k] for k in ("succ_pre", "succk15_pre", "value_pre", "geo_rel", "phase4", "grasp", "obj_id")}
    full = json.load(open(f"{OUT}/selection.json"))["full"]
    def zload(n):
        d = np.load(f"{od}/{n}.npz")
        return d["z_crossfit"] if "z_crossfit" in d.files else d["z"]
    srcs = [(n, zload(n)) for n in
            ("C-Goal", "C-Geo", "C-Phase", "C-Fusion", "C-RandomProj", "C-PCA", "C-Shuffle")]
    srcs += [(f"{n}-insample", np.load(f"{od}/{n}.npz")["z"]) for n in ("C-Goal", "C-Fusion")]
    srcs.append(("C-Full", load_feat(full)))
    for name, Z in srcs:
        for tc, fr in (("100%", 1.0), ("25%", 0.25), ("10%", 0.1)):
            for seed in range(3 if fr < 1 else 1):
                trm = (split == "train") if fr == 1 else episode_subsample(L, fr, seed)
                run_probe(Z, L, ev, split, trm, name, tc, rows, extra={"seed": seed})
        print("[eval_inputs]", name, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--stage", required=True, choices=["controls", "cand", "select", "eval_inputs"])
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    L = dict(np.load(f"{CACHE}/labels.npz"))
    split = L["split"]
    C = build_concepts(L)
    df = pd.read_csv(f"{OUT}/probe_scores_main.csv")
    rows = []
    if args.stage == "controls":
        stage_controls(L, C, split, df, rows)
        pd.DataFrame(rows).to_csv(f"{OUT}/controls.csv", index=False)
    elif args.stage == "cand":
        stage_cand(L, C, split, rows)
        pd.DataFrame(rows).to_csv(f"{OUT}/cand_rank.csv", index=False)
    elif args.stage == "select":
        print(stage_select(L, C, split, df))
    else:
        stage_eval_inputs(L, C, split, rows)
        pd.DataFrame(rows).to_csv(f"{OUT}/e04_inputs_eval.csv", index=False)


if __name__ == "__main__":
    main()

"""E03 (T018): linear probing of frozen pi0.5 features (CPU only).

Protocol: standardizer + probe fit on TRAIN episodes only; alpha / layer / subspace selection on VAL only;
TEST is evaluated for reporting only (never used for a choice).
Probe = ridge regression solved on an eigen-path (all alphas at once). Classification concepts use ridge
scores on one-hot / {0,1} targets (AUC / macro-F1); an L2-logistic check is run on the top features.

Outputs (/workspace/stochrl_t018/results): probe_scores.csv, curves.csv, controls.csv, cand_rank.csv,
selection.json, e04_inputs/*.npz + manifest.yaml.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

CACHE = "/workspace/stochrl_t018/cache"
OUT = "/workspace/stochrl_t018/results"
ALPHA_C = np.logspace(-4, 3, 15)  # alpha = c * N_train (standardized features)
DT = torch.float64

CTX_FEATS = ([f"vlm{L}_{p}" for L in (4, 9, 13, 17) for p in ("img_mean", "img_grid4", "txt_mean", "txt_last")]
             + [f"sig{L}_{p}" for L in (8, 17, 26) for p in ("mean", "grid4")]
             + [f"aectx{L}_mean" for L in (4, 9, 13, 17)] + ["aectx17_tok"])
CAND_FEATS = [f"aecand{L}_mean" for L in (4, 9, 13, 17)] + ["aecand17_tok"]
RAW_FEATS = ["raw_rgb32", "raw_rgb64", "raw_proprio", "raw_time", "raw_proprio_time"]
ALL_FEATS = CTX_FEATS + CAND_FEATS + RAW_FEATS


# ---------------------------------------------------------------- metrics
def auc(score, y, groups=None):
    """ROC AUC (ties = 0.5). If groups is given: only pairs within the same group (pooled)."""
    score = np.asarray(score, float); y = np.asarray(y).astype(bool)
    if groups is None:
        groups = np.zeros(len(y), int)
    num = den = 0.0
    for g in np.unique(groups):
        m = groups == g
        s, t = score[m], y[m]
        npos, nneg = t.sum(), (~t).sum()
        if npos == 0 or nneg == 0:
            continue
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
        # average ranks for ties
        ss = s[order]
        i = 0
        while i < len(ss):
            j = i
            while j + 1 < len(ss) and ss[j + 1] == ss[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        num += ranks[t].sum() - npos * (npos + 1) / 2
        den += npos * nneg
    return num / den if den > 0 else np.nan


def r2(pred, y, ytr_mean):
    ss = ((y - ytr_mean) ** 2).sum(0)
    return 1 - ((y - pred) ** 2).sum(0) / np.maximum(ss, 1e-12)


def macro_f1(pred, y, ncls):
    f = []
    for c in range(ncls):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        if (y == c).sum() == 0:
            continue
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def bal_acc(pred, y):
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


# ---------------------------------------------------------------- ridge eigen path
class RidgePath:
    """Standardize on train rows, eigendecompose once, predict any rows for any alpha / targets."""

    def __init__(self, X, tr):
        Xt = torch.from_numpy(X[tr]).to(DT)
        self.mu = Xt.mean(0)
        self.sd = Xt.std(0).clamp_min(1e-6)
        Xt = (Xt - self.mu) / self.sd
        self.N, self.D = Xt.shape
        self.Xt = Xt
        if self.D <= self.N:
            lam, V = torch.linalg.eigh(Xt.T @ Xt)
            self.mode, self.V = "primal", V
        else:
            lam, U = torch.linalg.eigh(Xt @ Xt.T)
            self.mode, self.U, self.V = "dual", U, Xt.T @ U  # V: D x N, columns = X^T u_i
        self.lam = lam.clamp_min(0)
        self.alphas = ALPHA_C * self.N

    def std(self, X):
        return (torch.from_numpy(X).to(DT) - self.mu) / self.sd

    def fit(self, Y):
        """Y [N, T] train targets -> B [r, T] (centered)."""
        Yt = torch.from_numpy(np.asarray(Y, np.float64)).to(DT)
        self.ym = Yt.mean(0)
        Yc = Yt - self.ym
        if self.mode == "primal":   # coef = V diag(1/(lam+a)) V^T X^T y
            self.B = self.V.T @ (self.Xt.T @ Yc)
        else:                       # coef = X^T U diag(1/(lam+a)) U^T y,  V = X^T U
            self.B = self.U.T @ Yc
        return self

    def proj(self, Xs):
        return Xs @ self.V  # [n, r]

    def predict(self, A, ai):
        """A = proj(std(X)); ai: alpha index per target (array len T) -> [n, T] numpy."""
        a = torch.as_tensor(self.alphas[np.asarray(ai)], dtype=DT)  # [T]
        w = 1.0 / (self.lam[:, None] + a[None, :])                 # [r, T]
        return (A @ (w * self.B) + self.ym).numpy()

    def predict_all(self, A):
        """All alphas: [n_alpha, n, T]."""
        return np.stack([self.predict(A, np.full(self.B.shape[1], k)) for k in range(len(self.alphas))])

    def coef(self, ai):
        a = torch.as_tensor(self.alphas[np.asarray(ai)], dtype=DT)
        w = 1.0 / (self.lam[:, None] + a[None, :])
        return (self.V @ (w * self.B)).numpy(), self.ym.numpy()  # [D, T] in standardized space


# ---------------------------------------------------------------- concepts
def winsor(Y, tr, q=(0.5, 99.5)):
    """Clip each target column to TRAIN percentiles (objects knocked off the table reach ~2 m)."""
    lo, hi = np.percentile(Y[tr], q, axis=0)
    return np.clip(Y, lo, hi)


def build_concepts(L):
    pre = ~L["success_before"]
    tr = L["split"] == "train"
    sone = L["success_once"].astype(float)
    C = {}
    # name: (rows_mask, Y [n,T], kind, extra)
    C["succ_all"] = (np.ones(len(sone), bool), sone[:, None], "bin")
    C["succ_pre"] = (pre, sone[:, None], "bin")
    C["succk15_pre"] = (pre, L["succ_k15"].astype(float)[:, None], "bin")
    C["value"] = (np.ones(len(sone), bool), L["value"][:, None], "reg")
    C["value_pre"] = (pre, L["value"][:, None], "reg")
    # objects knocked off the workspace (carrot-plate or gripper-carrot dist > 0.35 m) reach ~2 m and would
    # dominate R^2 -> excluded from geometry probes (label-side mask only; see summary for counts).
    inws = (L["geo"][:, 10] <= 0.35) & (L["geo"][:, 9] <= 0.35)
    C["geo_rel"] = (inws, L["geo"], "reg")
    C["abs_pos"] = (inws, L["abs_pos"], "reg")
    C["phase4"] = (np.ones(len(sone), bool), np.eye(4)[L["phase4"]], "cls")
    C["grasp"] = (np.ones(len(sone), bool), L["grasping_now"].astype(float)[:, None], "bin")
    C["contact"] = (np.ones(len(sone), bool), L["contact_any_finger"].astype(float)[:, None], "bin")
    C["obj_id"] = (np.ones(len(sone), bool), np.eye(16)[L["obj_id"]], "cls")
    return C


GEO_NAMES = ["gc_rel_x", "gc_rel_y", "gc_rel_z", "cp_rel_x", "cp_rel_y", "cp_rel_z", "gp_rel_x", "gp_rel_y", "gp_rel_z",
             "gc_dist", "cp_dist", "gp_dist"]
ABS_NAMES = ["grip_x", "grip_y", "grip_z", "obj_x", "obj_y", "obj_z", "plate_x", "plate_y", "plate_z"]


def score_set(kind, pred, Y, ytr_mean, groups=None):
    """Returns dict metric->value (main metric first)."""
    if kind == "bin":
        out = {"auc": auc(pred[:, 0], Y[:, 0] > 0.5)}
        if groups is not None:
            out["auc_within_t"] = auc(pred[:, 0], Y[:, 0] > 0.5, groups)
        return out
    if kind == "reg":
        rr = r2(pred, Y, ytr_mean)
        out = {"r2": float(np.mean(rr))}
        if Y.shape[1] == 12:
            out.update({"r2_gc": float(rr[[0, 1, 2, 9]].mean()), "r2_cp": float(rr[[3, 4, 5, 10]].mean()),
                        "r2_gp": float(rr[[6, 7, 8, 11]].mean())})
        if Y.shape[1] == 9:
            out.update({"r2_grip": float(rr[:3].mean()), "r2_obj": float(rr[3:6].mean()), "r2_plate": float(rr[6:].mean())})
        return out
    y = Y.argmax(1); p = pred.argmax(1)
    return {"macro_f1": macro_f1(p, y, Y.shape[1]), "bal_acc": bal_acc(p, y)}


MAIN_METRIC = {"bin": "auc", "reg": "r2", "cls": "macro_f1"}


def select_alpha(kind, P_val, Yv, ytr_mean):
    """P_val [n_alpha, n, T]. Returns alpha index per target (regression: per target; else shared)."""
    T = Yv.shape[1]
    if kind == "reg":
        sc = np.stack([r2(P_val[k], Yv, ytr_mean) for k in range(P_val.shape[0])])  # [A, T]
        return sc.argmax(0)
    sc = [score_set(kind, P_val[k], Yv, ytr_mean)[MAIN_METRIC[kind]] for k in range(P_val.shape[0])]
    return np.full(T, int(np.nanargmax(sc)))


def run_probe(X, L, C, split, tr_rows, name_feat, train_count_tag, rows_out, eval_test=True, keep=None, extra=None):
    """Fit all concepts on feature X with the given train rows. Appends rows; returns {concept: (path, ai)}."""
    paths = {}
    dec = L["decision"]
    for cname, (mask, Y, kind) in C.items():
        tr = tr_rows & mask
        va = (split == "val") & mask
        te = (split == "test") & mask
        key = tr.tobytes()
        if keep is not None and key in keep:
            rp = keep[key]
        else:
            rp = RidgePath(X, tr)
            if keep is not None:
                keep[key] = rp
        rp.fit(Y[tr])
        Av = rp.proj(rp.std(X[va]))
        Pv = rp.predict_all(Av)
        ytr_mean = Y[tr].mean(0)
        ai = select_alpha(kind, Pv, Y[va], ytr_mean)
        ev = {"val": va, "test": te} if eval_test else {"val": va}
        for sp, m in ev.items():
            P = rp.predict(rp.proj(rp.std(X[m])), ai)
            sc = score_set(kind, P, Y[m], ytr_mean, groups=dec[m] if kind == "bin" else None)
            for met, v in sc.items():
                rows_out.append(dict(feature=name_feat, concept=cname, metric=met, split=sp, value=float(v),
                                     train_count=train_count_tag, n_train=int(tr.sum()), alpha_c=float(ALPHA_C[ai[0]]),
                                     dim=X.shape[1], **(extra or {})))
        paths[cname] = (rp, ai)
    return paths


def load_feat(k):
    return np.load(f"{CACHE}/feat_{k}.npy").astype(np.float32)


def episode_subsample(L, frac, seed):
    tr_eps = np.unique(L["episode"][L["split"] == "train"])
    rs = np.random.RandomState(1000 + seed)
    n = max(2, int(round(frac * len(tr_eps))))
    sel = set(rs.choice(tr_eps, n, replace=False))
    return np.array([e in sel for e in L["episode"]]) & (L["split"] == "train")


def boot_ci(fn, ep_test, n=500, seed=0):
    rs = np.random.RandomState(seed)
    eps = np.unique(ep_test)
    idx_by = {e: np.where(ep_test == e)[0] for e in eps}
    vals = []
    for _ in range(n):
        s = rs.choice(eps, len(eps))
        ii = np.concatenate([idx_by[e] for e in s])
        vals.append(fn(ii))
    return np.nanpercentile(vals, [2.5, 97.5])


# ---------------------------------------------------------------- logistic check
def logreg_check(X, y_int, tr, va, te, ncls, Cs=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    """L2 logistic / softmax with LBFGS on standardized features; lambda chosen on val (main metric)."""
    Xt = torch.from_numpy(X[tr]).float()
    mu, sd = Xt.mean(0), Xt.std(0).clamp_min(1e-6)
    f = lambda m: ((torch.from_numpy(X[m]).float() - mu) / sd)
    Xtr, Xva, Xte = f(tr), f(va), f(te)
    ytr = torch.from_numpy(y_int[tr]).long()
    best = None
    K = max(ncls, 2)
    for lam in Cs:
        W = torch.zeros(X.shape[1], K, requires_grad=True); b = torch.zeros(K, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=200, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(Xtr @ W + b, ytr) + lam * (W ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            def met(Xs, ys):
                lo = (Xs @ W + b)
                if ncls == 2:
                    return auc((lo[:, 1] - lo[:, 0]).numpy(), ys == 1)
                return macro_f1(lo.argmax(1).numpy(), ys, ncls)
            v = met(Xva, y_int[va])
            if best is None or v > best[0]:
                best = (v, lam, met(Xte, y_int[te]))
    return best


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--stage", default="all", choices=["all", "probe", "curves", "controls", "cand", "select"])
    ap.add_argument("--feats", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(OUT, exist_ok=True)
    L = dict(np.load(f"{CACHE}/labels.npz"))
    split = L["split"]
    C = build_concepts(L)
    feats = args.feats.split(",") if args.feats else ALL_FEATS
    t0 = time.time()

    # leakage / split checks
    eps = {s: set(L["episode"][split == s]) for s in ("train", "val", "test")}
    assert not (eps["train"] & eps["val"]) and not (eps["train"] & eps["test"]) and not (eps["val"] & eps["test"])
    for e in np.unique(L["episode"]):
        assert len(set(split[L["episode"] == e])) == 1

    if args.stage in ("all", "probe"):
        rows = []
        for k in feats:
            X = load_feat(k)
            run_probe(X, L, C, split, split == "train", k, "100%", rows, keep={})
            print(f"[probe] {k} D={X.shape[1]} {time.time()-t0:.0f}s", flush=True)
        import pandas as pd
        pd.DataFrame(rows).to_csv(f"{OUT}/probe_scores_main.csv", index=False)

    if args.stage in ("all", "curves"):
        import pandas as pd
        rows = []
        cur_concepts = {k: C[k] for k in ("succ_pre", "value", "geo_rel", "phase4", "grasp", "succk15_pre")}
        for k in feats:
            X = load_feat(k)
            for frac in (0.1, 0.25, 0.5):
                for seed in range(3):
                    trm = episode_subsample(L, frac, seed)
                    run_probe(X, L, cur_concepts, split, trm, k, f"{int(frac*100)}%", rows, keep={},
                              extra={"seed": seed})
            print(f"[curves] {k} {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/curves.csv", index=False)


if __name__ == "__main__":
    main()

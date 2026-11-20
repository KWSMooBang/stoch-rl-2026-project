"""T036 (L1 decision 2026-11-21): refit the gate predictor as V_hat_end. CPU only.

T023's V_hat was fit on pre-success decisions with label success_once, so post-success ("placed") states are out
of distribution and score low -> a gate on it stays open after success. V_hat_end uses the same input and model
family, but
    label  = episode success_at_end (success flag on at the last step)
    states = ALL 16 decisions of the T017 train-split episodes, post-success states included
    lambda = val log-loss over the T023 grid;  Platt (a, b) on val.   test / held-out are never read.
Output (same file layout as /outputs/StochRL/t023/predictors, loadable by vla_residual.potential.Potential):
    /outputs/StochRL/t036/predictors/{vla.npz, calibration.json}, vhat_end_all.npz (train + val rows only), fit.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

CACHE = Path("/workspace/stochrl_t018/cache")
OUT = Path("/outputs/StochRL/t036")
ND = 16
LAMS = 10.0 ** np.arange(-5, 0.5, 0.5)
FEATURE = "vlm17_img_mean"


def fit_logreg(X, y, lam, w0=None):
    X = torch.as_tensor(X, dtype=torch.float64); y = torch.as_tensor(y, dtype=torch.float64)
    w = torch.zeros(X.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    if w0 is not None:
        w.data[:] = torch.as_tensor(w0)
    opt = torch.optim.LBFGS([w], lr=1, max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X @ w[:-1] + w[-1]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) + 0.5 * lam * (w[:-1] ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def auroc(s, y):
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def main():
    lab = np.load(CACHE / "labels.npz", allow_pickle=True)
    split = lab["split"].reshape(500, ND)[:, 0].astype(str)
    eps = lab["episode"].reshape(500, ND)[:, 0]
    y_end = lab["success_at_end"].reshape(500, ND)[:, 0].astype(float)
    assert (lab["success_at_end"].reshape(500, ND) == y_end[:, None]).all()
    y_once = lab["success_once"].reshape(500, ND)[:, 0].astype(bool)
    post = lab["success_before"].reshape(500, ND).astype(bool)
    X = np.load(CACHE / f"feat_{FEATURE}.npy").astype(np.float32).reshape(500, ND, -1)
    use = np.isin(split, ["train", "val"])  # test rows are dropped before anything is computed
    X, y_end, y_once, post, split, eps = X[use], y_end[use], y_once[use], post[use], split[use], eps[use]
    n = len(split)
    tr = np.repeat((split == "train")[:, None], ND, 1).reshape(-1)
    va = np.repeat((split == "val")[:, None], ND, 1).reshape(-1)
    Xs = X.reshape(-1, X.shape[-1]).astype(np.float64)
    y = np.repeat(y_end[:, None], ND, 1).reshape(-1)
    mu = Xs[tr].mean(0); sd = Xs[tr].std(0) + 1e-6
    Z = (Xs - mu) / sd
    best, w, grid = None, None, []
    for lam in LAMS[::-1]:
        w = fit_logreg(Z[tr], y[tr], lam, w)
        ll = logloss(sigmoid(Z[va] @ w[:-1] + w[-1]), y[va])
        grid.append(dict(lam=float(lam), val_logloss=ll))
        if best is None or ll < best[0]:
            best = (ll, lam, w.copy())
    ll, lam, w = best
    v_raw = sigmoid(Z @ w[:-1] + w[-1])
    pa = fit_logreg(logit(v_raw[va])[:, None], y[va], 0.0)
    a, b = float(pa[0]), float(pa[1])
    v = sigmoid(a * logit(v_raw) + b)
    (OUT / "predictors").mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "predictors" / "vla.npz", W=w[:-1], b=w[-1], mu=mu, sd=sd, lam=lam, val_logloss=ll, kind="linear",
             feature=FEATURE)
    json.dump({"vla": dict(platt_a=a, platt_b=b, label="success_at_end", states="all decisions (post-success included)")},
              open(OUT / "predictors" / "calibration.json", "w"), indent=1)
    np.savez(OUT / "vhat_end_all.npz", episodes=eps, split=split, vhat_end=v.reshape(n, ND).astype(np.float32),
             vhat_end_raw=v_raw.reshape(n, ND).astype(np.float32), y_end=y_end, y_once=y_once, post=post)
    P = post.reshape(-1)
    yo = np.repeat(y_once[:, None], ND, 1).reshape(-1)

    def rep(m):
        return dict(n=int(m.sum()), pos_rate=float(y[m].mean()), auroc=auroc(v[m], y[m]), logloss=logloss(v[m], y[m]),
                    brier=float(((v[m] - y[m]) ** 2).mean()), vhat_median=float(np.median(v[m])))

    out = dict(feature=FEATURE, label="success_at_end", lam=float(lam), platt_a=a, platt_b=b, lam_grid=grid,
               n_train_episodes=int((split == "train").sum()), n_val_episodes=int((split == "val").sum()),
               train_all=rep(tr), val_all=rep(va), val_pre_success=rep(va & ~P), val_post_success=rep(va & P),
               train_post_success=rep(tr & P),
               val_vhat_median=dict(pre_success_eps_end_success=float(np.median(v[va & ~P & (y == 1)])),
                                    pre_success_eps_end_fail=float(np.median(v[va & ~P & (y == 0)])),
                                    post_success_end_success=float(np.median(v[va & P & (y == 1)])),
                                    post_success_end_fail=float(np.median(v[va & P & (y == 0)])) if (va & P & (y == 0)).any() else None,
                                    n_post_success_end_fail=int((va & P & (y == 0)).sum())),
               once_vs_end=dict(train_once=float(y_once[split == "train"].mean()), train_end=float(y_end[split == "train"].mean()),
                                val_once=float(y_once[split == "val"].mean()), val_end=float(y_end[split == "val"].mean())),
               note="test split rows are dropped at load; nothing is fit, selected or reported on test / held-out")
    (OUT / "predictors" / "fit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "lam_grid"}, indent=1))


if __name__ == "__main__":
    main()

"""E04 (T019) diagnostic: does an ACTION-ONLY model (no state) already rank same-state candidates above chance?
Ridge / logistic on chunk features trained on TRAIN branch rows (+ buffer pre rows); alpha on VAL; test reported.
Also simple 1-feature heuristics (gripper mean, translation magnitude) with sign chosen on VAL.

  python -m repcritic.e04_actiononly
"""
import json

import numpy as np

from repcritic.e04_critic import conc, conc_parts
from repcritic.e04_report import boot_conc

OUT = "/workspace/stochrl_t019"


def main():
    D = dict(np.load(f"{OUT}/data.npz"))
    bA, y, st, sp = D["b_A"].reshape(len(D["b_A"]), -1), D["b_y"], D["b_state"], D["b_split"]
    res = {}
    # within-state demeaned ridge (pure action-difference model) on train branch rows
    def demean(X, g):
        X = X.copy()
        for s in np.unique(g):
            m = g == s
            X[m] -= X[m].mean(0)
        return X
    tr, va, te = sp == "train", sp == "val", sp == "test"
    Xtr, ytr = demean(bA[tr], st[tr]), demean(y[tr][:, None], st[tr])[:, 0]
    best = None
    for lam in 10.0 ** np.arange(-2, 5):
        w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
        v = conc(conc_parts(bA[va] @ w, y[va], st[va]))
        if best is None or v > best[0]:
            best = (v, lam, w)
    v, lam, w = best
    res["ridge_action_within"] = dict(val=v, lam=lam, test=boot_conc([conc_parts(bA[te] @ w, y[te], st[te])]))
    # heuristics
    A = D["b_A"]
    feats = {"gripper_mean": A[..., 6].mean(1), "trans_norm": np.linalg.norm(A[..., :3], axis=-1).mean(1),
             "rot_norm": np.linalg.norm(A[..., 3:6], axis=-1).mean(1), "z_mean": A[..., 2].mean(1)}
    for k, f in feats.items():
        vv = conc(conc_parts(f[va], y[va], st[va]))
        sgn = 1 if vv >= 0.5 else -1
        res[f"heur_{k}"] = dict(val=max(vv, 1 - vv), sign=sgn, test=boot_conc([conc_parts(sgn * f[te], y[te], st[te])]))
    # distance-to-logged (cand 0) and to candidate mean: are "typical" candidates better?
    for k, ref in (("dist_to_cand_mean", None),):
        f = np.zeros(len(y))
        for s in np.unique(st):
            m = st == s
            f[m] = -np.linalg.norm(bA[m] - bA[m].mean(0), axis=1)
        vv = conc(conc_parts(f[va], y[va], st[va]))
        res[f"heur_{k}"] = dict(val=vv, test=boot_conc([conc_parts(f[te], y[te], st[te])]))
    print(json.dumps(res, indent=1, default=float))
    json.dump(res, open("/outputs/StochRL/t019/action_only_baselines.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

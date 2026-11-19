"""T023 value audit: frozen-VLA success predictor V_hat(s) as PBRS potential (B) and gate (C).

CPU only. Inputs: T017 buffer (/workspace/stochrl_t017) + T018 feature cache (/workspace/stochrl_t018/cache).
Stages:
  fit    : fit + freeze predictors (train), select hyper-params on val, Platt / isotonic calibration on val.
  audit  : calibration, within-time discrimination, progress quality, gate lead time, PBRS terminal check (test reported).
Split use: train = fit, val = selection / calibration / thresholds, test = report only.

MDP convention (checked in env code, see summary): terminated == success (PutOnPlateInScene25Main-v3.evaluate has no
"fail" key -> ManiSkill BaseEnv terminated = success); failure = TimeLimit truncation at 80 steps (bootstrapped).
So the only states that exist in the RL MDP are pre-success states; predictors are trained/evaluated on them.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(16)

SRC = Path("/workspace/stochrl_t017")
CACHE = Path("/workspace/stochrl_t018/cache")
WORK = Path("/workspace/stochrl_t023")
OUT = Path("/outputs/StochRL/t023")
K = 5
ND = 16
GAMMA = 0.99          # RLinf maniskill_ppo_openpi_pi05 algorithm.gamma (env-step discount)
LAMS = 10.0 ** np.arange(-5, 0.5, 0.5)
VLA_CANDS = ["vlm9_img_mean", "vlm13_img_mean", "vlm17_img_mean"]
PHASES = ["approach", "grasp_contact", "transport", "post_grasp_lost", "placed"]
SEED = 0


# ----------------------------------------------------------------------------- data
def load_data():
    lab = np.load(CACHE / "labels.npz", allow_pickle=True)
    ep = lab["episode"]
    eps = list(dict.fromkeys(ep.tolist()))
    assert len(eps) == 500 and all((ep[i * ND:(i + 1) * ND] == eps[i]).all() for i in range(500))
    assert (lab["decision"].reshape(500, ND) == np.arange(ND)).all()
    E = dict(eps=np.array(eps), split=lab["split"].reshape(500, ND)[:, 0],
             y=lab["success_once"].reshape(500, ND)[:, 0].astype(int),
             post=lab["success_before"].reshape(500, ND), phase=lab["phase5"].reshape(500, ND))
    tfs = np.full(500, -1)
    grasp = np.zeros((500, ND + 1), bool)
    phase17 = np.empty((500, ND + 1), dtype=object)
    gc = np.zeros((500, ND + 1)); cp = np.zeros((500, ND + 1)); objpos = np.zeros((500, ND + 1, 3)); cpxy = np.zeros((500, ND + 1))
    for i, e in enumerate(eps):
        z = np.load(SRC / "episodes" / f"{e}.npz")
        ss = z["step_success"]
        tfs[i] = int(np.argmax(ss)) if ss.any() else -1
        p = np.load(SRC / "privileged" / f"{e}.npz")
        grasp[i] = p["dec_grasping_now"].astype(bool)
        phase17[i] = p["dec_phase"].astype(str)
        gc[i] = p["dec_gripper_carrot_dist"]; cp[i] = p["dec_carrot_plate_dist"]; objpos[i] = p["dec_pos_carrot"]
        cpxy[i] = np.linalg.norm(p["dec_carrot_plate_rel"][:, :2], axis=1)
    E.update(tfs=tfs, grasp=grasp, phase17=phase17.astype(str), gc=gc, cp=cp, objpos=objpos, cpxy=cpxy)
    assert ((E["tfs"] >= 0) == (E["y"] == 1)).all()
    # pre-success mask: state at t=5d exists in terminate-on-success MDP iff no success observed yet (tfs >= 5d)
    t = np.arange(ND)[None] * 5
    pre = (E["tfs"][:, None] < 0) | (E["tfs"][:, None] >= t)
    assert (pre == ~E["post"]).all()
    E["pre"] = pre
    E["dstar"] = np.where(tfs >= 0, tfs // K, -1)            # success chunk index
    E["keff"] = np.where(tfs >= 0, tfs % K + 1, K)           # env steps in success chunk
    return E


def feats(key):
    return np.load(CACHE / f"feat_{key}.npy").astype(np.float32).reshape(500, ND, -1)


def poly2(X):
    n, d = X.shape
    iu = np.triu_indices(d)
    return np.concatenate([X, (X[:, :, None] * X[:, None, :])[:, iu[0], iu[1]]], 1)


# ----------------------------------------------------------------------------- models
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


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def pav(x, y):
    """isotonic regression (increasing) of y on x; returns knots (xs, ys) for np.interp."""
    o = np.argsort(x, kind="mergesort"); xs = x[o]; ys = y[o].astype(float)
    vals, wts, xl, xr = [], [], [], []
    for xi, yi in zip(xs, ys):
        vals.append(yi); wts.append(1.0); xl.append(xi); xr.append(xi)
        while len(vals) > 1 and vals[-2] >= vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            w = wts[-2] + wts[-1]
            vals[-2:] = [v]; wts[-2:] = [w]; xr[-2:] = [xr[-1]]; xl[-2:] = [xl[-2]]
    kx, ky = [], []
    for v, a, b in zip(vals, xl, xr):
        kx += [a, b]; ky += [v, v]
    return np.array(kx), np.array(ky)


def fit_platt(p, y):
    w = fit_logreg(logit(p)[:, None], y, 0.0)
    return float(w[0]), float(w[1])


def load_potential(name="vla", calib="platt", root=WORK / "predictors"):
    """Frozen potential for T024. Returns f(X)->Phi in (0,1).
    X: [N, d] raw feature named in the npz `feature` (vla: vlm17_img_mean fp32, VLM prefix layer 17 mean over 256 image
    tokens, same hook as e02_rollout). time/phase use the lookup table instead (see lookup.npz)."""
    P = dict(np.load(Path(root) / f"{name}.npz", allow_pickle=True))
    cal = json.load(open(Path(root) / "calibration.json"))[name]

    def f(X):
        X = np.asarray(X, np.float64)
        if "Q" in P:
            X = ((X - P["mu0"]) / P["sd0"]) @ P["Q"]
        Z = (X - P["mu"]) / P["sd"]
        if str(P["kind"]) == "poly2":
            Z = (poly2(Z) - P["mu2"]) / P["sd2"]
        v = sigmoid(Z @ P["W"] + P["b"])
        if calib == "platt":
            v = sigmoid(cal["platt_a"] * logit(v) + cal["platt_b"])
        elif calib == "iso":
            v = np.clip(np.interp(v, cal["iso_x"], cal["iso_y"]), 1e-3, 1 - 1e-3)
        return v
    return f


# ----------------------------------------------------------------------------- stage fit
def stage_fit(E):
    WORK.mkdir(parents=True, exist_ok=True); (WORK / "predictors").mkdir(exist_ok=True)
    sp = E["split"]; pre = E["pre"]; Y = np.repeat(E["y"][:, None], ND, 1)
    tr = (sp == "train")[:, None] & pre; va = (sp == "val")[:, None] & pre
    D = np.repeat(np.arange(ND)[None], 500, 0)
    out = {}   # name -> V[500,16] (uncalibrated)
    sel = {}

    def lin_model(name, X, cands=None, kind="linear"):
        # X: [500,16,d]
        Xs = X.reshape(-1, X.shape[-1]); m = tr.reshape(-1)
        mu = Xs[m].mean(0); sd = Xs[m].std(0) + 1e-6
        Z = (Xs - mu) / sd
        if kind == "poly2":
            Z = poly2(Z); mu2 = Z[m].mean(0); sd2 = Z[m].std(0) + 1e-6; Z = (Z - mu2) / sd2
        else:
            mu2 = sd2 = None
        y = Y.reshape(-1); v = va.reshape(-1)
        best = None; w = None
        for lam in LAMS[::-1]:      # warm start from strong reg
            w = fit_logreg(Z[m], y[m], lam, w)
            ll = logloss(sigmoid(Z[v] @ w[:-1] + w[-1]), y[v])
            if best is None or ll < best[0]:
                best = (ll, lam, w.copy())
        ll, lam, w = best
        V = sigmoid(Z @ w[:-1] + w[-1]).reshape(500, ND)
        return V, dict(W=w[:-1], b=w[-1], mu=mu, sd=sd, mu2=mu2, sd2=sd2, lam=lam, val_logloss=ll, kind=kind)

    # VLA: choose layer on val log-loss
    res = {}
    for key in VLA_CANDS:
        V, prm = lin_model("vla", feats(key))
        res[key] = (V, prm); print(key, "val logloss", prm["val_logloss"], "lam", prm["lam"], flush=True)
    vkey = min(res, key=lambda k: res[k][1]["val_logloss"])
    sel["vla_layer"] = vkey; sel["vla_layer_val_logloss"] = {k: res[k][1]["val_logloss"] for k in res}
    out["vla"], P = res[vkey]; P["feature"] = vkey
    np.savez(WORK / "predictors" / "vla.npz", **{k: v for k, v in P.items() if v is not None})
    # random projection of the same layer (21 dims, orthogonal, seed 0) - T018 C-RandomProj convention
    Xv = feats(vkey); Xs = Xv.reshape(-1, Xv.shape[-1]); m = tr.reshape(-1)
    mu = Xs[m].mean(0); sd = Xs[m].std(0) + 1e-6
    Q, _ = np.linalg.qr(np.random.RandomState(SEED).randn(Xs.shape[1], 21))
    Xr = ((Xs - mu) / sd) @ Q
    out["randproj"], P = lin_model("randproj", Xr.reshape(500, ND, 21)); P.update(feature=vkey, Q=Q, mu0=mu, sd0=sd)
    np.savez(WORK / "predictors" / "randproj.npz", **{k: v for k, v in P.items() if v is not None})
    # raw: qpos (poly2), qpos+t (poly2), rgb 32x24 (linear)
    for name, key, kind in [("raw_q", "raw_proprio", "poly2"), ("raw_qt", "raw_proprio_time", "poly2"),
                            ("raw_rgb", "raw_rgb32", "linear")]:
        out[name], P = lin_model(name, feats(key), kind=kind); P["feature"] = key
        np.savez(WORK / "predictors" / f"{name}.npz", **{k: v for k, v in P.items() if v is not None})
        print(name, P["val_logloss"], P["lam"], flush=True)
    # time-only: per-decision rate on train pre-success rows, Beta(1,1)
    rate_t = np.array([(Y[tr & (D == d)].sum() + 1) / ((tr & (D == d)).sum() + 2) for d in range(ND)])
    out["time"] = np.repeat(rate_t[None], 500, 0)
    # phase-only (privileged, analysis only)
    ph = E["phase"]
    rate_p = {p: (Y[tr & (ph == p)].sum() + 1) / ((tr & (ph == p)).sum() + 2) for p in PHASES}
    out["phase"] = np.vectorize(rate_p.get)(ph).astype(float)
    # phase x time with shrinkage (m=5) to the phase rate
    rate_pt = {}
    for p in PHASES:
        for d in range(ND):
            mm = tr & (ph == p) & (D == d)
            rate_pt[(p, d)] = (Y[mm].sum() + 5 * rate_p[p]) / (mm.sum() + 5)
    out["phase_time"] = np.array([[rate_pt[(ph[i, d], d)] for d in range(ND)] for i in range(500)])
    np.savez(WORK / "predictors" / "lookup.npz", time=rate_t, phase_names=np.array(PHASES),
             phase=np.array([rate_p[p] for p in PHASES]),
             phase_time=np.array([[rate_pt[(p, d)] for d in range(ND)] for p in PHASES]))
    # calibration on val (Platt, isotonic) for every predictor
    cal = {}
    Vcal = {}
    for name, V in out.items():
        a, b = fit_platt(V[va], Y[va])
        kx, ky = pav(V[va], Y[va].astype(float))
        cal[name] = dict(platt_a=a, platt_b=b, iso_x=kx.tolist(), iso_y=ky.tolist())
        Vcal[name] = V
        Vcal[name + "@platt"] = sigmoid(a * logit(V) + b)
        Vcal[name + "@iso"] = np.clip(np.interp(V, kx, ky), 1e-3, 1 - 1e-3)
    json.dump(cal, open(WORK / "predictors" / "calibration.json", "w"), indent=1)
    np.savez(WORK / "vhat_all.npz", episodes=E["eps"], split=E["split"], **{k: v.astype(np.float32) for k, v in Vcal.items()})
    json.dump(sel, open(WORK / "selection.json", "w"), indent=1)
    print("fit done; vla layer", vkey)


# ----------------------------------------------------------------------------- metrics
def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_within(s, y, g):
    from scipy.stats import rankdata
    U = 0.0; P = 0.0
    for gv in np.unique(g):
        m = g == gv; yy = y[m]; n1 = yy.sum(); n0 = len(yy) - n1
        if n1 == 0 or n0 == 0:
            continue
        r = rankdata(s[m]); U += r[yy == 1].sum() - n1 * (n1 + 1) / 2; P += n1 * n0
    return float(U / P) if P > 0 else np.nan


def auprc(s, y):
    o = np.argsort(-s, kind="mergesort"); yy = y[o]
    tp = np.cumsum(yy); prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / max(yy.sum(), 1))


def ece(p, y, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1); e = 0.0
    for k in range(nb):
        m = b == k
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def boot(fn, ep_idx, nboot=1000, seed=1):
    """episode bootstrap: fn(list of episode indices) -> scalar"""
    rs = np.random.RandomState(seed); vals = []
    for _ in range(nboot):
        vals.append(fn(rs.choice(ep_idx, len(ep_idx), replace=True)))
    vals = np.array(vals, float)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


# ----------------------------------------------------------------------------- stage audit
PRED_MAIN = ["vla", "raw_qt", "raw_q", "raw_rgb", "randproj", "time", "phase", "phase_time"]
COL = {"vla": "#2a78d6", "raw_qt": "#eb6834", "time": "#1baf7a", "phase_time": "#eda100",
       "randproj": "#e87ba4", "raw_rgb": "#008300", "raw_q": "#6250d6", "phase": "#e34948",
       "vla_shuf": "#888888"}
LABEL = {"vla": "VLA", "raw_qt": "raw qpos+t", "raw_q": "raw qpos", "raw_rgb": "raw RGB", "randproj": "VLA randproj21",
         "time": "time-only", "phase": "phase-only*", "phase_time": "phase×time*", "vla_shuf": "VLA shuffled-in-t"}


def stage_audit(E, nboot=1000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6})
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "fig").mkdir(exist_ok=True)
    Vall = dict(np.load(WORK / "vhat_all.npz", allow_pickle=True))
    sp = E["split"]; pre = E["pre"]; y = E["y"]; D = np.repeat(np.arange(ND)[None], 500, 0)
    PH = E["phase"]
    rows = []

    def rec(section, pred, metric, value, lo=np.nan, hi=np.nan, split="test", **kw):
        rows.append(dict(section=section, predictor=pred, metric=metric, value=value, lo=lo, hi=hi, split=split,
                         extra=json.dumps(kw) if kw else ""))

    # shuffled control: VLA platt values permuted across test/val pre-success states within the same decision index
    rs = np.random.RandomState(3)
    sh = Vall["vla@platt"].copy()
    for s in ["val", "test"]:
        for d in range(ND):
            m = (sp == s) & pre[:, d]
            idx = np.where(m)[0]; sh[idx, d] = sh[rs.permutation(idx), d]
    Vall["vla_shuf@platt"] = sh

    # ------------------------------------------------ 1-3 calibration + within-time discrimination
    test_eps = np.where(sp == "test")[0]
    Y = np.repeat(y[:, None], ND, 1)

    def flat(V, eps):
        m = pre[eps]; return V[eps][m], Y[eps][m], D[eps][m], PH[eps][m]

    calib_tab = []
    for name in PRED_MAIN:
        for var in ["", "@platt", "@iso"]:
            V = Vall[name + var]
            for s in ["val", "test"]:
                eps = np.where(sp == s)[0]
                p, yy, dd, pp = flat(V, eps)
                met = dict(auroc=auroc(p, yy), auprc=auprc(p, yy), brier=float(((p - yy) ** 2).mean()),
                           ece=ece(p, yy), logloss=logloss(p, yy), auroc_within_t=auroc_within(p, yy, dd),
                           auroc_within_phase=auroc_within(p, yy, pp),
                           auroc_within_phase_t=auroc_within(p, yy, np.char.add(pp, dd.astype(str))))
                for k, v in met.items():
                    lo = hi = np.nan
                    if s == "test" and var in ("", "@platt") and k in ("auroc", "auroc_within_t", "brier", "ece",
                                                                         "auroc_within_phase"):
                        def f(b, V=V, k=k):
                            p2, y2, d2, ph2 = flat(V, b)
                            if k == "auroc": return auroc(p2, y2)
                            if k == "brier": return float(((p2 - y2) ** 2).mean())
                            if k == "ece": return ece(p2, y2)
                            if k == "auroc_within_t": return auroc_within(p2, y2, d2)
                            return auroc_within(p2, y2, ph2)
                        lo, hi = boot(f, eps, nboot)
                    rec("calibration", name + var, k, v, lo, hi, s)
                if s == "test":
                    calib_tab.append((name + var, met))
    # per-decision AUROC curve (early detection)
    per_d = {}
    for name in ["vla", "raw_qt", "raw_rgb", "randproj", "phase", "phase_time"]:
        V = Vall[name + "@platt"]; c = []
        for d in range(ND):
            m = pre[test_eps, d]
            c.append(auroc(V[test_eps, d][m], y[test_eps][m]))
            rec("per_decision_auroc", name, f"auroc_d{d}", c[-1], n=int(m.sum()), npos=int(y[test_eps][m].sum()))
        per_d[name] = c

    # reliability figure
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (title, names) in zip(axs, [("VLA: raw / Platt / isotonic", ["vla", "vla@platt", "vla@iso"]),
                                        ("Platt-calibrated, by source", ["vla@platt", "raw_qt@platt", "raw_rgb@platt",
                                                                         "randproj@platt"]),
                                        ("time / phase references", ["time@platt", "phase@platt", "phase_time@platt",
                                                                     "vla@platt"])]):
        ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1, ls="--")
        for n in names:
            p, yy, _, _ = flat(Vall[n], test_eps)
            bins = np.minimum((p * 10).astype(int), 9)
            xs = [p[bins == k].mean() for k in range(10) if (bins == k).sum() >= 5]
            ys = [yy[bins == k].mean() for k in range(10) if (bins == k).sum() >= 5]
            base = n.split("@")[0]; style = {"": "-", "@platt": "-", "@iso": ":"}
            ls = ":" if n.endswith("@iso") else ("--" if "@" not in n else "-")
            ax.plot(xs, ys, ls, marker="o", ms=4, lw=2, color=COL[base],
                    label=f"{LABEL[base]}{n[len(base):]} ECE={ece(p, yy):.3f}")
        ax.set_title(title); ax.set_xlabel("predicted V̂"); ax.set_ylabel("empirical success"); ax.legend(fontsize=7)
    fig.suptitle("Reliability (test, pre-success states, n=%d)" % pre[test_eps].sum())
    fig.tight_layout(); fig.savefig(OUT / "fig" / "reliability.png", dpi=150); plt.close(fig)

    # ------------------------------------------------ 4 progress quality
    # nonterminal transitions d -> d+1 among pre-success states with d+1 also pre-success (in terminate MDP)
    trans = []
    for i in range(500):
        for d in range(ND - 1):
            if pre[i, d] and pre[i, d + 1]:
                trans.append((i, d))
    trans = np.array(trans)
    ti, td = trans[:, 0], trans[:, 1]
    g0 = E["grasp"][ti, td]; g1 = E["grasp"][ti, td + 1]
    p0 = E["phase17"][ti, td]; p1 = E["phase17"][ti, td + 1]
    # events from the privileged phase label (grasp may be gained+lost inside one chunk)
    ev = np.where((p1 == "transport") & (p0 != "transport"), "grasp+",
         np.where((p1 == "post_grasp_lost") & (p0 != "post_grasp_lost"), "drop",
         np.where((p0 == "transport") & (p1 == "transport"), "carry",
         np.where((p0 == "approach") & (p1 == "approach"), "approach",
         np.where(p0 == p1, "stay_other", "other")))))
    dgc = E["gc"][ti, td + 1] - E["gc"][ti, td]; dcp = E["cpxy"][ti, td + 1] - E["cpxy"][ti, td]
    tsplit = sp[ti]
    evc = {e: int(((ev == e) & (tsplit == "test")).sum()) for e in ["grasp+", "drop", "carry", "approach", "stay_other", "other"]}
    print("test transition events", evc)
    prog_tab = {}
    from scipy.stats import spearmanr
    for name in PRED_MAIN + ["vla_shuf"]:
        V = Vall[name + "@platt"]
        dphi = V[ti, td + 1] - V[ti, td]
        res = {}
        for s in ["val", "test"]:
            m = tsplit == s
            none = m & np.isin(ev, ["carry", "approach", "stay_other"])
            noise = float(dphi[none].std())
            gp = float(dphi[m & (ev == "grasp+")].mean()); dr = float(dphi[m & (ev == "drop")].mean())
            r = dict(mean_abs_dphi=float(np.abs(dphi[m]).mean()), sd_dphi=float(dphi[m].std()),
                     sd_dphi_noevent=noise, mean_dphi_grasp=gp, mean_dphi_drop=dr,
                     snr_grasp=gp / noise if noise > 0 else np.nan, snr_drop=-dr / noise if noise > 0 else np.nan,
                     auroc_grasp_vs_stay=auroc(dphi[m & np.isin(ev, ["grasp+", "carry", "approach", "stay_other"])],
                                               (ev[m & np.isin(ev, ["grasp+", "carry", "approach", "stay_other"])] == "grasp+").astype(int)),
                     auroc_drop_vs_stay=auroc(-dphi[m & np.isin(ev, ["drop", "carry", "approach", "stay_other"])],
                                               (ev[m & np.isin(ev, ["drop", "carry", "approach", "stay_other"])] == "drop").astype(int)),
                     # does dphi predict eventual success beyond time? (pooled within same d)
                     auroc_dphi_success_within_t=auroc_within(dphi[m], y[ti][m], td[m]),
                     spearman_dphi_vs_gripper_approach=float(spearmanr(dphi[m & (ev == "approach")],
                                                                        -dgc[m & (ev == "approach")])[0]),
                     spearman_dphi_vs_obj_to_plate_xy_carry=float(spearmanr(dphi[m & (ev == "carry")],
                                                                          -dcp[m & (ev == "carry")])[0]))
            # within-d residual: Fisher ratio of success vs fail, dphi centred per d
            c = dphi[m].copy(); ddm = td[m]
            for d in range(ND):
                k = ddm == d
                if k.any(): c[k] -= c[k].mean()
            ys = y[ti][m]
            pooled = np.sqrt(0.5 * (c[ys == 1].var() + c[ys == 0].var()))
            r["cohen_d_succ_vs_fail_within_t"] = float((c[ys == 1].mean() - c[ys == 0].mean()) / pooled) if pooled > 0 else 0.0
            for k, v in r.items():
                lo = hi = np.nan
                if s == "test" and k in ("snr_grasp", "snr_drop", "auroc_dphi_success_within_t"):
                    te = np.where(sp == "test")[0]
                    pos = {e: np.where(ti == e)[0] for e in te}

                    def f(b, k=k):
                        idx = np.concatenate([pos[e] for e in b])
                        dp = dphi[idx]; evb = ev[idx]
                        nz = dp[np.isin(evb, ["carry", "approach", "stay_other"])].std()
                        if k == "snr_grasp": return dp[evb == "grasp+"].mean() / nz
                        if k == "snr_drop": return -dp[evb == "drop"].mean() / nz
                        # bootstrap duplicates episodes; within-d grouping on duplicated rows is fine
                        return auroc_within(dp, y[ti[idx]], td[idx])
                    lo, hi = boot(f, te, nboot)
                rec("progress", name, k, v, lo, hi, s)
            res[s] = r
        prog_tab[name] = res
    # fraction of progress-like dphi signal: success-chunk shaping is separate (PBRS section)

    # pre-success dip: transition s_{d*-1} -> s_{d*} (last non-terminal chunk) in success episodes
    for name in PRED_MAIN + ["vla_shuf"]:
        V = Vall[name + "@platt"]
        for s in ["val", "test"]:
            es = np.where((sp == s) & (y == 1) & (E["dstar"] >= 1))[0]
            dd = V[es, E["dstar"][es]] - V[es, E["dstar"][es] - 1]
            other = E["dstar"][es] >= 2
            dd2 = V[es[other], E["dstar"][es[other]] - 1] - V[es[other], E["dstar"][es[other]] - 2]
            rec("late_dip", name, "median_dphi_last_chunk_before_success", float(np.median(dd)), split=s, n=len(es))
            rec("late_dip", name, "frac_negative_last_chunk", float((dd < 0).mean()), split=s, n=len(es))
            rec("late_dip", name, "frac_below_-0.2_last_chunk", float((dd < -0.2).mean()), split=s, n=len(es))
            rec("late_dip", name, "median_dphi_second_last_chunk", float(np.median(dd2)), split=s, n=int(other.sum()))
            rec("late_dip", name, "median_phi_at_success_chunk_start", float(np.median(V[es, E["dstar"][es]])), split=s)

    # Φ curves figure
    names_curve = ["vla", "raw_qt", "time", "phase_time"]
    fig, axs = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
    for j, n in enumerate(names_curve):
        V = Vall[n + "@platt"]
        for yv, colr, lab in [(1, "#2a78d6", "success ep"), (0, "#eb6834", "failure ep")]:
            eps = test_eps[y[test_eps] == yv]
            med, lo, hi = [], [], []
            for d in range(ND):
                v = V[eps, d][pre[eps, d]]
                if len(v) >= 5:
                    med.append(np.median(v)); lo.append(np.percentile(v, 10)); hi.append(np.percentile(v, 90))
                else:
                    med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            axs[0, j].plot(range(ND), med, color=colr, lw=2, label=lab)
            axs[0, j].fill_between(range(ND), lo, hi, color=colr, alpha=0.15)
        axs[0, j].set_title(f"{LABEL[n]} — by decision index"); axs[0, j].set_xlabel("decision d (5 env steps)")
        # bottom: success aligned to success chunk; failure aligned to first drop (if any)
        eps = test_eps[y[test_eps] == 1]
        rel = np.arange(-10, 1)
        mat = np.full((len(eps), len(rel)), np.nan)
        for a, e in enumerate(eps):
            for b, r in enumerate(rel):
                d = E["dstar"][e] + r
                if 0 <= d < ND: mat[a, b] = V[e, d]
        cnt = (~np.isnan(mat)).sum(0); ok = cnt >= 5
        axs[1, j].plot(rel[ok], np.nanmedian(mat, 0)[ok], color="#2a78d6", lw=2, label="success: d − d*(success chunk)")
        axs[1, j].fill_between(rel[ok], np.nanpercentile(mat, 10, 0)[ok], np.nanpercentile(mat, 90, 0)[ok],
                               color="#2a78d6", alpha=0.15)
        epsf = [e for e in test_eps[y[test_eps] == 0] if (E["phase"][e] == "post_grasp_lost").any()]
        relf = np.arange(-6, 6); matf = np.full((len(epsf), len(relf)), np.nan)
        for a, e in enumerate(epsf):
            d0 = int(np.argmax(E["phase"][e] == "post_grasp_lost"))
            for b, r in enumerate(relf):
                if 0 <= d0 + r < ND: matf[a, b] = V[e, d0 + r]
        cnt = (~np.isnan(matf)).sum(0); ok = cnt >= 4
        axs[1, j].plot(relf[ok], np.nanmedian(matf, 0)[ok], color="#eb6834", lw=2, label="failure w/ drop: d − d_drop")
        axs[1, j].fill_between(relf[ok], np.nanpercentile(matf, 10, 0)[ok], np.nanpercentile(matf, 90, 0)[ok],
                               color="#eb6834", alpha=0.15)
        axs[1, j].axvline(0, color="#999999", lw=1, ls="--"); axs[1, j].set_xlabel("decisions relative to event")
    axs[0, 0].set_ylabel("Φ = V̂ (Platt)"); axs[1, 0].set_ylabel("Φ = V̂ (Platt)")
    axs[0, 0].legend(fontsize=7); axs[1, 0].legend(fontsize=7)
    fig.suptitle("Φ trajectories on test (median, 10–90% band; pre-success states only)")
    fig.tight_layout(); fig.savefig(OUT / "fig" / "phi_curves.png", dpi=150); plt.close(fig)

    # phase_progress figure: Φ by phase + ΔΦ by event
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    names_pp = ["vla", "raw_qt", "randproj", "time", "vla_shuf"]
    w = 0.15
    for j, n in enumerate(names_pp):
        V = Vall[n + "@platt"]
        data = [V[test_eps][pre[test_eps] & (PH[test_eps] == p)] for p in PHASES[:4]]
        pos = np.arange(4) + (j - 2) * w
        bp = axs[0].boxplot(data, positions=pos, widths=w * 0.9, patch_artist=True, showfliers=False)
        for b in bp["boxes"]: b.set_facecolor(COL[n]); b.set_alpha(0.6)
        for b in bp["medians"]: b.set_color("#0b0b0b")
        axs[0].plot([], [], color=COL[n], lw=6, alpha=0.6, label=LABEL[n])
        dphi = V[ti, td + 1] - V[ti, td]; mt = tsplit == "test"
        evs = ["approach", "grasp+", "carry", "drop"]
        data = [dphi[mt & (ev == e)] for e in evs]
        bp = axs[1].boxplot(data, positions=np.arange(4) + (j - 2) * w, widths=w * 0.9, patch_artist=True,
                            showfliers=False)
        for b in bp["boxes"]: b.set_facecolor(COL[n]); b.set_alpha(0.6)
        for b in bp["medians"]: b.set_color("#0b0b0b")
    axs[0].set_xticks(range(4)); axs[0].set_xticklabels(PHASES[:4]); axs[0].set_ylabel("Φ (Platt)")
    axs[0].set_title("Φ by phase (test, pre-success)"); axs[0].legend(fontsize=7, loc="upper left")
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels([f"{e}\n(n={evc[e]})" for e in evs])
    axs[1].axhline(0, color="#999999", lw=1); axs[1].set_ylabel("ΔΦ = Φ(s_{d+1}) − Φ(s_d)")
    axs[1].set_title("ΔΦ per K=5 chunk by event (test, non-terminal)")
    fig.tight_layout(); fig.savefig(OUT / "fig" / "phase_progress.png", dpi=150); plt.close(fig)

    # ------------------------------------------------ 5 gate audit (C)
    def ftype(e):
        if y[e] == 1: return "success"
        if (E["phase"][e] == "post_grasp_lost").any() or (E["phase17"][e] == "post_grasp_lost").any(): return "drop"
        if E["grasp"][e].any(): return "hold_timeout"
        return "no_grasp"
    FT = np.array([ftype(e) for e in range(500)])
    for s in ["train", "val", "test"]:
        for t in ["success", "drop", "hold_timeout", "no_grasp"]:
            rec("failure_types", "-", t, int(((sp == s) & (FT == t)).sum()), split=s)
    drop_d = np.array([int(np.argmax(E["phase"][e] == "post_grasp_lost")) if (E["phase"][e] == "post_grasp_lost").any()
                       else -1 for e in range(500)])

    def scores(V, gate):
        Ssc = np.full((500, ND), np.nan)
        if gate == "low":
            Ssc = -V.copy()
        else:
            Ssc[:, 1:] = -(V[:, 1:] - V[:, :-1]); Ssc[:, 0] = -np.inf
        Ssc[~pre] = np.nan
        return Ssc

    gate_rows = []
    lead_hist = {}
    budgets = [0.10, 0.25, 0.50]
    gate_preds = ["vla", "raw_qt", "raw_rgb", "randproj", "time", "phase_time"]
    for name in gate_preds:
        V = Vall[name + "@platt"]
        for gate in ["low", "drop"]:
            if gate == "drop" and name == "time":
                continue
            S = scores(V, gate)
            vmask = (sp == "val")[:, None] & pre
            vs = S[vmask]
            veps = np.where(sp == "val")[0]
            vmax_s = np.nanmax(np.where(pre[veps], S[veps], -np.inf), 1)[y[veps] == 1]
            for bud in budgets + ["fpr0.10", "fpr0.25"]:
                fin = vs[np.isfinite(vs)]
                if isinstance(bud, str):
                    # threshold on val: episode FPR (success eps alerted before success) = target
                    f_ = float(bud[3:]); thr = float(np.quantile(vmax_s[np.isfinite(vmax_s)], 1 - f_))
                else:
                    # threshold on val: alert when score > thr, alert rate = budget over val pre-success decisions
                    thr = np.quantile(fin, 1 - bud * len(vs) / len(fin)) if bud * len(vs) < len(fin) else -np.inf
                for s in ["val", "test"]:
                    eps = np.where(sp == s)[0]
                    A = (S[eps] > thr) & pre[eps]
                    first = np.where(A.any(1), A.argmax(1), -1)
                    fail = y[eps] == 0
                    r = dict(threshold=float(-thr) if gate == "low" else float(thr),
                             alert_rate=float(A.sum() / pre[eps].sum()),
                             recall=float((first[fail] >= 0).mean()), fpr=float((first[~fail] >= 0).mean()),
                             alert_precision=float(A[fail].sum() / max(A.sum(), 1)),
                             median_first_alert_d_fail=float(np.median(first[fail][first[fail] >= 0])) if (first[fail] >= 0).any() else np.nan)
                    dm = FT[eps] == "drop"
                    dd = drop_d[eps][dm]; fa = first[dm]
                    before = (fa >= 0) & (fa < dd)
                    r["drop_n"] = int(dm.sum())
                    r["drop_alert_before_frac"] = float(before.mean()) if dm.any() else np.nan
                    r["drop_lead_steps_median"] = float(np.median((dd - fa)[before]) * K) if before.any() else np.nan
                    # matched random gate expectation (same per-decision rate on test)
                    rr = r["alert_rate"]
                    r["random_drop_alert_before_frac"] = float(np.mean(1 - (1 - rr) ** dd)) if dm.any() else np.nan
                    npre = pre[eps].sum(1)
                    r["random_recall"] = float(np.mean(1 - (1 - rr) ** npre[fail]))
                    r["random_fpr"] = float(np.mean(1 - (1 - rr) ** npre[~fail]))
                    r["median_first_alert_d_no_grasp"] = float(np.median(first[FT[eps] == "no_grasp"][first[FT[eps] == "no_grasp"] >= 0])) if (first[FT[eps] == "no_grasp"] >= 0).any() else np.nan
                    for t in ["no_grasp", "hold_timeout"]:
                        mt_ = FT[eps] == t
                        r[f"recall_{t}"] = float((first[mt_] >= 0).mean()) if mt_.any() else np.nan
                    if s == "test":
                        lead_hist[(name, gate, bud)] = (dd - fa)[fa >= 0] * K
                    for k, v in r.items():
                        rec("gate", f"{name}:{gate}", k, v, split=s, budget=bud)
                    gate_rows.append(dict(pred=name, gate=gate, budget=bud, split=s, **r))
    # imminent-drop anticipation (decision level): states grasping at s_d, label = drop at s_{d+1}
    imm = {}
    for name in ["vla", "raw_qt", "raw_rgb", "randproj", "time", "phase_time"]:
        V = Vall[name + "@platt"]
        for s in ["val", "test"]:
            m = (tsplit == s) & np.isin(p0, ["transport", "grasp_contact"])
            lab_ = (ev[m] == "drop").astype(int)
            a1 = auroc(-V[ti[m], td[m]], lab_)
            prev = np.where(td[m] > 0, V[ti[m], np.maximum(td[m] - 1, 0)], V[ti[m], td[m]])
            a2 = auroc(-(V[ti[m], td[m]] - prev), lab_)
            rec("imminent_drop", name, "auroc_lowV", a1, split=s, n=int(m.sum()), npos=int(lab_.sum()))
            rec("imminent_drop", name, "auroc_Vdrop", a2, split=s, n=int(m.sum()), npos=int(lab_.sum()))
            imm[(name, s)] = (a1, a2, int(m.sum()), int(lab_.sum()))

    # gate figure: per-decision AUROC, episode recall-vs-FPR sweep (test), lead histogram
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for n, c in per_d.items():
        axs[0].plot(range(ND), c, marker="o", ms=3, lw=2 if n == "vla" else 1.2, color=COL[n], label=LABEL[n])
    axs[0].axhline(0.5, color="#999999", lw=1, ls="--")
    axs[0].set_xlabel("decision d"); axs[0].set_ylabel("AUROC eventual success (same d)")
    axs[0].set_title("Early detection: AUROC within each decision (test)"); axs[0].legend(fontsize=7)
    for n in ["vla", "raw_qt", "randproj", "time"]:
        V = Vall[n + "@platt"]
        for gate, ls in [("low", "-"), ("drop", "--")]:
            if gate == "drop" and n == "time": continue
            S = scores(V, gate)[test_eps]
            fin = np.unique(S[np.isfinite(S)])
            thr = np.quantile(fin, np.linspace(0, 1, 60))
            pts = []
            for th in thr:
                A = (S > th) & pre[test_eps]; any_ = A.any(1)
                pts.append(((any_[y[test_eps] == 1]).mean(), (any_[y[test_eps] == 0]).mean()))
            pts = np.array(pts)
            axs[1].plot(pts[:, 0], pts[:, 1], ls, color=COL[n], lw=1.6, label=f"{LABEL[n]} {gate}")
    for gr in gate_rows:
        if gr["split"] == "test" and gr["pred"] == "vla":
            axs[1].plot(gr["fpr"], gr["recall"], "o" if gr["gate"] == "low" else "s", color="#0b0b0b", ms=5)
    axs[1].plot([0, 1], [0, 1], color="#999999", lw=1, ls=":")
    axs[1].set_xlabel("episode FPR (success eps alerted before success)"); axs[1].set_ylabel("episode recall (failures alerted)")
    axs[1].set_title("Gate sweep (test); black = VLA at val budgets 10/25/50%"); axs[1].legend(fontsize=6)
    for n, c in [("vla", COL["vla"]), ("raw_qt", COL["raw_qt"])]:
        h = lead_hist.get((n, "low", 0.25))
        if h is not None and len(h):
            axs[2].hist(h, bins=np.arange(-80, 85, 10), alpha=0.6, color=c, label=f"{LABEL[n]} low-V @25% (n={len(h)})")
    axs[2].axvline(0, color="#999999", lw=1, ls="--")
    axs[2].set_xlabel("lead time = drop step − first alert step (env steps; >0 = before drop)")
    axs[2].set_ylabel("failed-by-drop episodes (test)"); axs[2].set_title("Lead time to grasp loss"); axs[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig" / "lead_time.png", dpi=150); plt.close(fig)

    # ------------------------------------------------ paired VLA - comparator differences (test, episode bootstrap)
    te = np.where(sp == "test")[0]
    pos = {e: np.where(ti == e)[0] for e in te}
    dphis = {n: Vall[n + "@platt"][ti, td + 1] - Vall[n + "@platt"][ti, td] for n in PRED_MAIN + ["vla_shuf"]}

    def m_within_t(n, b):
        p2, y2, d2, _ = flat(Vall[n + "@platt"], b); return auroc_within(p2, y2, d2)

    def m_within_pt(n, b):
        p2, y2, d2, ph2 = flat(Vall[n + "@platt"], b); return auroc_within(p2, y2, np.char.add(ph2, d2.astype(str)))

    def m_dphi(n, b):
        idx = np.concatenate([pos[e] for e in b]); return auroc_within(dphis[n][idx], y[ti[idx]], td[idx])

    def m_snr_drop(n, b):
        idx = np.concatenate([pos[e] for e in b]); dp = dphis[n][idx]; evb = ev[idx]
        return -dp[evb == "drop"].mean() / dp[np.isin(evb, ["carry", "approach", "stay_other"])].std()

    paired = {}
    for comp in ["raw_q", "raw_qt", "randproj", "time", "vla_shuf", "phase_time"]:
        for mn, fn in [("auroc_within_t", m_within_t), ("auroc_within_phase_t", m_within_pt),
                       ("dphi_auroc_success_within_t", m_dphi), ("snr_drop", m_snr_drop)]:
            if comp == "vla_shuf" and mn in ("auroc_within_t", "auroc_within_phase_t"):
                continue
            v = fn("vla", te) - fn(comp, te)
            lo, hi = boot(lambda b: fn("vla", b) - fn(comp, b), te, nboot, seed=7)
            rec("paired_diff", f"vla-{comp}", mn, v, lo, hi)
            paired[f"vla-{comp}|{mn}"] = (v, lo, hi)

    # ------------------------------------------------ 6 PBRS terminal handling (B)
    pbrs_rows = []
    hist_store = {}
    for name in ["vla", "raw_qt", "time", "phase_time", "vla_shuf"]:
        V = Vall[name + "@platt"]
        for phiT in [0.0, 1.0]:
            for beta in [0.1, 0.3, 0.5, 1.0]:
                for s in ["test"]:
                    eps = np.where(sp == s)[0]
                    succ_net, succ_shape, ep_sum_s, ep_sum_f, ret_s, ret_f, nonterm, nonterm_f = [], [], [], [], [], [], [], []
                    for e in eps:
                        tot = 0.0; disc = 0.0; tcur = 0
                        last = E["dstar"][e] if y[e] == 1 else ND - 1
                        for d in range(last + 1):
                            if y[e] == 1 and d == E["dstar"][e]:
                                ke = E["keff"][e]
                                r = GAMMA ** (ke - 1) * 1.0 + beta * (GAMMA ** ke * phiT - V[e, d])
                                succ_net.append(r); succ_shape.append(beta * (GAMMA ** ke * phiT - V[e, d]))
                            else:
                                nxt = V[e, d + 1] if d + 1 < ND else V[e, d]   # s_80 VLA feature not cached -> Φ(s_75)
                                r = beta * (GAMMA ** K * nxt - V[e, d])
                                nonterm.append(r)
                                if y[e] == 0: nonterm_f.append(r)
                                ke = K
                            tot += r; disc += GAMMA ** tcur * r; tcur += ke
                        (ep_sum_s if y[e] == 1 else ep_sum_f).append(tot)
                        (ret_s if y[e] == 1 else ret_f).append(disc)
                    succ_net = np.array(succ_net); nonterm = np.array(nonterm); nonterm_f = np.array(nonterm_f)
                    r = dict(pred=name, phiT=phiT, beta=beta, split=s,
                             succ_step_net_median=float(np.median(succ_net)), succ_step_net_min=float(succ_net.min()),
                             succ_step_net_p10=float(np.percentile(succ_net, 10)),
                             succ_step_shaping_median=float(np.median(succ_shape)),
                             frac_succ_net_below_0p2=float((succ_net < 0.2).mean()),
                             nonterm_p95=float(np.percentile(nonterm, 95)), nonterm_max_fail=float(nonterm_f.max()),
                             nonterm_mean_abs=float(np.abs(nonterm).mean()),
                             ratio_succnet_to_nonterm_p95=float(np.median(succ_net) / max(np.percentile(nonterm, 95), 1e-9)),
                             ep_sum_success_median=float(np.median(ep_sum_s)), ep_sum_fail_median=float(np.median(ep_sum_f)),
                             ep_sum_fail_max=float(np.max(ep_sum_f)),
                             disc_return_success_median=float(np.median(ret_s)), disc_return_fail_median=float(np.median(ret_f)),
                             disc_return_fail_max=float(np.max(ret_f)))
                    pbrs_rows.append(r)
                    for k, v in r.items():
                        if k not in ("pred", "phiT", "beta", "split"):
                            rec("pbrs", name, k, v, split=s, phiT=phiT, beta=beta)
                    if name == "vla" and beta == 1.0:
                        hist_store[phiT] = (succ_net, nonterm)
    # baseline: sparse only (beta=0): success step = gamma^(k-1)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
    for ax, phiT in zip(axs, [0.0, 1.0]):
        sn, nt = hist_store[phiT]
        bins = np.linspace(-1.2, 2.2, 60)
        ax.hist(nt, bins=bins, color="#888888", alpha=0.7, label=f"non-terminal chunks (n={len(nt)})", log=True)
        ax.hist(sn, bins=bins, color="#2a78d6", alpha=0.8, label=f"success chunk net (n={len(sn)})", log=True)
        ax.axvline(1.0, color="#0b0b0b", lw=1, ls="--")
        ax.set_title(f"VLA Φ, β=1, Φ_terminal={phiT:g} (test)"); ax.set_xlabel("shaped chunk reward"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig" / "pbrs_reward_hist.png", dpi=150); plt.close(fig)

    import csv
    with open(OUT / "value_audit.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    json.dump(dict(calib=calib_tab, prog=prog_tab, gate=gate_rows, pbrs=pbrs_rows, imminent={f"{k[0]}|{k[1]}": v for k, v in imm.items()},
                   per_d=per_d, events=evc, paired=paired), open(OUT / "audit.json", "w"), indent=1, default=float)
    print("audit done:", len(rows), "rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fit", "audit", "all"], default="all")
    ap.add_argument("--nboot", type=int, default=1000)
    a = ap.parse_args()
    E = load_data()
    if a.stage in ("fit", "all"):
        stage_fit(E)
    if a.stage in ("audit", "all"):
        stage_audit(E, a.nboot)

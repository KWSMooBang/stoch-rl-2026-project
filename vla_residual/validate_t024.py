"""T024 pre-training checks for PBRS (results -> /outputs/StochRL/t024/val/).

  --mode offline  (CPU)  torch Potential == T023 V_hat (vhat_all.npz, Platt) on the T018 cache; shaped_reward reproduces
                         the T023 success-chunk net reward table (Phi_T = 0 / 1); beta = 0 identity.
  --mode online   (GPU)  train-split ResidualVLAEnv with PBRS (vla, beta 1): rollouts with PPO's initial exploration
                         noise (u ~ N(0, exp(-1.5)^2), zero mean = B0). Checks success / truncation / beta=0 handling on
                         real transitions and tabulates Phi and beta*dPhi for vla, raw_q, time vs the sparse reward.
  CUDA_VISIBLE_DEVICES=0 python -m vla_residual.validate_t024 --mode online --batch 32 --rollouts 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vla_residual.potential import PRED_DIR, Potential, shaped_reward

OUT = Path("/outputs/StochRL/t024/val")
CACHE = Path("/workspace/stochrl_t018/cache")
T017 = Path("/workspace/stochrl_t017")
VHAT = Path("/outputs/StochRL/t023/vhat_all.npz")
ND, K = 16, 5
NAMES = ["vla", "raw_q", "time"]


def q(x, p):
    return float(np.percentile(np.asarray(x, float), p)) if len(x) else None


def offline():
    out = {}
    V = np.load(VHAT, allow_pickle=True)
    eps = V["episodes"]
    lab = np.load(CACHE / "labels.npz", allow_pickle=True)
    ep_c = list(dict.fromkeys(lab["episode"].tolist()))
    assert list(ep_c) == list(eps), "cache episode order differs from vhat_all"
    X = {"vla": np.load(CACHE / "feat_vlm17_img_mean.npy").astype(np.float32),
         "raw_q": np.load(CACHE / "feat_raw_proprio.npy").astype(np.float32)}
    dec = torch.arange(ND).repeat(len(eps))
    for n in NAMES:
        pot = Potential(n)
        phi = pot(feat_raw=torch.as_tensor(X["vla"]) if n == "vla" else None,
                  qpos=torch.as_tensor(X["raw_q"]) if n == "raw_q" else None, dec=dec).numpy().reshape(len(eps), ND)
        ref = V[n + "@platt"]
        out[f"port_maxabs_{n}"] = float(np.abs(phi - ref).max())
    # success-chunk net reward (T023 sec. 5 convention: test split, gamma 0.99/step, chunk Gamma = gamma^K_eff)
    vh = V["vla@platt"]
    sp = V["split"]
    rows = {"phi_prev": [], "keff": []}
    for i, e in enumerate(eps):
        if sp[i] != "test":
            continue
        z = np.load(T017 / "episodes" / f"{e}.npz")
        ss = z["step_success"]
        if not ss.any():
            continue
        tfs = int(np.argmax(ss))
        rows["phi_prev"].append(vh[i, tfs // K]); rows["keff"].append(tfs % K + 1)
    pp = torch.tensor(rows["phi_prev"], dtype=torch.float64); ke = torch.tensor(rows["keff"], dtype=torch.float64)
    term = torch.ones_like(pp, dtype=torch.bool)
    res = {}
    for beta in (0.1, 0.3, 1.0):
        for pt in (0.0, 1.0):
            g = 0.99 ** ke
            r = 0.99 ** (ke - 1)  # success at the last executed step of the chunk
            net, _ = shaped_reward(r, pp, torch.zeros_like(pp), term, beta, g, pt)
            res[f"t023conv_beta{beta}_phiT{int(pt)}"] = dict(median=float(net.median()), min=float(net.min()))
            # T024 RL convention: decision-level reward 1, gamma_dec = 0.96 (T022 PPO algorithm.gamma)
            net2, _ = shaped_reward(torch.ones_like(pp), pp, torch.zeros_like(pp), term, beta, 0.96, pt)
            res[f"t024conv_beta{beta}_phiT{int(pt)}"] = dict(median=float(net2.median()), min=float(net2.min()))
    out["success_chunk_net"] = res
    out["n_success_test"] = len(pp)
    # beta = 0: shaped_reward adds an exact zero; env bypasses it anyway (returns the sparse tensor object)
    r = torch.tensor([0.0, 1.0, 0.0, 1.0]); ph = torch.rand(4); pn = torch.rand(4)
    r2, shp = shaped_reward(r, ph, pn, torch.tensor([False, True, False, True]), 0.0, 0.96)
    out["beta0_identity_function"] = bool(torch.equal(r2, r))
    return out


def online(batch, rollouts, seed):
    from vla_residual import core as Vc
    from vla_residual.validate import make_cfg

    cfg = make_cfg("train", auto_reset=False, ignore_term=False,
                   pbrs=dict(potential="vla", beta=1.0, gamma=0.96, phi_term=1.0))
    cfg.seed = seed
    env = Vc.ResidualVLAEnv(cfg, batch, 0, 1, None)
    pots = {n: Potential(n, device=env.device) for n in NAMES}
    g = torch.Generator(device="cpu").manual_seed(seed)
    std = float(np.exp(-1.5))
    chk = dict(success_net_ok=True, trunc_uses_vhat=True, beta0_bitwise=True, no_shaping_after_end=True,
               sparse_eq_t022=True)
    rec = {n: dict(phi_d0=[], dphi=[], phi=[]) for n in NAMES}
    succ_net, trunc_phi, sparse_nz, n_tr = [], [], [], 0
    phi_succ_prev = []
    feat_mean = []
    for ro in range(rollouts):
        env.pbrs["beta"] = 0.0 if ro == rollouts - 1 and rollouts > 1 else 1.0  # last rollout: beta = 0 path
        beta = env.pbrs["beta"]
        env.reset()
        cur = {n: pots[n](env.feat_raw, env.qpos, env.dec) for n in NAMES}
        assert torch.allclose(cur["vla"], env.phi), "env Phi(s0) differs from Potential"
        for n in NAMES:
            rec[n]["phi_d0"] += cur[n].tolist()
        feat_mean.append(env.feat_raw.mean(0).cpu())
        for d in range(Vc.T_DEC):
            u = torch.randn(batch, 1, 7, generator=g).to(env.device) * std
            _, r, te, tr, inf = env.chunk_step(u)
            ri = env._last_rinfo
            r = r[:, 0]
            nxt = {n: pots[n](env.feat_raw, env.qpos, env.dec) for n in NAMES}
            act = ri["active"]
            newly = ri["sparse"]
            chk["sparse_eq_t022"] &= bool(((newly == 1) == (te[:, 0] & act)).all())
            if beta == 0.0:
                chk["beta0_bitwise"] &= bool(torch.equal(r, ri["sparse"]))
            else:
                exp_ = ri["sparse"] + beta * (0.96 * torch.where(te[:, 0], torch.ones_like(r), nxt["vla"]) - ri["phi_prev"])
                exp_ = torch.where(act, exp_, ri["sparse"])
                chk["success_net_ok"] &= bool(torch.allclose(r, exp_, atol=1e-6))
                s_m = te[:, 0] & act
                succ_net += r[s_m].tolist(); phi_succ_prev += ri["phi_prev"][s_m].tolist()
                t_m = tr[:, 0] & act
                if t_m.any():
                    chk["trunc_uses_vhat"] &= bool(torch.allclose(ri["phi_next"][t_m], nxt["vla"][t_m]))
                    trunc_phi += ri["phi_next"][t_m].tolist(); n_tr += int(t_m.sum())
            chk["no_shaping_after_end"] &= bool((ri["shp"][~act] == 0).all())
            nt = act & ~te[:, 0]  # non-terminal transitions (incl. truncation)
            for n in NAMES:
                rec[n]["dphi"] += (0.96 * nxt[n] - cur[n])[nt].tolist()
                rec[n]["phi"] += cur[n][act].tolist()
            sparse_nz += ri["sparse"][act].tolist()
            cur = nxt
        print(f"[online] rollout {ro} beta {beta} success_once {env.success_once.float().mean():.3f}", flush=True)
    st = np.load(Vc.REPO / "vla_residual/feat_stats_vlm17_img_mean.npz")
    fm = torch.stack(feat_mean).mean(0).numpy()
    z = (fm - st["mean"]) / st["std"]
    V = np.load(VHAT, allow_pickle=True)
    table = {}
    for n in NAMES:
        dp = np.array(rec[n]["dphi"])
        table[n] = dict(phi_d0_mean=float(np.mean(rec[n]["phi_d0"])), phi_d0_sd=float(np.std(rec[n]["phi_d0"])),
                        phi_mean=float(np.mean(rec[n]["phi"])), phi_p05=q(rec[n]["phi"], 5), phi_p95=q(rec[n]["phi"], 95),
                        dphi_absmean=float(np.abs(dp).mean()), dphi_sd=float(dp.std()), dphi_p05=q(dp, 5), dphi_p95=q(dp, 95),
                        t023_vhat_d0_mean=float(V[n + "@platt"][:, 0].mean()))
    nsp = np.array(sparse_nz)
    out = dict(checks=chk, n_transitions_active=len(nsp), sparse_rate_per_transition=float(nsp.mean()),
               sparse_abs_mean=float(np.abs(nsp).mean()),
               success_step_net_beta1=dict(n=len(succ_net), median=q(succ_net, 50), min=min(succ_net) if succ_net else None,
                                           phi_prev_median=q(phi_succ_prev, 50)),
               trunc=dict(n=n_tr, phi_next_mean=float(np.mean(trunc_phi)) if trunc_phi else None,
                          phi_next_max=max(trunc_phi) if trunc_phi else None),
               feat_raw_mean_z_vs_t017_stats=dict(absmean=float(np.abs(z).mean()), absmax=float(np.abs(z).max())),
               potential_table=table, batch=batch, rollouts=rollouts,
               peak_gb=torch.cuda.max_memory_allocated() / 1e9)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["offline", "online"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = offline() if a.mode == "offline" else online(a.batch, a.rollouts, a.seed)
    res["gpu"] = __import__("os").environ.get("CUDA_VISIBLE_DEVICES")
    Path(OUT, f"{a.mode}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

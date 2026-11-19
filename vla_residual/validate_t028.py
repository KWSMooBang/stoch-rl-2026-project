"""T028 pre-training checks: VLA-potential PBRS under RLinf GRPO (results -> /outputs/StochRL/t028/val/).

  --mode grpo     (GPU)  train-split ResidualVLAEnv, group 8 (T022 GRPO layout), PBRS vla beta 1, GRPO's initial
                         exploration noise. Feeds the real transitions through RLinf's own GRPO path
                         (calculate_adv_and_returns / calculate_scores) and checks the episode score against
                           (a) sum of shaped rewards up to the first done (undiscounted)  -> what RLinf computes
                           (b) closed form  sum r + beta (gamma Phi'_T - Phi_0) + beta (gamma - 1) sum_{t=1}^{T-1} Phi_t
                           (c) the discounted telescoping form  sum r + beta (gamma^T Phi'_T - Phi_0)  (NOT what RLinf does)
                         and decomposes the within-group advantage into sparse / terminal-Phi / path / Phi_0 parts.
  --mode beta0    (GPU)  same seed + same actions: T022 path (no pbrs cfg) vs pbrs beta 0 vs T022 path again
                         -> rewards / terminations / truncations / GRPO advantages torch.equal.
  CUDA_VISIBLE_DEVICES=0 python -m vla_residual.validate_t028 --mode grpo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

OUT = Path("/outputs/StochRL/t028/val")
G = 8
GAMMA = 0.96


def q(x, p):
    return float(np.percentile(np.asarray(x, float), p)) if len(x) else None


def rlinf_grpo(rew, dones):
    """rew [n_chunk, bsz, 1], dones [n_chunk+1, bsz, 1] (RLinf layout, rollout epochs stacked on bsz).
    Returns (scores [bsz], advantages [n_chunk, bsz, 1], loss_mask) exactly as embodied_fsdp_actor_worker does."""
    from rlinf.algorithms.registry import calculate_adv_and_returns
    from rlinf.algorithms.utils import calculate_scores, preprocess_embodied_advantages_inputs
    from rlinf.utils.metric_utils import compute_loss_mask

    loss_mask, loss_mask_sum = compute_loss_mask(dones)
    kw = dict(task_type="embodied", adv_type="grpo", rewards=rew, dones=dones, values=None, num_action_chunks=1,
              gamma=GAMMA, gae_lambda=0.95, group_size=G, reward_type="action_level", loss_mask=loss_mask,
              loss_mask_sum=loss_mask_sum, advantage_mode=None)
    k2 = calculate_scores(**preprocess_embodied_advantages_inputs(**dict(kw)))
    scores = k2["rewards"].reshape(-1)
    adv = calculate_adv_and_returns(**dict(kw))["advantages"]
    return scores, adv, loss_mask


def rollout(env, g, std, n_rollouts):
    """Returns per-step records stacked RLinf-style: [n_chunk, n_rollouts*B]."""
    from vla_residual import core as Vc

    recs = []
    for _ in range(n_rollouts):
        env.reset()
        R = dict(r=[], sparse=[], te=[], tr=[], phi_prev=[], phi_next=[], shp=[], active=[], eids=env.eids.clone())
        phi0 = env.phi.clone() if env.pbrs is not None else None
        for d in range(Vc.T_DEC):
            u = torch.randn(env.num_envs, 1, 7, generator=g).to(env.device) * std
            _, r, te, tr, _ = env.chunk_step(u)
            R["r"].append(r[:, 0].clone()); R["te"].append(te[:, 0]); R["tr"].append(tr[:, 0])
            if env.pbrs is not None:
                ri = env._last_rinfo
                for k in ("sparse", "phi_prev", "phi_next", "shp", "active"):
                    R[k].append(ri[k].clone())
        R = {k: (torch.stack(v) if isinstance(v, list) and v else v) for k, v in R.items()}
        R["phi0"] = phi0
        recs.append(R)
    cat = {}
    for k in recs[0]:
        if k == "eids":
            cat[k] = torch.cat([x[k] for x in recs])
        elif isinstance(recs[0][k], torch.Tensor):
            cat[k] = torch.cat([x[k] for x in recs], dim=-1)
    return cat


def to_rlinf(R, key="r"):
    rew = R[key][:, :, None].float().cpu()
    d = (R["te"] | R["tr"]).cpu()
    dones = torch.cat([torch.zeros_like(d[:1]), d])[:, :, None]
    return rew, dones


def mode_grpo(batch, rollouts, seed):
    from vla_residual import core as Vc
    from vla_residual.validate import make_cfg

    cfg = make_cfg("train", auto_reset=False, ignore_term=False, group_size=G,
                   pbrs=dict(potential="vla", beta=1.0, gamma=GAMMA, phi_term=1.0))
    cfg.seed = seed
    env = Vc.ResidualVLAEnv(cfg, batch, 0, 1, None)
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = rollout(env, g, float(np.exp(-1.5)), rollouts)
    nb = R["r"].shape[1]
    eids = R["eids"].reshape(-1, G)
    out = dict(batch=batch, rollouts=rollouts, n_traj=nb, n_groups=nb // G,
               groups_same_eid=bool((eids == eids[:, :1]).all()))
    # first done T (1-based length) per trajectory
    done = (R["te"] | R["tr"]).cpu()
    T = done.float().argmax(0) + 1  # every trajectory has a done by decision 16
    succ = R["te"].cpu().any(0)
    phi_prev = R["phi_prev"].cpu().double(); phi_next = R["phi_next"].cpu().double()
    sparse = R["sparse"].cpu().double()
    phi0 = R["phi0"].cpu().double()
    res = {}
    for beta in (1.0, 0.3):
        r_beta = (R["sparse"] + beta * R["shp"]).float()  # shp is linear in beta (logged at beta 1)
        if beta == 1.0:
            assert torch.allclose(r_beta, R["r"].float(), atol=1e-6)
        rew, dones = to_rlinf(dict(R, rb=r_beta), "rb")
        scores, adv, lm = rlinf_grpo(rew, dones)
        scores = scores.double()
        a_sum, b_form, c_disc, term_part, path_part, sp_sum = [], [], [], [], [], []
        for j in range(nb):
            t = int(T[j])
            s_sum = float(sparse[:t, j].sum())
            phiT = 1.0 if bool(succ[j]) else float(phi_next[t - 1, j])
            path = float(phi_prev[1:t, j].sum())  # Phi_1..Phi_{T-1}
            a_sum.append(float(r_beta[:t, j].double().sum()))
            b_form.append(s_sum + beta * (GAMMA * phiT - float(phi0[j])) + beta * (GAMMA - 1) * path)
            c_disc.append(s_sum + beta * (GAMMA ** t * phiT - float(phi0[j])))
            term_part.append(beta * GAMMA * phiT); path_part.append(beta * (GAMMA - 1) * path); sp_sum.append(s_sum)
        a_sum, b_form, c_disc = map(np.array, (a_sum, b_form, c_disc))
        sc = scores.numpy()
        # within-group decomposition (centered per group)
        def cen(x):
            x = np.asarray(x, float).reshape(-1, G)
            return (x - x.mean(1, keepdims=True)).reshape(-1)
        parts = dict(sparse=cen(sp_sum), terminal_phi=cen(term_part), path=cen(path_part),
                     phi0=cen(-beta * phi0.numpy()))
        tot = cen(sc)
        var_share = {k: float(np.dot(v, tot) / max(np.dot(tot, tot), 1e-12)) for k, v in parts.items()}
        # failures: score order vs terminal Phi inside each group
        rho = []
        for gi in range(nb // G):
            idx = [gi * G + i for i in range(G) if not bool(succ[gi * G + i])]
            if len(idx) >= 3:
                x = np.argsort(np.argsort(sc[idx])); y = np.argsort(np.argsort([term_part[i] for i in idx]))
                rho.append(float(np.corrcoef(x, y)[0, 1]))
        # successes: score vs T inside groups
        rho_s = []
        for gi in range(nb // G):
            idx = [gi * G + i for i in range(G) if bool(succ[gi * G + i])]
            if len(idx) >= 3 and len(set(int(T[i]) for i in idx)) > 1:
                x = np.argsort(np.argsort(sc[idx])); y = np.argsort(np.argsort([-int(T[i]) for i in idx]))
                rho_s.append(float(np.corrcoef(x, y)[0, 1]))
        # advantage is constant over the valid steps of a trajectory
        advc = []
        for j in range(nb):
            v = adv[:, j, 0][lm[:, j, 0]]
            advc.append(float(v.max() - v.min()) if len(v) else 0.0)
        g_sc = sc.reshape(-1, G)
        adv_expect = ((g_sc - g_sc.mean(1, keepdims=True)) / (g_sc.std(1, ddof=1, keepdims=True) + 1e-6)).reshape(-1)
        adv_first = np.array([float(adv[0, j, 0]) for j in range(nb)])
        same = [gi for gi in range(nb // G) if len(set(succ[gi * G:(gi + 1) * G].tolist())) == 1]
        same_out = dict(n_groups=len(same), outcomes=[bool(succ[gi * G]) for gi in same],
                        score_sd=[float(g_sc[gi].std(ddof=1)) for gi in same],
                        adv_absmean=[float(np.abs(adv_first.reshape(-1, G)[gi]).mean()) for gi in same],
                        sparse_only_adv_absmean=0.0)
        res[f"beta{beta}"] = dict(
            same_outcome_groups=same_out,
            score_eq_undiscounted_sum_maxerr=float(np.abs(sc - a_sum).max()),
            score_eq_closed_form_maxerr=float(np.abs(sc - b_form).max()),
            score_vs_discounted_telescoping=dict(maxabs=float(np.abs(sc - c_disc).max()),
                                                 meanabs=float(np.abs(sc - c_disc).mean())),
            adv_constant_within_traj_maxrange=float(max(advc)),
            adv_eq_group_zscore_maxerr=float(np.abs(adv_first - adv_expect).max()),
            within_group_score_sd=float(np.mean(g_sc.std(1, ddof=1))),
            share_of_within_group_score_variance=var_share,
            fail_rank_score_vs_terminal_phi_spearman=dict(mean=float(np.nanmean(rho)) if rho else None, n_groups=len(rho)),
            success_rank_score_vs_earliness_spearman=dict(mean=float(np.nanmean(rho_s)) if rho_s else None,
                                                          n_groups=len(rho_s)),
            example_group0=dict(success=succ[:G].tolist(), T=T[:G].tolist(), score=[round(x, 4) for x in sc[:G]],
                                closed_form=[round(x, 4) for x in b_form[:G]],
                                discounted_telescoping=[round(x, 4) for x in c_disc[:G]],
                                phi0=[round(float(x), 5) for x in phi0[:G]],
                                terminal_phi=[round(x / beta / GAMMA, 4) for x in term_part[:G]],
                                path_term=[round(x, 4) for x in path_part[:G]],
                                adv=[round(float(x), 3) for x in adv_first[:G]]),
        )
    out["phi0_within_group_sd_max"] = float(phi0.reshape(-1, G).std(1).max())
    out["phi0_between_group_sd"] = float(phi0.reshape(-1, G).mean(1).std())
    out["success_rate"] = float(succ.float().mean())
    out["groups_all_same_outcome"] = int(sum(len(set(succ[i * G:(i + 1) * G].tolist())) == 1 for i in range(nb // G)))
    out["T_success"] = dict(mean=float(T[succ].float().mean()) if succ.any() else None,
                            min=int(T[succ].min()) if succ.any() else None)
    out.update(res)
    out["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return out


def mode_beta0(batch, seed):
    from vla_residual import core as Vc
    from vla_residual.validate import make_cfg

    std = float(np.exp(-1.5))
    runs = {}
    for name, pb in (("t022_a", None), ("pbrs_beta0", dict(potential="vla", beta=0.0, gamma=GAMMA, phi_term=1.0)),
                     ("t022_b", None)):
        extra = {} if pb is None else dict(pbrs=pb)
        cfg = make_cfg("train", auto_reset=False, ignore_term=False, group_size=G, **extra)
        cfg.seed = seed
        torch.manual_seed(seed); np.random.seed(seed)
        env = Vc.ResidualVLAEnv(cfg, batch, 0, 1, None)
        assert (env.pbrs is None) == (pb is None)
        g = torch.Generator(device="cpu").manual_seed(seed)
        R = rollout(env, g, std, 1)
        rew, dones = to_rlinf(R)
        sc, adv, _ = rlinf_grpo(rew, dones)
        runs[name] = dict(r=R["r"].cpu(), te=R["te"].cpu(), tr=R["tr"].cpu(), eids=R["eids"], adv=adv, dtype=str(R["r"].dtype))
        print(f"[beta0] {name} success {R['te'].any(0).float().mean():.3f}", flush=True)
        del env
    def eq(a, b):
        return {k: bool(torch.equal(runs[a][k], runs[b][k])) for k in ("r", "te", "tr", "eids", "adv")}
    return dict(control_t022_vs_t022=eq("t022_a", "t022_b"), t022_vs_pbrs_beta0=eq("t022_a", "pbrs_beta0"),
                dtypes={k: v["dtype"] for k, v in runs.items()},
                success_rate=float(runs["t022_a"]["te"].any(0).float().mean()), batch=batch, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["grpo", "beta0"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = mode_grpo(a.batch, a.rollouts, a.seed) if a.mode == "grpo" else mode_beta0(a.batch, a.seed)
    res["gpu"] = __import__("os").environ.get("CUDA_VISIBLE_DEVICES")
    Path(OUT, f"{a.mode}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

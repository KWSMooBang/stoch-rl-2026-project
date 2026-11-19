"""T029 success-hold: pre-training checks for the "hold" (H1) train reward.

  --mode units   CPU only. Synthetic success sequences (incl. success -> loss -> re-success): sum of H1 rewards
                 = success_at_end; H2 sum = fraction of successful decisions; RLinf GRPO score (calculate_scores) = H1
                 sum; RLinf PPO GAE (critic values present) does not bootstrap through the decision-16 truncation;
                 loss_mask covers all 16 decisions.
  --mode eval    GPU. B0 (residual 0) and the hand-coded upper bound (expert - base within alpha) on the 128 held-out
                 episodes with reward=hold: per-episode H1 return vs success_once / success_at_end, losses, re-gains,
                 and agreement with the T022 validation files (same ids, same pi0.5 noise).
  --mode train   GPU. Train split, group 8, GRPO-like noise: no termination, truncation only at 16, env return =
                 success_at_end, identical initial state within GRPO groups, RLinf GRPO score = success_at_end, and
                 reward=first_success (default) is bit-identical to the T022 code (47a4d76) on the same rollout.
  CUDA_VISIBLE_DEVICES=0 python -m vla_residual.validate_t029 --mode eval
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import types
from pathlib import Path

import numpy as np
import torch

OUT = Path("/outputs/StochRL/t029/val")
T022_VAL = Path("/outputs/StochRL/t022/val")
G = 8
GAMMA = 0.96
T = 16


# ------------------------------------------------------------------ RLinf helpers
def to_rlinf(r, done):
    """r, done: [T, bsz] -> RLinf layout rewards [T, bsz, 1], dones [T+1, bsz, 1]."""
    rew = r[:, :, None].float().cpu()
    d = done.bool().cpu()
    dones = torch.cat([torch.zeros_like(d[:1]), d])[:, :, None]
    return rew, dones


def rlinf_grpo_scores(rew, dones, group=G):
    from rlinf.algorithms.registry import calculate_adv_and_returns
    from rlinf.algorithms.utils import calculate_scores, preprocess_embodied_advantages_inputs
    from rlinf.utils.metric_utils import compute_loss_mask

    loss_mask, loss_mask_sum = compute_loss_mask(dones)
    kw = dict(task_type="embodied", adv_type="grpo", rewards=rew, dones=dones, values=None, num_action_chunks=1,
              gamma=GAMMA, gae_lambda=0.95, group_size=group, reward_type="action_level", loss_mask=loss_mask,
              loss_mask_sum=loss_mask_sum, advantage_mode=None)
    scores = calculate_scores(**preprocess_embodied_advantages_inputs(**dict(kw)))["rewards"].reshape(-1)
    adv = calculate_adv_and_returns(**dict(kw))["advantages"]
    return scores, adv, loss_mask


def rlinf_gae_returns(rew, dones, values):
    from rlinf.algorithms.registry import calculate_adv_and_returns
    from rlinf.utils.metric_utils import compute_loss_mask

    loss_mask, loss_mask_sum = compute_loss_mask(dones)
    out = calculate_adv_and_returns(task_type="embodied", adv_type="gae", rewards=rew, dones=dones, values=values,
                                    num_action_chunks=1, gamma=GAMMA, gae_lambda=1.0, group_size=1,
                                    reward_type="action_level", loss_mask=loss_mask, loss_mask_sum=loss_mask_sum,
                                    normalize_advantages=False, advantage_mode=None)
    return out["returns"], loss_mask


# ------------------------------------------------------------------ units (CPU)
def mode_units():
    from vla_residual.core import hold_reward

    seqs = {
        "never": [0] * T,
        "success_at_16_only": [0] * 15 + [1],
        "hold_from_5": [0] * 4 + [1] * 12,
        "success_then_lost": [0] * 3 + [1] * 4 + [0] * 9,
        "success_lost_regain": [0] * 3 + [1] * 3 + [0] * 4 + [1] * 6,
        "success_lost_regain_lost": [0, 1, 0, 1, 1, 0, 0, 1] + [0] * 8,
        "flicker_end_success": [1, 0] * 7 + [0, 1],
        "success_from_start": [1] * T,
    }
    rows = {}
    H = torch.tensor(list(seqs.values()), dtype=torch.bool).T  # [T, n]
    prev = torch.zeros(H.shape[1], dtype=torch.bool)
    R1, R2 = [], []
    for t in range(T):
        R1.append(hold_reward(prev, H[t], "hold", T)); R2.append(hold_reward(prev, H[t], "hold_dense", T))
        prev = H[t]
    R1 = torch.stack(R1); R2 = torch.stack(R2)
    trunc = torch.zeros(T, H.shape[1], dtype=torch.bool); trunc[-1] = True
    # GRPO score through RLinf (group = 1 sequence each is not allowed by std; use group of all sequences)
    rew, dones = to_rlinf(R1, trunc)
    sc, adv, lm = rlinf_grpo_scores(rew, dones, group=H.shape[1])
    # PPO GAE with a critic: random values, huge value at the post-truncation state -> must not leak in
    vals = torch.randn(T + 1, H.shape[1], 1); vals[-1] = 1e3
    ret, lm2 = rlinf_gae_returns(rew, dones, vals)
    disc = (R1 * GAMMA ** torch.arange(T)[:, None]).sum(0)
    for j, k in enumerate(seqs):
        rows[k] = dict(h=seqs[k], r_h1=R1[:, j].tolist(), sum_h1=float(R1[:, j].sum()), success_at_end=int(H[-1, j]),
                       n_lost=int((H[:-1, j] & ~H[1:, j]).sum() + 0), sum_h2=float(R2[:, j].sum()),
                       frac_success=float(H[:, j].float().mean()), rlinf_grpo_score=float(sc[j]),
                       rlinf_gae_return_t0=float(ret[0, j, 0]), discounted_h1=float(disc[j]))
    checks = dict(
        h1_sum_eq_success_at_end=all(abs(v["sum_h1"] - v["success_at_end"]) < 1e-6 for v in rows.values()),
        h1_rewards_in_pm1=bool(((R1 == 1) | (R1 == 0) | (R1 == -1)).all()),
        h2_sum_eq_frac_success=all(abs(v["sum_h2"] - v["frac_success"]) < 1e-6 for v in rows.values()),
        grpo_score_eq_success_at_end=all(abs(v["rlinf_grpo_score"] - v["success_at_end"]) < 1e-6 for v in rows.values()),
        gae_no_bootstrap_at_truncation=all(abs(v["rlinf_gae_return_t0"] - v["discounted_h1"]) < 1e-4 for v in rows.values()),
        loss_mask_all_16=bool(lm.all()) and bool(lm2.all()),
    )
    assert all(checks.values()), checks
    return dict(checks=checks, sequences=rows,
                note="discounted H1 return (PPO, gamma 0.96) of success at t1 lost at t2 = g^t1 - g^t2 > 0 (small); "
                     "holding to the end gives g^t1. GRPO (undiscounted) sees exactly success_at_end.")


# ------------------------------------------------------------------ eval (GPU): B0 / upper under H1
def run_eval_hold(mode, n, batch):
    from vla_residual import core as V
    from vla_residual.validate import make_cfg

    extra = dict(expert_exec=1.0) if mode == "upper" else {}
    cfg = make_cfg("eval", n_eval=n, reward="hold", **extra)
    env = V.ResidualVLAEnv(cfg, batch, 0, 1, None)
    rows = []
    t0 = time.time()
    for b in range(n // batch):
        # A reset without env_idx restarts the eval id cursor (RLinf eval semantics); env_idx = all slots keeps it,
        # so batch b gets held-out ids [b*batch, (b+1)*batch). (eval.json of 2026-11-20 predates this fix: ids 0..63 x2.)
        env.reset() if b == 0 else env.reset(options={"env_idx": torch.arange(batch)})
        eids = env.eids.clone()
        R, HE, HA = [], [], []
        for d in range(V.T_DEC):
            _, r, te, tr, inf = env.chunk_step(torch.zeros(batch, 1, V.ACT_DIM, device=env.device))
            assert not te.any()
            R.append(r[:, 0].cpu()); HE.append(inf[0]["episode"]["success_at_end"].cpu())
        ep = inf[0]["episode"]
        R = torch.stack(R); HE = torch.stack(HE)
        for i in range(batch):
            h = HE[:, i]
            rows.append(dict(eid=int(eids[i]), success_once=bool(ep["success_once"][i]),
                             success_at_end=bool(ep["success_at_end"][i]), h1_return=float(R[:, i].sum()),
                             env_return=float(ep["return"][i]), n_lost=int(env.n_lost[i]),
                             n_regain=int(((~h[:-1]) & h[1:]).sum()) - 0, first_hold=int(h.float().argmax()) if h.any() else -1,
                             once_by_boundary=bool(h.any())))
        print(f"[{mode}] batch {b} {time.time() - t0:.0f}s", flush=True)
    ref = {e["eid"]: e for e in json.load(open(T022_VAL / f"{mode}.json"))["episodes"]}
    a = lambda k: np.array([r[k] for r in rows], float)  # noqa: E731
    same_once = sum(r["success_once"] == ref[r["eid"]]["success_once"] for r in rows)
    same_end = sum(r["success_at_end"] == ref[r["eid"]]["success_at_end"] for r in rows)
    return dict(mode=mode, n=len(rows), success_once=float(a("success_once").mean()),
                success_at_end=float(a("success_at_end").mean()), h1_return_mean=float(a("h1_return").mean()),
                h1_return_eq_success_at_end=bool(np.allclose(a("h1_return"), a("success_at_end"))),
                env_return_eq_h1=bool(np.allclose(a("env_return"), a("h1_return"))),
                once_but_not_end=int(((a("success_once") == 1) & (a("success_at_end") == 0)).sum()),
                episodes_with_loss=int((a("n_lost") > 0).sum()), episodes_with_regain=int((a("n_regain") > 0).sum()),
                mean_losses_given_once=float(a("n_lost")[a("success_once") == 1].mean()),
                once_only_inside_chunk=int(((a("success_once") == 1) & (a("once_by_boundary") == 0)).sum()),
                t022_val_agreement=dict(once=f"{same_once}/{len(rows)}", end=f"{same_end}/{len(rows)}",
                                        t022_once=float(np.mean([e["success_once"] for e in ref.values()])),
                                        t022_end=float(np.mean([e["success_at_end"] for e in ref.values()]))),
                wall_s=time.time() - t0, episodes=rows)


def mode_eval(n, batch, modes=("b0", "upper")):
    res = {}
    for m in modes:
        res[m] = run_eval_hold(m, n, batch)
        print(json.dumps({k: v for k, v in res[m].items() if k != "episodes"}, indent=1), flush=True)
    return res


# ------------------------------------------------------------------ train (GPU): GRPO groups, bitwise default path
def _old_core():
    """T022 core.py at 47a4d76 as a module sharing the loaded pi0.5 and ManiSkill scene with the current one."""
    from vla_residual import core as V

    src = subprocess.check_output(["git", "-C", str(V.REPO), "show", "47a4d76:vla_residual/core.py"], text=True)
    mod = types.ModuleType("core_t022_ref")
    mod.__file__ = V.__file__
    exec(compile(src, "core_t022_ref", "exec"), mod.__dict__)
    mod._VLA = V.get_vla()
    mod._INNER = V._INNER
    return mod


def rollout(env, seed, std):
    g = torch.Generator().manual_seed(seed)
    env.reset()
    eids = env.eids.clone()
    from vla_residual import core as V
    ps0 = V.priv_state(env.inner)
    obj = env.inner.env.unwrapped.select_carrot_ids.clone()
    R, TE, TR, HE, SO, ST = [], [], [], [], [], []
    for d in range(V.T_DEC):
        u = torch.randn(env.num_envs, 1, 7, generator=g).to(env.device) * std
        o, r, te, tr, inf = env.chunk_step(u)
        R.append(r[:, 0].cpu()); TE.append(te[:, 0].cpu()); TR.append(tr[:, 0].cpu())
        HE.append(inf[0]["episode"]["success_at_end"].cpu()); SO.append(inf[0]["episode"]["success_once"].cpu())
        ST.append(o[0]["states"].cpu())
    return dict(r=torch.stack(R), te=torch.stack(TE), tr=torch.stack(TR), he=torch.stack(HE), so=torch.stack(SO),
                states=torch.stack(ST), eids=eids, ps0=ps0, obj=obj, ret=inf[0]["episode"]["return"].cpu())


def mode_train(batch, seed, skip_hold=False):
    from vla_residual import core as V
    from vla_residual.validate import make_cfg

    std = float(np.exp(-1.5))
    out = {}

    def mk(mod, reward=None):
        extra = {} if reward is None else dict(reward=reward)
        cfg = make_cfg("train", auto_reset=False, ignore_term=False, group_size=G, **extra)
        cfg.seed = seed
        return mod.ResidualVLAEnv(cfg, batch, 0, 1, None)

    if skip_hold:  # bitwise diagnostic only; keep the hold results of the previous run
        out = {k: v for k, v in json.load(open(OUT / "train.json")).items() if k == "hold"}
        out["hold_from"] = "previous train.json (same code)"
    # 1) hold reward on the train split
    if not skip_hold:
        env = mk(V, "hold")
        h = rollout(env, seed, std)
        rew, dones = to_rlinf(h["r"], h["te"] | h["tr"])
        sc, adv, lm = rlinf_grpo_scores(rew, dones)
        cw = h["ps0"]["carrot_w"].view(-1, G, 3); pw = h["ps0"]["plate_w"].view(-1, G, 3)
        end = h["he"][-1].float()
        out["hold"] = dict(
            no_termination=bool(not h["te"].any()), truncation_only_at_16=bool(h["tr"][-1].all() and not h["tr"][:-1].any()),
            sum_reward_eq_success_at_end=bool(torch.equal(h["r"].sum(0), end)),
            env_return_eq_success_at_end=bool(torch.allclose(h["ret"], end)),
            rlinf_grpo_score_eq_success_at_end=bool(torch.allclose(sc, end)),
            loss_mask_all_16=bool(lm.all()),
            grpo_group_eids_equal=bool((h["eids"].view(-1, G) == h["eids"].view(-1, G)[:, :1]).all()),
            grpo_group_same_object=bool((h["obj"].view(-1, G) == h["obj"].view(-1, G)[:, :1]).all()),
            grpo_within_group_max_diff_m=float(max((cw - cw[:, :1]).abs().max(), (pw - pw[:, :1]).abs().max())),
            grpo_between_group_mean_diff_m=float((cw[1:, 0] - cw[:-1, 0]).norm(dim=1).mean()),
            grpo_group_success_at_end=[float(x) for x in end.view(-1, G).mean(1)],
            grpo_group_success_once=[float(x) for x in h["so"][-1].float().view(-1, G).mean(1)],
            n_episodes_with_loss=int((h["r"] < 0).any(0).sum()), reward_values=sorted(set(h["r"].flatten().tolist())),
            obs_has_no_success_flag="states = [feat 2048 | qpos 8 | base chunk 35 | t/T] (unchanged from T022)",
            obs_dim=int(h["states"].shape[-1]))
        print(json.dumps(out["hold"], indent=1), flush=True)
        del env
    # 2) default reward (first_success) vs T022 code 47a4d76: bitwise on the same rollout
    old = _old_core()
    runs = {}
    for name, mod in (("t022_ref", old), ("t029_default", V), ("t022_ref_again", old), ("t029_default_again", V)):
        env = mk(mod)
        runs[name] = rollout(env, seed, std)
        del env

    def eq(a, b):
        return {k: bool(torch.equal(runs[a][k], runs[b][k])) for k in ("r", "te", "tr", "eids", "states")}
    def maxdiff(a, b):  # per obs block: feature 2048 | qpos 8 | base chunk 35 | t
        d = (runs[a]["states"] - runs[b]["states"]).abs()
        blocks = dict(feat=d[..., :2048], qpos=d[..., 2048:2056], base=d[..., 2056:2091], t=d[..., 2091:])
        return {k: float(v.max()) for k, v in blocks.items()} | dict(
            frac_decisions_feat_differs=float((blocks["feat"].amax(-1) > 0).float().mean()))
    pairs = [("t022_ref", "t022_ref_again"), ("t029_default", "t029_default_again"), ("t022_ref", "t029_default"),
             ("t022_ref_again", "t029_default_again"), ("t022_ref_again", "t029_default")]
    out["bitwise"] = {f"{a}_vs_{b}": eq(a, b) for a, b in pairs}
    out["states_maxabs"] = {f"{a}_vs_{b}": maxdiff(a, b) for a, b in pairs}
    print(json.dumps(out["bitwise"], indent=1), json.dumps(out["states_maxabs"], indent=1), flush=True)
    out["batch"] = batch; out["seed"] = seed
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["units", "eval", "train"])
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--skip_hold", action="store_true")
    ap.add_argument("--modes", default="b0,upper")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "units":
        res = mode_units()
    elif a.mode == "eval":
        res = mode_eval(a.n, a.batch, tuple(a.modes.split(",")))
    else:
        res = mode_train(a.batch, a.seed, a.skip_hold)
    res["gpu"] = __import__("os").environ.get("CUDA_VISIBLE_DEVICES")
    res["rc_policy_mb"] = __import__("os").environ.get("RC_POLICY_MB")
    Path(OUT, f"{a.mode}{a.tag}.json").write_text(json.dumps(res, indent=1))
    if a.mode == "units":
        print(json.dumps(res["checks"], indent=1))


if __name__ == "__main__":
    main()

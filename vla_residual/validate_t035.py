"""T035 pre-training checks for base-anchored GRPO.

--mode cpu  (no GPU, RLinf code paths on synthetic data):
  1 advantage formula A = (R - mean base) / (std residual + eps) against hand-computed numbers, sigma == 0 handling
  2 base members: loss_mask 0 and advantage 0 after the same two RLinf calls the actor makes
    (_process_received_rollout_batch -> calculate_adv_and_returns, mask cleared in place); loss_mask_sum untouched;
    RLinf PPO actor loss has zero gradient w.r.t. base rows, finite loss
  3 n_base = 0: "grpo_anchor" == RLinf "grpo" bit by bit (random scores, through calculate_adv_and_returns)
--mode env  (GPU, single process, one scene; train split, group 8 = 2 base + 6 residual, CRN on):
  4 within-group initial state identical (object / plate pose, episode id, object model)
  5 CRN: pi0.5 noise identical within a group at every decision; decision-0 base chunk identical within a group
  6 base members: executed action == clip(a_base) bit by bit at every sim step, although the policy output is not 0;
    residual members: executed action differs from a_base
  7 advantage-sign diagnostic (POMO caution): residual output drawn from the untrained policy (mean 0, log-std -1.5),
    n rollouts; share of A > 0 / < 0 / == 0, base-pair agreement, compared with RLinf GRPO on the same scores
  CUDA_VISIBLE_DEVICES=0 python -m vla_residual.validate_t035 --mode env --out /outputs/StochRL/t035/val
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from vla_residual import anchor as A

G, NB = 8, 2


def scores_from(rewards, dones):
    """RLinf calculate_scores: undiscounted reward sum up to the first done. rewards [T,B], dones [T+1,B]."""
    s = torch.zeros(rewards.shape[1])
    for t in reversed(range(rewards.shape[0])):
        s = s * ~dones[t + 1]
        s += rewards[t]
    return s


def cpu_tests():
    from omegaconf import OmegaConf
    from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
    from rlinf.utils.metric_utils import compute_loss_mask

    from vla_residual import rlinf_ext as X

    X.register()
    out = {}
    # ---- 1 formula
    sc = torch.tensor([[1., 0., 1., 1., 0., 0., 1., 0.],   # b .5, res 3/6 -> std .5477
                       [0., 0., 1., 1., 1., 1., 1., 1.],   # all residual succeed, base fails -> sigma 0 -> floor
                       [1., 1., 0., 0., 0., 0., 0., 0.],   # all residual fail, base succeeds  -> sigma 0 -> floor
                       [0., 0., 0., 0., 0., 0., 0., 0.],   # everything fails -> 0
                       [1., 1., 1., 1., 1., 1., 1., 0.]])  # one regression: res 5/6 -> std .4082
    adv, st = A.anchor_advantages(sc, G, NB)
    fl = 6 ** -0.5
    exp = torch.zeros(5, 8)
    s0 = torch.tensor([1., 1., 0., 0., 1., 0.]).std()
    exp[0, 2:] = (sc[0, 2:] - 0.5) / (s0 + 1e-6)
    exp[1, 2:] = 1.0 / (fl + 1e-6)
    exp[2, 2:] = -1.0 / (fl + 1e-6)
    s4 = torch.tensor([1., 1., 1., 1., 1., 0.]).std()
    exp[4, 2:] = (sc[4, 2:] - 1.0) / (s4 + 1e-6)
    out["formula_max_abs_err"] = float((adv - exp).abs().max())
    out["formula_ok"] = bool(torch.allclose(adv, exp, atol=1e-6))
    out["formula_examples"] = dict(group0=[round(float(x), 4) for x in adv[0]], all_res_succ_base_fail=float(adv[1, 2]),
                                   all_res_fail_base_succ=float(adv[2, 2]), one_regression=[round(float(x), 4) for x in adv[4]],
                                   sigma=[round(float(x), 4) for x in st["sigma"]], sigma_floor=fl)
    out["sigma_zero_flags"] = st["sigma_zero"].tolist()
    out["base_adv_zero"] = bool((adv[:, :NB] == 0).all())

    # ---- 2 / 3 through RLinf (rollout layout: [rollout_epoch x T(+1), bsz, 1])
    T, B, E = 16, 32, 2
    g = torch.Generator().manual_seed(0)
    succ_t = torch.randint(0, 2 * T, (E, B), generator=g)  # first-success decision; >= T -> never succeeds
    rewards = torch.zeros(E, T, B, 1)
    dones = torch.zeros(E, T + 1, B, 1, dtype=torch.bool)
    for e in range(E):
        for b in range(B):
            t = int(succ_t[e, b])
            if t < T:
                rewards[e, t, b, 0] = 1.0
                dones[e, t + 1, b, 0] = True
            dones[e, T, b, 0] = True  # truncation at the end
    raw = dict(rewards=rewards.reshape(E * T, B, 1), dones=dones.reshape(E * (T + 1), B, 1))

    def run(adv_type, n_base):
        cfg = OmegaConf.create(dict(
            env=dict(train=dict(rollout_epoch=E, auto_reset=False, ignore_terminations=False)),
            algorithm=dict(adv_type=adv_type, group_size=G, reward_type="action_level")))
        from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

        # same two calls, same arguments as EmbodiedFSDPActor.recv_rollout_trajectories / compute_advantages_and_returns
        fake = SimpleNamespace(cfg=cfg)
        rb = EmbodiedFSDPActor._process_received_rollout_batch(fake, {k: v.clone() for k, v in raw.items()})
        kwargs = dict(
            task_type="embodied", adv_type=adv_type, rewards=rb["rewards"], dones=rb["dones"], values=None,
            prev_logprobs=None, teacher_logprobs=None, num_action_chunks=1, gamma=0.96, gae_lambda=0.95,
            group_size=G, reward_type="action_level", loss_mask=rb.get("loss_mask"), loss_mask_sum=rb.get("loss_mask_sum"),
            advantage_mode=(f"anchor_nbase={n_base}" if adv_type == "grpo_anchor" else None))
        res = calculate_adv_and_returns(**kwargs)
        rb.update(res)
        rb.update({"loss_mask": kwargs["loss_mask"], "loss_mask_sum": kwargs["loss_mask_sum"]})
        return rb, res["advantages"]

    os.environ.pop("VR_LOG_DIR", None)
    rb_ref, a_ref = run("grpo", 0)
    rb0, a0 = run("grpo_anchor", 0)
    out["nbase0_bitwise_equal_rlinf_grpo"] = bool(torch.equal(a_ref, a0) and torch.equal(rb_ref["loss_mask"], rb0["loss_mask"]))
    rb2, a2 = run("grpo_anchor", NB)
    bc = A.base_columns(E * B, G, NB)
    ref_mask, _ = compute_loss_mask(rb2["dones"])
    out["base_loss_mask_all_zero"] = bool((~rb2["loss_mask"][:, bc]).all())
    out["residual_loss_mask_unchanged"] = bool(torch.equal(rb2["loss_mask"][:, ~bc], ref_mask[:, ~bc]))
    out["loss_mask_sum_unchanged"] = bool(torch.equal(rb2["loss_mask_sum"], rb_ref["loss_mask_sum"]) and (rb2["loss_mask_sum"] > 0).all())
    out["base_adv_all_zero_rlinf_path"] = bool((a2[:, bc] == 0).all())
    # advantage of residual members = formula on the scores, broadcast over their valid steps
    sc2 = scores_from(rb2["rewards"][..., 0], rb2["dones"][..., 0]).view(-1, G)
    adv2, st2 = A.anchor_advantages(sc2, G, NB)
    expect = adv2.view(1, -1) * rb2["loss_mask"][..., 0]
    out["rlinf_path_matches_formula"] = bool(torch.equal(a2[..., 0], expect))
    out["residual_samples"] = int(rb2["loss_mask"].sum()); out["all_samples_grpo"] = int(rb_ref["loss_mask"].sum())
    out["residual_sample_frac"] = out["residual_samples"] / max(out["all_samples_grpo"], 1)
    # ---- PPO actor loss: zero gradient w.r.t. base rows, finite
    n = T * E * B
    flat = lambda x: x[:T].reshape(n, *x.shape[2:])  # noqa: E731
    logp = torch.zeros(n, 7, requires_grad=True)
    loss, md = policy_loss(
        loss_type="actor", logprob_type="action_level", reward_type="action_level", single_action_dim=7,
        logprobs=logp + 0.01, values=None, old_logprobs=torch.zeros(n, 7), advantages=flat(a2), returns=None,
        prev_values=None, clip_ratio_high=0.2, clip_ratio_low=0.2, value_clip=1.0, huber_delta=10.0,
        loss_mask=flat(rb2["loss_mask"]), loss_mask_sum=flat(rb2["loss_mask_sum"]), max_episode_steps=T,
        task_type="embodied", critic_warmup=False, logprob_mask=None,
        loss_agg_func=__import__("rlinf.utils.utils", fromlist=["x"]).get_loss_agg_func("token-mean"),
        log_logprob_diagnostics=False)
    loss.backward()
    gr = logp.grad.view(T, E * B, 7)
    out["loss_finite"] = bool(torch.isfinite(loss))
    out["base_rows_zero_grad"] = bool((gr[:, bc] == 0).all())
    out["residual_rows_nonzero_grad_frac"] = float((gr[:, ~bc].abs().sum(-1) > 0).float().mean())
    out["pass"] = bool(all(out[k] for k in (
        "formula_ok", "base_adv_zero", "nbase0_bitwise_equal_rlinf_grpo", "base_loss_mask_all_zero",
        "residual_loss_mask_unchanged", "loss_mask_sum_unchanged", "base_adv_all_zero_rlinf_path",
        "rlinf_path_matches_formula", "loss_finite", "base_rows_zero_grad")))
    return out


def env_tests(batch, n_roll, seed, logstd):
    from vla_residual import core as V
    from vla_residual.validate import make_cfg

    cfg = make_cfg("train", group_size=G, auto_reset=False, ignore_term=False, anchor_n_base=NB, anchor_crn=True)
    cfg.seed = seed
    env = V.ResidualVLAEnv(cfg, batch, 0, 1, None)
    bm = env.base_member.clone()
    assert int(bm.sum()) == batch // G * NB
    dev = env.device
    g = torch.Generator().manual_seed(4242 + seed)
    out = dict(batch=batch, n_roll=n_roll, group_size=G, n_base=NB, logstd=logstd)
    init_diff, obj_same, eid_same, dec0_chunk_eq, noise_eq = [], [], [], [], []
    base_exec_eq, res_exec_diff, base_u_abs = [], [], []
    S, t0 = [], time.time()
    # record the noise handed to pi0.5
    orig_noise, seen = env._noise, []

    def rec_noise(idx):
        z = orig_noise(idx)
        seen.append(z.detach().cpu())
        return z

    env._noise = rec_noise
    for r in range(n_roll):
        env.reset()
        ps = V.priv_state(env.inner)
        cw, pw = ps["carrot_w"].view(-1, G, 3), ps["plate_w"].view(-1, G, 3)
        init_diff.append(float(max((cw - cw[:, :1]).abs().max(), (pw - pw[:, :1]).abs().max())))
        ids = env.inner.env.unwrapped.select_carrot_ids.view(-1, G)
        obj_same.append(bool((ids == ids[:, :1]).all()))
        e = env.eids.view(-1, G)
        eid_same.append(bool((e == e[:, :1]).all()))
        ab = env.a_base.view(-1, G, V.C.K, V.ACT_DIM)
        dec0_chunk_eq.append(float((ab == ab[:, :1]).flatten(2).all(-1).float().mean()))
        R, D = [], [torch.zeros(batch, dtype=torch.bool)]
        for d in range(V.T_DEC):
            z = seen[-1].view(batch // G, G, -1)
            noise_eq.append(bool((z == z[:, :1]).all()))
            u = math_exp(logstd) * torch.randn(batch, 1, V.ACT_DIM, generator=g)
            a_base = env.a_base.clone()
            _, rew, te, tr, _ = env.chunk_step(u.to(dev))
            want = torch.maximum(torch.minimum(a_base, env.hi), env.lo)
            eq = (env.last_a_exec == want).flatten(1).all(1).cpu()
            base_exec_eq.append(bool(eq[bm].all()))
            res_exec_diff.append(float((~eq[~bm]).float().mean()))
            base_u_abs.append(float(torch.tanh(u[bm]).abs().mean()))
            R.append(rew[:, 0].cpu()); D.append((te | tr)[:, 0].cpu())
        sc = scores_from(torch.stack(R), torch.stack(D)).view(-1, G)
        S.append(sc)
        print(f"[env] rollout {r}: base succ {sc[:, :NB].mean():.3f} residual succ {sc[:, NB:].mean():.3f} "
              f"{time.time()-t0:.0f}s", flush=True)
    S = torch.cat(S)
    adv, st = A.anchor_advantages(S, G, NB)
    a = st["a_res"]
    # RLinf GRPO (group mean, 8 members) on the same scores, for the sign comparison
    zs = (S - S.mean(1, keepdim=True)) / (S.std(1, keepdim=True) + 1e-6)
    out.update(
        init_within_group_max_diff_m=max(init_diff), init_same_object=all(obj_same), init_same_eid=all(eid_same),
        crn_noise_equal_within_group_all_decisions=all(noise_eq), crn_n_noise_checks=len(noise_eq),
        crn_dec0_base_chunk_equal_frac=min(dec0_chunk_eq),
        base_exec_eq_a_base_all_decisions=all(base_exec_eq), n_exec_checks=len(base_exec_eq),
        base_policy_output_abs_mean=sum(base_u_abs) / len(base_u_abs),
        residual_exec_differs_frac=min(res_exec_diff),
        scores=S.tolist(), b=st["b"].tolist(), sigma=[round(float(x), 4) for x in st["sigma"]],
        base_success=float(S[:, :NB].mean()), residual_success=float(S[:, NB:].mean()),
        base_pair_disagree_groups=int((S[:, 0] != S[:, 1]).sum()), n_groups=int(S.shape[0]),
        adv_sign_anchor=dict(pos=float((a > 0).float().mean()), neg=float((a < 0).float().mean()),
                             zero=float((a == 0).float().mean()), mean=float(a.mean()), abs_mean=float(a.abs().mean()),
                             min=float(a.min()), max=float(a.max()), n=int(a.numel())),
        adv_sign_grpo_same_scores=dict(pos=float((zs > 0).float().mean()), neg=float((zs < 0).float().mean()),
                                       zero=float((zs == 0).float().mean()), n=int(zs.numel())),
        groups_all_zero_adv_anchor=int((a.abs().sum(1) == 0).sum()), groups_all_zero_adv_grpo=int((zs.abs().sum(1) == 0).sum()),
        sigma_zero_groups=int(st["sigma_zero"].sum()),
        wall_s=time.time() - t0, peak_gb=torch.cuda.max_memory_allocated() / 1e9, gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    out["pass"] = bool(out["init_within_group_max_diff_m"] < 1e-4 and out["init_same_object"] and out["init_same_eid"]
                       and out["crn_noise_equal_within_group_all_decisions"] and out["base_exec_eq_a_base_all_decisions"]
                       and out["residual_exec_differs_frac"] > 0.99 and out["base_policy_output_abs_mean"] > 0.05)
    return out


def math_exp(x):
    import math

    return math.exp(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["cpu", "env"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n_roll", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logstd", type=float, default=-1.5)
    ap.add_argument("--out", default="/outputs/StochRL/t035/val")
    a = ap.parse_args()
    res = cpu_tests() if a.mode == "cpu" else env_tests(a.batch, a.n_roll, a.seed, a.logstd)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, f"units_t035_{a.mode}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "scores"}, indent=1))
    raise SystemExit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()

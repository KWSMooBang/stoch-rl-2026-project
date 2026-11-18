"""One SAC training run for (condition, method, lambda, seed). Returns a metrics dict."""
from __future__ import annotations

import time

import numpy as np
import torch

from .env import MAX_STEP_DISP, R_SUM, PushEnv, make_cfg
from .rewards import (H_MH, NEUTRAL, CounterfactualEvaluator, base_reward, method_spec,
                      shaping_value)
from .sac import SAC, Replay

GAMMA = 0.99
EVAL_SEED0 = 10_000          # evaluation episodes use seeds disjoint from training seeds
HELP_SEED0 = 20_000


def wind_helps_init(cond: str, rng: np.random.Generator):
    """Initial states where the wind blows the object toward the goal (doc 12 counterexample)."""
    cfg = make_cfg(cond)
    x = rng.uniform(-0.5, 0.0, 2)
    g = x + np.array([0.55, 0.0])
    th = np.arctan2(*(g - x)[::-1])
    p = x + 0.4 * np.array([np.cos(th + np.pi / 2), np.sin(th + np.pi / 2)])
    if cfg.wind == "bias":
        w = cfg.bias_mu * np.array([1.0, 0.0])
    elif cfg.wind == "ar1":
        w = 1.5 * cfg.sigma * np.array([1.0, 0.0])
    else:
        return None
    return dict(p=p, x=x, g=g, w=w)


def run_episode(env, policy, seed=None, init=None, record=False):
    env.reset(seed=seed, options=dict(init=init) if init is not None else None)
    ret_effort, traj = 0.0, []
    while True:
        o = env._obs()
        a = policy(o, env)
        if record:
            traj.append((env.get_snapshot(), a.copy()))
        _, _, term, trunc, info = env.step(a)
        ret_effort += float(a @ a)
        if term or trunc:
            return dict(success=bool(info["success"]), steps=env.t, effort=ret_effort, traj=traj)


def evaluate(agent, cond, n, seed0=EVAL_SEED0, helps=False, record=False):
    env = PushEnv(make_cfg(cond))
    pol = lambda o, e: agent.act(o, deterministic=True)
    out, trajs = [], []
    rng = np.random.default_rng(seed0)
    for i in range(n):
        init = wind_helps_init(cond, rng) if helps else None
        if helps and init is None:
            return None
        r = run_episode(env, pol, seed=seed0 + i, init=init, record=record)
        out.append(r)
        if record:
            trajs.append(r["traj"])
    res = dict(sr=float(np.mean([r["success"] for r in out])),
               steps=float(np.mean([r["steps"] for r in out])),
               effort=float(np.mean([r["effort"] for r in out])))
    return (res, trajs) if record else res


def attribution_stats(cond, trajs, n_states=200, m_ref=32, seed=0, eps=0.05, delta=0.01):
    """Attribution MAE and SPR on states visited by the policy (doc 9.3).

    Reference C_ref^(K)(s,a) = mean over m_ref fresh noise draws of the paired difference
    (noise independent of the training/eval draws). Estimators evaluated with a single draw:
      prog    : realised d_t - d_{t+1}  (= the progress reward)
      aapr1   : paired one-step AAPR     (realised noise)
      mh      : paired multi-horizon
      mh_ind  : independent-noise multi-horizon
    SPR = P(prog/0.1 > eps | |C_ref^(1)|/0.1 < delta).
    """
    rng = np.random.default_rng(seed)
    pool = [sa for tr in trajs for sa in tr]
    if not pool:
        return {}
    idx = rng.choice(len(pool), min(n_states, len(pool)), replace=False)
    env = PushEnv(make_cfg(cond))
    ev = CounterfactualEvaluator(env, seed=seed + 1)
    rows = []
    for j in idx:
        snap, a = pool[j]
        refs = []
        for _ in range(m_ref):
            st = np.random.default_rng(rng.integers(2 ** 63)).bit_generator.state
            da = ev.trace(snap, a, max(H_MH), rng_state=st)
            d0 = ev.trace(snap, NEUTRAL, max(H_MH), rng_state=st)
            refs.append([d0[k - 1] - da[k - 1] for k in H_MH])
        refs = np.mean(refs, 0)
        c1_ref, mh_ref = refs[0], refs.mean()
        env.set_snapshot(snap)
        d_prev = env.dist(); env.step(a); prog = d_prev - env.dist()
        c_pair = ev.contributions(snap, a, H_MH, paired=True)
        c_ind = ev.contributions(snap, a, H_MH, paired=False)
        rows.append([c1_ref, mh_ref, prog, c_pair[0], c_pair.mean(), c_ind.mean()])
    R = np.array(rows) / MAX_STEP_DISP
    small = np.abs(R[:, 0]) < delta
    return dict(
        mae_prog_vs_c1=float(np.abs(R[:, 2] - R[:, 0]).mean()),
        mae_aapr1_vs_c1=float(np.abs(R[:, 3] - R[:, 0]).mean()),
        mae_prog_vs_mh=float(np.abs(R[:, 2] - R[:, 1]).mean()),
        mae_mh_vs_mh=float(np.abs(R[:, 4] - R[:, 1]).mean()),
        mae_mhind_vs_mh=float(np.abs(R[:, 5] - R[:, 1]).mean()),
        spr=float(np.mean(R[small, 2] > eps)) if small.any() else float("nan"),
        frac_c1_small=float(small.mean()), n_states=int(len(R)))


def train(cond, method, lam=1.0, seed=0, total_steps=60_000, eval_every=5_000, eval_eps=50,
          final_eval_eps=100, learning_starts=2_000, batch=256, aug_n=16, alpha0=0.05,
          attribution=True, log=print):
    torch.set_num_threads(1)
    t0 = time.time()
    rng = np.random.default_rng(seed)
    env = PushEnv(make_cfg(cond), seed=seed)
    spec = method_spec(method)
    need_snap = spec["cf"] or method == "aug"
    ev = CounterfactualEvaluator(env, seed=seed + 7) if need_snap else None
    agent = SAC(PushEnv.OBS_DIM, 2, gamma=GAMMA, alpha0=alpha0, seed=seed)
    cap = total_steps * (1 + (aug_n if method == "aug" else 0))
    rb = Replay(PushEnv.OBS_DIM, 2, cap)

    o, _ = env.reset(seed=seed)
    curve, shaping_abs, shaping_sum, rew_sum = [], [], 0.0, []
    ep_ret, n_eps, train_succ = 0.0, 0, []
    upd = (0.0, 0.0, alpha0, 0.0)
    for step in range(1, total_steps + 1):
        a = rng.uniform(-1, 1, 2) if step <= learning_starts else agent.act(o)
        snap = env.get_snapshot() if need_snap else None
        d_prev, gap_prev = env.dist(), env.reach_gap()
        o2, _, term, trunc, info = env.step(a)
        d_next, gap_next = env.dist(), env.reach_gap()
        contrib = ev.contributions(snap, a, spec["horizons"], paired=spec["paired"]) if spec["cf"] else None
        sh = shaping_value(method, d_prev, d_next, info["success"], GAMMA, contrib)
        r = base_reward(info["success"], a, d_prev, d_next, gap_prev, gap_next) + lam * sh
        rb.add(o, a, r, o2, float(term))
        shaping_abs.append(abs(lam * sh)); rew_sum.append(r)
        if method == "aug":  # matched-compute model-based augmentation: random one-step branches
            for _ in range(aug_n):
                ab = rng.uniform(-1, 1, 2)
                ob2, tb, ib, dp, gp, dn, gn = ev.one_step_branch(snap, ab)
                rbv = base_reward(ib["success"], ab, dp, dn, gp, gn) + lam * shaping_value(
                    "prog", dp, dn, ib["success"], GAMMA)
                rb.add(o, ab, rbv, ob2, float(tb))
        ep_ret += r
        o = o2
        if term or trunc:
            train_succ.append(info["success"]); n_eps += 1; ep_ret = 0.0
            o, _ = env.reset()
        if step > learning_starts:
            upd = agent.update(rb.sample(rng, batch))
        if step % eval_every == 0:
            res = evaluate(agent, cond, eval_eps)
            res["step"] = step
            res["alpha"] = upd[2]; res["q"] = upd[3]
            res["train_sr_recent"] = float(np.mean(train_succ[-50:])) if train_succ else 0.0
            curve.append(res)
            log(f"[{cond}/{method}/lam{lam}/s{seed}] step {step} SR {res['sr']:.2f} "
                f"alpha {upd[2]:.3f} q {upd[3]:.2f} |sh| {np.mean(shaping_abs[-eval_every:]):.3f} "
                f"t {time.time() - t0:.0f}s")
    train_time = time.time() - t0
    final, trajs = evaluate(agent, cond, final_eval_eps, seed0=EVAL_SEED0 + 5000, record=attribution)
    helps = evaluate(agent, cond, 50, seed0=HELP_SEED0, helps=True)
    out = dict(cond=cond, method=method, lam=lam, seed=seed, total_steps=total_steps,
               curve=curve, auc=float(np.mean([c["sr"] for c in curve])), final=final,
               wind_helps=helps,
               sim_transitions=int(env.n_sim_steps + (ev.n_sim_steps if ev else 0)),
               sim_transitions_branch=int(ev.n_sim_steps if ev else 0),
               wall_clock_train_s=train_time,
               mean_abs_shaping=float(np.mean(shaping_abs)), mean_reward=float(np.mean(rew_sum)),
               final_alpha=upd[2], n_train_episodes=n_eps)
    if attribution:
        out["attribution"] = attribution_stats(cond, trajs, seed=seed)
    out["wall_clock_total_s"] = time.time() - t0
    return out

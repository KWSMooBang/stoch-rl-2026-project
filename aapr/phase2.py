"""Phase 2: expectation check (L1 fix 1), reward unit test (L1 fix 5), paired vs independent variance.

Under CRN the one-step AAPR is, draw by draw,
    C1 = [d_t - d(s^a_{t+1})] - [d_t - d(s^0_{t+1})] = prog(a) - prog(a0)
so E[C1 | s, a] = E[prog | s, a] - b(s),  b(s) = E[prog | s, a0]  (neutral-action progress).
b(s) is a state-dependent offset; it is not a potential difference, so it can change the optimal
policy through which states are visited. This script measures b(s) and the reward statistics.

usage: python -m aapr.phase2 OUT_DIR
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np

from .env import MAX_STEP_DISP as U, R_SUM, PushEnv, make_cfg
from .rewards import (H_MH, NEUTRAL, CounterfactualEvaluator, R_SUCC, base_reward, method_spec,
                      shaping_value)
from .scripted import random_action, scripted_action, zero_action

CONDS = ["iid", "ar1", "bias", "ar1_hi"]
GOAL = np.array([0.6, 0.0])


def canonical_states(cond):
    """(name, init, actions{label: a}). Goal at +x. 'wind away' = wind latent pointing -x."""
    cfg = make_cfg(cond)
    if cfg.wind == "ar1":
        w_mag = 1.5 * cfg.sigma
    elif cfg.wind == "bias":
        w_mag = cfg.bias_mu
    else:
        w_mag = 0.0   # iid: no latent, the wind is zero-mean every step
    away, toward = np.array([-w_mag, 0.0]), np.array([w_mag, 0.0])
    behind = [-R_SUM - 0.002, 0.0]
    far = [-0.8, 0.8]
    push, hold, back = np.array([1.0, 0.0]), NEUTRAL, np.array([-1.0, 0.0])
    S = [
        ("S1 rest, pusher far", dict(p=far, x=[0, 0], g=GOAL, w=away * 0), {"a0": hold}),
        ("S2 push from rest", dict(p=behind, x=[0, 0], g=GOAL, w=away * 0), {"push": push, "a0": hold}),
        ("S3 coasting toward goal (v=0.5)", dict(p=far, x=[0, 0], v=[0.5, 0], g=GOAL, w=away * 0), {"a0": hold}),
        ("S4 wind away, pusher in contact (hold against wind)", dict(p=behind, x=[0, 0], g=GOAL, w=away),
         {"hold(a0)": hold, "push": push, "let go": back}),
        ("S5 wind toward goal, pusher in contact", dict(p=behind, x=[0, 0], g=GOAL, w=toward),
         {"wait(a0)": hold, "push": push}),
        ("S6 wind away, pusher far (drift)", dict(p=far, x=[0, 0], g=GOAL, w=away), {"a0": hold}),
        ("S7 wind toward goal, pusher far", dict(p=far, x=[0, 0], g=GOAL, w=toward), {"a0": hold}),
        # steady-state drift: object already moving at the wind's terminal velocity w/(1-rho)
        ("S8 steady drift away, pusher far", dict(p=far, x=[0, 0], v=away / (1 - cfg.rho), g=GOAL, w=away),
         {"a0": hold}),
        ("S9 steady drift toward goal, pusher far", dict(p=far, x=[0, 0], v=toward / (1 - cfg.rho), g=GOAL, w=toward),
         {"a0": hold}),
        ("S10 steady drift toward goal, pusher in contact behind", dict(p=behind, x=[0, 0], v=toward / (1 - cfg.rho), g=GOAL, w=toward),
         {"wait(a0)": hold, "push": push}),
    ]
    return S


def _absent_trace(ev, snap, K):
    """Difference-rewards style counterfactual: the pusher is removed (no contact) for K steps."""
    e = ev.env
    e.set_snapshot(snap, restore_rng=True)
    out = []
    for _ in range(K):
        e.step(NEUTRAL, contact=False)
        out.append(e.dist())
    return np.array(out)


def state_stats(cond, init, a, n=2000, seed=0):
    env = PushEnv(make_cfg(cond))
    env.reset(options=dict(init=init))
    ev = CounterfactualEvaluator(env)
    rng = np.random.default_rng(seed)
    base = env.get_snapshot()
    d0 = env.dist()
    rows = []
    for _ in range(n):
        snap = dict(base)
        snap["rng"] = np.random.default_rng(rng.integers(2 ** 63)).bit_generator.state
        da = ev.trace(snap, a, max(H_MH))
        dn = ev.trace(snap, NEUTRAL, max(H_MH))
        dab = _absent_trace(ev, snap, 1)
        rows.append([d0 - da[0], d0 - dn[0], dn[0] - da[0], np.mean([dn[k - 1] - da[k - 1] for k in H_MH]),
                     dab[0] - da[0]])
    R = np.array(rows) / U
    return dict(prog_mean=R[:, 0].mean(), prog_std=R[:, 0].std(), b=R[:, 1].mean(), b_std=R[:, 1].std(),
                aapr_mean=R[:, 2].mean(), aapr_std=R[:, 2].std(), mh_mean=R[:, 3].mean(), mh_std=R[:, 3].std(),
                aapr_absent_mean=R[:, 4].mean())


def canonical_table(cond):
    out = []
    for name, init, acts in canonical_states(cond):
        for lab, a in acts.items():
            st = state_stats(cond, init, a)
            out.append(dict(cond=cond, state=name, action=lab, **{k: float(v) for k, v in st.items()}))
    return out


def rollout_states(cond, policy, n_eps, seed):
    env = PushEnv(make_cfg(cond))
    rng = np.random.default_rng(seed)
    S = []
    for i in range(n_eps):
        env.reset(seed=seed + i)
        while True:
            a = policy(env, rng)
            S.append((env.get_snapshot(), a))
            _, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    return S


def b_distribution(cond, n_states=600, m=64, seed=0):
    """b(s) = E[prog | s, a0] and E[AAPR], E[prog] on states visited by scripted / random policies.
    Also paired vs independent variance of C^(K) for K in {1,4,8} on the same states."""
    out = {}
    pols = {"scripted": lambda e, r: scripted_action(e), "random": random_action}
    for pname, pol in pols.items():
        S = rollout_states(cond, pol, 40, seed=50_000 + seed)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(S), min(n_states, len(S)), replace=False)
        env = PushEnv(make_cfg(cond))
        ev = CounterfactualEvaluator(env, seed=seed)
        rows, var_rows = [], []
        for j in idx:
            snap, a = S[j]
            env.set_snapshot(snap)
            d0 = env.dist()
            pa, pn, cp, ci = [], [], [], []
            for _ in range(m):
                st = np.random.default_rng(rng.integers(2 ** 63)).bit_generator.state
                st2 = np.random.default_rng(rng.integers(2 ** 63)).bit_generator.state
                da = ev.trace(snap, a, 8, rng_state=st)
                dn = ev.trace(snap, NEUTRAL, 8, rng_state=st)
                dn_ind = ev.trace(snap, NEUTRAL, 8, rng_state=st2)
                pa.append(d0 - da[0]); pn.append(d0 - dn[0])
                cp.append([dn[k - 1] - da[k - 1] for k in (1, 4, 8)])
                ci.append([dn_ind[k - 1] - da[k - 1] for k in (1, 4, 8)])
            pa, pn, cp, ci = map(lambda z: np.array(z) / U, (pa, pn, cp, ci))
            touching = np.linalg.norm(snap["p"] - snap["x"]) < R_SUM + 0.02
            rows.append([pa.mean(), pn.mean(), cp[:, 0].mean(), touching, pa.var(), cp[:, 0].var()])
            var_rows.append(np.concatenate([cp.var(0), ci.var(0)]))
        R = np.array(rows, float)
        V = np.array(var_rows)
        b = R[:, 1]
        out[pname] = dict(
            n_states=int(len(R)),
            b_mean=float(b.mean()), b_std=float(b.std()), b_q05=float(np.quantile(b, 0.05)),
            b_q50=float(np.quantile(b, 0.5)), b_q95=float(np.quantile(b, 0.95)),
            frac_b_pos=float(np.mean(b > 0.01)), frac_b_neg=float(np.mean(b < -0.01)),
            frac_absb_gt_0p1=float(np.mean(np.abs(b) > 0.1)),
            frac_absb_gt_0p25=float(np.mean(np.abs(b) > 0.25)),
            b_mean_touching=float(b[R[:, 3] == 1].mean()) if (R[:, 3] == 1).any() else float("nan"),
            b_mean_free=float(b[R[:, 3] == 0].mean()) if (R[:, 3] == 0).any() else float("nan"),
            frac_touching=float(R[:, 3].mean()),
            Eprog_mean=float(R[:, 0].mean()), Eaapr_mean=float(R[:, 2].mean()),
            within_state_var_prog=float(R[:, 4].mean()), within_state_var_aapr=float(R[:, 5].mean()),
            var_paired_K1_4_8=[float(x) for x in V[:, :3].mean(0)],
            var_indep_K1_4_8=[float(x) for x in V[:, 3:].mean(0)],
        )
    return out


def unit_test(cond, n_eps=100, gamma=0.99, lam=1.0):
    """Undiscounted and discounted episode returns of each method's total reward under
    scripted / random / zero policies (same episodes, same seeds). Pass if scripted > both."""
    methods = ["sparse", "dist", "prog", "pbrs", "aapr1", "mh", "mh_ind"]
    pols = {"scripted": lambda e, r: scripted_action(e), "random": random_action, "zero": zero_action}
    res = {}
    for pname, pol in pols.items():
        env = PushEnv(make_cfg(cond))
        ev = CounterfactualEvaluator(env, seed=3)
        rng = np.random.default_rng(0)
        R = {m: [] for m in methods}; G = {m: [] for m in methods}
        steps_prog, steps_aapr, succ = [], [], []
        for i in range(n_eps):
            env.reset(seed=30_000 + i)
            ret = {m: 0.0 for m in methods}; disc = {m: 0.0 for m in methods}
            t = 0
            while True:
                a = pol(env, rng)
                snap = env.get_snapshot()
                d_prev, g_prev = env.dist(), env.reach_gap()
                _, _, term, trunc, info = env.step(a)
                d_next, g_next = env.dist(), env.reach_gap()
                cp = ev.contributions(snap, a, H_MH, paired=True)
                ci = ev.contributions(snap, a, H_MH, paired=False)
                base = base_reward(info["success"], a, d_prev, d_next, g_prev, g_next)
                for m in methods:
                    c = {"aapr1": cp[:1], "mh": cp, "mh_ind": ci}.get(m)
                    r = base + lam * shaping_value(m, d_prev, d_next, info["success"], gamma, c)
                    ret[m] += r; disc[m] += gamma ** t * r
                steps_prog.append((d_prev - d_next) / U); steps_aapr.append(cp[0] / U)
                t += 1
                if term or trunc:
                    break
            succ.append(info["success"])
            for m in methods:
                R[m].append(ret[m]); G[m].append(disc[m])
        res[pname] = dict(sr=float(np.mean(succ)),
                          ret={m: float(np.mean(R[m])) for m in methods},
                          disc={m: float(np.mean(G[m])) for m in methods},
                          step_prog_mean=float(np.mean(steps_prog)), step_prog_var=float(np.var(steps_prog)),
                          step_aapr_mean=float(np.mean(steps_aapr)), step_aapr_var=float(np.var(steps_aapr)))
    res["pass_undisc"] = {m: bool(res["scripted"]["ret"][m] > max(res["random"]["ret"][m], res["zero"]["ret"][m]))
                          for m in methods}
    res["pass_disc"] = {m: bool(res["scripted"]["disc"][m] > max(res["random"]["disc"][m], res["zero"]["disc"][m]))
                        for m in methods}
    return res


def zero_policy_wind_helps(cond, n=50):
    from .train import HELP_SEED0, wind_helps_init
    env = PushEnv(make_cfg(cond))
    rng = np.random.default_rng(HELP_SEED0)
    out = {}
    for pname, pol in {"zero": zero_action, "scripted": lambda e, r: scripted_action(e)}.items():
        rng = np.random.default_rng(HELP_SEED0); s = []
        for i in range(n):
            init = wind_helps_init(cond, rng)
            if init is None:
                return None
            env.reset(seed=HELP_SEED0 + i, options=dict(init=init))
            while True:
                _, _, term, trunc, info = env.step(pol(env, rng))
                if term or trunc:
                    break
            s.append(info["success"])
        out[pname] = float(np.mean(s))
    return out


def _job(args):
    kind, cond = args
    if kind == "canon":
        return kind, cond, canonical_table(cond)
    if kind == "bdist":
        return kind, cond, b_distribution(cond)
    if kind == "unit":
        return kind, cond, unit_test(cond)
    if kind == "helps":
        return kind, cond, zero_policy_wind_helps(cond)


def main(out_dir):
    jobs = [(k, c) for k in ("canon", "bdist", "unit", "helps") for c in CONDS]
    res = {k: {} for k in ("canon", "bdist", "unit", "helps")}
    with Pool(len(jobs)) as p:
        for kind, cond, r in p.imap_unordered(_job, jobs):
            res[kind][cond] = r
            print("done", kind, cond, flush=True)
    with open(f"{out_dir}/phase2.json", "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1])

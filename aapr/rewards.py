"""Reward functions and the counterfactual branch evaluator (design doc 4, 5, 8).

All shaping terms are expressed in units of MAX_STEP_DISP (= max pusher displacement per
step, 0.1), so one step of full-speed pushing straight at the goal gives ~+1.

Common to every method (doc 6.1, "no free shaping for AAPR"):
    r = R_SUCC * success + lam * shaping + ETA * reach - C_ACT * |a|^2
    reach = (e_t - e_{t+1}) / 0.1,  e = |p - (x - n_goal * R_SUM)|  (pusher to push point)

Shaping per method:
    sparse  : 0
    dist    : -d(s_{t+1})                                  (M1)
    prog    : (d_t - d_{t+1}) / 0.1                        (M2)
    pbrs    : (d_t - gamma * d_{t+1}) / 0.1, Phi = -d/0.1, Phi(success terminal) = 0   (M3)
    aapr1   : C^(1) / 0.1     paired (CRN) one-step AAPR   (M4)
    mh      : sum_K alpha_K C^(K) / 0.1, H = {1,4,8}, paired (M5)
    mh_ind  : same as mh with independent branch noise    (M6)

Counterfactual branches:
  * run on a separate env copy (the evaluator), never on the training env.
  * factual branch: first action a_t, then the neutral action (hold) for K-1 steps.
  * neutral branch: neutral action for K steps. (= "fixed continuation", both branches get the
    same continuation; doc 4.1 default / section 3 analysis.)
  * paired: both branches start from the full snapshot *including the RNG state*, so the first
    wind draw equals the real step's draw and later draws are shared (CRN).
    independent: the neutral branch uses a fresh RNG stream.
  * a branch that reaches the success terminal becomes absorbing (state frozen).
  * the episode time limit is ignored inside branches.
  * one trace of length K_max per branch is reused for all K in H (doc 8, compute saving).
"""
from __future__ import annotations

import numpy as np

from .env import MAX_STEP_DISP, PushEnv

R_SUCC = 10.0
ETA = 0.5
C_ACT = 0.01
NEUTRAL = np.zeros(2)


class CounterfactualEvaluator:
    def __init__(self, env: PushEnv, seed: int = 12345):
        self.env = env.clone()
        self.rng = np.random.default_rng(seed)

    @property
    def n_sim_steps(self) -> int:
        return self.env.n_sim_steps

    def trace(self, snap: dict, first_action, K: int, rng_state=None, neutral=NEUTRAL) -> np.ndarray:
        """Distances to goal after 1..K steps. rng_state=None -> restore snapshot RNG (paired)."""
        e = self.env
        e.set_snapshot(snap, restore_rng=True)
        if rng_state is not None:
            e.rng.bit_generator.state = rng_state
        out = np.empty(K)
        e.step(first_action)
        out[0] = e.dist()
        for k in range(1, K):
            e.step(neutral)
            out[k] = e.dist()
        return out

    def contributions(self, snap, action, horizons=(1,), paired=True, neutral=NEUTRAL):
        """C^(K) = d(s^0_{t+K}) - d(s^a_{t+K}) for K in horizons."""
        K = max(horizons)
        da = self.trace(snap, action, K, neutral=neutral)
        if paired:
            d0 = self.trace(snap, neutral, K, neutral=neutral)
        else:
            fresh = np.random.default_rng(self.rng.integers(2 ** 63)).bit_generator.state
            d0 = self.trace(snap, neutral, K, rng_state=fresh, neutral=neutral)
        return np.array([d0[k - 1] - da[k - 1] for k in horizons])

    def one_step_branch(self, snap, action):
        """One transition from snap with `action` (paired noise). Returns (obs, info, env-snapshot)."""
        e = self.env
        e.set_snapshot(snap, restore_rng=True)
        d_prev, gap_prev = e.dist(), e.reach_gap()
        obs, _, term, trunc, info = e.step(action)
        return obs, term, info, d_prev, gap_prev, e.dist(), e.reach_gap()


SHAPINGS = ("sparse", "dist", "prog", "pbrs", "aapr1", "mh", "mh_ind", "aug")
H_MH = (1, 4, 8)


def method_spec(method: str) -> dict:
    """horizons/paired for counterfactual methods, base shaping otherwise."""
    if method == "aapr1":
        return dict(cf=True, horizons=(1,), paired=True)
    if method == "mh":
        return dict(cf=True, horizons=H_MH, paired=True)
    if method == "mh_ind":
        return dict(cf=True, horizons=H_MH, paired=False)
    if method.startswith("k") and method[1:].isdigit():   # fixed-K ablation, e.g. k4, k8
        return dict(cf=True, horizons=(int(method[1:]),), paired=True)
    return dict(cf=False)


def base_reward(success, a, d_prev, d_next, gap_prev, gap_next):
    return R_SUCC * float(success) + ETA * (gap_prev - gap_next) / MAX_STEP_DISP - C_ACT * float(a @ a)


def shaping_value(method, d_prev, d_next, success, gamma, contrib=None):
    if method == "sparse":
        return 0.0
    if method == "dist":
        return -d_next
    if method in ("prog", "aug"):
        return (d_prev - d_next) / MAX_STEP_DISP
    if method == "pbrs":
        phi_next = 0.0 if success else -d_next / MAX_STEP_DISP
        return gamma * phi_next + d_prev / MAX_STEP_DISP
    # counterfactual: equal-weight mean over horizons
    return float(np.mean(contrib)) / MAX_STEP_DISP

"""Counterfactual branch evaluator J^{(K)}_{g,mu}(s,a) and decision-aware gate (design 3.4-3.5).

For each state snapshot, every candidate action a (all 6) is executed in a *cloned* simulator,
then the continuation policy mu is followed for K-1 steps; bootstrap V=0 at K (finite-horizon
rollout, design 3.4 "충분한 finite-horizon rollout").

mu = 'oracle'  : exact optimal policy (see oracle.py). Name: "oracle evaluator".
mu = 'learned' : greedy policy of the *current* online Q-network (ablation; self-confirmation risk).
"""
from __future__ import annotations

import numpy as np
import torch

from .env import N_ACT, State, step, observe, assert_train_goals
from .oracle import oracle_action

TIE = 1e-9


def branch_returns(s: State, go, gt, K: int, gamma: float, mu: str, net=None):
    """J (B, 6) and number of simulator branch steps consumed."""
    B = s.B
    r = s.repeat(N_ACT)
    r.go = np.repeat(np.asarray(go, np.int64), N_ACT)
    r.gt = np.repeat(np.asarray(gt, np.int64), N_ACT)
    r.done[:] = False
    r.t[:] = 0
    ret = np.zeros(r.B)
    disc = 1.0
    nsteps = 0
    act = np.tile(np.arange(N_ACT), B)
    for k in range(K):
        live = ~r.done
        if not live.any():
            break
        if k > 0:
            act = np.zeros(r.B, np.int64)
            sub = r.index(live)
            if mu == "oracle":
                act[live] = oracle_action(sub, sub.go, sub.gt, gamma)
            else:
                with torch.no_grad():
                    q = net(torch.from_numpy(observe(sub)), torch.from_numpy(sub.go), torch.from_numpy(sub.gt))
                act[live] = q.argmax(1).numpy()
        nsteps += int(live.sum())
        rew, _, _ = step(r, act, horizon=K)
        ret += disc * rew
        disc *= gamma
    return ret.reshape(B, N_ACT), nsteps


def pair_choice(Ji: np.ndarray, Jj: np.ndarray):
    """a_i, a_j, Delta_i, Delta_j with deterministic tie-breaking.

    a_i is chosen among the argmax set of J_i, preferring the action that is worst for g_j
    (so that a shared optimal action is detected as shared: then Delta=0 -> gate off).
    """
    Ai = Ji >= Ji.max(1, keepdims=True) - TIE
    Aj = Jj >= Jj.max(1, keepdims=True) - TIE
    ai = np.where(Ai, -Jj, -np.inf).argmax(1)
    aj = np.where(Aj, -Ji, -np.inf).argmax(1)
    ar = np.arange(len(ai))
    # if some action is optimal for both goals, it is a shared action
    shared = (Ai & Aj).any(1)
    sh_a = (Ai & Aj).argmax(1)
    ai = np.where(shared, sh_a, ai)
    aj = np.where(shared, sh_a, aj)
    di = Ji[ar, ai] - Ji[ar, aj]
    dj = Jj[ar, aj] - Jj[ar, ai]
    return ai, aj, di, dj, shared


def gate(ai, aj, di, dj, tau):
    return (ai != aj) & (di > tau) & (dj > tau)


def build_entries(s: State, gi, gj, K, gamma, tau, mu, net=None):
    """Evaluate a batch of (state, g_i, g_j) triples. Only TRAIN goals allowed."""
    assert_train_goals(gi[0], gi[1], "CGC goal pair (g_i)")
    assert_train_goals(gj[0], gj[1], "CGC goal pair (g_j)")
    Ji, n1 = branch_returns(s, gi[0], gi[1], K, gamma, mu, net)
    Jj, n2 = branch_returns(s, gj[0], gj[1], K, gamma, mu, net)
    ai, aj, di, dj, shared = pair_choice(Ji, Jj)
    w = gate(ai, aj, di, dj, tau)
    return dict(Ji=Ji, Jj=Jj, ai=ai, aj=aj, di=di, dj=dj, gate=w, shared=shared), n1 + n2

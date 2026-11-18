"""Exact optimal values for the deterministic gridworld ("oracle").

Reward is 1 only on the successful commit, so with discount gamma
    V*(s, g) = gamma ** (D(s, g) - 1),     Q*(s, a, g) = r + gamma * V*(s', g) = gamma ** D(s', g)
where D is the minimum number of actions to the successful Drop.  D has a closed form
(Manhattan distances; moving is unobstructed).  This is the fixed point of value iteration
on this MDP; tests/test_oracle.py checks it against brute-force BFS (= VI with unit costs)
over the real simulator.  Time is not part of the state, so Q* is the infinite-horizon value.
"""
from __future__ import annotations

import numpy as np

from .env import N, N_CELLS, N_ACT, State, step, CFG

CELLS = np.stack(np.divmod(np.arange(N_CELLS), N), -1)  # (64, 2)


def _manh(a, b):
    return np.abs(a - b).sum(-1)


def dist_to_success(s: State, go: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """D(s, g) for every row (assumes row not terminal)."""
    B = s.B
    ar = np.arange(B)
    P = s.obj[ar, go]  # goal object position (== agent if held)
    T = s.tgt[ar, gt]
    A = s.agent
    D = np.empty(B, np.int64)

    h_o = s.held == go
    D[h_o] = _manh(A[h_o], T[h_o]) + 1

    h_n = s.held < 0
    D[h_n] = _manh(A[h_n], P[h_n]) + 1 + _manh(P[h_n], T[h_n]) + 1

    h_x = ~h_o & ~h_n
    if CFG["drop_only_on_target"]:
        D[h_x] = 10**6  # holding another object: it can only be committed (failure) -> V*=0
    elif h_x.any():
        idx = np.where(h_x)[0]
        # free cells to drop the held (other) object: not a target, not occupied by a
        # non-held object.  The agent's own cell counts (held object sits there).
        occ = np.zeros((idx.size, N_CELLS), bool)
        for k in range(s.obj.shape[1]):
            notheld = s.held[idx] != k
            cid = s.obj[idx, k, 0] * N + s.obj[idx, k, 1]
            occ[np.arange(idx.size)[notheld], cid[notheld]] = True
        for k in range(s.tgt.shape[1]):
            occ[np.arange(idx.size), s.tgt[idx, k, 0] * N + s.tgt[idx, k, 1]] = True
        c = _manh(A[idx][:, None, :], CELLS[None]) + _manh(CELLS[None], P[idx][:, None, :])
        c = np.where(occ, 10**6, c).min(1)
        D[idx] = c + 1 + 1 + _manh(P[idx], T[idx]) + 1
    return D


def q_star(s: State, go: np.ndarray, gt: np.ndarray, gamma: float) -> np.ndarray:
    """Q*(s, a, g) for all 6 actions -> (B, 6). Rows must be non-terminal."""
    B = s.B
    r = s.repeat(N_ACT)
    r.go = np.repeat(np.asarray(go, np.int64), N_ACT)
    r.gt = np.repeat(np.asarray(gt, np.int64), N_ACT)
    r.done[:] = False
    r.t[:] = 0
    acts = np.tile(np.arange(N_ACT), B)
    rew, term, _ = step(r, acts, horizon=10**9)
    q = np.where(term, rew, 0.0).astype(np.float64)
    nt = ~term
    if nt.any():
        q[nt] = gamma ** dist_to_success(r.index(nt), r.go[nt], r.gt[nt])
    return q.reshape(B, N_ACT)


def oracle_action(s: State, go, gt, gamma: float) -> np.ndarray:
    """Deterministic optimal action (lowest index among ties)."""
    return q_star(s, go, gt, gamma).argmax(1)

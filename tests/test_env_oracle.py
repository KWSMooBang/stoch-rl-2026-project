import numpy as np
import pytest

from dacgc.env import (GridPickPlaceEnv, State, sample_layouts, step, observe, N, N_ACT, OUT_SUCCESS,
                       TRAIN_GOALS, HELDOUT_GOALS, assert_train_goals, LeakageError, goal_probs)
from dacgc.oracle import dist_to_success, q_star, oracle_action

G = 0.95


def _key(s, i):
    return (tuple(s.agent[i]), tuple(s.obj[i].ravel()), int(s.held[i]))


def bfs_distance(s0: State, max_depth=40):
    """Brute-force shortest #actions to success over the real simulator (VI w/ unit costs)."""
    frontier = s0.clone()
    frontier.done[:] = False
    seen = {_key(frontier, 0)}
    for d in range(1, max_depth + 1):
        B = frontier.B
        r = frontier.repeat(N_ACT)
        rew, term, _ = step(r, np.tile(np.arange(N_ACT), B), horizon=10**9)
        if (rew > 0).any():
            return d
        keep = []
        for i in range(r.B):
            if term[i]:
                continue
            k = _key(r, i)
            if k not in seen:
                seen.add(k)
                keep.append(i)
        if not keep:
            return None
        frontier = r.index(np.array(keep))
    return None


def random_states(n, seed, max_rand_steps=30):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        g = TRAIN_GOALS[rng.integers(len(TRAIN_GOALS))]
        s = sample_layouts(rng, 1, [g[0]], [g[1]])
        for _ in range(rng.integers(0, max_rand_steps)):
            a = rng.integers(N_ACT)
            sc = s.clone()
            step(sc, np.array([a]), 10**9)
            if sc.done[0]:
                break
            s = sc
        out.append(s)
    return out


def test_clone_restore_exact():
    env = GridPickPlaceEnv(horizon=50)
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(5):
        env.step(int(rng.integers(4)))
    snap = env.clone_state()
    acts = rng.integers(0, N_ACT, 20)
    traj1 = [env.step(int(a)) for a in acts]
    env.restore_state(snap)
    traj2 = [env.step(int(a)) for a in acts]
    for (o1, r1, te1, tr1, _), (o2, r2, te2, tr2, _) in zip(traj1, traj2):
        assert np.array_equal(o1, o2) and r1 == r2 and te1 == te2 and tr1 == tr2
        if te1 or tr1:
            break
    # restoring must not alias
    env.restore_state(snap)
    env.step(4)
    env.restore_state(snap)
    assert np.array_equal(env.clone_state().agent, snap.agent)


def test_dynamics_pick_drop_commit():
    rng = np.random.default_rng(1)
    s = sample_layouts(rng, 1, [0], [1])
    s.agent[0] = s.obj[0, 0]
    step(s, np.array([4]), 50)
    assert s.held[0] == 0
    s.agent[0] = s.tgt[0, 1]; s.obj[0, 0] = s.agent[0]
    r, te, tr = step(s, np.array([5]), 50)
    assert r[0] == 1 and te[0] and s.outcome[0] == OUT_SUCCESS
    # wrong target
    s = sample_layouts(rng, 1, [0], [1])
    s.held[0] = 0; s.agent[0] = s.tgt[0, 2]; s.obj[0, 0] = s.agent[0]
    r, te, _ = step(s, np.array([5]), 50)
    assert r[0] == 0 and te[0] and s.outcome[0] == 2
    # timeout
    s = sample_layouts(rng, 1, [0], [1])
    for _ in range(50):
        r, te, tr = step(s, np.array([4 if s.held[0] >= 0 else 0]), 50)
    assert tr[0] and s.outcome[0] == 4


def test_oracle_matches_bfs():
    states = random_states(60, seed=0)
    n_other = 0
    for s in states:
        D = dist_to_success(s, s.go, s.gt)[0]
        if D > 16:
            continue
        n_other += int(s.held[0] >= 0 and s.held[0] != s.go[0])
        assert bfs_distance(s, max_depth=D + 1) == D
    assert n_other >= 1


def test_oracle_policy_reaches_in_D_steps():
    rng = np.random.default_rng(5)
    B = 256
    go = rng.integers(0, 3, B); gt = rng.integers(0, 3, B)
    s = sample_layouts(rng, B, go, gt)
    D = dist_to_success(s, go, gt)
    steps = np.zeros(B, int)
    for _ in range(60):
        live = ~s.done
        a = np.zeros(B, int)
        if live.any():
            a[live] = oracle_action(s.index(live), go[live], gt[live], G)
        steps += live
        step(s, a, 60)
    assert (s.outcome == OUT_SUCCESS).all()
    assert np.array_equal(steps, D)


def test_q_star_bellman():
    for s in random_states(30, seed=2):
        q = q_star(s, s.go, s.gt, G)[0]
        D = dist_to_success(s, s.go, s.gt)[0]
        assert np.isclose(q.max(), G ** (D - 1))


def test_leakage_assert():
    assert_train_goals(np.array([g[0] for g in TRAIN_GOALS]), np.array([g[1] for g in TRAIN_GOALS]), "t")
    with pytest.raises(LeakageError):
        assert_train_goals(np.array([HELDOUT_GOALS[0][0]]), np.array([HELDOUT_GOALS[0][1]]), "t")
    for sk in (0.5, 0.8, 0.95):
        p = goal_probs(sk).reshape(3, 3)
        for o, t in HELDOUT_GOALS:
            assert p[o, t] == 0


def test_observe_shape():
    s = sample_layouts(np.random.default_rng(0), 4, [0] * 4, [0] * 4)
    o = observe(s)
    assert o.shape[0] == 4 and np.isfinite(o).all()


def test_drop_only_on_target_variant():
    from dacgc.env import CFG
    CFG["drop_only_on_target"] = True
    try:
        n_checked = 0
        for s in random_states(60, seed=7):
            D = dist_to_success(s, s.go, s.gt)[0]
            if s.held[0] >= 0 and s.held[0] != s.go[0]:
                assert D >= 10**6 and bfs_distance(s, max_depth=14) is None
                continue
            if D > 16:
                continue
            assert bfs_distance(s, max_depth=D + 1) == D
            n_checked += 1
        assert n_checked > 10
        # drop off-target is a no-op
        rng = np.random.default_rng(0)
        s = sample_layouts(rng, 1, [0], [0])
        s.agent[0] = s.obj[0, 0]
        step(s, np.array([4]), 50)
        step(s, np.array([5]), 50)
        assert s.held[0] == 0
    finally:
        CFG["drop_only_on_target"] = False


def test_phi_obs():
    from dacgc.train import phi_obs
    rng = np.random.default_rng(3)
    s = sample_layouts(rng, 64, rng.integers(0, 3, 64), rng.integers(0, 3, 64), p_hold=0.5)
    p = phi_obs(observe(s), s.go)
    ar = np.arange(64)
    d = np.abs(s.obj[ar, s.go] - s.agent).sum(1)
    exp = np.where(s.held == s.go, 1.0, np.where(s.held < 0, -d / (2.0 * (N - 1)), 0.0))
    assert np.allclose(p, exp, atol=1e-5)

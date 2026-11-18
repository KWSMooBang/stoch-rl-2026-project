"""Phase 0 dynamics tests (design doc 13, Phase 0; 7.2 sanity checks)."""
import numpy as np
import pytest

from aapr.env import DT, R_SUM, PushEnv, make_cfg
from aapr.rewards import CounterfactualEvaluator, NEUTRAL

CONDS = ["none", "iid", "ar1", "bias", "ar1_hi"]


def _state(e):
    return np.concatenate([e.p, e.x, e.v, e.w, [e.t, e.dwell]])


@pytest.mark.parametrize("cond", CONDS)
def test_snapshot_restore_exact(cond):
    """(1) full state + RNG restore reproduces the rollout bit-for-bit, also on a separate copy."""
    env = PushEnv(make_cfg(cond), seed=3)
    env.reset(seed=3)
    acts = np.random.default_rng(0).uniform(-1, 1, (60, 2))
    for a in acts[:20]:
        env.step(a)
    snap = env.get_snapshot()
    ref = []
    for a in acts[20:]:
        env.step(a)
        ref.append(_state(env))
    # restore on the same env
    env.set_snapshot(snap)
    for i, a in enumerate(acts[20:]):
        env.step(a)
        assert np.array_equal(_state(env), ref[i])
    # restore on an independent copy (the evaluator path)
    other = PushEnv(make_cfg(cond), seed=999)
    other.reset(seed=999)
    other.set_snapshot(snap)
    for i, a in enumerate(acts[20:]):
        other.step(a)
        assert np.array_equal(_state(other), ref[i])


@pytest.mark.parametrize("cond", CONDS)
def test_paired_factual_branch_equals_real_step(cond):
    env = PushEnv(make_cfg(cond), seed=5)
    env.reset(seed=5)
    ev = CounterfactualEvaluator(env)
    rng = np.random.default_rng(1)
    for _ in range(40):
        a = rng.uniform(-1, 1, 2)
        snap = env.get_snapshot()
        d_branch = ev.trace(snap, a, 1)[0]
        env.step(a)
        assert d_branch == env.dist()


def test_contact_impulse():
    """(2) no wind: no contact -> no force; pushing into the object transfers velocity."""
    cfg = make_cfg("none")
    env = PushEnv(cfg)
    # pusher far away, moving in all directions: object stays at rest
    env.reset(options=dict(init=dict(p=[-0.8, 0.5], x=[0.0, 0.0], g=[0.6, 0.0])))
    for a in [[1, 0], [0, -1], [-1, 1], [0.3, 0.3]]:
        env.step(np.array(a, float))
        assert np.all(env.v == 0) and np.all(env.x == 0)
    # pusher just behind object, push +x for 3 steps then stop
    env.reset(options=dict(init=dict(p=[-R_SUM - 0.02, 0.0], x=[0.0, 0.0], g=[0.6, 0.0])))
    vs = []
    for k in range(8):
        a = np.array([1.0, 0.0]) if k < 3 else NEUTRAL
        env.step(a)
        vs.append(env.v.copy())
    vs = np.array(vs)
    assert vs[2, 0] == pytest.approx(1.0)   # object moves at pusher speed while pushed
    assert np.allclose(vs[:, 1], 0)
    # after the pusher stops, the object coasts with v *= rho
    assert vs[4, 0] == pytest.approx(cfg.rho * vs[3, 0])
    # pusher moving away from a touching object does not pull it
    env.reset(options=dict(init=dict(p=[-R_SUM, 0.0], x=[0.0, 0.0], g=[0.6, 0.0])))
    env.step(np.array([-1.0, 0.0]))
    assert np.all(env.v == 0)
    # stationary pusher blocks an object moving into it (hold neutral resists)
    env.reset(options=dict(init=dict(p=[-R_SUM - 0.001, 0.0], x=[0.0, 0.0], v=[-1.0, 0.0], g=[0.6, 0.0])))
    env.step(NEUTRAL)
    assert env.v[0] >= 0 and np.linalg.norm(env.x - env.p) >= R_SUM - 1e-9


@pytest.mark.parametrize("cond", ["iid", "ar1", "bias"])
def test_wind_only_factual_equals_neutral(cond):
    """(3) wind only (pusher never touches): paired factual and neutral branches coincide."""
    env = PushEnv(make_cfg(cond), seed=7)
    rng = np.random.default_rng(2)
    ev = CounterfactualEvaluator(env)
    for trial in range(30):
        env.reset(options=dict(init=dict(p=[-0.85, -0.85], x=[0.3, 0.3], g=[-0.3, 0.4])))
        for _ in range(trial % 5):
            env.step(NEUTRAL)
        a = rng.uniform(-1, 1, 2)
        snap = env.get_snapshot()
        c = ev.contributions(snap, a, horizons=(1, 4, 8), paired=True)
        assert np.all(c == 0.0), c
        # the wind did move the object (the test is not vacuous)
        assert ev.trace(snap, a, 8)[-1] != env.dist()
        # independent noise does NOT cancel
    c_ind = [ev.contributions(snap, rng.uniform(-1, 1, 2), (8,), paired=False)[0] for _ in range(20)]
    assert np.std(c_ind) > 0


def test_inertia_delayed_effect():
    """(4) high inertia: a single push keeps paying off over K; ratio matches the linear formula."""
    for rho in (0.8, 0.95):
        env = PushEnv(make_cfg("none", rho=rho))
        env.reset(options=dict(init=dict(p=[-0.6 - R_SUM - 0.05, 0.0], x=[-0.6, 0.0], v=[0.0, 0.0], g=[0.9, 0.0])))
        # bring the pusher into contact
        env.step(np.array([0.5, 0.0]))
        ev = CounterfactualEvaluator(env)
        snap = env.get_snapshot()
        C = ev.contributions(snap, np.array([1.0, 0.0]), horizons=tuple(range(1, 31)), paired=True)
        assert np.all(np.diff(C) >= -1e-12)            # contribution accumulates
        dv = C[0] / DT                                 # one-step velocity difference
        for K in (4, 8, 16):
            pred = DT * dv * (1 - rho ** K) / (1 - rho)
            assert C[K - 1] == pytest.approx(pred, rel=1e-6)
    # with high inertia the one-step contribution is a much smaller fraction of the 16-step one

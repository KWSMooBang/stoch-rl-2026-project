"""Hand-written near-optimal pushing controller (reward unit test reference) + trivial policies."""
import numpy as np

from .env import DT, R_SUM, PushEnv


def scripted_action(env: PushEnv, gain: float = 6.0) -> np.ndarray:
    p, x, v, g = env.p, env.x, env.v, env.g
    to_g = g - x
    dg = np.linalg.norm(to_g)
    if dg < 1e-6:
        return np.zeros(2)
    n = to_g / dg
    # desired object velocity: toward the goal, slowing near it; compensate drift/inertia
    k = (1.0 - env.cfg.rho) / DT          # speed at which coasting stops right at the goal
    v_des = n * min(1.0, k * dg) - 0.5 * v
    # push point behind the object along the desired direction
    vd = v_des / max(np.linalg.norm(v_des), 1e-6)
    push_pt = x - vd * (R_SUM - 0.01)
    approach_pt = x - vd * (R_SUM + 0.06)
    rel = p - x
    behind = (rel @ vd) < -(R_SUM * 0.7) and np.linalg.norm(rel - (rel @ vd) * vd) < R_SUM * 0.6
    if behind:
        u = gain * (push_pt - p) * 0.1 + vd * min(1.0, k * dg)
    else:
        # go around the object: target the approach point, sidestep if the object is in the way
        tgt = approach_pt
        d = tgt - p
        if np.linalg.norm(rel) < R_SUM + 0.05 and (rel @ vd) > -R_SUM * 0.7:
            side = np.array([-vd[1], vd[0]])
            side = side if (rel @ side) >= 0 else -side
            d = side * 0.5 + (tgt - p)
        u = d / max(np.linalg.norm(d), 1e-6)
        u *= min(1.0, np.linalg.norm(tgt - p) / 0.1 + 0.3)
    return np.clip(u, -1, 1)


def zero_action(env, rng=None):
    return np.zeros(2)


def random_action(env, rng):
    return rng.uniform(-1, 1, 2)

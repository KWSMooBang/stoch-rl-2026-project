"""2D stochastic inertial pushing environment (numpy, Gymnasium API).

Design doc section 7. A kinematic disk pusher (velocity command) pushes a disk object
with inertia toward a goal. Wind acts on the object only (exogenous disturbance).

Dynamics per step (dt = 0.1):
    p_r' = clip(p_r + dt * u)                          pusher (kinematic, position-controlled)
    w_t  = wind process sample                         (iid / AR(1) / per-episode bias)
    v_f  = rho * v_o + w_t                             free object velocity (inertia + wind)
    contact: if the object would overlap the pusher, the normal component of the object
             velocity is raised to the pusher's normal velocity (inelastic, beta = 1) and the
             object is projected out of the pusher. No contact -> no force on the object.
    x_o' = clip(x_o + dt * v_o')                       walls zero the normal velocity

Neutral action a0 = (0, 0): zero velocity command = the pusher holds its position
(a physically realisable "do nothing" for a velocity-controlled robot).

Full snapshot (get_snapshot/set_snapshot) contains every piece of mutable state including
the numpy Generator state, so a restored env reproduces the same rollout bit-for-bit.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

DT = 0.1
ARENA = 1.0
R_PUSHER = 0.05
R_OBJECT = 0.08
R_SUM = R_PUSHER + R_OBJECT
GOAL_R = 0.15
DWELL = 3            # consecutive steps inside the goal region -> success (terminal)
MAX_STEP_DISP = DT * 1.0   # max pusher displacement per step; reward unit


@dataclass
class EnvCfg:
    wind: str = "ar1"          # none | iid | ar1 | bias
    rho: float = 0.8           # object velocity retention per step (inertia)
    sigma: float = 0.04        # wind scale (per-step velocity impulse units)
    kappa: float = 0.95        # AR(1) persistence
    bias_mu: float = 0.04      # bias-wind magnitude (per episode, random direction)
    bias_sigma: float = 0.02   # small iid part on top of the bias
    max_steps: int = 100
    # optional fixed reset (used by tests / analysis)
    init: dict | None = field(default=None)


WIND_PRESETS = {
    "none": dict(wind="none"),
    "iid": dict(wind="iid", sigma=0.10),
    "ar1": dict(wind="ar1", sigma=0.04, kappa=0.95),
    "bias": dict(wind="bias", bias_mu=0.04, bias_sigma=0.02),
    "ar1_hi": dict(wind="ar1", sigma=0.02, kappa=0.95, rho=0.95),  # high inertia (auxiliary)
}


def make_cfg(cond: str, **kw) -> EnvCfg:
    d = dict(WIND_PRESETS[cond])
    d.update(kw)
    return EnvCfg(**d)


class PushEnv(gym.Env):
    metadata = {"render_modes": []}
    OBS_DIM = 16

    def __init__(self, cfg: EnvCfg | None = None, seed: int | None = None):
        self.cfg = cfg or EnvCfg()
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.OBS_DIM,), np.float32)
        self.rng = np.random.default_rng(seed)
        self.n_sim_steps = 0   # counts every physics transition executed by this instance
        self._zero_state()

    # ------------------------------------------------------------------ state
    def _zero_state(self):
        self.p = np.zeros(2)
        self.x = np.zeros(2)
        self.v = np.zeros(2)
        self.g = np.zeros(2)
        self.w = np.zeros(2)      # wind latent: last AR(1) sample / bias mean / zeros for iid
        self.t = 0
        self.dwell = 0
        self.done = False

    def get_snapshot(self) -> dict:
        return dict(p=self.p.copy(), x=self.x.copy(), v=self.v.copy(), g=self.g.copy(),
                    w=self.w.copy(), t=self.t, dwell=self.dwell, done=self.done,
                    rng=copy.deepcopy(self.rng.bit_generator.state))

    def set_snapshot(self, s: dict, restore_rng: bool = True):
        self.p, self.x, self.v = s["p"].copy(), s["x"].copy(), s["v"].copy()
        self.g, self.w = s["g"].copy(), s["w"].copy()
        self.t, self.dwell, self.done = s["t"], s["dwell"], s["done"]
        if restore_rng:
            self.rng.bit_generator.state = copy.deepcopy(s["rng"])

    def clone(self) -> "PushEnv":
        e = PushEnv(copy.deepcopy(self.cfg))
        e.set_snapshot(self.get_snapshot())
        return e

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._zero_state()
        init = (options or {}).get("init", self.cfg.init)
        rng = self.rng
        if init is not None:
            self.p = np.asarray(init.get("p", [-0.5, 0.0]), float).copy()
            self.x = np.asarray(init.get("x", [0.0, 0.0]), float).copy()
            self.v = np.asarray(init.get("v", [0.0, 0.0]), float).copy()
            self.g = np.asarray(init.get("g", [0.6, 0.0]), float).copy()
            self.w = np.asarray(init.get("w", self._init_wind(rng)), float).copy()
        else:
            while True:
                x = rng.uniform(-0.6, 0.6, 2)
                g = rng.uniform(-0.6, 0.6, 2)
                if np.linalg.norm(g - x) > 0.4:
                    break
            while True:
                p = rng.uniform(-0.9, 0.9, 2)
                if np.linalg.norm(p - x) > R_SUM + 0.1:
                    break
            self.p, self.x, self.g = p, x, g
            self.w = self._init_wind(rng)
        return self._obs(), {}

    def _init_wind(self, rng) -> np.ndarray:
        c = self.cfg
        if c.wind == "ar1":
            return rng.normal(0.0, c.sigma, 2)            # stationary distribution
        if c.wind == "bias":
            th = rng.uniform(0, 2 * np.pi)
            return c.bias_mu * np.array([np.cos(th), np.sin(th)])
        return np.zeros(2)

    def sample_wind(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Returns (impulse w_t, new latent). Uses exactly 2 normal draws for every wind type
        so that RNG streams stay aligned across branches."""
        c = self.cfg
        eps = rng.standard_normal(2)
        if c.wind == "none":
            return np.zeros(2), self.w
        if c.wind == "iid":
            return c.sigma * eps, self.w
        if c.wind == "ar1":
            w = c.kappa * self.w + np.sqrt(1 - c.kappa ** 2) * c.sigma * eps
            return w, w
        if c.wind == "bias":
            return self.w + c.bias_sigma * eps, self.w
        raise ValueError(c.wind)

    def physics(self, u: np.ndarray, wind: np.ndarray, contact: bool = True):
        """One physics transition with a given wind impulse. Mutates state. Returns info."""
        c = self.cfg
        u = np.clip(np.asarray(u, float), -1.0, 1.0)
        p_new = np.clip(self.p + DT * u, -ARENA + R_PUSHER, ARENA - R_PUSHER)
        v_r = (p_new - self.p) / DT
        v = c.rho * self.v + wind
        x_free = self.x + DT * v
        touched = False
        if contact:
            dvec = x_free - p_new
            dist = np.linalg.norm(dvec)
            if dist < R_SUM:
                touched = True
                if dist > 1e-9:
                    n = dvec / dist
                else:
                    d0 = self.x - self.p
                    n = d0 / max(np.linalg.norm(d0), 1e-9)
                vn, rn = v @ n, v_r @ n
                if vn < rn:
                    v = v + (rn - vn) * n
                x_new = self.x + DT * v
                dvec2 = x_new - p_new
                d2 = np.linalg.norm(dvec2)
                if d2 < R_SUM:
                    x_new = p_new + n * R_SUM if d2 < 1e-9 else p_new + dvec2 / d2 * R_SUM
            else:
                x_new = x_free
        else:
            x_new = x_free
        # walls
        lim = ARENA - R_OBJECT
        for i in range(2):
            if x_new[i] > lim:
                x_new[i] = lim; v[i] = min(v[i], 0.0)
            elif x_new[i] < -lim:
                x_new[i] = -lim; v[i] = max(v[i], 0.0)
        self.p, self.x, self.v = p_new, x_new, v
        self.n_sim_steps += 1
        return touched

    def step(self, action, contact: bool = True):
        if self.done:  # absorbing
            return self._obs(), 0.0, True, False, dict(success=True, touched=False, absorbing=True)
        d_prev = self.dist()
        wind, new_latent = self.sample_wind(self.rng)
        touched = self.physics(action, wind, contact=contact)
        self.w = new_latent
        self.t += 1
        self.dwell = self.dwell + 1 if self.dist() < GOAL_R else 0
        success = self.dwell >= DWELL
        self.done = success
        truncated = (self.t >= self.cfg.max_steps) and not success
        info = dict(success=success, touched=touched, d_prev=d_prev, d=self.dist(),
                    wind=wind)
        return self._obs(), float(success), success, truncated, info

    # ------------------------------------------------------------------ helpers
    def dist(self) -> float:
        return float(np.linalg.norm(self.x - self.g))

    def reach_gap(self) -> float:
        """Distance from the pusher to the push point behind the object (opposite the goal).
        Shared reach shaping for every method (doc 6.1)."""
        to_g = self.g - self.x
        n = to_g / max(np.linalg.norm(to_g), 1e-9)
        return float(np.linalg.norm(self.p - (self.x - n * R_SUM)))

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.p, self.x, self.v, self.g, self.w * 10.0,
                               self.x - self.p, self.g - self.x, [self.dwell / DWELL, self.t / self.cfg.max_steps]]
                              ).astype(np.float32)

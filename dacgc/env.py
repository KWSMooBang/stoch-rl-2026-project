"""8x8 pick-and-place gridworld (design doc 4.1), batched numpy core + Gymnasium wrapper.

Objects: 0 Bottle, 1 Bowl, 2 Box.  Targets: 0 Rack, 1 Plate, 2 Stove.
Actions: 0 Up, 1 Down, 2 Left, 3 Right, 4 Pick, 5 Drop.

Dynamics (deterministic):
  * moves clip at the wall; a held object moves with the agent.
  * Pick: if hand empty and an object is on the agent cell -> hold it (else no-op).
  * Drop: if holding and no *other* object is on the cell -> put it down (else no-op).
    Dropping onto a target cell is a *commit*: the episode terminates.
    reward = 1 iff (dropped object, target) == goal, else 0.
  * Objects never share a cell; objects/targets never start on each other.
  * Timeout (truncation) at horizon H.

Episode outcome categories (used for SR / SER):
  success, wrong_target (goal object on a different target), wrong_object
  (another object committed on any target), timeout.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

N = 8
N_CELLS = N * N
N_OBJ = 3
N_TGT = 3
N_ACT = 6
OBJ_NAMES = ["Bottle", "Bowl", "Box"]
TGT_NAMES = ["Rack", "Plate", "Stove"]
ACT_NAMES = ["Up", "Down", "Left", "Right", "Pick", "Drop"]
MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)
PLANE_DIM = N_CELLS * (1 + N_OBJ + N_TGT) + (N_OBJ + 1)
# object-centric features: abs coords (14), pairwise offsets agent-obj(6), agent-tgt(6), obj-tgt(18),
# on-cell indicators agent/obj(3), agent/tgt(3), held one-hot (4, LAST)
OBS_DIM = 2 + 2 * N_OBJ + 2 * N_TGT + 2 * N_OBJ + 2 * N_TGT + 2 * N_OBJ * N_TGT + N_OBJ + N_TGT + (N_OBJ + 1)

# ---- compositional split (design doc 4.2) ----
ALL_GOALS = [(o, t) for o in range(N_OBJ) for t in range(N_TGT)]
HELDOUT_GOALS = [(0, 2), (1, 0), (2, 1)]  # Bottle->Stove, Bowl->Rack, Box->Plate
TRAIN_GOALS = [g for g in ALL_GOALS if g not in HELDOUT_GOALS]
# frequent ("preferred") target of each object under skew; a perfect matching so every
# target is preferred by exactly one object.
PREF_TARGET = {0: 0, 1: 1, 2: 2}  # Bottle->Rack, Bowl->Plate, Box->Stove
RARE_TARGET = {0: 1, 1: 2, 2: 0}  # Bottle->Plate, Bowl->Stove, Box->Rack
assert all((o, PREF_TARGET[o]) in TRAIN_GOALS and (o, RARE_TARGET[o]) in TRAIN_GOALS for o in range(N_OBJ))
HELDOUT_ID = np.zeros((N_OBJ, N_TGT), dtype=bool)
for _o, _t in HELDOUT_GOALS:
    HELDOUT_ID[_o, _t] = True

OUT_NONE, OUT_SUCCESS, OUT_WRONG_TARGET, OUT_WRONG_OBJECT, OUT_TIMEOUT = 0, 1, 2, 3, 4


# Dynamics variant (set once per process from the config): if True, Drop is a no-op except on a
# target cell, i.e. a picked object is carried until it is committed somewhere.
CFG = {"drop_only_on_target": False}


class LeakageError(AssertionError):
    pass


LEAK_CHECKS = {"n": 0}


def assert_train_goals(go: np.ndarray, gt: np.ndarray, where: str) -> None:
    """Hard assert used at every point where a goal enters a learning signal."""
    go = np.asarray(go).reshape(-1)
    gt = np.asarray(gt).reshape(-1)
    LEAK_CHECKS["n"] += go.size
    if HELDOUT_ID[go, gt].any():
        bad = sorted({(int(a), int(b)) for a, b in zip(go, gt) if HELDOUT_ID[a, b]})
        raise LeakageError(f"held-out goal(s) {bad} leaked into {where}")


def goal_probs(skew: float) -> np.ndarray:
    """Sampling distribution over the 9 goals (held-out get 0).

    skew=None/0.5 -> balanced over 6 train goals; otherwise object uniform, preferred
    target with prob `skew`, the other train target with 1-skew.
    """
    p = np.zeros((N_OBJ, N_TGT))
    for o in range(N_OBJ):
        p[o, PREF_TARGET[o]] = skew / N_OBJ
        p[o, RARE_TARGET[o]] = (1 - skew) / N_OBJ
    assert np.isclose(p.sum(), 1) and (p[HELDOUT_ID] == 0).all()
    return p.reshape(-1)


class State:
    """Batched simulator state (all arrays shape (B, ...)). Cheap to clone."""

    __slots__ = ("agent", "obj", "held", "tgt", "go", "gt", "t", "done", "outcome")

    def __init__(self, agent, obj, held, tgt, go, gt, t, done, outcome):
        self.agent, self.obj, self.held, self.tgt = agent, obj, held, tgt
        self.go, self.gt, self.t, self.done, self.outcome = go, gt, t, done, outcome

    @property
    def B(self):
        return self.agent.shape[0]

    def clone(self) -> "State":
        return State(*(getattr(self, k).copy() for k in self.__slots__))

    def index(self, idx) -> "State":
        return State(*(getattr(self, k)[idx].copy() for k in self.__slots__))

    def repeat(self, r: int) -> "State":
        """Each row repeated r times consecutively (row i -> rows i*r..i*r+r-1)."""
        return State(*(np.repeat(getattr(self, k), r, axis=0) for k in self.__slots__))

    def set_rows(self, idx, other: "State") -> None:
        for k in self.__slots__:
            getattr(self, k)[idx] = getattr(other, k)

    @staticmethod
    def concat(states) -> "State":
        return State(*(np.concatenate([getattr(s, k) for s in states]) for k in State.__slots__))


def sample_layouts(rng: np.random.Generator, B: int, go, gt, p_hold: float = 0.0) -> State:
    """Random layout. With prob p_hold (training-only exploration aid, goal-agnostic) the agent
    starts on a uniformly random object's cell holding it."""
    s = _sample_layouts(rng, B, go, gt)
    if p_hold > 0:
        h = rng.random(B) < p_hold
        k = rng.integers(N_OBJ, size=B)
        s.held[h] = k[h]
        s.agent[h] = s.obj[np.arange(B), k][h]
    return s


def _sample_layouts(rng: np.random.Generator, B: int, go, gt) -> State:
    cells = np.argsort(rng.random((B, N_CELLS)), axis=1)[:, : 1 + N_OBJ + N_TGT]
    rc = np.stack([cells // N, cells % N], -1)
    agent = rc[:, 0].copy()
    obj = rc[:, 1 : 1 + N_OBJ].copy()
    tgt = rc[:, 1 + N_OBJ :].copy()
    z = np.zeros(B, dtype=np.int64)
    return State(agent, obj, np.full(B, -1, np.int64), tgt, np.asarray(go, np.int64).copy(),
                 np.asarray(gt, np.int64).copy(), z.copy(), np.zeros(B, bool), z.copy())


def step(s: State, a: np.ndarray, horizon: int):
    """In-place step of all non-done rows. Returns (reward, terminated, truncated)."""
    a = np.asarray(a, np.int64)
    B = s.B
    live = ~s.done
    ar = np.arange(B)
    rew = np.zeros(B, np.float32)
    term = np.zeros(B, bool)

    mv = live & (a < 4)
    if mv.any():
        d = MOVES[np.minimum(a, 3)]
        newpos = np.clip(s.agent + d, 0, N - 1)
        s.agent[mv] = newpos[mv]
        hm = mv & (s.held >= 0)
        s.obj[ar[hm], s.held[hm]] = s.agent[hm]

    pk = live & (a == 4) & (s.held < 0)
    if pk.any():
        on = (s.obj == s.agent[:, None, :]).all(-1)  # (B, 3)
        has = on.any(1)
        sel = pk & has
        s.held[sel] = on[sel].argmax(1)

    dp = live & (a == 5) & (s.held >= 0)
    if dp.any():
        on = (s.obj == s.agent[:, None, :]).all(-1)  # held object is always on agent cell
        n_on = on.sum(1)
        ok = dp & (n_on == 1)  # no other object on this cell
        on_t = (s.tgt == s.agent[:, None, :]).all(-1)  # (B, 3)
        if CFG["drop_only_on_target"]:
            ok &= on_t.any(1)
        commit = ok & on_t.any(1)
        tid = on_t.argmax(1)
        dropped = s.held.copy()
        s.held[ok] = -1
        if commit.any():
            succ = commit & (dropped == s.go) & (tid == s.gt)
            wt = commit & (dropped == s.go) & (tid != s.gt)
            wo = commit & (dropped != s.go)
            rew[succ] = 1.0
            s.outcome[succ] = OUT_SUCCESS
            s.outcome[wt] = OUT_WRONG_TARGET
            s.outcome[wo] = OUT_WRONG_OBJECT
            term = commit

    s.t[live] += 1
    trunc = live & ~term & (s.t >= horizon)
    s.outcome[trunc] = OUT_TIMEOUT
    s.done |= term | trunc
    return rew, term, trunc


def commit_info(s: State, a: np.ndarray):
    """Before stepping: which (object, target) would a Drop commit? (-1 if none)."""
    on_t = (s.tgt == s.agent[:, None, :]).all(-1)
    on = (s.obj == s.agent[:, None, :]).all(-1)
    c = (np.asarray(a) == 5) & (s.held >= 0) & (on.sum(1) == 1) & on_t.any(1) & ~s.done
    return np.where(c, s.held, -1), np.where(c, on_t.argmax(1), -1)


def observe(s: State) -> np.ndarray:
    """Object-centric goal-agnostic features (goal NOT included). Held one-hot is the last 4 dims."""
    B = s.B
    f = 1.0 / (N - 1)
    A = s.agent[:, None, :].astype(np.float32)
    O = s.obj.astype(np.float32)
    Tg = s.tgt.astype(np.float32)
    held = np.zeros((B, N_OBJ + 1), np.float32)
    held[np.arange(B), s.held + 1] = 1
    parts = [A.reshape(B, -1) * f, O.reshape(B, -1) * f, Tg.reshape(B, -1) * f,
             (O - A).reshape(B, -1) * f, (Tg - A).reshape(B, -1) * f,
             (Tg[:, None, :, :] - O[:, :, None, :]).reshape(B, -1) * f,
             (s.obj == s.agent[:, None, :]).all(-1).astype(np.float32),
             (s.tgt == s.agent[:, None, :]).all(-1).astype(np.float32), held]
    out = np.concatenate(parts, 1)
    assert out.shape[1] == OBS_DIM
    return out


def observe_planes(s: State) -> np.ndarray:
    """(unused) Flat one-hot planes (agent, 3 objects, 3 targets) + held one-hot. Goal NOT included."""
    B = s.B
    obs = np.zeros((B, PLANE_DIM), np.float32)
    ar = np.arange(B)
    obs[ar, s.agent[:, 0] * N + s.agent[:, 1]] = 1
    for k in range(N_OBJ):
        obs[ar, N_CELLS * (1 + k) + s.obj[:, k, 0] * N + s.obj[:, k, 1]] = 1
    for k in range(N_TGT):
        obs[ar, N_CELLS * (1 + N_OBJ + k) + s.tgt[:, k, 0] * N + s.tgt[:, k, 1]] = 1
    obs[ar, N_CELLS * (1 + N_OBJ + N_TGT) + s.held + 1] = 1
    return obs


def phase_of(s: State) -> np.ndarray:
    """0 approach (hand empty), 1 carry (holding goal object), 2 holding other object."""
    return np.where(s.held < 0, 0, np.where(s.held == s.go, 1, 2))


class GridPickPlaceEnv(gym.Env):
    """Single-env Gymnasium API wrapper (obs includes goal one-hots).

    clone_state()/restore_state() give exact simulator snapshots for branching.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, horizon: int = 50, goal_p: np.ndarray | None = None):
        self.horizon = horizon
        self.goal_p = goal_probs(0.5) if goal_p is None else goal_p
        self.observation_space = spaces.Box(0, 1, (OBS_DIM + N_OBJ + N_TGT,), np.float32)
        self.action_space = spaces.Discrete(N_ACT)
        self.s: State | None = None

    def _obs(self):
        o = observe(self.s)[0]
        g = np.zeros(N_OBJ + N_TGT, np.float32)
        g[self.s.go[0]] = 1
        g[N_OBJ + self.s.gt[0]] = 1
        return np.concatenate([o, g])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "goal" in options:
            go, gt = options["goal"]
        else:
            gi = self.np_random.choice(len(ALL_GOALS), p=self.goal_p)
            go, gt = ALL_GOALS[gi]
        self.s = sample_layouts(self.np_random, 1, [go], [gt])
        return self._obs(), {}

    def step(self, action):
        r, te, tr = step(self.s, np.array([action]), self.horizon)
        return self._obs(), float(r[0]), bool(te[0]), bool(tr[0]), {"outcome": int(self.s.outcome[0])}

    def clone_state(self) -> State:
        return self.s.clone()

    def restore_state(self, st: State) -> None:
        self.s = st.clone()

    def render(self):
        g = [["." for _ in range(N)] for _ in range(N)]
        for k in range(N_TGT):
            g[self.s.tgt[0, k, 0]][self.s.tgt[0, k, 1]] = "RPS"[k]
        for k in range(N_OBJ):
            g[self.s.obj[0, k, 0]][self.s.obj[0, k, 1]] = "bwx"[k]
        r, c = self.s.agent[0]
        g[r][c] = "@" if self.s.held[0] < 0 else "&"
        return "\n".join("".join(row) for row in g)

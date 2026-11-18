"""Goal-conditioned Double DQN + auxiliary losses for all compared methods.

method:
  B0  plain GC Double DQN
  B1  B0 + HER ('final' relabel to the committed goal, TRAIN goals only)
  B2  B0 + naive action separation: overlap of Boltzmann policies of two train goals at every
      replay state, sum_a p_i(a) p_j(a), minimised (ignores whether actions should differ)
  B3  B0 + ungated evaluator Q-ranking: same evaluator/pairs as B4 but applied at every state.
      For goal g_i: positive a_i (evaluator best); negative a_j if a_j is strictly worse for
      g_i, else a uniformly random strictly-worse action.  Symmetric for g_j.
  B4  B0 + decision-aware gated Q-ranking (proposed): only gate-on entries, negative = other
      goal's preferred action (design 3.6)
  B6  B0 + Q* regression auxiliary loss (exact Q*, all 6 actions) at every replay state for
      two train goals (supervision-matched control)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import torch.nn.functional as F

from .env import (TRAIN_GOALS, ALL_GOALS, OBS_DIM, N_ACT, N_OBJ, N_TGT, PREF_TARGET, State, sample_layouts,
                  step, observe, commit_info, goal_probs, assert_train_goals, phase_of, LEAK_CHECKS, CFG as ENV_CFG)
from .nets import QNet
from .oracle import q_star, oracle_action
from .evaluator import build_entries
from . import metrics as MX

HELD0 = OBS_DIM - (N_OBJ + 1)  # obs[HELD0 + 1 + k] = holding object k
OFF_AO = 2 + 2 * N_OBJ + 2 * N_TGT  # obs offset of (object k - agent) vectors (scaled by 1/(N-1))


def phi_obs(obs: np.ndarray, go: np.ndarray) -> np.ndarray:
    """Unit potential (multiplied by cfg.shaping): +1 when holding the goal object; when the hand is
    empty, -(Manhattan distance agent->goal object)/(2(N-1)).  Depends only on the goal *object*, never on
    the target, so the target choice (where the shortcut lives) stays sparse."""
    ar = np.arange(len(go))
    hold = obs[ar, HELD0 + 1 + go]
    empty = obs[:, HELD0]
    d = np.abs(obs[ar, OFF_AO + 2 * go]) + np.abs(obs[ar, OFF_AO + 2 * go + 1])  # in units of 1/7
    return hold - empty * d * 0.5  # obs offsets are scaled by 1/(N-1)


TRAIN_GO = np.array([g[0] for g in TRAIN_GOALS])
TRAIN_GT = np.array([g[1] for g in TRAIN_GOALS])


@dataclass
class Cfg:
    method: str = "B0"
    skew: float = 0.5
    bc_init: bool = False  # T013 biased BC: demos of the 3 preferred goals only (== demo="pref")
    # T013b common demo warm start: "" none | "cond" demos of the 6 TRAIN goals drawn from the
    # condition's own goal distribution (skew ratio matched) | "pref" preferred goals only (biased BC)
    demo: str = ""
    demo_ratio: float = 0.0  # fraction of each RL batch drawn from the demo buffer (DQfD keeps demos)
    eps_start: float = 1.0
    enc: str = "film"
    mu: str = "oracle"
    lam: float = 1.0
    tau: float = 0.005
    K: int = 50
    margin: float = 0.3  # in Q~ units (see td_loss)
    sep_temp: float = 1.0  # in Q~ units
    drop_only_on_target: bool = False
    p_hold: float = 0.5  # training episodes starting with a random (not goal-specific) object in hand
    n_step: int = 3
    shaping: float = 0.1  # PBRS weight, see phi_obs (no target information)
    seed: int = 0
    total_steps: int = 300_000
    n_envs: int = 16
    horizon: int = 30
    gamma: float = 0.95
    lr: float = 3e-4
    batch: int = 128
    aux_batch: int = 128
    replay: int = 200_000
    learn_start: int = 5_000
    train_every: int = 4  # env steps per gradient update
    target_every: int = 1_000  # updates
    eps_end: float = 0.05
    eps_frac: float = 0.3
    eps_steps: int = 250_000  # if >0: linear decay over this many env steps (overrides eps_frac)
    cache_every: int = 250  # updates between evaluator refreshes
    cache_states: int = 256
    cache_cap: int = 16_384
    eval_every: int = 25_000
    eval_eps_curve: int = 20
    eval_eps_final: int = 100
    bc_demos: int = 2_000
    bc_updates: int = 5_000
    hidden: int = 256
    do_switch: bool = True
    do_rank: bool = True
    tag: str = ""


class Replay:
    def __init__(self, cap):
        self.cap = cap
        self.obs = np.zeros((cap, OBS_DIM), np.float32)
        self.nobs = np.zeros((cap, OBS_DIM), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.go = np.zeros(cap, np.int64)
        self.gt = np.zeros(cap, np.int64)
        self.real = np.zeros(cap, bool)
        self.nxt = np.full(cap, -1, np.int64)  # next transition of the same episode (-1: none)
        # raw simulator state of s (for counterfactual branching)
        self.agent = np.zeros((cap, 2), np.int64)
        self.objp = np.zeros((cap, 3, 2), np.int64)
        self.held = np.zeros(cap, np.int64)
        self.tgt = np.zeros((cap, 3, 2), np.int64)
        self.ptr = 0
        self.n = 0

    def add(self, obs, a, r, nobs, d, go, gt, st: State | None, real=True):
        assert_train_goals(go, gt, "replay (RL data / HER relabel)")
        B = len(a)
        idx = (self.ptr + np.arange(B)) % self.cap
        self.obs[idx], self.nobs[idx] = obs, nobs
        self.a[idx], self.r[idx], self.d[idx] = a, r, d
        self.go[idx], self.gt[idx] = go, gt
        self.real[idx] = real
        self.nxt[idx] = -1
        if st is not None:
            self.agent[idx], self.objp[idx], self.held[idx], self.tgt[idx] = st.agent, st.obj, st.held, st.tgt
        self.ptr = (self.ptr + B) % self.cap
        self.n = min(self.n + B, self.cap)
        return idx

    def sample(self, rng, B):
        return rng.integers(0, self.n, B)

    def states(self, idx) -> State:
        B = len(idx)
        z = np.zeros(B, np.int64)
        return State(self.agent[idx].copy(), self.objp[idx].copy(), self.held[idx].copy(), self.tgt[idx].copy(),
                     z.copy(), z.copy(), z.copy(), np.zeros(B, bool), z.copy())


def sample_goal_pairs(rng, B):
    i = rng.integers(len(TRAIN_GOALS), size=B)
    j = (i + rng.integers(1, len(TRAIN_GOALS), size=B)) % len(TRAIN_GOALS)
    gi = (TRAIN_GO[i], TRAIN_GT[i])
    gj = (TRAIN_GO[j], TRAIN_GT[j])
    assert_train_goals(*gi, "aux goal pair")
    assert_train_goals(*gj, "aux goal pair")
    return gi, gj


def T(x, dt=None):
    t = torch.from_numpy(np.ascontiguousarray(x))
    return t.float() if dt == "f" else t


class Trainer:
    def __init__(self, cfg: Cfg):
        self.c = cfg
        ENV_CFG["drop_only_on_target"] = cfg.drop_only_on_target
        torch.manual_seed(cfg.seed)
        torch.set_num_threads(1)
        self.rng = np.random.default_rng(cfg.seed)
        self.net = QNet(cfg.enc, cfg.hidden)
        self.tgt = QNet(cfg.enc, cfg.hidden)
        self.tgt.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.rb = Replay(cfg.replay)
        self.gp = goal_probs(cfg.skew)
        self.updates = 0
        self.env_steps = 0
        self.branch_steps = 0
        self.cache = None
        self.gate_log = []
        self.curve = []
        self.loss_log = []
        self.demo = None
        self.demo_mode = "pref" if cfg.bc_init else cfg.demo
        assert self.demo_mode in ("", "pref", "cond"), self.demo_mode
        self.bc_eval = True
        self.demo_n_eps = 0
        self.demo_goal_counts = None

    # ---------------- acting ----------------
    def sample_goals(self, B):
        gid = self.rng.choice(len(ALL_GOALS), size=B, p=self.gp)
        go = np.array([ALL_GOALS[i][0] for i in gid])
        gt = np.array([ALL_GOALS[i][1] for i in gid])
        assert_train_goals(go, gt, "training episode goal")
        return go, gt

    def eps(self):
        span = self.c.eps_steps if self.c.eps_steps > 0 else self.c.eps_frac * self.c.total_steps
        f = min(1.0, self.env_steps / span)
        return self.c.eps_start + f * (self.c.eps_end - self.c.eps_start)

    def act(self, s: State, eps):
        a = MX.greedy(self.net, s)
        rnd = self.rng.random(s.B) < eps
        a[rnd] = self.rng.integers(N_ACT, size=rnd.sum())
        return a

    # ---------------- losses ----------------
    def td_loss(self, idx):
        """n-step Double DQN on the Q~ scale.

        The network outputs Q~ = (Q - 1)/(1 - gamma), an affine transform of the sparse-reward Q
        (== HER-style -1/step reward with failed commits absorbing at -1/(1-gamma)). Same optimal
        policy, 1/(1-gamma) larger action gaps.  Per-step reward on that scale incl. PBRS shaping
        Phi = shaping*phi_obs(s,g) (Phi(terminal)=0, computed from obs so HER
        relabels stay consistent):  r~ = (r - 1 + g(1-d) + g Phi' - Phi)/(1-g).
        """
        rb, c = self.rb, self.c
        g = c.gamma
        go_np = rb.go[idx]
        cur = idx.copy()
        alive = np.ones(len(idx), bool)
        G = np.zeros(len(idx))
        disc = np.ones(len(idx))
        last = idx.copy()
        for k in range(c.n_step):
            d = rb.d[cur]
            phi = c.shaping * phi_obs(rb.obs[cur], go_np)
            nphi = c.shaping * phi_obs(rb.nobs[cur], go_np) * (1 - d)
            rt = (rb.r[cur] - 1 + g * (1 - d) + g * nphi - phi) / (1 - g)
            G += np.where(alive, disc * rt, 0.0)
            last = np.where(alive, cur, last)
            disc = np.where(alive, disc * g, disc)
            nx = rb.nxt[cur]
            alive = alive & (d == 0) & (nx >= 0)
            cur = np.where(alive, nx, cur)
        B = len(idx)
        o = T(rb.obs[idx], "f")
        no = T(rb.nobs[last], "f")
        go, gt = T(rb.go[idx]), T(rb.gt[idx])
        qq = self.net(torch.cat([o, no]), torch.cat([go, go]), torch.cat([gt, gt]))  # one pass
        q = qq[:B].gather(1, T(rb.a[idx])[:, None])[:, 0]
        with torch.no_grad():
            na = qq[B:].detach().argmax(1, keepdim=True)
            nq = self.tgt(no, go, gt).gather(1, na)[:, 0]
            y = T(G, "f") + T(disc * (1 - rb.d[last]), "f") * nq
        return F.smooth_l1_loss(q, y)

    def refresh_cache(self):
        c = self.c
        idx = self.rb.sample(self.rng, c.cache_states)
        s = self.rb.states(idx)
        gi, gj = sample_goal_pairs(self.rng, c.cache_states)
        net = self.net if c.mu == "learned" else None
        e, n = build_entries(s, gi, gj, c.K, c.gamma, c.tau, c.mu, net)
        self.branch_steps += n
        e["obs"] = self.rb.obs[idx]
        e["gi_o"], e["gi_t"], e["gj_o"], e["gj_t"] = gi[0], gi[1], gj[0], gj[1]
        ph = phase_of_pair(s, gi, gj)
        log = dict(step=self.env_steps, gate_rate=float(e["gate"].mean()))
        for k, m in ph.items():
            log[f"gate_rate_{k}"] = float(e["gate"][m].mean()) if m.any() else None
            log[f"frac_{k}"] = float(m.mean())
        if c.mu == "learned":
            # reference: oracle evaluator on the same entries (not counted as branch budget)
            eo, _ = build_entries(s, gi, gj, c.K, c.gamma, c.tau, "oracle")
            og = eo["gate"]
            rare_pair = is_rare(gi) | is_rare(gj)
            log["oracle_gate_rate"] = float(og.mean())
            log["learned_gate_recall"] = float(e["gate"][og].mean()) if og.any() else None
            log["learned_gate_precision"] = float(og[e["gate"]].mean()) if e["gate"].any() else None
            m = og & rare_pair & (s.held >= 0)
            log["learned_gate_recall_rare_holding"] = float(e["gate"][m].mean()) if m.any() else None
            # "shortcut states": oracle gate on but the learned continuation never reaches g_j/g_i
            sc = og & ((e["Ji"].max(1) <= 0) | (e["Jj"].max(1) <= 0))
            log["frac_oracle_gated_with_failed_learned_continuation"] = float(sc[og].mean()) if og.any() else None
            log["learned_gate_on_shortcut_states"] = float(e["gate"][sc].mean()) if sc.any() else None
            # label agreement where both gates on
            both = og & e["gate"]
            log["learned_label_agree"] = float(((e["ai"] == eo["ai"]) & (e["aj"] == eo["aj"]))[both].mean()) if both.any() else None
        self.gate_log.append(log)
        keys = ["obs", "ai", "aj", "gate", "Ji", "Jj", "gi_o", "gi_t", "gj_o", "gj_t"]
        new = {k: e[k] for k in keys}
        if self.cache is None:
            self.cache = new
        else:
            self.cache = {k: np.concatenate([self.cache[k], new[k]])[-c.cache_cap:] for k in keys}

    def rank_loss(self):
        c = self.c
        C = self.cache
        if C is None:
            return None
        if c.method == "B4":
            pool = np.where(C["gate"])[0]
        else:
            pool = np.arange(len(C["ai"]))
        if pool.size == 0:
            return None
        b = pool[self.rng.integers(pool.size, size=c.aux_batch)]
        o = T(C["obs"][b], "f")
        ai, aj = C["ai"][b], C["aj"][b]
        Ji, Jj = C["Ji"][b], C["Jj"][b]
        if c.method == "B4":
            pos_i, neg_i, ok_i = ai, aj, np.ones(len(b), bool)
            pos_j, neg_j, ok_j = aj, ai, np.ones(len(b), bool)
        else:
            pos_i, neg_i, ok_i = self._neg(Ji, ai, aj)
            pos_j, neg_j, ok_j = self._neg(Jj, aj, ai)
        qi = self.net(o, T(C["gi_o"][b]), T(C["gi_t"][b]))
        qj = self.net(o, T(C["gj_o"][b]), T(C["gj_t"][b]))
        ar = torch.arange(len(b))
        li = F.relu(c.margin - qi[ar, T(pos_i)] + qi[ar, T(neg_i)]) * T(ok_i, "f")
        lj = F.relu(c.margin - qj[ar, T(pos_j)] + qj[ar, T(neg_j)]) * T(ok_j, "f")
        return (li + lj).mean()

    def _neg(self, J, pos, other):
        best = J[np.arange(len(pos)), pos][:, None]
        worse = J < best - 1e-9
        ok = worse.any(1)
        other_worse = worse[np.arange(len(pos)), other]
        r = np.where(worse, self.rng.random(worse.shape), -1).argmax(1)
        neg = np.where(other_worse, other, r)
        return pos, neg, ok

    def sep_loss(self):
        c = self.c
        idx = self.rb.sample(self.rng, c.aux_batch)
        o = T(self.rb.obs[idx], "f")
        gi, gj = sample_goal_pairs(self.rng, c.aux_batch)
        pi = torch.softmax(self.net(o, T(gi[0]), T(gi[1])) / c.sep_temp, 1)
        pj = torch.softmax(self.net(o, T(gj[0]), T(gj[1])) / c.sep_temp, 1)
        return (pi * pj).sum(1).mean()

    def qstar_loss(self):
        c = self.c
        idx = self.rb.sample(self.rng, c.aux_batch)
        s = self.rb.states(idx)
        o = T(self.rb.obs[idx], "f")
        gi, gj = sample_goal_pairs(self.rng, c.aux_batch)
        loss = 0.0
        for g in (gi, gj):
            assert_train_goals(*g, "B6 Q* regression target")
            # exact Q* of the shaped MDP: Q*_shaped(s,a) = Q*(s,a) - Phi(s)
            qs = q_star(s, g[0], g[1], c.gamma) - c.shaping * phi_obs(observe(s), g[0])[:, None]
            qs = (qs - 1) / (1 - c.gamma)  # Q~ scale
            loss = loss + F.mse_loss(self.net(o, T(g[0]), T(g[1])), T(qs, "f"))
        return loss

    def update(self):
        c = self.c
        idx = self.rb.sample(self.rng, c.batch)
        loss = self.td_loss(idx)
        aux = None
        if c.method in ("B3", "B4"):
            if self.cache is None or self.updates % c.cache_every == 0:
                self.refresh_cache()
            aux = self.rank_loss()
        elif c.method == "B2":
            aux = self.sep_loss()
        elif c.method == "B6":
            aux = self.qstar_loss()
        tot = loss + (c.lam * aux if aux is not None else 0.0)
        if self.demo is not None and c.demo_ratio > 0:
            nd = max(1, int(round(c.batch * c.demo_ratio)))
            dtd, dje = self.demo_loss(self.demo.sample(self.rng, nd))
            tot = tot + c.demo_ratio * (dtd + dje)
        self.opt.zero_grad()
        tot.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        if self.updates % c.target_every == 0:
            self.tgt.load_state_dict(self.net.state_dict())
        if self.updates % 1000 == 0:
            self.loss_log.append(dict(step=self.env_steps, td=float(loss.detach()), aux=None if aux is None else float(aux.detach())))

    # ---------------- demo warm start (DQfD-style: n-step TD + large-margin on scripted BFS-optimal demos) ----
    def demo_goals(self, B):
        """Goals of demo episodes.  'pref' (T013 biased BC): preferred goals only.  'cond' (T013b
        common warm start): the condition's own TRAIN-goal distribution.  Held-out never."""
        if self.demo_mode == "pref":
            go = self.rng.integers(N_OBJ, size=B)
            gt = np.array([PREF_TARGET[o] for o in go])
        else:
            go, gt = self.sample_goals(B)
        assert_train_goals(go, gt, "demo goals")
        return go, gt

    def collect_demos(self):
        c = self.c
        demo = Replay(c.bc_demos * c.horizon)
        n_ep = 0
        B = 64
        while n_ep < c.bc_demos:
            B = min(B, c.bc_demos - n_ep)
            go, gt = self.demo_goals(B)
            s = sample_layouts(self.rng, B, go, gt)
            last = np.full(B, -1, np.int64)
            while not s.done.all():
                live = ~s.done
                a = np.zeros(B, np.int64)
                a[live] = oracle_action(s.index(live), go[live], gt[live], c.gamma)
                ob = observe(s)
                prev = s.clone()
                r, te, tr = step(s, a, c.horizon)
                idx = demo.add(ob[live], a[live], r[live], observe(s)[live], te[live].astype(np.float32), go[live],
                               gt[live], prev.index(live))
                lv = np.where(live)[0]
                hp = last[lv] >= 0
                demo.nxt[last[lv][hp]] = idx[hp]  # n-step links inside each demo episode
                last[lv] = idx
            n_ep += B
        self.demo_n_eps = n_ep
        self.demo_goal_counts = {f"{o}{t}": int(((demo.go[:demo.n] == o) & (demo.gt[:demo.n] == t) & (demo.d[:demo.n] > 0)).sum())
                                 for o, t in TRAIN_GOALS}
        return demo

    def demo_loss(self, idx):
        """n-step TD on the demo buffer + DQfD large-margin J_E."""
        c = self.c
        rb, self.rb = self.rb, self.demo
        td = self.td_loss(idx)
        self.rb = rb
        d = self.demo
        q = self.net(T(d.obs[idx], "f"), T(d.go[idx]), T(d.gt[idx]))
        ae = T(d.a[idx])
        marg = torch.full_like(q, c.margin)
        marg[torch.arange(len(idx)), ae] = 0
        je = ((q + marg).max(1).values - q.gather(1, ae[:, None])[:, 0]).mean()
        return td, je

    def bc_pretrain(self):
        c = self.c
        self.demo = self.collect_demos()
        for u in range(c.bc_updates):
            idx = self.demo.sample(self.rng, c.batch)
            td, je = self.demo_loss(idx)
            self.opt.zero_grad()
            (td + je).backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
            self.opt.step()
            if (u + 1) % c.target_every == 0:
                self.tgt.load_state_dict(self.net.state_dict())
            if (u + 1) % 1000 == 0:
                self.loss_log.append(dict(step=0, bc_update=u + 1, td=float(td.detach()), je=float(je.detach())))
        self.tgt.load_state_dict(self.net.state_dict())

    # ---------------- main loop ----------------
    def run(self):
        c = self.c
        t0 = time.time()
        if self.demo_mode:
            self.bc_pretrain()
            if self.bc_eval:
                self.curve.append(dict(self.eval_curve(), pretrain=True))
        go, gt = self.sample_goals(c.n_envs)
        s = sample_layouts(self.rng, c.n_envs, go, gt, c.p_hold)
        ep_idx = [[] for _ in range(c.n_envs)]
        last_idx = np.full(c.n_envs, -1, np.int64)
        next_eval = 0
        upd_debt = 0.0
        while self.env_steps < c.total_steps:
            if self.env_steps >= next_eval:
                self.curve.append(self.eval_curve())
                next_eval += c.eval_every
            ob = observe(s)
            a = self.act(s, self.eps())
            co, ct = commit_info(s, a)
            prev = s.clone()
            r, te, tr = step(s, a, c.horizon)
            idx = self.rb.add(ob, a, r, observe(s), te.astype(np.float32), s.go, s.gt, prev)
            self.env_steps += c.n_envs
            done = te | tr
            hp = last_idx >= 0
            self.rb.nxt[last_idx[hp]] = idx[hp]
            last_idx = np.where(done, -1, idx)
            for i in range(c.n_envs):
                ep_idx[i].append(idx[i])
                if done[i]:
                    if c.method == "B1" and co[i] >= 0 and not (co[i] == s.go[i] and ct[i] == s.gt[i]):
                        self.her(ep_idx[i], co[i], ct[i])
                    ep_idx[i] = []
            if done.any():
                d = np.where(done)[0]
                ngo, ngt = self.sample_goals(d.size)
                s.set_rows(d, sample_layouts(self.rng, d.size, ngo, ngt, c.p_hold))
            if self.rb.n >= c.learn_start:
                upd_debt += c.n_envs / c.train_every
                while upd_debt >= 1:
                    self.update()
                    upd_debt -= 1
        self.curve.append(self.eval_curve())
        wall_train = time.time() - t0
        return self.final_eval(wall_train)

    def her(self, ids, o, t):
        """'final' relabel: the whole episode relabelled with the committed goal, if it is a TRAIN goal."""
        if (o, t) not in TRAIN_GOALS:
            return  # never relabel to held-out goals
        ids = np.array(ids)
        rb = self.rb
        n = len(ids)
        r = np.zeros(n, np.float32)
        r[-1] = 1.0
        go = np.full(n, o)
        gt = np.full(n, t)
        st = rb.states(ids)
        new = rb.add(rb.obs[ids].copy(), rb.a[ids].copy(), r, rb.nobs[ids].copy(), rb.d[ids].copy(), go, gt, st,
                     real=False)
        rb.nxt[new[:-1]] = new[1:]

    def eval_curve(self):
        _, S = MX.eval_policy(self.net, self.c.eval_eps_curve, self.c.horizon)
        return dict(step=self.env_steps, SR_seen=S["SR_seen"], SR_unseen=S["SR_unseen"],
                    SR_seen_rare=S["seen_rare_sr"], SER_unseen=S["SER_unseen"], SER_unseen_freq=S["SER_unseen_freq"])

    def final_eval(self, wall_train):
        c = self.c
        t0 = time.time()
        pg, S = MX.eval_policy(self.net, c.eval_eps_final, c.horizon)
        out = dict(cfg=asdict(c), summary=S, per_goal=pg, curve=self.curve,
                   env_steps=self.env_steps, branch_steps=self.branch_steps,
                   demo_eps=self.demo_n_eps, demo_goal_counts=self.demo_goal_counts, updates=self.updates,
                   wall_train_s=wall_train, leak_checks=LEAK_CHECKS["n"], loss_log=self.loss_log[-20:])
        if c.do_switch:
            res, agg = MX.goal_switch(self.net, horizon=c.horizon)
            out["switch"] = agg
            out["switch_detail"] = res
        if c.do_rank:
            ref = MX.reference_states(gamma=c.gamma)
            out["rank"] = MX.rank_metrics(self.net, ref, c.gamma)
        if self.gate_log:
            out["gate_log"] = self.gate_log[:: max(1, len(self.gate_log) // 40)]
            keys = [k for k in self.gate_log[0] if k != "step"]
            late = self.gate_log[len(self.gate_log) // 2:]
            out["gate_summary"] = {k: _nanmean([g.get(k) for g in late]) for k in keys}
        out["wall_eval_s"] = time.time() - t0
        return out


def _nanmean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def is_rare(g):
    o, t = g
    return np.array([PREF_TARGET[a] != b for a, b in zip(o, t)])


def phase_of_pair(s: State, gi, gj):
    same_obj = gi[0] == gj[0]
    hold_i = s.held == gi[0]
    hold_j = s.held == gj[0]
    return {
        "handempty": s.held < 0,
        "holding_shared_obj": (s.held >= 0) & same_obj & hold_i,
        "holding_goal_obj_diffobj": (s.held >= 0) & ~same_obj & (hold_i | hold_j),
        "holding_other": (s.held >= 0) & ~hold_i & ~hold_j,
    }


def run_cfg(cfg_dict: dict) -> dict:
    cfg = Cfg(**cfg_dict)
    t0 = time.time()
    out = Trainer(cfg).run()
    out["wall_total_s"] = time.time() - t0
    return out


if __name__ == "__main__":
    import sys
    cfg = Cfg(**json.loads(sys.argv[1])) if len(sys.argv) > 1 else Cfg()
    res = run_cfg(asdict(cfg))
    print(json.dumps(res["summary"], indent=1))
    print(json.dumps({k: res[k] for k in ["env_steps", "branch_steps", "updates", "wall_train_s", "wall_total_s"]}))

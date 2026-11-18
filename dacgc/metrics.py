"""Evaluation: SR / SER decomposition, goal-switch test, RankAcc & shared-action preservation."""
from __future__ import annotations

import numpy as np
import torch

from .env import (ALL_GOALS, TRAIN_GOALS, HELDOUT_GOALS, PREF_TARGET, RARE_TARGET, N_OBJ, N_TGT, N_ACT,
                  OUT_SUCCESS, OUT_WRONG_TARGET, OUT_WRONG_OBJECT, OUT_TIMEOUT, State, sample_layouts,
                  step, observe, commit_info, phase_of)
from .oracle import q_star, oracle_action
from .evaluator import pair_choice

EVAL_SEED = 10_000  # evaluation layouts are disjoint from training rng streams


def greedy(net, s: State):
    with torch.no_grad():
        q = net(torch.from_numpy(observe(s)), torch.from_numpy(s.go), torch.from_numpy(s.gt))
    return q.argmax(1).numpy()


def rollout(net, s: State, horizon: int):
    """Greedy rollout in place; returns committed (obj, tgt) per row (-1 if none)."""
    B = s.B
    co = np.full(B, -1)
    ct = np.full(B, -1)
    while not s.done.all():
        a = greedy(net, s)
        o, t = commit_info(s, a)
        m = o >= 0
        co[m], ct[m] = o[m], t[m]
        step(s, a, horizon)
    return co, ct


def eval_layouts(n: int, seed: int = EVAL_SEED):
    rng = np.random.default_rng(seed)
    return sample_layouts(rng, n, np.zeros(n, np.int64), np.zeros(n, np.int64))


def eval_policy(net, n_per_goal=100, horizon=50, seed=EVAL_SEED):
    """Same n layouts for every one of the 9 goals (seen/unseen differ only in the goal)."""
    base = eval_layouts(n_per_goal, seed)
    per_goal = {}
    for (o, t) in ALL_GOALS:
        s = base.clone()
        s.go[:] = o
        s.gt[:] = t
        co, ct = rollout(net, s, horizon)
        out = s.outcome
        wt = out == OUT_WRONG_TARGET
        per_goal[f"{o}{t}"] = dict(
            sr=float((out == OUT_SUCCESS).mean()),
            wt=float(wt.mean()),
            wt_freq=float((wt & (ct == PREF_TARGET[o])).mean()),  # object's frequent target
            wt_other=float((wt & (ct != PREF_TARGET[o])).mean()),
            wrong_obj=float((out == OUT_WRONG_OBJECT).mean()),
            timeout=float((out == OUT_TIMEOUT).mean()),
        )
    return per_goal, summarize(per_goal)


def summarize(pg):
    def avg(goals, k):
        return float(np.mean([pg[f"{o}{t}"][k] for o, t in goals]))

    pref = [(o, PREF_TARGET[o]) for o in range(N_OBJ)]
    rare = [(o, RARE_TARGET[o]) for o in range(N_OBJ)]
    S = {}
    for name, gl in [("seen", TRAIN_GOALS), ("seen_freq", pref), ("seen_rare", rare), ("unseen", HELDOUT_GOALS)]:
        for k in ["sr", "wt", "wt_freq", "wt_other", "wrong_obj", "timeout"]:
            S[f"{name}_{k}"] = avg(gl, k)
    S["SR_seen"], S["SR_unseen"] = S["seen_sr"], S["unseen_sr"]
    S["SER_unseen"] = S["unseen_wt"]  # goal object committed on a wrong target
    S["SER_unseen_freq"] = S["unseen_wt_freq"]
    return S


def goal_switch(net, n=100, horizon=50, seed=EVAL_SEED + 1):
    """Same snapshot, only the goal changes. Snapshots: t=0 (approach) and the step right after
    the goal object is picked (carry), both taken from a rollout with the object's frequent
    (train) goal.  Returns P(commit target | requested target, phase) per object, and follow rate
    for seen vs unseen requested goals."""
    res = {}
    base = eval_layouts(n, seed)
    for o in range(N_OBJ):
        src = base.clone()
        src.go[:] = o
        src.gt[:] = PREF_TARGET[o]
        snaps = {"approach": src.clone()}
        picked = np.zeros(n, bool)
        carry = src.clone()
        while not src.done.all():
            a = greedy(net, src)
            step(src, a, horizon)
            new = (~picked) & (src.held == o) & ~src.done
            if new.any():
                carry.set_rows(np.where(new)[0], src.index(np.where(new)[0]))
                picked |= new
        snaps["carry"] = carry.index(np.where(picked)[0]) if picked.any() else None
        for ph, sn in snaps.items():
            if sn is None or sn.B == 0:
                continue
            for t in range(N_TGT):
                s = sn.clone()
                s.gt[:] = t
                s.go[:] = o
                s.t[:] = 0
                s.done[:] = False
                s.outcome[:] = 0
                co, ct = rollout(net, s, horizon)
                dist = [float(((co == o) & (ct == tt)).mean()) for tt in range(N_TGT)]
                res[f"o{o}_{ph}_req{t}"] = dict(n=int(sn.B), commit_dist=dist,
                                                 follow=dist[t], heldout=bool((o, t) in HELDOUT_GOALS),
                                                 to_freq=dist[PREF_TARGET[o]])
    agg = {}
    for ph in ["approach", "carry"]:
        for kind, flag in [("seen", False), ("unseen", True)]:
            v = [r["follow"] for k, r in res.items() if f"_{ph}_" in k and r["heldout"] == flag]
            f = [r["to_freq"] for k, r in res.items() if f"_{ph}_" in k and r["heldout"] == flag]
            agg[f"switch_{ph}_{kind}_follow"] = float(np.mean(v)) if v else None
            agg[f"switch_{ph}_{kind}_to_freq"] = float(np.mean(f)) if f else None
    return res, agg


def reference_states(n=1500, gamma=0.95, seed=EVAL_SEED + 2, eps=0.3, horizon=50):
    """States from eps-oracle rollouts on independent layouts (not used for training)."""
    rng = np.random.default_rng(seed)
    B = 200
    out = []
    while sum(x.B for x in out) < n:
        gid = rng.integers(len(TRAIN_GOALS), size=B)
        go = np.array([TRAIN_GOALS[i][0] for i in gid])
        gt = np.array([TRAIN_GOALS[i][1] for i in gid])
        s = sample_layouts(rng, B, go, gt)
        for _ in range(horizon):
            live = ~s.done
            if not live.any():
                break
            take = live & (rng.random(B) < 0.15)
            if take.any():
                out.append(s.index(take))
            a = oracle_action(s, s.go, s.gt, gamma)
            rnd = rng.random(B) < eps
            a[rnd] = rng.integers(N_ACT, size=rnd.sum())
            # avoid committing on wrong targets while collecting
            co, ct = commit_info(s, a)
            bad = (co >= 0) & ~((co == s.go) & (ct == s.gt))
            a[bad] = 0
            step(s, a, horizon)
    ref = State.concat(out).index(np.arange(n))
    ref.done[:] = False
    return ref


def rank_metrics(net, ref: State, gamma=0.95):
    """RankAcc on branch states / common states (reference = exact Q*), shared-action
    preservation (SAP), split into seen pairs (both goals train) and pairs involving a held-out goal."""
    goals = ALL_GOALS
    obs = torch.from_numpy(observe(ref))
    Qs, Qn, Aopt = {}, {}, {}
    for (o, t) in goals:
        go = np.full(ref.B, o)
        gt = np.full(ref.B, t)
        Qs[(o, t)] = q_star(ref, go, gt, gamma)
        with torch.no_grad():
            Qn[(o, t)] = net(obs, torch.from_numpy(go), torch.from_numpy(gt)).numpy()
    ph = ref.held >= 0
    rows = []
    for i, gi in enumerate(goals):
        for gj in goals[i + 1:]:
            ai, aj, di, dj, shared = pair_choice(Qs[gi], Qs[gj])
            ar = np.arange(ref.B)
            branch = (ai != aj) & (di > 1e-6) & (dj > 1e-6)
            c_i = Qn[gi][ar, ai] > Qn[gi][ar, aj]
            c_j = Qn[gj][ar, aj] > Qn[gj][ar, ai]
            kind = "seen" if (gi in TRAIN_GOALS and gj in TRAIN_GOALS) else "heldout"
            # common: a shared optimal action exists. RankAcc_common = fraction of (shared best,
            # strictly worse) pairs ordered correctly, both goals.
            ca = []
            for g in (gi, gj):
                q, qs = Qn[g], Qs[g]
                best = qs[ar, ai][:, None]
                worse = qs < best - 1e-6
                corr = q[ar, ai][:, None] > q
                ca.append(np.where(worse, corr, np.nan))
            ca = np.nanmean(np.concatenate(ca, 1), 1)
            pi = Qn[gi].argmax(1)
            pj = Qn[gj].argmax(1)
            opt_i = Qs[gi][ar, pi] >= Qs[gi].max(1) - 1e-9
            opt_j = Qs[gj][ar, pj] >= Qs[gj].max(1) - 1e-9
            rows.append(dict(kind=kind, branch=branch, shared=shared, ci=c_i, cj=c_j, ca=ca,
                             sap=opt_i & opt_j, div=pi != pj, held=ph))
    M = {}
    for kind in ["seen", "heldout"]:
        R = [r for r in rows if r["kind"] == kind]
        br = np.concatenate([np.concatenate([r["ci"][r["branch"]], r["cj"][r["branch"]]]) for r in R])
        cm = np.concatenate([r["ca"][r["shared"]] for r in R])
        sap = np.concatenate([r["sap"][r["shared"]] for r in R])
        div = np.concatenate([r["div"][r["shared"]] for r in R])
        brc = np.concatenate([np.concatenate([r["ci"][r["branch"] & r["held"]], r["cj"][r["branch"] & r["held"]]]) for r in R])
        bra = np.concatenate([np.concatenate([r["ci"][r["branch"] & ~r["held"]], r["cj"][r["branch"] & ~r["held"]]]) for r in R])
        M[f"rankacc_branch_{kind}"] = float(br.mean()) if br.size else None
        M[f"rankacc_branch_holding_{kind}"] = float(brc.mean()) if brc.size else None
        M[f"rankacc_branch_handempty_{kind}"] = float(bra.mean()) if bra.size else None
        M[f"rankacc_common_{kind}"] = float(np.nanmean(cm)) if cm.size else None
        M[f"sap_{kind}"] = float(sap.mean()) if sap.size else None
        M[f"shared_divergence_{kind}"] = float(div.mean()) if div.size else None
        M[f"n_branch_{kind}"] = int(br.size // 2)
        M[f"n_common_{kind}"] = int(cm.size)
    return M

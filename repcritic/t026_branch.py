"""T026 — branch-advantage mini: VLA-noise candidates vs constrained perturbations (snapshot branches).

Pre-registration: repcritic/reports/t026_prereg.md (fixed before any branch was run).

States: 50 decision states of the T017 buffer, TRAIN split only, one state per episode, episodes NOT used
by the T017 branch/oracle sets (so no overlap with T019 critic training states). Phase-stratified
(privileged labels, label-only):
  A0_init 8 | A1_far 12 | A2_near 12 | C_contact 8 | L_lost 6 | T_transport 4   (approach = 32/50)
Candidates per state (NC = 12):
  0      VLA sample = the logged executed chunk (stored noise, executed exactly as logged) = "base"
  1..7   fresh frozen-policy samples (fresh initial noise)
  8..11  constrained perturbations of cand 0's executed K=5 steps (env space):
         +u, -u, +v, -v, per-step translation offset DELTA=0.005 m on all K steps (2.5 cm total target shift;
         the target-based controller accumulates it). u = horizontal unit vector tcp->target in the robot ROOT
         frame (target = object if not grasping now, else plate), v = z_up x u. Rotation and gripper unchanged.
         Bounds: xyz clipped to [q01, q99] of the policy's action norm stats (= model-normalised [-1, 1]) and
         to the env action space.
Branch = restore snapshot -> execute candidate's K=5 steps -> frozen pi0.5 until t=80.
Replicates: NREP = 4 continuations per candidate. Continuation noise is CRN: initial noise of continuation
decision k for replicate r of state s is seeded by (s, r, k) and is identical across the 12 candidates.

usage: .venv/bin/python -m repcritic.t026_branch --select
       CUDA_VISIBLE_DEVICES=1 RC_POLICY_MB=16 .venv/bin/python -m repcritic.t026_branch --restore_test
       CUDA_VISIBLE_DEVICES=1 RC_POLICY_MB=16 .venv/bin/python -m repcritic.t026_branch --E 96
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

SRC = Path("/workspace/stochrl_t017")
OUT = Path(os.environ.get("RC_T026_OUT", "/workspace/stochrl_t026"))
NC_VLA = 8
PERTS = ("+u", "-u", "+v", "-v")
NC = NC_VLA + len(PERTS)
NREP = 4
DELTA = 0.005
SEL_SEED = 2026
STRATA = dict(A0_init=8, A1_far=12, A2_near=12, C_contact=8, L_lost=6, T_transport=4)
NEAR_M = 0.08


def seed_of(*xs):
    return int(hashlib.md5("_".join(map(str, xs)).encode()).hexdigest()[:8], 16)


def stratum(ph, gd, gn, d):
    p = ph[d]
    if p == "placed" or d > 12:  # d<=12: at least 20 steps (4 decisions) left after the branch point
        return None
    if p == "approach":
        if d == 0:
            return "A0_init"
        return "A2_near" if gd[d] < NEAR_M else "A1_far"
    if p == "grasp_contact" or (p == "transport" and d > 0 and not gn[d - 1]):
        return "C_contact"
    if p == "post_grasp_lost":
        return "L_lost"
    if p == "transport":
        return "T_transport"
    return None


def select_states():
    train = [int(l.strip().replace("ep", "")) for l in open(SRC / "split_train.txt") if l.strip()]
    used = set()
    for f in ("state_select", "state_select_ext"):
        used |= {s["eid"] for s in json.loads((SRC / "branch" / f"{f}.json").read_text())["states"]}
    pool = {k: [] for k in STRATA}
    succ = {}
    for r in (json.loads(l) for f in sorted(SRC.glob("decisions_*.jsonl")) for l in open(f)):
        if r["decision"] == 0:
            succ[r["eid"]] = bool(r["success_once"])
    for e in sorted(set(train) - used):
        z = np.load(SRC / "privileged" / f"ep{e:06d}.npz")
        ph, gd, gn = z["dec_phase"], z["dec_gripper_carrot_dist"], z["dec_grasping_now"]
        for d in range(16):
            k = stratum(ph, gd, gn, d)
            if k:
                pool[k].append((e, d, float(gd[d]), str(ph[d])))
    rng = np.random.RandomState(SEL_SEED)
    taken, states = set(), []
    # rarest stratum first so that one-state-per-episode does not starve it
    for k in sorted(STRATA, key=lambda k: len(pool[k])):
        cand = [x for x in pool[k] if x[0] not in taken]
        idx = rng.permutation(len(cand))
        got = 0
        for i in idx:
            e, d, g, p = cand[i]
            if e in taken:
                continue
            taken.add(e)
            states.append(dict(eid=e, decision=d, t=d * 5, stratum=k, phase=p, gripper_obj_dist=g,
                               episode_success_once=succ[e], split="train", state_id=f"ep{e:06d}_d{d:02d}"))
            got += 1
            if got == STRATA[k]:
                break
    states.sort(key=lambda s: (s["t"], s["state_id"]))
    meta = dict(sel_seed=SEL_SEED, strata=STRATA, near_m=NEAR_M, pool_sizes={k: len(v) for k, v in pool.items()},
                n_excluded_eids=len(used), NC=NC, NC_VLA=NC_VLA, PERTS=PERTS, NREP=NREP, DELTA=DELTA,
                n_branch_total=len(states) * NC * NREP, states=states)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "state_select.json").write_text(json.dumps(meta, indent=0))
    print({k: v for k, v in meta.items() if k != "states"})
    return meta


def pack(states, E, restore_test=False):
    """One batch = E//(NC*NREP) states with similar t (sorted). Jobs: (state_idx, cand, rep)."""
    per = NC * NREP
    spb = max(1, E // per)
    order = sorted(range(len(states)), key=lambda i: (states[i]["t"], i))
    batches = []
    for b in range(0, len(order), spb):
        jobs = []
        for si in order[b:b + spb]:
            for c in range(NC):
                for r in range(NREP):
                    jobs.append((si, 0, r) if restore_test else (si, c, r))
        batches.append(jobs)
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--restore_test", action="store_true",
                    help="every env runs cand 0 with the replicate's CRN noise (12 identical copies per rep)")
    ap.add_argument("--E", type=int, default=96)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--first_batch", type=int, default=0)
    args = ap.parse_args()
    if args.select:
        select_states()
        return

    from repcritic import rc_common as C

    C.setup_env_vars()
    import torch

    sel = json.loads((OUT / "state_select.json").read_text())
    states = sel["states"]
    batches = pack(states, args.E, args.restore_test)
    if args.restore_test:
        batches = [batches[0], batches[len(batches) // 2]]
    batches = batches[args.first_batch:]
    if args.max_batches:
        batches = batches[: args.max_batches]
    tag = "restore_test" if args.restore_test else "main"
    print(f"{tag}: {len(batches)} batches; envs {sum(len(b) for b in batches)}", flush=True)
    cdir = OUT / "candidates"
    cdir.mkdir(parents=True, exist_ok=True)
    res_path = OUT / f"branch_results_{tag}.jsonl"
    done = set()
    if res_path.exists() and not args.restore_test:
        done = {(r["state_id"], r["cand"], r["rep"]) for r in map(json.loads, open(res_path))}

    torch.cuda.reset_peak_memory_stats()
    pol = C.Policy(C.build_policy())
    E = args.E
    env = C.build_env(E)
    base = env.env.unwrapped
    dev = env.device
    lo = torch.as_tensor(env.env.action_space.low[0]).float()
    hi = torch.as_tensor(env.env.action_space.high[0]).float()
    ns = json.loads(Path(C.ckpt_path(), "physical-intelligence/maniskill/norm_stats.json").read_text())["norm_stats"]["actions"]
    q01 = torch.tensor(ns["q01"][:7]).float()
    q99 = torch.tensor(ns["q99"][:7]).float()
    from repcritic import e02_common as E2

    def restore_multi(snaps):
        cat = lambda *ts: torch.cat(ts, 0).to(dev)  # noqa: E731

        def tree_cat(objs):
            o0 = objs[0]
            if isinstance(o0, dict):
                return {k: tree_cat([o[k] for o in objs]) for k in o0}
            if isinstance(o0, torch.Tensor):
                return cat(*objs)
            return o0
        sim = tree_cat([s["sim"] for s in snaps])
        ctrl = sim.pop("controller", None)
        base.set_state_dict(sim)
        if ctrl is not None:
            base.agent.set_controller_state(ctrl)
        for a in C.TASK_STATE_ATTRS:
            getattr(base, a)[:] = cat(*[s["task"][a] for s in snaps])
        es = tree_cat([s["episode_stats"] for s in snaps])
        for k, v in es.items():
            base.episode_stats[k] = v
        base._elapsed_steps[:] = cat(*[s["elapsed"] for s in snaps])
        env.prev_step_reward[:] = cat(*[s["prev_rew"] for s in snaps])
        return env._wrap_obs(base.get_obs())

    def directions(e, priv):
        """u (toward target, horizontal) and v = z x u in the robot root frame, for env e."""
        tgt = priv["pos_plate"][e] if priv["grasping_now"][e] else priv["pos_carrot"][e]
        w = torch.tensor(tgt - priv["pos_gripper"][e]).float()
        R = base.agent.robot.pose.to_transformation_matrix()[e, :3, :3].float().cpu()
        r = R.T @ w
        h = torch.tensor([r[0], r[1], 0.0])
        fallback = bool(h.norm() < 0.005)
        u = torch.tensor([1.0, 0.0, 0.0]) if fallback else h / h.norm()
        v = torch.cross(torch.tensor([0.0, 0.0, 1.0]), u, dim=0)
        return u, v, fallback, float(w.norm())

    def get_candidates(si, obs, e, priv):
        s = states[si]
        path = cdir / f"{s['state_id']}.npz"
        if path.exists():
            return torch.from_numpy(np.load(path)["exec_env"])
        ep = np.load(SRC / "episodes" / f"ep{s['eid']:06d}.npz")
        d = s["decision"]
        g = torch.Generator().manual_seed(seed_of("cand26", s["state_id"]))
        nz = pol.noise(NC_VLA, g)
        nz[0] = torch.from_numpy(ep["noise"][d]).to(nz.device)
        acts, macts = pol.act(C.obs_repeat(obs, e, NC_VLA), nz, full_h=True)
        acts = acts.cpu()
        exe = torch.zeros(NC, C.K, 7)
        exe[:NC_VLA] = torch.max(torch.min(acts[:, : C.K], hi), lo)
        exe[0] = torch.from_numpy(ep["exec_env"][d])
        u, v, fb, dist = directions(e, priv)
        offs = {"+u": u, "-u": -u, "+v": v, "-v": -v}
        for j, name in enumerate(PERTS):
            x = exe[0].clone()
            x[:, :3] = x[:, :3] + DELTA * offs[name]
            x[:, :3] = torch.max(torch.min(x[:, :3], q99[:3]), q01[:3])
            x = torch.max(torch.min(x, hi), lo)
            exe[NC_VLA + j] = x
        realized = (exe[NC_VLA:, :, :3] - exe[0, :, :3]).sum(1)  # [4,3] total xyz shift
        exec_model = 2 * (exe - q01) / (q99 - q01) - 1  # quantile normalisation (model space, 7 dims)
        recompute = float((acts[0] - torch.from_numpy(ep["chunk_env"][d])).abs().max())
        img_diff = float(np.abs(obs["main_images"][e].cpu().numpy().astype(np.float32) - ep["rgb"][d].astype(np.float32)).mean())
        np.savez(path, exec_env=exe.numpy(), exec_model=exec_model.numpy(), chunk_env_vla=acts.numpy(),
                 chunk_model_vla=macts.float().cpu().numpy(), noise_vla=nz.cpu().numpy(), u_root=u.numpy(),
                 v_root=v.numpy(), dir_fallback=fb, tcp_target_dist=dist, realized_shift=realized.numpy(),
                 target_is_plate=bool(priv["grasping_now"][e]), cand0_recompute_maxdiff=recompute,
                 restored_img_meanabsdiff=img_diff)
        return exe

    T0 = time.time()
    with open(res_path, "a") as fout:
        for bi, jobs in enumerate(batches):
            jobs = [j for j in jobs if (states[j[0]]["state_id"], j[1], j[2]) not in done]
            if not jobs:
                continue
            tb = time.time()
            n = len(jobs)
            pad = jobs + [jobs[-1]] * (E - n)
            eids = torch.tensor([states[j[0]]["eid"] for j in pad])
            env.reset(seed=0, options={"episode_id": eids.to(dev)})
            snaps_by_ep = {}
            snaps = []
            for (si, c, r) in pad:
                eid = states[si]["eid"]
                if eid not in snaps_by_ep:
                    snaps_by_ep[eid] = torch.load(SRC / "snapshots" / f"ep{eid:06d}.pt", weights_only=False)
                snaps.append(snaps_by_ep[eid][states[si]["decision"]])
            obs = restore_multi(snaps)
            priv = E2.privileged(env)
            t_env = torch.tensor([states[j[0]]["t"] for j in pad])
            assert bool((base._elapsed_steps.cpu() == t_env).all())
            first = torch.zeros(E, C.K, 7)
            seen = {}
            for e, (si, c, r) in enumerate(pad):
                if si not in seen:
                    seen[si] = get_candidates(si, obs, e, priv)
                first[e] = seen[si][c]
            t = int(t_env.min())
            s_once = torch.zeros(E, dtype=torch.bool)
            s_last = torch.zeros(E, dtype=torch.bool)
            first_t = torch.full((E,), -1)
            ret = torch.zeros(E)
            chunk = first
            afterK = None
            step, kdec = 0, 0
            n_steps = C.MAX_STEPS - t
            while step < n_steps:
                for k in range(chunk.shape[1]):
                    obs, rwd, te, tr, info = env.step(chunk[:, k], auto_reset=False)
                    live = (t_env + step) < C.MAX_STEPS
                    sc = info["success"].cpu() & live
                    first_t = torch.where(sc & (first_t < 0), t_env + step, first_t)
                    s_once |= sc
                    s_last = torch.where(live, info["success"].cpu(), s_last)
                    ret += rwd.float().cpu() * live
                    step += 1
                    if step >= n_steps:
                        break
                if afterK is None:
                    afterK = E2.privileged(env)
                    afterK["success_within_K"] = s_once.numpy().copy()
                if step >= n_steps:
                    break
                kdec += 1
                # CRN continuation noise: depends on (state, rep, continuation decision), not on the candidate
                nz = torch.stack([torch.randn(*pol.noise_shape, generator=torch.Generator().manual_seed(
                    seed_of("cont26", states[si]["state_id"], r, kdec))) for (si, c, r) in pad]).to(pol.device)
                chunk, _ = pol.act(obs, nz)
                chunk = chunk.cpu()
            for e, (si, c, r) in enumerate(jobs):
                s = states[si]
                fout.write(json.dumps(dict(
                    state_id=s["state_id"], eid=s["eid"], decision=s["decision"], t=s["t"], stratum=s["stratum"],
                    phase=s["phase"], cand=c, kind=("vla" if c < NC_VLA else "pert"),
                    pert=(PERTS[c - NC_VLA] if c >= NC_VLA else None), rep=r,
                    success_once=bool(s_once[e]), success_at_end=bool(s_last[e]), first_success_t=int(first_t[e]),
                    ret=float(ret[e]),
                    afterK={k: (v[e].tolist() if hasattr(v[e], "tolist") else v[e]) for k, v in afterK.items()
                            if k in ("gripper_carrot_dist", "carrot_plate_dist", "grasping_now", "ever_grasped",
                                     "src_on_target", "contact_any_finger", "pos_gripper", "success_within_K")},
                )) + "\n")
            fout.flush()
            print(f"batch {bi+1}/{len(batches)} t={t}..{int(t_env.max())} envs {n} states {len(seen)} "
                  f"succ {s_once[:n].float().mean():.2f} | {time.time()-tb:.0f}s peak {C.gpu_mem_gb():.1f}GB "
                  f"elapsed {(time.time()-T0)/60:.1f}min", flush=True)
    print("done", (time.time() - T0) / 3600, "h", flush=True)


if __name__ == "__main__":
    main()

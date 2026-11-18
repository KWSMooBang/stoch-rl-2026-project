"""E02 candidate-level labels + multi-continuation Oracle via snapshot branches.

States = (episode, decision) pairs of the E02 rollout buffer (state_select.json, made by --select).
Per state s: NC=16 candidates on the same (obs, instruction):
  cand 0 = the logged executed chunk (its stored noise; executed exactly as stored),
  cand 1..15 = fresh frozen-policy samples (fresh initial noise).
Branch = restore snapshot -> execute candidate's first K=5 steps -> frozen policy continuation
         (fresh noise per env) until t=80 -> success_once / success_at_end.
  label states  : every candidate x 1 continuation               (16 branches)
  oracle states : cand 0..7 x 4 continuations + cand 8..15 x 1   (40 branches)  -> p_i for 8 candidates
Branches of many states (same decision index d => same t) are packed into one env batch (E sub-envs),
because GPU PhysX cost per control step (100 substeps) is ~independent of the env count.

usage: python -m repcritic.e02_branch --select
       CUDA_VISIBLE_DEVICES=0 RC_POLICY_MB=8 python -m repcritic.e02_branch --shard 0 --nshard 2 --E 64
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("RC_E02_OUT", "/workspace/stochrl_t017"))
NC = 16
NC_OR = 8
NCONT_OR = 4
SEL_SEED = 31
DEC_GRID = (0, 3, 6, 9, 12)


def select_states(n_ep=dict(train=20, val=15, test=15), n_oracle=50, name="state_select", exclude=(), seed=SEL_SEED):
    rows = [json.loads(l) for f in sorted(OUT.glob("decisions_*.jsonl")) if os.environ.get("RC_E02_OUT") or "smoke" not in f.name
            for l in open(f)]
    eps = {}
    for r in rows:
        eps.setdefault(r["eid"], []).append(r)
    rng = np.random.RandomState(seed)
    states = []
    chosen = []
    for split, k in n_ep.items():
        cand = sorted(e for e, rs in eps.items() if rs[0]["split"] == split and e not in exclude)
        pick = rng.choice(cand, size=k, replace=False)
        chosen += [(int(e), split) for e in pick]
    for eid, split in chosen:
        rs = sorted(eps[eid], key=lambda r: r["decision"])
        pre = [r["decision"] for r in rs if not r["success_before"]]
        grid = [d for d in DEC_GRID if d in pre]
        if len(grid) >= 4:
            ds = sorted(rng.choice(grid, size=4, replace=False).tolist())
        else:
            rest = [d for d in pre if d not in grid and d <= 12]
            extra = sorted(rng.choice(rest, size=min(4 - len(grid), len(rest)), replace=False).tolist()) if rest else []
            ds = sorted(grid + extra)
        for d in ds:
            states.append(dict(eid=eid, split=split, decision=int(d), t=int(d) * 5,
                               episode_success_once=rs[0]["success_once"]))
    # oracle states: one random state per chosen episode (up to n_oracle)
    by_ep = {}
    for i, s in enumerate(states):
        by_ep.setdefault(s["eid"], []).append(i)
    for eid in list(by_ep)[:n_oracle]:
        j = by_ep[eid][rng.randint(len(by_ep[eid]))]
        states[j]["oracle"] = True
    for i, s in enumerate(states):
        s["oracle"] = bool(s.get("oracle", False))
        s["state_id"] = f"ep{s['eid']:06d}_d{s['decision']:02d}"
        s["n_branch"] = NC_OR * NCONT_OR + (NC - NC_OR) if s["oracle"] else NC
    meta = dict(sel_seed=seed, excluded_eids=sorted(exclude), dec_grid=DEC_GRID, n_ep=n_ep, n_states=len(states),
                n_oracle=sum(s["oracle"] for s in states), NC=NC, NC_OR=NC_OR, NCONT_OR=NCONT_OR,
                n_branch_total=sum(s["n_branch"] for s in states), states=states)
    (OUT / "branch").mkdir(exist_ok=True)
    (OUT / "branch" / f"{name}.json").write_text(json.dumps(meta, indent=0))
    print({k: v for k, v in meta.items() if k != "states"})
    return meta


def branch_jobs(states):
    jobs = []  # (state_idx, cand, cont)
    for si, s in enumerate(states):
        for c in range(NC):
            ncont = NCONT_OR if (s["oracle"] and c < NC_OR) else 1
            for j in range(ncont):
                jobs.append((si, c, j))
    return jobs


def pack(states, jobs, E):
    """Sort by (decision, state) and chunk into batches of <=E envs (adjacent decisions share a batch;
    each env keeps its own clock and stops recording at its own t=80)."""
    js = sorted(jobs, key=lambda jb: (states[jb[0]]["decision"], jb[0], jb[1], jb[2]))
    return [js[i:i + E] for i in range(0, len(js), E)]


def seed_of(*xs):
    return int(hashlib.md5("_".join(map(str, xs)).encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--E", type=int, default=64)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--n_ep", default="train=20,val=15,test=15")
    ap.add_argument("--n_oracle", type=int, default=50)
    ap.add_argument("--sel", default="state_select")
    ap.add_argument("--sel_seed", type=int, default=SEL_SEED)
    ap.add_argument("--exclude_from", nargs="*", default=[])
    args = ap.parse_args()
    if args.select:
        excl = set()
        for f in args.exclude_from:
            excl |= {s["eid"] for s in json.loads((OUT / "branch" / f"{f}.json").read_text())["states"]}
        select_states({k: int(v) for k, v in (x.split("=") for x in args.n_ep.split(","))}, args.n_oracle,
                      name=args.sel, exclude=excl, seed=args.sel_seed)
        return

    from repcritic import rc_common as C

    C.setup_env_vars()
    import torch

    from repcritic import e02_common as E2

    sel = json.loads((OUT / "branch" / f"{args.sel}.json").read_text())
    states = sel["states"]
    batches = pack(states, branch_jobs(states), args.E)
    mine = [b for i, b in enumerate(batches) if i % args.nshard == args.shard]
    if args.max_batches:
        mine = mine[: args.max_batches]
    print(f"shard {args.shard}/{args.nshard}: {len(mine)} batches of {len(batches)}; envs "
          f"{sum(len(b) for b in mine)}", flush=True)
    cdir = OUT / "branch" / "candidates"
    cdir.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "branch" / f"branch_results_{args.sel}_shard{args.shard}.jsonl"
    done_keys = set()
    if res_path.exists():
        for l in open(res_path):
            r = json.loads(l)
            done_keys.add((r["state_id"], r["cand"], r["cont"]))

    torch.cuda.reset_peak_memory_stats()
    pol = E2.FeatPolicy(C.build_policy())
    env = C.build_env(args.E)
    base = env.env.unwrapped
    dev = env.device
    lo = torch.as_tensor(env.env.action_space.low[0]).float()
    hi = torch.as_tensor(env.env.action_space.high[0]).float()
    snap_cache = {}
    ep_cache = {}

    def load_ep(eid):
        if eid not in ep_cache:
            if len(ep_cache) > 8:
                ep_cache.pop(next(iter(ep_cache)))
            z = np.load(OUT / "episodes" / f"ep{eid:06d}.npz")
            ep_cache[eid] = {k: z[k] for k in ("rgb", "noise", "chunk_env", "exec_env")}
        return ep_cache[eid]

    def load_snaps(eid):
        if eid not in snap_cache:
            snap_cache[eid] = torch.load(OUT / "snapshots" / f"ep{eid:06d}.pt", weights_only=False)
        return snap_cache[eid]

    def restore_multi(snaps):
        """snaps: list of E single-env snaps (CPU). Each sub-env must already be reset to its episode."""
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

    def get_candidates(si, obs, env_idx):
        """16 candidates for state si (cached on disk)."""
        s = states[si]
        path = cdir / f"{s['state_id']}.npz"
        if path.exists():
            z = np.load(path)
            return torch.from_numpy(z["chunk_env"]), torch.from_numpy(z["exec_env"])
        ep = load_ep(s["eid"])
        d = s["decision"]
        g = torch.Generator().manual_seed(seed_of("cand", s["state_id"]))
        nz = pol.noise(NC, g)
        nz[0] = torch.from_numpy(ep["noise"][d]).to(nz.device)
        o = C.obs_repeat(obs, env_idx, NC)
        acts, macts, F = pol.act_feats(o, nz, full_h=True)
        F.pop("_img_slot")
        acts = acts.cpu()
        exe = torch.max(torch.min(acts[:, : C.K], hi), lo)
        exe[0] = torch.from_numpy(ep["exec_env"][d])  # cand 0 = logged executed actions, exactly
        recompute_diff = float((acts[0] - torch.from_numpy(ep["chunk_env"][d])).abs().max())
        img_diff = float(np.abs(obs["main_images"][env_idx].cpu().numpy().astype(np.float32)
                                - ep["rgb"][d].astype(np.float32)).mean())
        np.savez(path, chunk_env=acts.numpy(), chunk_model=macts.float().cpu().numpy(), noise=nz.cpu().numpy(),
                 exec_env=exe.numpy(), cand0_recompute_maxdiff=recompute_diff, restored_img_meanabsdiff=img_diff,
                 **{k: v.numpy() for k, v in F.items() if k.startswith("ae_cand")})
        return acts, exe

    T0 = time.time()
    with open(res_path, "a") as fout:
        for bi, jobs in enumerate(mine):
            jobs = [j for j in jobs if (states[j[0]]["state_id"], j[1], j[2]) not in done_keys]
            if not jobs:
                continue
            tb = time.time()
            n = len(jobs)
            pad = jobs + [jobs[-1]] * (args.E - n)  # padding envs replay the last job; results discarded
            eids = torch.tensor([states[j[0]]["eid"] for j in pad])
            env.reset(seed=0, options={"episode_id": eids.to(dev)})
            snaps = [load_snaps(states[si]["eid"])[states[si]["decision"]] for (si, c, j) in pad]
            obs = restore_multi(snaps)
            first = torch.zeros(args.E, C.K, 7)
            seen = {}
            for e, (si, c, j) in enumerate(pad):
                if si not in seen:
                    seen[si] = get_candidates(si, obs, e)
                first[e] = seen[si][1][c]
            t_env = torch.tensor([states[j[0]]["t"] for j in pad])
            assert bool((base._elapsed_steps.cpu() == t_env).all())
            t = int(t_env.min())
            gen = torch.Generator().manual_seed(seed_of("cont", args.shard, bi, pad[0][0], t))
            s_once = torch.zeros(args.E, dtype=torch.bool)
            s_last = torch.zeros(args.E, dtype=torch.bool)
            first_t = torch.full((args.E,), -1)
            ret = torch.zeros(args.E)
            chunk = first
            afterK = None
            step = 0
            n_steps = C.MAX_STEPS - t
            while step < n_steps:
                for k in range(chunk.shape[1]):
                    obs, r, te, tr, info = env.step(chunk[:, k], auto_reset=False)
                    live = (t_env + step) < C.MAX_STEPS  # env-local time of this step
                    sc = info["success"].cpu() & live
                    first_t = torch.where(sc & (first_t < 0), t_env + step, first_t)
                    s_once |= sc
                    s_last = torch.where(live, info["success"].cpu(), s_last)
                    ret += r.float().cpu() * live
                    step += 1
                    if step >= n_steps:
                        break
                if afterK is None:
                    afterK = E2.privileged(env)
                    afterK["success_within_K"] = s_once.numpy().copy()
                if step >= n_steps:
                    break
                chunk, _ = pol.act(obs, pol.noise(args.E, gen))
            for e, (si, c, j) in enumerate(jobs):
                s = states[si]
                fout.write(json.dumps(dict(
                    state_id=s["state_id"], eid=s["eid"], split=s["split"], decision=s["decision"], t=s["t"],
                    oracle=s["oracle"], cand=c, cont=j, success_once=bool(s_once[e]), success_at_end=bool(s_last[e]),
                    first_success_t=int(first_t[e]), ret=float(ret[e]),
                    afterK={k: (v[e].tolist() if hasattr(v[e], "tolist") else v[e]) for k, v in afterK.items()
                            if k in ("gripper_carrot_dist", "carrot_plate_dist", "gripper_plate_dist", "grasping_now",
                                     "ever_grasped", "src_on_target", "contact_any_finger", "pos_carrot",
                                     "pos_gripper", "success_within_K")},
                )) + "\n")
            fout.flush()
            print(f"batch {bi+1}/{len(mine)} t={t}..{int(t_env.max())} envs {n} states {len(seen)} "
                  f"succ {s_once[:n].float().mean():.2f} | {time.time()-tb:.0f}s peak {C.gpu_mem_gb():.1f}GB "
                  f"elapsed {(time.time()-T0)/60:.1f}min", flush=True)
    print("done", (time.time() - T0) / 3600, "h", flush=True)


if __name__ == "__main__":
    main()

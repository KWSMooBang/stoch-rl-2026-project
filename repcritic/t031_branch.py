"""T031 — counterfactual P1 (intervention sensitivity): one-shot residual chunk, frozen pi0.5 continuation (mu = pi_VLA).

Pre-registration: repcritic/reports/t031_prereg.md (committed before any candidate branch was run).
States: /workspace/stochrl_t031/state_select.json (repcritic/t031_select.py), 48 = 4 strata x 12.
Candidates per state (NC = 13); base = the logged executed chunk of the T017 episode at decision d (env units):
  0  base       u = 0, CRN continuation noise (reference branch R^(0))
  1  u0_ind     u = 0, INDEPENDENT continuation noise stream (noise baseline: discordance with no intervention)
  2  E1         expert residual, this chunk only:  a[k] = clip(base[k] + clip(a_exp - mean_k base, +-alpha), bounds)
  3  E3         same expert residual on 3 consecutive chunks (this + next 2 VLA chunks, expert re-evaluated online)
  4-7  T1_{+u,-u,+v,-v}  translation 1.0 alpha_pos (0.02 m/step -> 10 cm/chunk) along u (horizontal tcp->target, root
                         frame; target = object, or plate if grasping) and v = z x u
  8-9  T05_{+u,-u}       translation 0.5 alpha_pos along +-u
  10-11 yaw_{+,-}        rotation dim 5 (yaw) +- alpha_rot (0.1 rad/step -> 0.5 rad/chunk)
  12 grip_flip  gripper inverted for this chunk: g' = clip(g - 2 sign(g), -1, 1)   (alpha_g = 2)
alpha = T022 (vla_residual/core.py: a_pos 0.02, a_rot 0.1, a_grip 2.0). Expert = T022 ScriptedExpert (privileged), with
phase memory initialised from the snapshot (see expert_init).
Branch = restore snapshot -> candidate chunk(s) -> frozen pi0.5 to t = 80 (16 decisions). NREP = 4 replicates. CRN:
continuation noise of decision k for replicate r of state s is seeded by (s, r, k), identical for all candidates except
u0_ind which uses an independent stream (s, r, k, "ind").
Sim: intermediate K-1 steps of a chunk run with obs_mode "none" (render skip, T022 _sim_step); physics unchanged.

usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m repcritic.t031_branch --restore_test
       CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m repcritic.t031_branch --budget_h 0.5
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
OUT = Path(os.environ.get("RC_T031_OUT", "/workspace/stochrl_t031"))
CANDS = ["base", "u0_ind", "E1", "E3", "T1_+u", "T1_-u", "T1_+v", "T1_-v", "T05_+u", "T05_-u", "yaw_+", "yaw_-",
         "grip_flip"]
NC = len(CANDS)
NREP = 4
A_POS, A_ROT, A_GRIP = 0.02, 0.1, 2.0
E3_CHUNKS = 3


def seed_of(*xs):
    return int(hashlib.md5("_".join(map(str, xs)).encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore_test", action="store_true",
                    help="all envs run cand 0 (base) with the replicate's CRN noise (13 identical copies per rep)")
    ap.add_argument("--budget_h", type=float, default=0.5, help="stop starting new batches after this wall time")
    ap.add_argument("--max_states", type=int, default=0)
    args = ap.parse_args()

    from repcritic import rc_common as C

    C.setup_env_vars()
    import torch

    from repcritic import e02_common as E2
    from vla_residual.core import ScriptedExpert, priv_state

    sel = json.loads((OUT / "state_select.json").read_text())
    states = sorted(sel["states"], key=lambda s: s["run_order"])
    if args.restore_test:
        s0 = [s for s in states if s["stratum"] == "S0_init"][0]
        s1 = [s for s in states if s["stratum"] == "S2_post"][0]
        states = [s0, s1]
    if args.max_states:
        states = states[: args.max_states]
    tag = "restore_test" if args.restore_test else "main"
    res_path = OUT / f"branch_results_{tag}.jsonl"
    done = set()
    if res_path.exists() and not args.restore_test:
        done = {r["state_id"] for r in map(json.loads, open(res_path))}
    cdir = OUT / "candidates"
    cdir.mkdir(parents=True, exist_ok=True)
    E = NC * NREP
    jobs = [(c, r) for c in range(NC) for r in range(NREP)]
    print(f"{tag}: {len(states)} states ({len(done)} done) E={E}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    pol = C.Policy(C.build_policy())
    env = C.build_env(E)
    base = env.env.unwrapped
    dev = env.device
    lo = torch.as_tensor(env.env.action_space.low[0]).float()
    hi = torch.as_tensor(env.env.action_space.high[0]).float()
    alpha = torch.tensor([A_POS] * 3 + [A_ROT] * 3 + [A_GRIP]).float()
    X = ScriptedExpert(E, dev)

    def restore_all(snap):
        n = E
        rep = lambda t: t.expand(n, *t.shape[1:]).clone().to(dev)  # noqa: E731
        sim = C._tree(rep, snap["sim"])
        ctrl = sim.pop("controller", None)
        base.set_state_dict(sim)
        if ctrl is not None:
            base.agent.set_controller_state(ctrl)
        for a in C.TASK_STATE_ATTRS:
            getattr(base, a)[:] = rep(snap["task"][a])
        for k, v in C._tree(rep, snap["episode_stats"]).items():
            base.episode_stats[k] = v
        base._elapsed_steps[:] = rep(snap["elapsed"])
        env.prev_step_reward[:] = rep(snap["prev_rew"])
        return env._wrap_obs(base.get_obs())

    def sim_step(a, render):
        if render:
            return env.step(a, auto_reset=False)
        ub = env.env.unwrapped
        mode = ub._obs_mode
        ub._obs_mode = "none"
        try:
            o, r, te, tr, info = env.env.step(a.to(dev))
        finally:
            ub._obs_mode = mode
        return None, r, te, tr, info

    def directions(priv):
        """u (toward target, horizontal) and v = z x u in the robot root frame (env 0; all envs share the state)."""
        tgt = priv["pos_plate"][0] if priv["grasping_now"][0] else priv["pos_carrot"][0]
        w = torch.tensor(tgt - priv["pos_gripper"][0]).float()
        R = base.agent.robot.pose.to_transformation_matrix()[0, :3, :3].float().cpu()
        r = R.T @ w
        h = torch.tensor([r[0], r[1], 0.0])
        fallback = bool(h.norm() < 0.005)
        u = torch.tensor([1.0, 0.0, 0.0]) if fallback else h / h.norm()
        v = torch.cross(torch.tensor([0.0, 0.0, 1.0]), u, dim=0)
        return u, v, fallback, float(w.norm())

    def expert_init(ps, rest_carrot_wz):
        """Phase memory from the snapshot: not grasping -> APPROACH (transitions to DESCEND/CLOSE happen inside the
        call when the geometry says so); grasping -> LIFT with cz0 = object rest height (base frame), ez0 = current ee
        z, grasp offset = ee - object (LIFT -> CARRY inside the call if already lifted)."""
        allm = torch.ones(E, dtype=torch.bool, device=dev)
        X.reset(allm)
        g = ps["grasp"]
        cz_rest = ps["carrot"][:, 2] - (ps["carrot_w"][:, 2] - rest_carrot_wz)
        X.phase = torch.where(g, torch.full_like(X.phase, X.LIFT), X.phase)
        X.cz0 = cz_rest.clone()
        X.ez0 = ps["ee"][:, 2].clone()
        X.off = ps["ee"] - ps["carrot"]
        X.cnt.zero_()

    def expert_delta(base_chunk):
        """T022 residual: clip(a_exp - mean_k base, +-alpha). Returns delta [E,7] (cpu), alpha-clip fraction [E]."""
        a_exp = X(priv_state(env)).float().cpu()
        raw = a_exp - base_chunk.mean(1)
        d = torch.max(torch.min(raw, alpha), -alpha)
        return d, (raw.abs() > alpha + 1e-9).float().mean(1), a_exp

    T0 = time.time()
    n_run = 0
    with open(res_path, "a") as fout:
        for s in states:
            if s["state_id"] in done:
                continue
            el = (time.time() - T0) / 3600
            if n_run and el + el / n_run > args.budget_h:
                print(f"BUDGET STOP after {n_run} states ({el:.3f} h)", flush=True)
                break
            tb = time.time()
            eid, d = s["eid"], s["decision"]
            env.reset(seed=0, options={"episode_id": torch.full((E,), eid).to(dev)})
            snaps = torch.load(SRC / "snapshots" / f"ep{eid:06d}.pt", weights_only=False)
            obs = restore_all(snaps[d])
            t0 = int(base._elapsed_steps[0])
            assert t0 == s["t"], (t0, s["t"])
            priv = E2.privileged(env)
            ep = np.load(SRC / "episodes" / f"ep{eid:06d}.npz")
            rest_wz = float(np.load(SRC / "privileged" / f"ep{eid:06d}.npz")["dec_pos_carrot"][0, 2])
            b0 = torch.from_numpy(ep["exec_env"][d]).float()  # [K,7] logged executed chunk
            img_diff = float(np.abs(obs["main_images"][0].cpu().numpy().astype(np.float32)
                                    - ep["rgb"][d].astype(np.float32)).mean())
            # ---- candidate first chunks
            ps = priv_state(env)
            expert_init(ps, rest_wz)
            dlt, aclip, a_exp = expert_delta(b0[None].expand(E, -1, -1))
            ph0 = int(X.phase[0])
            u, v, fb, dist = directions(priv)
            cand = torch.zeros(NC, C.K, 7)
            for c, name in enumerate(CANDS):
                x = b0.clone()
                if name in ("E1", "E3"):
                    x = x + dlt[0]
                elif name.startswith("T"):
                    m = 1.0 if name.startswith("T1_") else 0.5
                    dirv = {"+u": u, "-u": -u, "+v": v, "-v": -v}[name.split("_")[1]]
                    x[:, :3] = x[:, :3] + m * A_POS * dirv
                elif name.startswith("yaw"):
                    x[:, 5] = x[:, 5] + (A_ROT if name == "yaw_+" else -A_ROT)
                elif name == "grip_flip":
                    sg = torch.where(x[:, 6] >= 0, 1.0, -1.0)
                    x[:, 6] = x[:, 6] - A_GRIP * sg
                cand[c] = x
            unclipped = cand.clone()
            cand = torch.max(torch.min(cand, hi), lo)
            bound_clip = ((unclipped - cand).abs() > 1e-9).float().mean((1, 2))
            if args.restore_test:
                cand[:] = cand[0]
            np.savez(cdir / f"{s['state_id']}{'_rt' if args.restore_test else ''}.npz", exec_env=cand.numpy(),
                     base=b0.numpy(), expert_action=a_exp[0].numpy(), expert_delta=dlt[0].numpy(),
                     expert_alpha_clip=float(aclip[0]), expert_phase0=ph0, u_root=u.numpy(), v_root=v.numpy(),
                     dir_fallback=fb, tcp_target_dist=dist, bound_clip=bound_clip.numpy(),
                     restored_img_meanabsdiff=img_diff)
            first = torch.stack([cand[c] for (c, r) in jobs])
            is_e3 = torch.tensor([CANDS[c] == "E3" and not args.restore_test for (c, r) in jobs])
            # ---- roll out
            s_once = torch.zeros(E, dtype=torch.bool)
            s_last = torch.zeros(E, dtype=torch.bool)
            first_t = torch.full((E,), -1)
            chunk = first
            afterK = None
            e3_log = [[] for _ in range(E)]
            step, kdec = 0, 0
            n_steps = C.MAX_STEPS - t0
            while step < n_steps:
                for k in range(C.K):
                    _o, _r, _te, _tr, info = sim_step(chunk[:, k], render=(k == C.K - 1))
                    if k == C.K - 1:
                        obs = _o
                    sc = info["success"].cpu().bool()
                    first_t = torch.where(sc & (first_t < 0), torch.tensor(t0 + step), first_t)
                    s_once |= sc
                    s_last = sc
                    step += 1
                if afterK is None:
                    afterK = E2.privileged(env)
                if step >= n_steps:
                    break
                kdec += 1
                nz = torch.stack([torch.randn(*pol.noise_shape, generator=torch.Generator().manual_seed(
                    seed_of("cont31", s["state_id"], r, kdec, *(("ind",) if CANDS[c] == "u0_ind" and not
                                                              args.restore_test else ())))) for (c, r) in jobs])
                chunk, _ = pol.act(obs, nz.to(pol.device))
                chunk = chunk.cpu()
                if kdec < E3_CHUNKS and bool(is_e3.any()):
                    dl, ac, _ = expert_delta(chunk)
                    newc = torch.max(torch.min(chunk + dl[:, None], hi), lo)
                    chunk = torch.where(is_e3[:, None, None], newc, chunk)
                    for e in torch.nonzero(is_e3).flatten().tolist():
                        e3_log[e].append(dict(kdec=kdec, delta=[round(float(x), 5) for x in dl[e]],
                                              alpha_clip=float(ac[e]), phase=int(X.phase[e])))
            for e, (c, r) in enumerate(jobs):
                row = dict(state_id=s["state_id"], eid=eid, decision=d, t=t0, stratum=s["stratum"], phase=s["phase"],
                           vhat=s["vhat"], episode_success_once=s["episode_success_once"], cand=c, cand_name=CANDS[c],
                           rep=r, success_once=bool(s_once[e]), success_at_end=bool(s_last[e]),
                           first_success_t=int(first_t[e]), bound_clip=float(bound_clip[c]),
                           afterK={k: (v[e].tolist() if hasattr(v[e], "tolist") else v[e]) for k, v in afterK.items()
                                   if k in ("gripper_carrot_dist", "carrot_plate_dist", "grasping_now", "ever_grasped",
                                            "src_on_target", "pos_gripper", "pos_carrot")})
                if CANDS[c] in ("E1", "E3"):
                    row.update(expert_alpha_clip=float(aclip[0]), expert_phase0=ph0)
                if e3_log[e]:
                    row["e3_later"] = e3_log[e]
                fout.write(json.dumps(row) + "\n")
            fout.flush()
            n_run += 1
            print(f"state {n_run} {s['state_id']} {s['stratum']} t={t0} succ base {s_once[:NREP].float().mean():.2f} "
                  f"all {s_once.float().mean():.2f} expert_ph {ph0} | {time.time()-tb:.0f}s peak {C.gpu_mem_gb():.1f}GB "
                  f"elapsed {(time.time()-T0)/60:.1f}min", flush=True)
    print("done", (time.time() - T0) / 3600, "h", flush=True)


if __name__ == "__main__":
    main()

"""E02 rollout buffer: frozen pi0.5 SFT, PutOnPlateInScene25Main-v3 (obj_set=train), fixed 80-step episodes.

Episode set: first N of randperm(total_num_trials, seed=EP_SEED); split 70/15/15 by episode with SPLIT_SEED.
Each sub-env = one episode. Per decision d (16 per episode, t = 5d) we store:
  inputs   : rgb (3rd_view 640x480 uint8), qpos state, instruction, frozen features (fp16; e02_common doc)
  actions  : initial noise [8,32], raw chunk H=8 (env + normalized model space), executed K=5 (env clip)
  transition: per-step reward (RLinf rel. shaped reward), success, terminated, truncated; next obs = obs[d+1]
  labels   : privileged sim labels (label-only file), phase, success flags
  snapshot : full restorable sim state (for snapshot branches)

usage: CUDA_VISIBLE_DEVICES=0 RC_POLICY_MB=8 python -m repcritic.e02_rollout --start 0 --n 250 --batch 25
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path

from repcritic import rc_common as C

C.setup_env_vars()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from repcritic import e02_common as E  # noqa: E402

import os
OUT = Path(os.environ.get("RC_E02_OUT", "/workspace/stochrl_t017"))
EP_SEED = 20261119
SPLIT_SEED = 7
N_TOTAL = 500
N_DEC = C.MAX_STEPS // C.K  # 16


def episode_table(total, n=N_TOTAL):
    g = torch.Generator().manual_seed(EP_SEED)
    eids = torch.randperm(total, generator=g)[:n].tolist()
    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(n)
    n_tr, n_va = int(round(0.7 * n)), int(round(0.15 * n))
    split = {}
    for r, j in enumerate(perm):
        split[eids[j]] = "train" if r < n_tr else ("val" if r < n_tr + n_va else "test")
    return [(i, e, split[e]) for i, e in enumerate(eids)]


class Writer(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue(maxsize=64)
        self.err = None
        self.start()

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            fn, args = item
            try:
                fn(*args)
            except Exception as e:  # noqa: BLE001
                self.err = e
                print("WRITER ERROR", e, flush=True)

    def put(self, fn, *args):
        if self.err:
            raise self.err
        self.q.put((fn, args))

    def close(self):
        self.q.put(None)
        self.join()
        if self.err:
            raise self.err


def _savez(path, compressed, d):
    tmp = path.with_suffix(".tmp.npz")
    (np.savez_compressed if compressed else np.savez)(tmp, **d)
    tmp.rename(path)


def _save_pt(path, obj):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.rename(path)


def run_batch(env, pol, batch, noise_seed, writer, rows_f, check_hook=False):
    """batch: list of (idx, eid, split). Runs all as parallel sub-envs."""
    n = len(batch)
    eids = torch.tensor([b[1] for b in batch])
    obs, _ = env.reset(seed=0, options={"episode_id": eids.to(env.device)})
    instr = list(obs["task_descriptions"])
    gen = torch.Generator().manual_seed(noise_seed)
    lo = torch.as_tensor(env.env.action_space.low[0]).float()
    hi = torch.as_tensor(env.env.action_space.high[0]).float()

    rgb = np.zeros((n, N_DEC + 1, 480, 640, 3), np.uint8)
    state = np.zeros((n, N_DEC + 1, 8), np.float32)
    noise_all = np.zeros((n, N_DEC, C.H, 32), np.float32)
    ch_env = np.zeros((n, N_DEC, C.H, 7), np.float32)
    ch_mod = np.zeros((n, N_DEC, C.H, 32), np.float32)
    ex_env = np.zeros((n, N_DEC, C.K, 7), np.float32)
    st_rew = np.zeros((n, C.MAX_STEPS), np.float32)
    st_succ = np.zeros((n, C.MAX_STEPS), bool)
    st_term = np.zeros((n, C.MAX_STEPS), bool)
    st_trunc = np.zeros((n, C.MAX_STEPS), bool)
    st_qpos = np.zeros((n, C.MAX_STEPS, 8), np.float32)
    st_pos = {k: np.zeros((n, C.MAX_STEPS, 3), np.float32) for k in ("pos_carrot", "pos_plate", "pos_gripper")}
    st_grasp = np.zeros((n, C.MAX_STEPS), bool)
    dpriv = []
    feats = []
    snaps = [[] for _ in range(n)]
    t = 0
    tim = dict(policy=0.0, env=0.0, priv=0.0, snap=0.0)
    hook_check = None
    for d in range(N_DEC + 1):
        t0 = time.perf_counter()
        sn = E.snapshot_batch(env)
        for i in range(n):
            snaps[i].append(sn[i])
        tim["snap"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        dpriv.append(E.privileged(env))
        tim["priv"] += time.perf_counter() - t0
        rgb[:, d] = obs["main_images"].cpu().numpy()
        state[:, d] = obs["states"].float().cpu().numpy()
        if d == N_DEC:
            break
        t0 = time.perf_counter()
        nz = pol.noise(n, gen)
        acts, macts, F = pol.act_feats(obs, nz, full_h=True)
        if check_hook and d in (0, 8):
            a_plain, _ = pol.act(obs, nz, full_h=True)
            pol.ctx_pass = False
            a_noctx, _, _ = pol.act_feats(obs, nz, full_h=True)
            pol.ctx_pass = True
            hook_check = (hook_check or []) + [dict(d=d, max_abs_diff_plain_vs_feats=float((a_plain - acts).abs().max()),
                                                    max_abs_diff_noctx_vs_ctx=float((a_noctx - acts).abs().max()))]
            print("hook check", hook_check[-1], flush=True)
        tim["policy"] += time.perf_counter() - t0
        img_slot = F.pop("_img_slot")
        feats.append(F)
        noise_all[:, d] = nz.cpu().numpy()
        ch_env[:, d] = acts.cpu().numpy()
        ch_mod[:, d] = macts.float().cpu().numpy()
        exe = torch.max(torch.min(acts[:, : C.K].cpu(), hi), lo)
        ex_env[:, d] = exe.numpy()
        t0 = time.perf_counter()
        for k in range(C.K):
            obs, r, te, tr, info = env.step(acts[:, k], auto_reset=False)
            st_rew[:, t] = r.float().cpu().numpy()
            st_succ[:, t] = info["success"].cpu().numpy()
            st_term[:, t] = te.cpu().numpy()
            st_trunc[:, t] = tr.cpu().numpy() if torch.is_tensor(tr) else tr
            st_qpos[:, t] = obs["states"].float().cpu().numpy()
            base = env.env.unwrapped
            xs = base.extra_stats
            st_pos["pos_carrot"][:, t] = xs["extra_pos_carrot"].float().cpu().numpy()
            st_pos["pos_plate"][:, t] = xs["extra_pos_plate"].float().cpu().numpy()
            st_pos["pos_gripper"][:, t] = xs["extra_pos_gripper"].float().cpu().numpy()
            st_grasp[:, t] = (base.consecutive_grasp > 0).cpu().numpy()
            t += 1
        tim["env"] += time.perf_counter() - t0

    # stack decision privileged labels -> [n, 17, ...]
    P = {k: np.stack([p[k] for p in dpriv], 1) for k in dpriv[0]}
    phase = np.stack([E.phase_from_priv(p) for p in dpriv], 1)  # [n,17] object
    FE = {k: torch.stack([f[k] for f in feats], 1).numpy() for k in feats[0]}  # [n,16,...] fp16

    s_once = st_succ.any(1)
    s_end = st_succ[:, -1]
    for i, (idx, eid, split) in enumerate(batch):
        name = f"ep{eid:06d}"
        writer.put(_savez, OUT / "episodes" / f"{name}.npz", True, dict(
            rgb=rgb[i], state=state[i], noise=noise_all[i], chunk_env=ch_env[i], chunk_model=ch_mod[i],
            exec_env=ex_env[i], step_reward=st_rew[i], step_success=st_succ[i], step_terminated=st_term[i],
            step_truncated=st_trunc[i], step_qpos=st_qpos[i]))
        writer.put(_savez, OUT / "features" / f"{name}.npz", False, {k: v[i] for k, v in FE.items()})
        writer.put(_savez, OUT / "privileged" / f"{name}.npz", True, dict(
            **{f"dec_{k}": v[i] for k, v in P.items()}, dec_phase=phase[i].astype("U16"),
            **{f"step_{k}": v[i] for k, v in st_pos.items()}, step_grasping=st_grasp[i]))
        writer.put(_save_pt, OUT / "snapshots" / f"{name}.pt", snaps[i])
        succ_before = False
        for d in range(N_DEC):
            ts = slice(d * C.K, (d + 1) * C.K)
            rows_f.write(json.dumps(dict(
                episode_id=f"{name}", eid=int(eid), episode_index=int(idx), split=split, decision=d, t_env=d * C.K,
                task_id=C.ENV_ID, obj_set="train", policy_id="pi05_sft_frozen", policy_revision=C.CKPT_REV,
                language_goal=instr[i], goal_type="G0_task_success", goal_ref=None, target_source="env success predicate",
                H_model=C.H, K_executed=C.K, K_actual=C.K,
                obs_ref=f"episodes/{name}.npz:rgb[{d}],state[{d}]", next_obs_ref=f"episodes/{name}.npz:rgb[{d+1}],state[{d+1}]",
                action_candidate_ref=f"episodes/{name}.npz:chunk_env[{d}],chunk_model[{d}],noise[{d}]",
                action_executed_ref=f"episodes/{name}.npz:exec_env[{d}]",
                feature_ref=f"features/{name}.npz:[{d}]", sim_labels_ref=f"privileged/{name}.npz:dec_*[{d}]",
                snapshot_ref=f"snapshots/{name}.pt:[{d}]",
                rewards=[float(x) for x in st_rew[i, ts]], reward_sum=float(st_rew[i, ts].sum()),
                success_in_chunk=bool(st_succ[i, ts].any()), success_at_chunk_end=bool(st_succ[i, ts][-1]),
                success_before=bool(succ_before),
                terminated=bool(st_term[i, ts].any()), truncated=bool(st_trunc[i, ts].any()),
                done=bool(d == N_DEC - 1),
                success_once=bool(s_once[i]), success_at_end=bool(s_end[i]),
            )) + "\n")
            succ_before = succ_before or bool(st_succ[i, ts].any())
    rows_f.flush()
    return s_once, s_end, tim, hook_check, img_slot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--noise_seed", type=int, default=1000)
    ap.add_argument("--check_hook", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    for sub in ("episodes", "features", "privileged", "snapshots", "logs"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    pol = E.FeatPolicy(C.build_policy())
    env = C.build_env(args.batch)
    total = env.env.unwrapped.total_num_trials
    table = episode_table(total)
    (OUT / "split_manifest.json").write_text(json.dumps(dict(
        ep_seed=EP_SEED, split_seed=SPLIT_SEED, total_num_trials=int(total), n=len(table),
        episodes=[dict(index=i, eid=e, split=s) for i, e, s in table]), indent=0))
    mine = table[args.start: args.start + args.n]
    writer = Writer()
    T0 = time.time()
    with open(OUT / f"decisions_{args.tag or args.start}.jsonl", "a") as rows_f:
        for b in range(0, len(mine), args.batch):
            batch = mine[b: b + args.batch]
            assert len(batch) == args.batch, "n must be a multiple of batch"
            pad = batch
            t0 = time.time()
            so, se, tim, hc, slot = run_batch(env, pol, pad, args.noise_seed + args.start + b, writer, rows_f,
                                              check_hook=args.check_hook and b == 0)
            so, se = so[: len(batch)], se[: len(batch)]
            print(f"batch {b//args.batch} eps {batch[0][0]}..{batch[-1][0]} succ_once {so.mean():.3f} "
                  f"succ_end {se.mean():.3f} | {time.time()-t0:.0f}s {json.dumps({k: round(v,1) for k,v in tim.items()})} "
                  f"img_slot {slot} peak {C.gpu_mem_gb():.1f}GB | elapsed {(time.time()-T0)/60:.1f}min", flush=True)
            if hc:
                (OUT / "logs" / f"hook_check_{args.start}.json").write_text(json.dumps(hc, indent=1))
    writer.close()
    print("done", (time.time() - T0) / 3600, "h", flush=True)


if __name__ == "__main__":
    main()

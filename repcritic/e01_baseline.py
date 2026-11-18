"""E01: frozen pi0.5 SFT baseline on PutOnPlateInScene25Main-v3.

All episodes of one noise seed run as parallel sub-envs (one per episode_id).
Episode initial state is fully determined by episode_id (RLinf 25Main design);
policy stochasticity = per-decision Gaussian initial noise from a seeded CPU generator.

usage: CUDA_VISIBLE_DEVICES=0 python -m repcritic.e01_baseline --n_episodes 30 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json

from repcritic import rc_common as C

C.setup_env_vars()

import numpy as np  # noqa: E402
import torch  # noqa: E402


def episode_ids(n, total, seed=12345):
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(total, generator=g)[:n]


def run_seed(env, pol, eids, noise_seed, record_video=True, log_actions=None):
    n = len(eids)
    obs, info = env.reset(seed=0, options={"episode_id": eids.to(env.device)})
    gen = torch.Generator().manual_seed(noise_seed)
    succ_steps = torch.zeros(n, C.MAX_STEPS, dtype=torch.bool)
    rets = torch.zeros(n)
    frames = [[] for _ in range(n)] if record_video else None
    lat = []
    instr = list(obs["task_descriptions"])
    step = 0
    decisions = 0
    amin = np.full(7, np.inf); amax = np.full(7, -np.inf)
    mmin = np.full(7, np.inf); mmax = np.full(7, -np.inf)
    while step < C.MAX_STEPS:
        if record_video:
            im = obs["main_images"].cpu().numpy()
            for i in range(n):
                frames[i].append(im[i])
        torch.cuda.synchronize(); t0 = C.now()
        acts, macts = pol.act(obs, pol.noise(n, gen))
        torch.cuda.synchronize(); lat.append((C.now() - t0) * 1000)
        decisions += 1
        a = acts.cpu().numpy(); ma = macts[..., :7].float().cpu().numpy()
        amin = np.minimum(amin, a.reshape(-1, 7).min(0)); amax = np.maximum(amax, a.reshape(-1, 7).max(0))
        mmin = np.minimum(mmin, ma.reshape(-1, 7).min(0)); mmax = np.maximum(mmax, ma.reshape(-1, 7).max(0))
        if log_actions is not None:
            log_actions.append(a)
        for k in range(acts.shape[1]):
            obs, r, term, trunc, info = env.step(acts[:, k], auto_reset=False)
            succ_steps[:, step] = info["success"].cpu()
            rets += r.float().cpu()
            step += 1
            if record_video and k < acts.shape[1] - 1:
                im = obs["main_images"].cpu().numpy()
                for i in range(n):
                    frames[i].append(im[i])
            if step >= C.MAX_STEPS:
                break
    if record_video:
        im = obs["main_images"].cpu().numpy()
        for i in range(n):
            frames[i].append(im[i])
    s_once = succ_steps.any(1)
    s_end = succ_steps[:, -1]
    first = torch.where(s_once, succ_steps.float().argmax(1) + 1, torch.full((n,), C.MAX_STEPS))
    return dict(s_once=s_once.numpy(), s_end=s_end.numpy(), steps=first.numpy(), ret=rets.numpy(),
                lat_ms=float(np.mean(lat)), decisions=decisions, frames=frames, instr=instr,
                env_act_min=amin.tolist(), env_act_max=amax.tolist(),
                model_act_min=mmin.tolist(), model_act_max=mmax.tolist(), succ_steps=succ_steps.numpy())


def save_video(path, frames, fps=5):
    import imageio

    imageio.mimsave(path, [f.astype(np.uint8) for f in frames], fps=fps)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--obj_set", default="train")
    ap.add_argument("--tag", default="e01")
    ap.add_argument("--max_videos", type=int, default=15)
    ap.add_argument("--repro_check", action="store_true")
    args = ap.parse_args()

    out = C.OUT_BIG / args.tag
    vid = C.OUT_BIG / "videos" / args.tag
    out.mkdir(parents=True, exist_ok=True); vid.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    pol = C.Policy(C.build_policy())
    env = C.build_env(args.n_episodes, args.obj_set)
    base = env.env.unwrapped
    total = base.total_num_trials
    eids = episode_ids(args.n_episodes, total)
    meta = dict(env_id=C.ENV_ID, obj_set=args.obj_set, total_num_trials=int(total), episode_ids=eids.tolist(),
                control_mode=str(base.control_mode), obs_mode=str(base.obs_mode),
                sim_freq=base.sim_freq, control_freq=base.control_freq,
                camera="3rd_view_camera rgb 640x480 (green-screen overlay)",
                policy_checkpoint=C.CKPT_ID, policy_revision=C.CKPT_REV, H=C.H, K=C.K, num_denoise=C.NUM_DENOISE,
                action_norm="openpi pi05_maniskill norm_stats (physical-intelligence/maniskill/norm_stats.json), output Unnormalize",
                action_space=str(env.env.action_space))
    print(json.dumps(meta, indent=1))

    rows = []
    n_succ_vid = n_fail_vid = 0
    per_seed = {}
    for s in args.seeds:
        r = run_seed(env, pol, eids, noise_seed=s)
        per_seed[s] = r
        for i in range(args.n_episodes):
            rows.append(dict(episode_id=int(eids[i]), seed=s, env_id=C.ENV_ID, policy_checkpoint=C.CKPT_ID,
                             policy_revision=C.CKPT_REV, success_once=bool(r["s_once"][i]),
                             success_at_end=bool(r["s_end"][i]), env_return=float(r["ret"][i]),
                             steps=int(r["steps"][i]), decisions=r["decisions"], action_horizon_H=C.H,
                             execute_steps_K=C.K, obs_mode=meta["obs_mode"], control_mode=meta["control_mode"],
                             action_norm="pi05_maniskill_norm_stats", latency_ms=r["lat_ms"],
                             batch=args.n_episodes, gpu_peak_mem_gb=C.gpu_mem_gb(), instruction=r["instr"][i]))
            ok = bool(r["s_once"][i])
            if (ok and n_succ_vid < args.max_videos) or ((not ok) and n_fail_vid < args.max_videos):
                tag = "succ" if ok else "fail"
                save_video(vid / f"{tag}_ep{int(eids[i])}_s{s}.mp4", r["frames"][i])
                n_succ_vid += ok; n_fail_vid += (not ok)
        print(f"seed {s}: success_once {r['s_once'].mean():.3f} success_at_end {r['s_end'].mean():.3f} "
              f"lat {r['lat_ms']:.1f}ms/decision(batch {args.n_episodes})", flush=True)

    # latency at batch 1
    obs1, _ = env.reset(seed=0, options={"episode_id": eids.to(env.device)})
    o1 = C.obs_select(obs1, [0])
    g = torch.Generator().manual_seed(0)
    ts = []
    for _ in range(12):
        torch.cuda.synchronize(); t0 = C.now(); pol.act(o1, pol.noise(1, g)); torch.cuda.synchronize()
        ts.append((C.now() - t0) * 1000)
    lat1 = float(np.median(ts[2:]))

    repro = None
    if args.repro_check:
        r2 = run_seed(env, pol, eids, noise_seed=args.seeds[0], record_video=False)
        r1 = per_seed[args.seeds[0]]
        repro = dict(same_success_once=bool((r1["s_once"] == r2["s_once"]).all()),
                     same_step_success=bool((r1["succ_steps"] == r2["succ_steps"]).all()),
                     n_diff_once=int((r1["s_once"] != r2["s_once"]).sum()))
        print("repro", repro)

    import csv

    with open(out / "baseline_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(out / "episode_manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    so = np.array([r["success_once"] for r in rows]); se = np.array([r["success_at_end"] for r in rows])
    summ = dict(meta=meta, n=len(rows), success_once=float(so.mean()), success_once_ci=wilson(so.sum(), len(so)),
                success_at_end=float(se.mean()), success_at_end_ci=wilson(se.sum(), len(se)),
                per_seed={s: dict(success_once=float(v["s_once"].mean()), success_at_end=float(v["s_end"].mean()),
                                  mean_steps=float(v["steps"].mean()), mean_return=float(v["ret"].mean()),
                                  lat_ms_batch=v["lat_ms"], env_act_min=v["env_act_min"], env_act_max=v["env_act_max"],
                                  model_act_min=v["model_act_min"], model_act_max=v["model_act_max"])
                          for s, v in per_seed.items()},
                per_episode_any_seed=float(np.mean([any(per_seed[s]["s_once"][i] for s in args.seeds)
                                                    for i in range(args.n_episodes)])),
                lat_ms_batch1=lat1, gpu_peak_mem_gb=C.gpu_mem_gb(), repro=repro,
                videos=dict(succ=n_succ_vid, fail=n_fail_vid, dir=str(vid)))
    with open(out / "summary.json", "w") as f:
        json.dump(summ, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k != "meta"}, indent=1, default=float))


if __name__ == "__main__":
    main()

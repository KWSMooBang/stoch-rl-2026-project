"""Candidate diversity (06 "후보 다양성 진단") + Oracle-N (test-only diagnostic).

One episode at a time, E = 2*NO sub-envs all reset to the same episode_id.
At each decision d of the main (baseline) trajectory:
  1. snapshot main state S (env 0) and broadcast it to all E envs.
  2. sample NC=32 candidate chunks (H=8) from the frozen policy on the same (obs, instruction)
     -> diversity metrics.
  3. Oracle branch: env i<NO executes candidate i's first K steps; envs NO..2NO-1 execute candidate 0
     (control: same first chunk, only the continuation noise differs). Then every env continues with
     the frozen policy (fresh noise) until step 80. Outcome = success_once / success_at_end of that branch.
  4. restore S everywhere and execute candidate 0's first K steps -> main trajectory continues.
OraclePass@N(d) = any(success of candidates 0..N-1). N=1 is the baseline policy itself.

usage: CUDA_VISIBLE_DEVICES=0 python -m repcritic.oracle_diversity --test_restore
       CUDA_VISIBLE_DEVICES=0 python -m repcritic.oracle_diversity --n_episodes 30
"""
from __future__ import annotations

import argparse
import json

from repcritic import rc_common as C

C.setup_env_vars()

import numpy as np  # noqa: E402
import torch  # noqa: E402

NC = 32  # candidates for diversity
NO = 16  # oracle candidates


def env_probe(env):
    base = env.env.unwrapped
    q = base.agent.robot.get_qpos().float().cpu()
    st = base.extra_stats if hasattr(base, "extra_stats") else {}
    car = st.get("extra_pos_carrot")
    return q, (car.float().cpu() if car is not None else None)


def rollout_from(env, pol, obs, first_chunk, gen, t0, record_frames=False):
    """Execute first_chunk [E,K,7] then continue with policy until MAX_STEPS. Returns success_once, success_at_end."""
    E = first_chunk.shape[0]
    s_once = torch.zeros(E, dtype=torch.bool)
    s_last = torch.zeros(E, dtype=torch.bool)
    t = t0
    chunk = first_chunk
    frames = []
    while t < C.MAX_STEPS:
        for k in range(chunk.shape[1]):
            obs, r, te, tr, info = env.step(chunk[:, k], auto_reset=False)
            s = info["success"].cpu()
            s_once |= s; s_last = s
            if record_frames:
                frames.append(obs["main_images"][0].cpu().numpy())
            t += 1
            if t >= C.MAX_STEPS:
                break
        if t >= C.MAX_STEPS:
            break
        chunk, _ = pol.act(obs, pol.noise(E, gen))
    return s_once, s_last, obs, frames


def test_restore(env, pol, eid, out):
    E = env.env.unwrapped.num_envs
    obs, _ = env.reset(seed=0, options={"episode_id": torch.full((E,), eid, device=env.device)})
    g = torch.Generator().manual_seed(0)
    # drive 3 decisions with an identical chunk in all envs
    for _ in range(3):
        a, _ = pol.act(C.obs_select(obs, [0]), pol.noise(1, g))
        a = a.expand(E, -1, -1)
        for k in range(C.K):
            obs, *_ = env.step(a[:, k], auto_reset=False)
    q, _ = env_probe(env)
    cross_env_qpos_diff = float((q - q[0:1]).abs().max())
    snap = C.snapshot(env, 0)
    # fixed action sequence, 4 chunks
    seq = []
    g2 = torch.Generator().manual_seed(1)
    o = C.obs_select(obs, [0])
    for _ in range(4):
        a, _ = pol.act(o, pol.noise(1, g2))
        seq.append(a[0])
    seq = torch.cat(seq, 0)  # [20,7]

    def play():
        qs, imgs, succ = [], [], []
        for k in range(seq.shape[0]):
            ob, r, te, tr, info = env.step(seq[k : k + 1].expand(E, -1), auto_reset=False)
            qq, car = env_probe(env)
            qs.append(qq); imgs.append(ob["main_images"].float().cpu()); succ.append(info["success"].cpu())
        return torch.stack(qs), torch.stack(imgs), torch.stack(succ)

    q_nat, im_nat, s_nat = play()          # natural continuation (no restore)
    obs_r = C.restore_all(env, snap)
    im0_diff = float((obs_r["main_images"].float().cpu() - obs["main_images"].float().cpu()).abs().max())
    q_r1, im_r1, s_r1 = play()             # after restore #1
    C.restore_all(env, snap)
    q_r2, im_r2, s_r2 = play()             # after restore #2
    res = dict(
        episode_id=eid,
        cross_env_qpos_maxdiff_before_snapshot=cross_env_qpos_diff,
        restored_obs_image_maxdiff=im0_diff,
        natural_vs_restore1_qpos_maxdiff=float((q_nat - q_r1).abs().max()),
        natural_vs_restore1_img_maxdiff=float((im_nat - im_r1).abs().max()),
        restore1_vs_restore2_qpos_maxdiff=float((q_r1 - q_r2).abs().max()),
        restore1_vs_restore2_img_maxdiff=float((im_r1 - im_r2).abs().max()),
        cross_env_after_restore_qpos_maxdiff=float((q_r1 - q_r1[:, 0:1]).abs().max()),
        success_same=bool((s_nat == s_r1).all() and (s_r1 == s_r2).all()),
    )
    print(json.dumps(res, indent=1), flush=True)
    return res


def diversity_metrics(acts, macts):
    """acts: env-space [N,H,7]; macts: normalized model-space [N,H,7]."""
    def mean_pdist(x):
        x = x.reshape(x.shape[0], -1)
        d = torch.cdist(x, x)
        n = x.shape[0]
        return float(d.sum() / (n * (n - 1)))

    out = {}
    for name, a in (("env", acts), ("norm", macts)):
        out[f"l2_{name}_K"] = mean_pdist(a[:, : C.K])
        out[f"l2_{name}_H"] = mean_pdist(a)
        out[f"l2_{name}_K_perstep"] = out[f"l2_{name}_K"] / np.sqrt(C.K)
        out[f"l2_{name}_H_perstep"] = out[f"l2_{name}_H"] / np.sqrt(C.H)
    # predicted EE displacement = cumulative xyz deltas (env action units)
    for L, tag in ((C.K, "K"), (C.H, "H")):
        disp = acts[:, :L, :3].sum(1)  # [N,3]
        out[f"ee_disp_var_{tag}"] = float(disp.var(0).sum())
        out[f"ee_disp_std_{tag}"] = disp.std(0).tolist()
        out[f"ee_disp_mean_norm_{tag}"] = float(disp.norm(dim=1).mean())
        rot = acts[:, :L, 3:6].sum(1)
        out[f"rot_disp_var_{tag}"] = float(rot.var(0).sum())
    grip = acts[:, :, 6]
    pat = (grip > 0).int()  # binarized per step
    pats = ["".join(map(str, p.tolist())) for p in pat]
    uniq, cnt = np.unique(pats, return_counts=True)
    p = cnt / cnt.sum()
    out["grip_unique_patterns_H"] = int(len(uniq))
    out["grip_pattern_entropy_H"] = float(-(p * np.log2(p)).sum())
    patK = ["".join(map(str, q[: C.K].tolist())) for q in pat]
    uK, cK = np.unique(patK, return_counts=True)
    pK = cK / cK.sum()
    out["grip_unique_patterns_K"] = int(len(uK))
    out["grip_pattern_entropy_K"] = float(-(pK * np.log2(pK)).sum())
    out["grip_majority_frac_H"] = float(cnt.max() / cnt.sum())
    out["grip_value_std_mean"] = float(grip.std(0).mean())
    return out


def phase_label(env, t):
    base = env.env.unwrapped
    es = base.episode_stats
    ever = bool(es["is_src_obj_grasped"][0])
    now = bool(base.consecutive_grasp[0] > 0)
    on = bool(es.get("src_on_target", torch.zeros(1, dtype=torch.bool))[0]) if "src_on_target" in es else False
    if on:
        ph = "placed"
    elif now:
        ph = "transport"
    elif ever:
        ph = "post_grasp_lost"
    else:
        ph = "approach"
    return dict(t=t, ever_grasped=ever, grasping_now=now, on_target=on, phase=ph,
                gripper_carrot_dist=float(es["gripper_carrot_dist"][0]) if "gripper_carrot_dist" in es else None)


def run_episode(env, pol, eid, gen, args, video_dir=None):
    E = env.env.unwrapped.num_envs
    obs, _ = env.reset(seed=0, options={"episode_id": torch.full((E,), eid, device=env.device)})
    main_once = False
    main_last = False
    recs = []
    t = 0
    main_frames = [obs["main_images"][0].cpu().numpy()]
    for d in range(C.MAX_STEPS // C.K):
        snap = C.snapshot(env, 0)
        obs = C.restore_all(env, snap)
        ph = phase_label(env, t) if d > 0 else dict(t=0, ever_grasped=False, grasping_now=False, on_target=False,
                                                    phase="approach", gripper_carrot_dist=None)
        o1 = C.obs_select(obs, [0])
        oN = C.obs_repeat(o1, 0, NC)
        acts, macts = pol.act(oN, pol.noise(NC, gen), full_h=True)
        rec = dict(episode_id=eid, decision=d, main_success_before=main_once, **ph)
        rec.update(diversity_metrics(acts.cpu(), macts[..., :7].float().cpu()))
        if d % args.oracle_every == 0:
            first = torch.cat([acts[:NO, : C.K], acts[0:1, : C.K].expand(E - NO, -1, -1)], 0)
            s_once, s_last, _, _ = rollout_from(env, pol, obs, first, gen, t)
            rec["cand_success_once"] = s_once[:NO].int().tolist()
            rec["ctrl_success_once"] = [int(s_once[0])] + s_once[NO:].int().tolist()[: NO - 1]
            rec["cand_success_end"] = s_last[:NO].int().tolist()
            rec["ctrl_success_end"] = [int(s_last[0])] + s_last[NO:].int().tolist()[: NO - 1]
            obs = C.restore_all(env, snap)
        recs.append(rec)
        # main trajectory: candidate 0 (the baseline policy sample)
        for k in range(C.K):
            obs, r, te, tr, info = env.step(acts[0:1, k].expand(E, -1), auto_reset=False)
            s = bool(info["success"][0])
            main_once |= s; main_last = s
            main_frames.append(obs["main_images"][0].cpu().numpy())
            t += 1
    for r in recs:
        r["main_success_once"] = main_once
        r["main_success_end"] = main_last
    if video_dir is not None:
        import imageio

        imageio.mimsave(video_dir / f"{'succ' if main_once else 'fail'}_ep{eid}_oracle_main.mp4",
                        [f.astype(np.uint8) for f in main_frames], fps=5)
    return recs, main_once, main_last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--oracle_every", type=int, default=1)
    ap.add_argument("--test_restore", action="store_true")
    ap.add_argument("--noise_seed", type=int, default=100)
    ap.add_argument("--tag", default="oracle")
    args = ap.parse_args()
    out = C.OUT_BIG / args.tag
    out.mkdir(parents=True, exist_ok=True)
    vid = C.OUT_BIG / "videos" / args.tag
    vid.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    pol = C.Policy(C.build_policy())
    env = C.build_env(2 * NO)
    from repcritic.e01_baseline import episode_ids

    eids = episode_ids(args.start + args.n_episodes, env.env.unwrapped.total_num_trials)[args.start:].tolist()
    if args.test_restore:
        res = [test_restore(env, pol, e, out) for e in eids[:3]]
        (out / "restore_test.json").write_text(json.dumps(res, indent=1))
        return
    gen = torch.Generator().manual_seed(args.noise_seed)
    with open(out / f"decisions_{args.start}.jsonl", "w") as f:
        for j, eid in enumerate(eids):
            t0 = C.now()
            recs, mo, ml = run_episode(env, pol, eid, gen, args, video_dir=vid if j < 12 else None)
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            o = [r for r in recs if "cand_success_once" in r]
            p1 = np.mean([r["cand_success_once"][0] for r in o])
            p16 = np.mean([max(r["cand_success_once"]) for r in o])
            print(f"ep {j} id {eid} main_once {mo} | dec pass@1 {p1:.2f} pass@16 {p16:.2f} | "
                  f"{C.now() - t0:.0f}s peak {C.gpu_mem_gb():.1f}GB", flush=True)


if __name__ == "__main__":
    main()

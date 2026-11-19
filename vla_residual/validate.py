"""T022 pre-training checks (single process, no Ray), on the held-out eval episodes unless noted.

  --mode b0        residual 0                         (B0 reproduction, T016/T020 ~0.53-0.55)
  --mode expert    scripted privileged expert alone   (expert success)
  --mode upper     base + clip((expert - base)/alpha) (hand-coded upper bound within alpha)
  --mode units     reward / termination / truncation / auto-reset / GRPO-group checks (train split)
  CUDA_VISIBLE_DEVICES=1 python -m vla_residual.validate --mode b0 --n 128 --batch 64 --out /outputs/StochRL/t022/val
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from omegaconf import OmegaConf

from vla_residual import core as V

import torch  # noqa: E402
import numpy as np  # noqa: E402


def make_cfg(split, n_eval=128, auto_reset=False, ignore_term=True, group_size=1, a_pos=0.02, a_rot=0.1,
             a_grip=2.0, squash="tanh", **extra):
    vr = dict(split=split, a_pos=a_pos, a_rot=a_rot, a_grip=a_grip, squash=squash, n_eval=n_eval)
    vr.update(extra)
    return OmegaConf.create(dict(seed=0, auto_reset=auto_reset, ignore_terminations=ignore_term,
                                 group_size=group_size, video_cfg=dict(save_video=False), vla_residual=vr))


def run_eval(mode, n, batch, a_pos, a_rot, a_grip, rep=0):
    extra = {}
    if mode == "expert":
        extra = dict(expert_only=True, expert_exec=1.0)
    elif mode == "upper":
        extra = dict(expert_exec=1.0)
    cfg = make_cfg("eval", n_eval=n, a_pos=a_pos, a_rot=a_rot, a_grip=a_grip, eval_noise_rep=rep, **extra)
    env = V.ResidualVLAEnv(cfg, batch, 0, 1, None)
    rows = []
    t0 = time.time()
    for b in range(n // batch):
        obs, _ = env.reset()
        eids = env.eids.clone()
        for d in range(V.T_DEC):
            u = torch.zeros(batch, 1, V.ACT_DIM, device=env.device)
            obs_l, r, te, tr, inf = env.chunk_step(u)
        ep = inf[0]["episode"]
        for i in range(batch):
            rows.append(dict(eid=int(eids[i]), success_once=bool(ep["success_once"][i]),
                             success_at_end=bool(ep["success_at_end"][i])))
        print(f"[{mode}] batch {b} once {ep['success_once'].float().mean():.3f} end "
              f"{ep['success_at_end'].float().mean():.3f} {time.time()-t0:.0f}s", flush=True)
    so = np.mean([r["success_once"] for r in rows]); se = np.mean([r["success_at_end"] for r in rows])
    return dict(mode=mode, n=len(rows), success_once=float(so), success_at_end=float(se),
                alpha=dict(pos=a_pos, rot=a_rot, grip=a_grip), wall_s=time.time() - t0,
                peak_gb=torch.cuda.max_memory_allocated() / 1e9, episodes=rows)


def units(batch):
    """Unit checks on the train split (residual 0)."""
    out = {}
    # 1) termination on success: reward 1 exactly at first success, terminated, no reward afterwards
    cfg = make_cfg("train", auto_reset=False, ignore_term=False)
    env = V.ResidualVLAEnv(cfg, batch, 0, 1, None)
    obs, _ = env.reset()
    assert obs["states"].shape == (batch, V.OBS_DIM), obs["states"].shape
    assert not (set(env.eids.tolist()) & env.excluded), "train id in held-out/excluded set"
    R = []; TE = []; TR = []; SO = []
    for d in range(V.T_DEC):
        _, r, te, tr, inf = env.chunk_step(torch.zeros(batch, 1, 7, device=env.device))
        R.append(r[:, 0].cpu()); TE.append(te[:, 0].cpu()); TR.append(tr[:, 0].cpu())
        SO.append(inf[0]["episode"]["success_once"].cpu())
    R = torch.stack(R, 1); TE = torch.stack(TE, 1); TR = torch.stack(TR, 1); SO = torch.stack(SO, 1)
    first = SO.float().argmax(1)
    ok_r = all((R[i].sum() == float(SO[i, -1])) and (not SO[i, -1] or (R[i, first[i]] == 1 and TE[i, first[i]]))
               for i in range(batch))
    out["reward_eq_first_success"] = bool(ok_r)
    # truncation only at decision 16 and only for envs not terminated at that same decision (envs terminated
    # earlier are masked by RLinf's loss_mask when auto_reset=False)
    out["truncation_only_at_16"] = bool((TR[:, :-1].sum() == 0) and (TR[:, -1] == ~TE[:, -1]).all())
    out["no_reward_after_termination"] = bool(all(R[i, first[i] + 1:].sum() == 0 for i in range(batch)))
    out["train_success_once_b0"] = float(SO[:, -1].float().mean())
    # obs scale sanity
    st = obs["states"]
    out["obs_absmax_feat"] = float(st[:, :2048].abs().max()); out["obs_absmax_rest"] = float(st[:, 2048:].abs().max())
    # 2) auto-reset: done slots get new train ids, final_observation present, decision counter restarts
    cfg = make_cfg("train", auto_reset=True, ignore_term=False)
    env2 = V.ResidualVLAEnv(cfg, batch, 0, 1, None)
    env2.reset()
    seen_reset = False
    for d in range(V.T_DEC + 2):
        e_before = env2.eids.clone()
        _, r, te, tr, inf = env2.chunk_step(torch.zeros(batch, 1, 7, device=env2.device))
        dn = (te | tr)[:, 0].cpu()
        if dn.any():
            seen_reset = True
            assert "final_observation" in inf[0]
            assert (env2.dec[dn.to(env2.device)] == 0).all()
            assert not (set(env2.eids[dn].tolist()) & env2.excluded)
            assert (env2.eids[~dn] == e_before[~dn]).all()
    out["auto_reset_ok"] = seen_reset
    del env, env2
    # 3) GRPO groups: same episode id in a group -> identical initial state
    G = 8
    cfg = make_cfg("train", group_size=G)
    env3 = V.ResidualVLAEnv(cfg, batch, 0, 1, None)
    env3.reset()
    ps = V.priv_state(env3.inner)
    cw = ps["carrot_w"].view(-1, G, 3); pw = ps["plate_w"].view(-1, G, 3)
    out["grpo_group_eids_equal"] = bool((env3.eids.view(-1, G) == env3.eids.view(-1, G)[:, :1]).all())
    out["grpo_within_group_max_diff_m"] = float(max((cw - cw[:, :1]).abs().max(), (pw - pw[:, :1]).abs().max()))
    out["grpo_between_group_mean_diff_m"] = float((cw[1:, 0] - cw[:-1, 0]).norm(dim=1).mean())
    ids = env3.inner.env.unwrapped.select_carrot_ids.view(-1, G)
    out["grpo_group_same_object"] = bool((ids == ids[:, :1]).all())
    # 4) residual mapping: tanh / clip squash and expert label inverse
    a = torch.tensor([[0.5, -2.0, 0.0, 0.0, 0.0, 0.0, 3.0]], device=env3.device)
    s = torch.tanh(a)
    out["delta_example_tanh"] = (env3.alpha * s).cpu().tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["b0", "expert", "upper", "units"])
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--a_pos", type=float, default=0.02)
    ap.add_argument("--a_rot", type=float, default=0.1)
    ap.add_argument("--a_grip", type=float, default=2.0)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--out", default="/outputs/StochRL/t022/val")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    if args.mode == "units":
        res = units(args.batch)
    else:
        res = run_eval(args.mode, args.n, args.batch, args.a_pos, args.a_rot, args.a_grip, args.rep)
    res["gpu"] = __import__("os").environ.get("CUDA_VISIBLE_DEVICES")
    name = f"{args.mode}{args.tag}.json"
    Path(args.out, name).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "episodes"}, indent=1))


if __name__ == "__main__":
    main()

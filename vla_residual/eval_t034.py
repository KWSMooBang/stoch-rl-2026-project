"""T034: evaluation-only rollouts with per-decision residual / success logging (no training, no Ray).

Reproduces the RLinf held-out evaluation of the GRPO residual runs (T022 config): 128 held-out episodes in 4 blocks
of 32 env slots, env seed 1000, ignore_terminations=True, 16 decisions, deterministic action u = actor mean,
pi0.5 noise seeded by (episode id, decision), RC_POLICY_MB=8. E0 runs use zero_vlm=True. "b0" = residual 0.
One process: pi0.5 and the ManiSkill scene are built once and shared by all runs.

Per decision (row) it stores: u, residual alpha*tanh(u), base chunk, executed chunk, per-sim-step success, success
after the decision (success_now), ever_success, privileged state before / after the decision (label only; never fed
to the policy), the 44 non-VLM obs dims, and (npz) the true standardized VLM feature (also for E0, where the policy
sees zeros).

  CUDA_VISIBLE_DEVICES=0 python -m vla_residual.eval_t034 --out /outputs/StochRL/t034/raw
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("RC_POLICY_MB", "8")  # same pi0.5 micro-batch as vla_residual/run.sh

from vla_residual import core as V  # noqa: E402
from vla_residual.validate import make_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

RUNS = {  # name -> (cond, seed, run_id)
    "e0-s0": ("E0", 0, "t032-grpo-e0-s0"), "e0-s1": ("E0", 1, "t032-grpo-e0-s1"), "e0-s2": ("E0", 2, "t032-grpo-e0-s2"),
    "e2-s0": ("E2", 0, "t022-grpo-s0"), "e2-s1": ("E2", 1, "t030-grpo-s1"), "e2-s2": ("E2", 2, "t030-grpo-s2"),
    "b0": ("B0", -1, None),
}
RUN_ROOT = Path("/outputs/StochRL/runs")
EVAL_SEED, BATCH = 1000, 32


def ckpt_file(run_id):
    cks = sorted((RUN_ROOT / run_id / "rlinf_logs" / run_id / "checkpoints").glob("global_step_*"),
                 key=lambda p: int(p.name.split("_")[-1]))
    return cks[-1] / "actor/model_state_dict/full_weights.pt"


def load_actor(run_id, device):
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy

    m = MLPPolicy(V.OBS_DIM, V.ACT_DIM, num_action_chunks=1, add_value_head=False, add_q_head=False)
    m.load_state_dict(torch.load(ckpt_file(run_id), map_location="cpu"), strict=True)
    return m.to(device).float().eval()


def priv(env, pre):
    ps = V.priv_state(env.inner)
    off = ps["carrot_w"] - ps["plate_w"]
    d = dict(cp_xy=torch.linalg.norm(off[:, :2], dim=1), cp_dz=off[:, 2], cp_dist=torch.linalg.norm(off, dim=1),
             carrot_z=ps["carrot"][:, 2], ee_x=ps["ee"][:, 0], ee_y=ps["ee"][:, 1], ee_z=ps["ee"][:, 2],
             ee_carrot=torch.linalg.norm(ps["ee"] - ps["carrot"], dim=1),
             ee_plate_xy=torch.linalg.norm(ps["ee"][:, :2] - ps["plate"][:, :2], dim=1),
             ee_dz_plate=ps["ee"][:, 2] - ps["plate"][:, 2], fingers=ps["fingers"], grasp=ps["grasp"])
    return {f"{pre}_{k}": v.detach().cpu().numpy() for k, v in d.items()}


@torch.no_grad()
def rollout(name, n_blocks, n_dec):
    cond, seed, run_id = RUNS[name]
    cfg = make_cfg("eval", n_eval=V.N_EVAL_DEFAULT, auto_reset=False, ignore_term=True, zero_vlm=(cond == "E0"))
    cfg.seed = EVAL_SEED
    env = V.ResidualVLAEnv(cfg, BATCH, 0, 1, None)
    assert env.zero_vlm == (cond == "E0") and env.squash == "tanh"
    actor = load_actor(run_id, env.device) if run_id else None
    steps = []
    orig = env._sim_step

    def sim_step(a, render):  # observe-only hook: executed action + success flag of every sim step
        out = orig(a, render)
        steps.append((a.detach().cpu(), out[4]["success"].detach().cpu().bool()))
        return out

    env._sim_step = sim_step
    rows, feats = [], []
    B = BATCH
    for b in range(n_blocks):
        # RLinf: external full reset for the first block, auto-reset (all slots at once) for the next ones
        obs, _ = env.reset() if b == 0 else env.reset(options={"env_idx": torch.arange(B)})
        eids = env.eids.clone().numpy()
        prev_now = np.zeros(B, bool)
        prev_ever = np.zeros(B, bool)
        for d in range(n_dec):
            st = obs["states"].to(env.device).float()
            if cond == "E0":
                assert bool((st[:, : V.FEAT_DIM] == 0).all())
            u = actor._sample_actions(st)[0] if actor is not None else torch.zeros(B, V.ACT_DIM, device=env.device)
            res = env.alpha * torch.tanh(u)
            a_base = env.a_base.clone()
            feats.append(env.feat.detach().cpu().numpy().astype(np.float16))
            pre = priv(env, "pre")
            steps.clear()
            o, r, te, tr, inf = env.chunk_step(u[:, None])
            obs = o[0]
            post = priv(env, "post")
            ex = torch.stack([s[0] for s in steps], 1)  # [B,K,7]
            ss = torch.stack([s[1] for s in steps], 1).numpy()  # [B,K]
            inf = inf[0]
            now = inf["success"].cpu().numpy().astype(bool)
            ever = inf["episode"]["success_once"].cpu().numpy().astype(bool)
            assert (ss[:, -1] == now).all() and ((prev_ever | ss.any(1)) == ever).all()
            un, rn, bn, exn = u.cpu().numpy(), res.cpu().numpy(), a_base.cpu().numpy(), ex.numpy()
            o44 = st[:, V.FEAT_DIM:].cpu().numpy()
            for i in range(B):
                row = dict(name=name, cond=cond, seed=seed, run_id=run_id or "b0", eid=int(eids[i]), block=b, slot=i,
                           dec=d, success_now=bool(now[i]), success_any=bool(ss[i].any()), ever_success=bool(ever[i]),
                           prev_success_now=bool(prev_now[i]), prev_ever_success=bool(prev_ever[i]),
                           n_succ_steps=int(ss[i].sum()), succ_steps=ss[i].tolist(),
                           u=un[i].tolist(), base_chunk=bn[i].reshape(-1).tolist(), exec_chunk=exn[i].reshape(-1).tolist(),
                           obs44=o44[i].tolist())
                for j, c in enumerate(("x", "y", "z", "rx", "ry", "rz", "grip")):
                    row[f"res_{c}"] = float(rn[i, j])
                    row[f"base_{c}"] = float(bn[i, :, j].mean())
                    row[f"exec_{c}"] = float(exn[i, :, j].mean())
                row["res_pos_norm"] = float(np.linalg.norm(rn[i, :3]))
                row["res_rot_norm"] = float(np.linalg.norm(rn[i, 3:6]))
                row["res_tanh_norm"] = float(np.linalg.norm(np.tanh(un[i])))  # alpha-normalised, 0..sqrt(7)
                for k, v in {**pre, **post}.items():
                    row[k] = v[i].item()
                rows.append(row)
            prev_now, prev_ever = now, ever
        print(f"[{name}] block {b} once {ever.mean():.3f} end {now.mean():.3f}", flush=True)
    env._sim_step = orig
    return rows, np.concatenate(feats, 0), dict(lo=env.lo.cpu().tolist(), hi=env.hi.cpu().tolist(),
                                                alpha=env.alpha.cpu().tolist())


def repro(name, rows):
    """Episode-level comparison with the stored final evaluation (or t022/val/b0.json for b0)."""
    cond, seed, run_id = RUNS[name]
    last = {}
    for r in rows:
        last[r["eid"]] = r  # rows are in decision order -> last write = final decision
    if run_id:
        m = json.loads((RUN_ROOT / run_id / "metrics.json").read_text())
        ref = {e["eid"]: e for e in m["final_eval_episodes"]}
        ref_once, ref_end = m["heldout_success_once"], m["heldout_success_at_end"]
    else:
        m = json.loads(Path("/outputs/StochRL/t022/val/b0.json").read_text())
        ref = {e["eid"]: e for e in m["episodes"]}
        ref_once, ref_end = m["success_once"], m["success_at_end"]
    common = [e for e in last if e in ref]
    once = np.array([last[e]["ever_success"] for e in common]); end = np.array([last[e]["success_now"] for e in common])
    ro = np.array([ref[e]["success_once"] for e in common]); re = np.array([ref[e]["success_at_end"] for e in common])
    return dict(name=name, n=len(last), n_common=len(common),
                once=float(np.mean([v["ever_success"] for v in last.values()])),
                end=float(np.mean([v["success_now"] for v in last.values()])),
                ref_once=float(ref_once), ref_end=float(ref_end),
                agree_once=float((once == ro).mean()), agree_end=float((end == re).mean()),
                ckpt=str(ckpt_file(run_id)) if run_id else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="e0-s1,e2-s1,b0,e0-s0,e2-s0,e0-s2,e2-s2")
    ap.add_argument("--blocks", type=int, default=V.N_EVAL_DEFAULT // BATCH)
    ap.add_argument("--n_dec", type=int, default=V.T_DEC)
    ap.add_argument("--out", default="/outputs/StochRL/t034/raw")
    ap.add_argument("--budget_s", type=float, default=1e9, help="do not start a run if the projected end exceeds this")
    a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "T034: physical GPU 0 only"
    import pandas as pd

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); per = []
    for name in a.runs.split(","):
        if (out / f"{name}.parquet").exists():
            print(f"[{name}] exists, skip", flush=True); continue
        est = max(per) if per else 300.0
        if time.time() - t0 + est > a.budget_s:
            print(f"[{name}] SKIP: budget ({time.time()-t0:.0f}s used, est {est:.0f}s)", flush=True); continue
        t1 = time.time()
        rows, feat, meta = rollout(name, a.blocks, a.n_dec)
        rp = repro(name, rows); rp["wall_s"] = time.time() - t1; rp["gpu"] = os.environ["CUDA_VISIBLE_DEVICES"]
        rp.update(meta); rp["policy_mb"] = os.environ["RC_POLICY_MB"]
        pd.DataFrame(rows).to_parquet(out / f"{name}.parquet")
        np.savez_compressed(out / f"{name}_feat.npz", feat=feat)
        (out / f"{name}_repro.json").write_text(json.dumps(rp, indent=1))
        if len(per) or name == a.runs.split(",")[0]:
            per.append(time.time() - t1)
        print(json.dumps(rp), flush=True)
    print(f"total wall {time.time()-t0:.0f}s peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()

"""T039 checks for noise-space steering (vla_residual.noise_steer). Single process, no Ray, one shared scene.

  --mode units  prefix-cache denoise == pi0.5 sample_actions (same noise), add-mode u=0 returns the B0 chunk,
                u != 0 changes the chunk, obs format unchanged (train split GRPO setting and eval split)
  --mode sens   action change caused by a noise shift beta*s (s in {-1,+1}^7 and exploration-size s), per beta,
                on states visited by a B0 eval rollout (pos / rot in env units, gripper sign flips)
  --mode eval   held-out episodes, deterministic u = 0 unless noted:
                  none      B0 (no steering)            add      add-mode, u = 0 (must be B0)
                  replace   DSRL-style z = beta*tanh(0) = "noise 0" (NOT B0; what u = 0 means there)
                  explore   add-mode with u ~ N(0, exp(-1.5)^2) per decision (= GRPO train-time exploration at init)
  CUDA_VISIBLE_DEVICES=0 python -m vla_residual.validate_t039 --mode units --out /outputs/StochRL/t039/val
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from vla_residual import core as V
from vla_residual.validate import make_cfg

ALPHA_POS = 0.02


def mk_env(split, batch, mode, beta, n_eval=128, seed=0):
    kw = dict(group_size=8, auto_reset=False, ignore_term=False) if split == "train" else dict(ignore_term=True)
    cfg = make_cfg(split, n_eval=n_eval, noise_steer=dict(mode=mode, beta=beta, dims=7), **kw)
    cfg.seed = seed
    return V.ResidualVLAEnv(cfg, batch, 0, 1, None)


def units(batch):
    out = {}
    for split in ("train", "eval"):
        env = mk_env(split, batch, "add", 1.0, n_eval=batch)
        obs, _ = env.reset()
        r = dict(obs_shape=list(obs["states"].shape), obs_dim_ok=obs["states"].shape[1] == V.OBS_DIM)
        g = torch.Generator().manual_seed(7)
        diffs, ident, changed, n_steps = [], [], [], 6
        for d in range(n_steps):
            # 1) cached-prefix denoise with the env's own noise reproduces the B0 chunk of pi0.5 sample_actions
            a_re = env.vla.denoise(env.z_ref).to(env.device)
            diffs.append(float((a_re - env.a_base).abs().max()))
            # 2) add-mode, s = 0 -> the B0 chunk object itself
            s0 = torch.zeros(batch, 7, device=env.device)
            ident.append(env.steered_chunk(s0) is env.a_base)
            # 3) s != 0 changes the chunk
            s1 = torch.tanh(torch.randn(batch, 7, generator=g)).to(env.device)
            changed.append(float((env.steered_chunk(s1) - env.a_base).abs().amax((1, 2)).gt(1e-6).float().mean()))
            u = 0.3 * torch.randn(batch, 1, 7, generator=g)
            env.chunk_step(u.to(env.device))
        r.update(denoise_vs_sample_actions_maxabs=diffs, denoise_bitwise=all(x == 0 for x in diffs),
                 u0_returns_b0_chunk=all(ident), frac_envs_changed_by_s=changed,
                 vla_cache_owner_ok=env.vla._cache_owner == id(env))
        out[split] = r
        print(split, json.dumps(r), flush=True)
        del env
    # 4) replace-mode with s=0 == pi0.5 with all-zero noise
    env = mk_env("eval", batch, "replace", 1.0, n_eval=batch)
    env.reset()
    a_z0 = env.steered_chunk(torch.zeros(batch, 7, device=env.device))
    out["replace_u0_is_noise0"] = float((a_z0 - env.vla.denoise(torch.zeros_like(env.z_ref)).to(env.device)).abs().max()) == 0.0
    out["replace_u0_vs_b0_pos_absdiff_m"] = float((a_z0 - env.a_base)[..., :3].abs().mean())
    return out


def sens(batch, betas, n_dir=8):
    env = mk_env("eval", batch, "add", 1.0, n_eval=batch)
    env.reset()
    g = torch.Generator().manual_seed(11)
    dirs = (torch.randint(0, 2, (n_dir, 7), generator=g) * 2 - 1).float()
    rows = []
    for d in range(V.T_DEC):
        if d in (0, 3, 6, 9, 12):
            for beta in betas:
                env.ns_beta = beta
                for kind in ("sign", "explore"):
                    dp, dr, gf = [], [], []
                    for j in range(n_dir):
                        if kind == "sign":
                            s = dirs[j].expand(batch, 7)
                        else:
                            s = torch.tanh(math.exp(-1.5) * torch.randn(batch, 7, generator=g))
                        a = env.steered_chunk(s.to(env.device))
                        dlt = a - env.a_base
                        dp.append(float(dlt[..., :3].abs().mean()))
                        dr.append(float(dlt[..., 3:6].abs().mean()))
                        gf.append(float((torch.sign(a[..., 6]) != torch.sign(env.a_base[..., 6])).any(1).float().mean()))
                    rows.append(dict(dec=d, beta=beta, kind=kind, dpos_m=np.mean(dp), drot_rad=np.mean(dr),
                                     grip_flip_frac=np.mean(gf)))
            env.ns_beta = 1.0
        env.chunk_step(torch.zeros(batch, 1, 7, device=env.device))
    base_pos = None
    summ = {}
    for beta in betas:
        for kind in ("sign", "explore"):
            rr = [r for r in rows if r["beta"] == beta and r["kind"] == kind]
            summ[f"{kind}_b{beta}"] = dict(dpos_m=float(np.mean([r["dpos_m"] for r in rr])),
                                           dpos_over_alpha=float(np.mean([r["dpos_m"] for r in rr]) / ALPHA_POS),
                                           drot_rad=float(np.mean([r["drot_rad"] for r in rr])),
                                           grip_flip_frac=float(np.mean([r["grip_flip_frac"] for r in rr])))
            print(kind, beta, json.dumps(summ[f"{kind}_b{beta}"]), flush=True)
    return dict(summary=summ, rows=rows)


def run_eval(cond, n, batch, beta, seed=0):
    mode = dict(none="none", add="add", replace="replace", explore="add")[cond]
    env = mk_env("eval", batch, mode, beta, n_eval=n)
    g = torch.Generator().manual_seed(1234 + seed)
    rows, t0 = [], time.time()
    for b in range(n // batch):
        # reset() without options rewinds the eval id cursor -> use env_idx for later batches (T029 gotcha)
        env.reset() if b == 0 else env.reset(options={"env_idx": torch.arange(batch)})
        eids = env.eids.clone()
        for d in range(V.T_DEC):
            u = torch.zeros(batch, 1, 7)
            if cond == "explore":
                u = math.exp(-1.5) * torch.randn(batch, 1, 7, generator=g)
            o, r, te, tr, inf = env.chunk_step(u.to(env.device))
        ep = inf[0]["episode"]
        for i in range(batch):
            row = dict(eid=int(eids[i]), success_once=bool(ep["success_once"][i]),
                       success_at_end=bool(ep["success_at_end"][i]))
            if mode != "none":
                row.update(env._ns_row(i))
            rows.append(row)
        print(f"[{cond}] batch {b} once {ep['success_once'].float().mean():.3f} end "
              f"{ep['success_at_end'].float().mean():.3f} {time.time()-t0:.0f}s", flush=True)
    so = float(np.mean([r["success_once"] for r in rows])); se = float(np.mean([r["success_at_end"] for r in rows]))
    return dict(cond=cond, mode=mode, beta=beta, n=len(rows), success_once=so, success_at_end=se,
                wall_s=time.time() - t0, peak_gb=torch.cuda.max_memory_allocated() / 1e9, episodes=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["units", "sens", "eval"])
    ap.add_argument("--conds", default="none,add,none")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--betas", default="0.5,1,1.5,2,3")
    ap.add_argument("--out", default="/outputs/StochRL/t039/val")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.mode == "units":
        res = units(min(a.batch, 16))
        res["wall_s"] = time.time() - t0
        json.dump(res, open(f"{a.out}/units_t039.json", "w"), indent=1)
    elif a.mode == "sens":
        res = sens(a.batch, [float(x) for x in a.betas.split(",")])
        res["wall_s"] = time.time() - t0
        json.dump(res, open(f"{a.out}/sens_t039.json", "w"), indent=1)
    else:
        allr = {}
        for k, c in enumerate(a.conds.split(",")):
            key = c if c not in allr else f"{c}_{k}"
            allr[key] = run_eval(c, a.n, a.batch, a.beta)
            json.dump(allr[key], open(f"{a.out}/eval_{key}_b{a.beta}.json", "w"), indent=1)
        ks = list(allr)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                x = {r["eid"]: r for r in allr[ks[i]]["episodes"]}
                y = {r["eid"]: r for r in allr[ks[j]]["episodes"]}
                com = sorted(set(x) & set(y))
                print(f"agree {ks[i]} vs {ks[j]}: once "
                      f"{np.mean([x[e]['success_once'] == y[e]['success_once'] for e in com]):.3f} end "
                      f"{np.mean([x[e]['success_at_end'] == y[e]['success_at_end'] for e in com]):.3f} n={len(com)}")
        print(json.dumps({k: (v["success_once"], v["success_at_end"]) for k, v in allr.items()}))
    print(f"wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

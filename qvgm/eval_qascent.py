"""T040 closed-loop Q-ascent (test-time, no learning) on the T022 held-out 128 + action-expert memory probe.

Same procedure as vla_residual/eval_t034.py (B0 path): env seed 1000, 32 slots x 4 blocks, 16 decisions,
ignore_terminations, pi0.5 noise fixed per (episode id, decision). One process, physical GPU 1, conditions run
sequentially sharing pi0.5 and the scene.  Per decision the base env chunk (5x7) is mapped to model-normalized space,
modified (prereg 2), mapped back, clipped to the action bounds and executed; the next decision is pi0.5 again.
  QA(alpha): j<=3 steps a += alpha*sqrt(35)*g/|g|, keep only if mean-ensemble Q rises (keep-best gate)
  RG(alpha): same loop along a fixed random unit direction (seed eid*64+dec), with the Q gate
  RN(alpha): 3 steps along the same random direction, no gate

  CUDA_VISIBLE_DEVICES=1 python -m qvgm.eval_qascent --out /workspace/stochrl_t040/closed_loop
  CUDA_VISIBLE_DEVICES=1 python -m qvgm.eval_qascent --memprobe_only --out ...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("RC_POLICY_MB", "8")

from vla_residual import core as V  # noqa: E402
from vla_residual.validate import make_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from qvgm.critic import QEns, VNet  # noqa: E402

EVAL_SEED, BATCH, J = 1000, 32, 3
CONDS = {"b0": ("B0", 0.0), "qa0.05": ("QA", 0.05), "qa0.1": ("QA", 0.1), "rg0.1": ("RG", 0.1), "rn0.1": ("RN", 0.1),
         "rg0.05": ("RG", 0.05), "rn0.05": ("RN", 0.05), "b0rep": ("B0", 0.0)}


def smi():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader", "-i", "1"],
                              capture_output=True, text=True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return str(e)


class Critic:
    def __init__(self, path, dev):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.q = QEns().to(dev); self.q.load_state_dict(ck["q"]); self.q.eval()
        self.v = VNet().to(dev); self.v.load_state_dict(ck["v"]); self.v.eval()
        for p in [*self.q.parameters(), *self.v.parameters()]:
            p.requires_grad_(False)
        self.s = torch.as_tensor(ck["act_scale"], dtype=torch.float32, device=dev)
        self.b = torch.as_tensor(ck["act_bias"], dtype=torch.float32, device=dev)

    def to_model(self, a_env):  # [B,5,7] -> [B,35]
        return ((a_env - self.b) / self.s).reshape(a_env.shape[0], -1)

    def to_env(self, a_m):
        return a_m.reshape(-1, 5, 7) * self.s + self.b

    def qbar(self, z, a):
        return self.q(z, a).mean(0)

    def grad(self, z, a):
        with torch.enable_grad():
            a = a.detach().requires_grad_(True)
            qv = self.qbar(z, a)
            g, = torch.autograd.grad(qv.sum(), a)
        return qv.detach(), g


def modify(kind, alpha, crit, z, a0, eids, dec):
    """Returns modified model-space action and per-env diagnostics."""
    B = a0.shape[0]
    step = alpha * (a0.shape[1] ** 0.5)
    q0 = crit.qbar(z, a0)
    if kind == "B0":
        return a0, dict(n_acc=torch.zeros(B), q0=q0, q1=q0)
    if kind in ("RG", "RN"):
        r = torch.stack([torch.randn(a0.shape[1], generator=torch.Generator().manual_seed(int(e) * 64 + dec))
                         for e in eids]).to(a0.device)
        r = r / r.norm(dim=1, keepdim=True)
    a, q, n_acc = a0.clone(), q0.clone(), torch.zeros(B, device=a0.device)
    active = torch.ones(B, dtype=torch.bool, device=a0.device)
    for _ in range(J):
        if kind == "QA":
            _, g = crit.grad(z, a)
            d = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
        else:
            d = r
        cand = a + step * d
        qc = crit.qbar(z, cand)
        acc = active & ((qc > q) if kind in ("QA", "RG") else torch.ones_like(active))
        a = torch.where(acc[:, None], cand, a)
        q = torch.where(acc, qc, q)
        n_acc += acc.float()
        active = acc  # stop at the first rejected step
    return a, dict(n_acc=n_acc.cpu(), q0=q0, q1=q)


@torch.no_grad()
def rollout(name, crit, n_blocks, n_dec, env):
    kind, alpha = CONDS[name]
    steps = []
    orig = env._sim_step

    def sim_step(a, render):
        out = orig(a, render)
        steps.append(a.detach().clone())
        return out

    env._sim_step = sim_step
    rows, B = [], BATCH
    for b in range(n_blocks):
        obs, _ = env.reset() if b == 0 else env.reset(options={"env_idx": torch.arange(B)})
        eids = env.eids.clone().numpy()
        for d in range(n_dec):
            a_base = env.a_base.clone()  # env units [B,5,7]
            z = env.feat.float()
            a0 = crit.to_model(a_base)
            a1, info = modify(kind, alpha, crit, z, a0, eids, d)
            a_env = (a_base if kind == "B0" else crit.to_env(a1)).clamp(env.lo, env.hi)
            env.a_base = a_env.clone()  # executed as a_base + alpha*tanh(0) = a_env
            steps.clear()
            o, r, te, tr, inf = env.chunk_step(torch.zeros(B, 1, V.ACT_DIM, device=env.device))
            ex = torch.stack(steps, 1)
            assert torch.allclose(ex, a_env, atol=1e-6)
            inf = inf[0]
            now = inf["success"].cpu().numpy().astype(bool)
            ever = inf["episode"]["success_once"].cpu().numpy().astype(bool)
            dm = (crit.to_model(a_env) - a0)
            dxyz = (a_env - a_base)[:, :, :3].norm(dim=2).mean(1) * 1000  # mm/step
            for i in range(B):
                rows.append(dict(cond=name, eid=int(eids[i]), block=b, slot=i, dec=d, success_now=bool(now[i]),
                                 ever_success=bool(ever[i]), n_acc=int(info["n_acc"][i]),
                                 q0=float(info["q0"][i]), q1=float(info["q1"][i]),
                                 d_rms=float(dm[i].pow(2).mean().sqrt()), d_xyz_mm=float(dxyz[i]),
                                 d_grip=float((a_env - a_base)[i, :, 6].abs().mean()),
                                 clip=float((crit.to_env(a1)[i] != a_env[i]).float().mean())))
        print(f"[{name}] block {b} once {ever.mean():.3f} end {now.mean():.3f}", flush=True)
    env._sim_step = orig
    return rows


def memprobe(env, out):
    """Peak VRAM of forward+backward+Adam step on the action expert (VLM frozen), batch {4,8}, full / LoRA r=16."""
    import dataclasses

    import torch.nn as nn
    from torch.utils._pytree import tree_map

    m = env.vla.pol.m
    return memprobe_model(m, m.env_obs_to_observation(env._last_wobs), out, dict(scene_loaded=True))


def memprobe_model(m, observation, out, extra, path="kv"):
    import dataclasses

    import torch.nn as nn
    from torch.utils._pytree import tree_map

    res = dict(gpu_smi_before=smi(), model_dtype=str(next(m.llm.parameters()).dtype),
               embed_dtype=str(m.embed_dtype), runs=[], path=path, **extra)
    llm = m.llm

    def expert_params():
        ps = []
        for blk in llm.layers:
            for sub in (blk.pre_attention_norms[1], blk.pre_ffw_norms[1], blk.mlps[1]):
                ps += list(sub.parameters())
            for pl in (blk.attn.q_proj, blk.attn.k_proj, blk.attn.v_proj, blk.attn.o_proj):
                if pl[1] is not None:
                    ps += list(pl[1].parameters())
        if llm.final_norms[1] is not None:
            ps += list(llm.final_norms[1].parameters())
        for nm in ("action_in_proj", "action_out_proj", "state_proj", "action_time_mlp_in", "action_time_mlp_out",
                   "time_mlp_in", "time_mlp_out"):
            mod = getattr(m, nm, None)
            if mod is not None:
                ps += list(mod.parameters())
        return ps

    class LoRALinear(nn.Module):
        def __init__(self, base, r=16):
            super().__init__()
            self.base = base
            dt, dev = base.weight.dtype, base.weight.device
            self.A = nn.Parameter(torch.randn(r, base.in_features, dtype=dt, device=dev) / base.in_features ** 0.5)
            self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=dt, device=dev))

        def forward(self, x):
            return self.base(x) + (x @ self.A.T.to(x.dtype)) @ self.B.T.to(x.dtype)

    class LoRAFFN(nn.Module):
        def __init__(self, base, r=16):
            super().__init__()
            self.base = base
            F_, H = base.features, base.hidden_dim
            dt, dev = base.w_linear.dtype, base.w_linear.device
            self.Ag = nn.Parameter(torch.randn(2, F_, r, dtype=dt, device=dev) / F_ ** 0.5)
            self.Bg = nn.Parameter(torch.zeros(2, r, H, dtype=dt, device=dev))
            self.Al = nn.Parameter(torch.randn(H, r, dtype=dt, device=dev) / H ** 0.5)
            self.Bl = nn.Parameter(torch.zeros(r, F_, dtype=dt, device=dev))

        def forward(self, x):
            from rlinf.models.embodiment.openpi.modules.utils import gelu_glu

            dt = x.dtype
            wg = self.base.w_gating.to(dt) + (self.Ag.to(dt) @ self.Bg.to(dt))
            wl = self.base.w_linear.to(dt) + self.Al.to(dt) @ self.Bl.to(dt)
            return torch.matmul(gelu_glu(torch.matmul(x, wg[0]), torch.matmul(x, wg[1])), wl)

    def set_lora(on):
        for blk in llm.layers:
            for pl in (blk.attn.q_proj, blk.attn.k_proj, blk.attn.v_proj, blk.attn.o_proj):
                if pl[1] is None:
                    continue
                if on and isinstance(pl[1], nn.Linear):
                    pl[1] = LoRALinear(pl[1])
                elif not on and isinstance(pl[1], LoRALinear):
                    pl[1] = pl[1].base
            if on and not isinstance(blk.mlps[1], LoRAFFN):
                blk.mlps[1] = LoRAFFN(blk.mlps[1])
            elif not on and isinstance(blk.mlps[1], LoRAFFN):
                blk.mlps[1] = blk.mlps[1].base

    def sub_obs(B):
        return dataclasses.replace(observation, **{
            f.name: tree_map(lambda x: x[:B] if isinstance(x, torch.Tensor) else x, getattr(observation, f.name))
            for f in dataclasses.fields(observation)})

    all_p = list(m.parameters())
    for mode in ("full", "lora16"):
        for B in (4, 8):
            for p in all_p:
                p.requires_grad_(False)
            if mode == "full":
                train_p = expert_params()
            else:
                set_lora(True)
                train_p = [p for blk in llm.layers for mod in (blk.mlps[1], blk.attn.q_proj[1], blk.attn.k_proj[1],
                                                                  blk.attn.v_proj[1], blk.attn.o_proj[1])
                           if mod is not None for n_, p in mod.named_parameters() if not n_.startswith("base.")]
            for p in train_p:
                p.requires_grad_(True)
            ntrain = sum(p.numel() for p in train_p)
            opt = torch.optim.Adam(train_p, lr=1e-5)
            torch.cuda.synchronize(); torch.cuda.empty_cache()
            base_alloc = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            err = None
            try:
                ob = sub_obs(B)
                acts = torch.randn(B, m.action_horizon, m.action_dim, device=m.device)
                if path == "joint":  # compute_loss: prefix+suffix in one forward (keeps VLM activations)
                    with torch.enable_grad():
                        loss = m.compute_loss(ob, acts, train=False).mean()
                        loss.backward()
                else:  # Q-VGM-style: frozen VLM prefix KV cache under no_grad, grad only through the expert suffix
                    from rlinf.models.embodiment.openpi.modules import model as M

                    with torch.no_grad():
                        obp = M._observation_to_dtype(M.preprocess_observation(ob, train=False), m.embed_dtype)
                        _, pmask, kv = m.build_prefix_cache(obp)
                    with torch.enable_grad():
                        noise = torch.randn_like(acts)
                        tt = torch.rand(B, device=m.device) * 0.999 + 0.001
                        x_t = tt[:, None, None] * noise + (1 - tt[:, None, None]) * acts
                        v = m.velocity_from_suffix(m.run_suffix(obp, x_t, tt, kv, pmask))
                        loss = ((v - (noise - acts)) ** 2).mean()
                        loss.backward()
                opt.step()
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                smi_peak = smi()
            except torch.cuda.OutOfMemoryError as e:  # noqa: PERF203
                err, peak, smi_peak = str(e)[:200], torch.cuda.max_memory_allocated(), smi()
            st_bytes = sum(t.numel() * t.element_size() for s in opt.state.values() for t in s.values()
                           if torch.is_tensor(t))
            res["runs"].append(dict(path=path, mode=mode, batch=B, n_trainable=ntrain, base_alloc_GB=base_alloc / 2 ** 30,
                                    peak_alloc_GB=peak / 2 ** 30, extra_over_base_GB=(peak - base_alloc) / 2 ** 30,
                                    adam_state_GB=st_bytes / 2 ** 30, smi_at_peak=smi_peak, error=err))
            print(json.dumps(res["runs"][-1]), flush=True)
            opt.zero_grad(set_to_none=True)
            for p in train_p:
                p.grad = None
            del opt
            for p in all_p:
                p.requires_grad_(False)
            if mode == "lora16":
                set_lora(False)
            torch.cuda.empty_cache()
    res["gpu_smi_after"] = smi()
    fn = "memory_probe.json" if extra.get("scene_loaded") else "memory_probe_noscene.json"
    (out / fn).write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="b0,qa0.05,qa0.1,rg0.1,rn0.1,rg0.05,rn0.05")
    ap.add_argument("--critic", default="/workspace/stochrl_t040/critic/all_s0.pt")
    ap.add_argument("--blocks", type=int, default=V.N_EVAL_DEFAULT // BATCH)
    ap.add_argument("--n_dec", type=int, default=V.T_DEC)
    ap.add_argument("--out", default="/workspace/stochrl_t040/closed_loop")
    ap.add_argument("--budget_s", type=float, default=1e9)
    ap.add_argument("--memprobe", action="store_true")
    ap.add_argument("--memprobe_only", action="store_true")
    a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "T040: physical GPU 1 only"
    import pandas as pd

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg = make_cfg("eval", n_eval=V.N_EVAL_DEFAULT, auto_reset=False, ignore_term=True)
    cfg.seed = EVAL_SEED
    env = V.ResidualVLAEnv(cfg, BATCH, 0, 1, None)
    assert env.squash == "tanh" and not env.zero_vlm
    orig_make = env._make_obs

    def make_obs(wobs, idx=None):  # keep the last wrapped obs for the memory probe
        env._last_wobs = wobs
        return orig_make(wobs, idx)

    env._make_obs = make_obs
    crit = Critic(a.critic, env.device)
    print(f"[t040] load {time.time()-t0:.0f}s smi {smi()}", flush=True)
    per = []
    if not a.memprobe_only:
        for name in a.conds.split(","):
            if (out / f"{name}.parquet").exists():
                print(f"[{name}] exists, skip", flush=True); continue
            est = max(per) if per else 400.0
            if time.time() - t0 + est > a.budget_s:
                print(f"[{name}] SKIP: budget ({time.time()-t0:.0f}s used, est {est:.0f}s)", flush=True); continue
            t1 = time.time()
            rows = rollout(name, crit, a.blocks, a.n_dec, env)
            pd.DataFrame(rows).to_parquet(out / f"{name}.parquet")
            per.append(time.time() - t1)
            print(json.dumps(dict(cond=name, wall_s=round(per[-1], 1), smi=smi(),
                                  peak_GB=torch.cuda.max_memory_allocated() / 2 ** 30)), flush=True)
    if a.memprobe or a.memprobe_only:
        if not hasattr(env, "_last_wobs"):
            env.reset()
        memprobe(env, out)
    print(f"[t040] total wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

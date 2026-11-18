"""Smoke: (1) ManiSkill core PickCube-v1, (2) pi0.5 load + frozen/hook invariance, (3) 1 episode on 25Main-v3.

usage: CUDA_VISIBLE_DEVICES=0 python -m repcritic.smoke
"""
from __future__ import annotations

import json
import sys

from repcritic import rc_common as C

C.setup_env_vars()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402


def pickcube():
    import mani_skill.envs  # noqa: F401

    env = gym.make("PickCube-v1", num_envs=1, obs_mode="rgb", reward_mode="dense")
    obs, info = env.reset(seed=42)
    for _ in range(5):
        obs, r, te, tr, info = env.step(env.action_space.sample())
    rgb = obs["sensor_data"]["base_camera"]["rgb"]
    out = dict(action_space=str(env.action_space), rgb_shape=list(rgb.shape), rgb_mean=float(rgb.float().mean()),
               reward=float(r.mean()))
    env.close()
    return out


def hook_invariance(pol, obs):
    """Forward hooks on VLM / action-expert layers must not change actions."""
    m = pol.m
    g = torch.Generator().manual_seed(7)
    nz = pol.noise(obs["states"].shape[0], g)
    a0, _ = pol.act(obs, nz)
    feats = {}
    handles = []
    layers = [(n, mod) for n, mod in m.named_modules() if n.endswith("layers.9") or n.endswith("layers.17")]
    def mk(n):
        def hook(mod, i, o):
            x = o
            while isinstance(x, (tuple, list)):
                x = next((e for e in x if e is not None), None)
            if isinstance(x, torch.Tensor):
                feats.setdefault(n, x.detach().float().cpu())
            return None  # observe only
        return hook

    for n, mod in layers:
        handles.append(mod.register_forward_hook(mk(n)))
    a1, _ = pol.act(obs, nz)
    for h in handles:
        h.remove()
    a2, _ = pol.act(obs, nz)
    return dict(hooked_layers=[n for n, _ in layers], feat_shapes={k: list(v.shape) if hasattr(v, "shape") else str(type(v)) for k, v in feats.items()},
                max_abs_diff_hook=float((a0 - a1).abs().max()), max_abs_diff_after=float((a0 - a2).abs().max()),
                any_requires_grad=any(p.requires_grad for p in m.parameters()), training=m.training)


def main():
    if "--pickcube" in sys.argv:
        r = pickcube(); print(json.dumps(r, indent=1))
        (C.OUT_BIG / "smoke_pickcube.json").write_text(json.dumps(r, indent=1)); return
    res = {}
    t0 = C.now()
    pol = C.Policy(C.build_policy())
    res["policy_load_s"] = C.now() - t0
    n_params = sum(p.numel() for p in pol.m.parameters())
    res["n_params_B"] = n_params / 1e9
    env = C.build_env(2)
    base = env.env.unwrapped
    obs, info = env.reset(seed=0, options={"episode_id": torch.tensor([0, 1], device=env.device)})
    res["obs_keys"] = {k: (list(v.shape) if hasattr(v, "shape") else v) for k, v in obs.items()}
    res["control_mode"] = str(base.control_mode)
    res["action_space"] = str(env.env.action_space)
    res["total_num_trials"] = int(base.total_num_trials)
    res["hook"] = hook_invariance(pol, obs)
    g = torch.Generator().manual_seed(0)
    succ = torch.zeros(2, dtype=torch.bool)
    for d in range(C.MAX_STEPS // C.K):
        acts, macts = pol.act(obs, pol.noise(2, g))
        if d == 0:
            res["env_action_chunk0"] = acts[0].tolist()
            res["model_action_shape"] = list(macts.shape)
        for k in range(C.K):
            obs, r, te, tr, info = env.step(acts[:, k], auto_reset=False)
            succ |= info["success"].cpu()
    res["episode_success_once"] = succ.tolist()
    res["peak_mem_gb"] = C.gpu_mem_gb()
    print(json.dumps(res, indent=1, default=str))
    (C.OUT_BIG / "smoke.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()

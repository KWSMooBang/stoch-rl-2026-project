"""Shared adapter: RLinf ManiSkill PutOnPlateInScene25Main-v3 + frozen pi0.5 SFT.

Single-process (no Ray). Mirrors RLinf's maniskill_ppo_openpi_pi05 eval path:
  env: ManiskillEnv default obs wrap (3rd_view_camera rgb, robot qpos, instruction)
  policy: Pi0Eval (deterministic Euler ODE from Gaussian initial noise)
          H=8 (action_horizon), K=5 (num_action_chunks), 4 denoise steps.
Stochasticity of the policy = initial noise only; we pass it explicitly.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RLINF = REPO / "third_party" / "RLinf"
OUT_BIG = Path("/workspace/stochrl_t016")
OUT_SUM = Path("/outputs/StochRL/t016")

CKPT_ID = "RLinf/RLinf-Pi05-ManiSkill-25Main-SFT"
CKPT_REV = "eb0e2c90726f7f0bdf99e8c9b6656917f0eb5030"
ENV_ID = "PutOnPlateInScene25Main-v3"
H = 8
K = 5
NUM_DENOISE = 4
MAX_STEPS = 80


def setup_env_vars():
    icd = str(REPO / "repcritic/vulkan/nvidia_icd.json")
    os.environ["VK_ICD_FILENAMES"] = icd
    os.environ["VK_DRIVER_FILES"] = icd
    os.environ.setdefault("HF_HOME", "/cache/huggingface")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("EMBODIED_PATH", str(RLINF / "examples/embodiment"))
    # GPU guard: only physical GPU 0 or 1 allowed (human instruction).
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd not in ("0", "1"):
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must be 0 or 1, got {cvd!r}")
    if str(RLINF) not in sys.path:
        sys.path.insert(0, str(RLINF))


def ckpt_path() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(CKPT_ID, revision=CKPT_REV)


def env_cfg(num_envs: int, obj_set: str = "train", max_steps: int = MAX_STEPS):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "env_type": "maniskill",
            "total_num_envs": num_envs,
            "auto_reset": False,
            "ignore_terminations": True,
            "use_rel_reward": True,
            "seed": 0,
            "group_size": 1,
            "use_fixed_reset_state_ids": False,
            "max_steps_per_rollout_epoch": max_steps,
            "max_episode_steps": max_steps,
            "video_cfg": {"save_video": False, "info_on_video": False, "video_base_dir": str(OUT_BIG / "videos")},
            "enable_offload": False,
            "init_params": {
                "id": ENV_ID,
                "num_envs": None,
                "obs_mode": "rgb+segmentation",
                "control_mode": None,
                "sim_backend": "gpu",
                "sim_config": {"sim_freq": 500, "control_freq": 5},
                "max_episode_steps": max_steps,
                "sensor_configs": {"shader_pack": "default"},
                "render_mode": "all",
                "obj_set": obj_set,
                "use_multiple_plates": False,
            },
        }
    )


def build_env(num_envs: int, obj_set: str = "train"):
    from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv

    cfg = env_cfg(num_envs, obj_set)
    env = ManiskillEnv(cfg, num_envs=num_envs, seed_offset=0, total_num_processes=1,
                       worker_info=None, record_metrics=True)
    return env


def model_cfg(path: str):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "model_path": path,
            "model_type": "openpi",
            "precision": None,
            "pi05": True,
            "is_lora": False,
            "num_action_chunks": K,
            "action_dim": 7,
            "use_proprio": True,
            "num_steps": NUM_DENOISE,
            "add_value_head": False,
            "openpi": {
                "task": "eval",
                "config_name": "pi05_maniskill",
                "num_images_in_input": 1,
                "action_chunk": K,
                "num_steps": NUM_DENOISE,
                "action_horizon": H,
                "action_env_dim": 7,
                "model_action_dim": 32,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "max_token_len": 200,
                "torch_compile": False,
                "add_value_head": False,
                "value_after_vlm": True,
            },
        }
    )


def build_policy(device="cuda"):
    import torch
    from rlinf.models.embodiment.openpi import get_model

    model = get_model(model_cfg(ckpt_path()))
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    assert not model.training
    return model


class Policy:
    """Frozen pi0.5 wrapper with explicit initial-noise control."""

    def __init__(self, model):
        import torch

        self.m = model
        self.device = next(model.parameters()).device
        self.noise_shape = (model.action_horizon, model.action_dim)

    def noise(self, B, gen):
        import torch

        return torch.randn(B, *self.noise_shape, generator=gen, device="cpu").to(self.device)

    def act(self, obs, noise, full_h=False):
        """Returns env actions [B, K, 7] (or [B, H, 7] if full_h) and model actions [B, H, 32]."""
        import torch
        from rlinf.models.embodiment.openpi.transforms.apply import apply_output_transform

        import dataclasses

        from torch.utils._pytree import tree_map

        with torch.no_grad():
            observation = self.m.env_obs_to_observation(obs)
            B = observation.state.shape[0]
            mb = int(os.environ.get("RC_POLICY_MB", "0")) or B
            outs = []
            for s in range(0, B, mb):
                sl = slice(s, min(B, s + mb))
                sub = dataclasses.replace(observation, **{
                    f.name: tree_map(lambda x: x[sl] if isinstance(x, torch.Tensor) else x,
                                     getattr(observation, f.name))
                    for f in dataclasses.fields(observation)})
                outs.append(self.m.sample_actions(sub, num_steps=self.m.num_steps, noise=noise[sl]))
            model_actions = torch.cat(outs, 0)
            out = apply_output_transform(
                self.m._output_transform_fn,
                {"actions": model_actions, "state": observation.state},
                action_chunk=None,
            )["actions"].to(torch.float32)
        acts = out if full_h else out[:, :K]
        return acts, model_actions


def obs_select(obs, idx):
    """Select a subset of envs from a wrapped obs dict."""
    out = {}
    for k, v in obs.items():
        if k == "task_descriptions":
            out[k] = [v[i] for i in idx]
        elif v is None:
            out[k] = None
        else:
            out[k] = v[idx]
    return out


def obs_repeat(obs, i, n):
    import torch

    out = {}
    for k, v in obs.items():
        if k == "task_descriptions":
            out[k] = [v[i]] * n
        elif v is None:
            out[k] = None
        else:
            out[k] = v[i : i + 1].repeat_interleave(n, dim=0)
    return out


def _tree(fn, x):
    import torch

    if isinstance(x, dict):
        return {k: _tree(fn, v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return fn(x)
    return x


TASK_STATE_ATTRS = ("consecutive_grasp",)


def snapshot(env, i):
    """Full restorable state of sub-env i: physx state + controller targets + task counters + step count."""
    base = env.env.unwrapped
    sd = base.get_state_dict()  # includes 'controller' key (target pose / target qpos)
    snap = {"sim": _tree(lambda t: t[i : i + 1].clone(), sd)}
    snap["task"] = {a: getattr(base, a)[i : i + 1].clone() for a in TASK_STATE_ATTRS}
    snap["episode_stats"] = _tree(lambda t: t[i : i + 1].clone(), dict(base.episode_stats))
    snap["elapsed"] = base._elapsed_steps[i : i + 1].clone()
    snap["prev_rew"] = env.prev_step_reward[i : i + 1].clone()
    return snap


def restore_all(env, snap):
    """Broadcast a single-env snapshot into all sub-envs; returns fresh wrapped obs."""
    base = env.env.unwrapped
    n = base.num_envs
    rep = lambda t: t.expand(n, *t.shape[1:]).clone()  # noqa: E731
    sim = _tree(rep, snap["sim"])
    ctrl = sim.pop("controller", None)
    base.set_state_dict(sim)
    if ctrl is not None:
        base.agent.set_controller_state(ctrl)
    for a, v in snap["task"].items():
        getattr(base, a)[:] = rep(v)
    for k, v in _tree(rep, snap["episode_stats"]).items():
        base.episode_stats[k] = v
    base._elapsed_steps[:] = rep(snap["elapsed"])
    env.prev_step_reward[:] = rep(snap["prev_rew"])
    raw = base.get_obs()
    return env._wrap_obs(raw)


def gpu_mem_gb():
    import torch

    return torch.cuda.max_memory_allocated() / 1e9


def now():
    return time.perf_counter()

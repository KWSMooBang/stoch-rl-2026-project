"""RLinf extension for T022 (loaded in every Ray worker via RLINF_EXT_MODULE=vla_residual.rlinf_ext).

No RLinf source edits beyond T021's patch 0001. Everything below is opt-in through config keys that only our
configs set, so stock configs behave exactly as before:
  env.{train,eval}.vla_residual          -> env class = vla_residual.core.ResidualVLAEnv
  actor.model.vr_init                    -> MLP init: actor mean last layer zero (vr_zero_last), initial log-std
                                            (vr_init_logstd; PPO/GRPO state-free param, SAC/CrossQ via the log-std
                                            layer bias with zero weights)
  (obs key "expert_u" is added to RLinf's env->rollout observation whitelist)
  rollout.expert_model.vr_expert_obs_key -> "expert" MLP returns env_obs[key] as its action (scripted privileged
                                            expert computed in the env; DAgger labels). Student never sees the key.
  algorithm.vr_anchor (float)            -> SAC/CrossQ actor loss += vr_anchor * mean ||tanh(mu)||^2 (T011 lesson)
  algorithm.adv_type: grpo_commit + algorithm.advantage_mode: "commit_k=<int>"
                                         -> T038 committed residual: GRPO loss only at sample decisions (t % k == 0);
                                            reuse decisions get loss_mask 0 (rewards / episode scores unchanged).
"""
from __future__ import annotations

import math


def _patch_env_cls():
    import rlinf.envs as envs

    orig = envs.get_env_cls
    if getattr(orig, "_vr", False):
        return

    def get_env_cls(env_type, env_cfg=None):
        if env_cfg is not None and env_cfg.get("vla_residual", None) is not None:
            from vla_residual.core import ResidualVLAEnv

            return ResidualVLAEnv
        return orig(env_type, env_cfg)

    get_env_cls._vr = True
    envs.get_env_cls = get_env_cls
    try:
        import rlinf.workers.env.env_worker as ew

        ew.get_env_cls = get_env_cls
    except Exception:  # pragma: no cover - only workers that host envs need it
        pass


def _patch_mlp():
    import torch
    import rlinf.models.embodiment.mlp_policy as mp
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy

    if getattr(mp.get_model, "_vr", False):
        return
    orig_get = mp.get_model

    def get_model(cfg, torch_dtype=torch.bfloat16):
        model = orig_get(cfg, torch_dtype)
        vi = cfg.get("vr_init", None)
        if vi is not None and isinstance(model, MLPPolicy):
            with torch.no_grad():
                if vi.get("zero_last", False):
                    model.actor_mean.weight.zero_()
                    model.actor_mean.bias.zero_()
                ls = vi.get("init_logstd", None)
                if ls is not None:
                    if model.independent_std:
                        model.actor_logstd.fill_(float(ls))
                    else:  # logstd = lo + 0.5 (hi - lo) (tanh(x) + 1)
                        lo, hi = model.logstd_range
                        y = 2 * (float(ls) - lo) / (hi - lo) - 1
                        model.actor_logstd.weight.zero_()
                        model.actor_logstd.bias.fill_(math.atanh(max(-0.999, min(0.999, y))))
        key = cfg.get("vr_expert_obs_key", None)
        if key is not None and model is not None:
            model.vr_expert_obs_key = str(key)
        return model

    get_model._vr = True
    mp.get_model = get_model

    orig_pred = MLPPolicy.predict_action_batch

    def predict_action_batch(self, env_obs, *args, **kwargs):
        chunk_actions, result = orig_pred(self, env_obs, *args, **kwargs)
        key = getattr(self, "vr_expert_obs_key", None)
        if key is not None:
            a = env_obs[key].to(chunk_actions.device, chunk_actions.dtype)
            a = a.reshape(chunk_actions.shape[0], -1)
            result["forward_inputs"]["action"] = a.clone()
            result["forward_inputs"]["model_action"] = a.clone()
            chunk_actions = a.reshape(chunk_actions.shape)
        return chunk_actions, result

    MLPPolicy.predict_action_batch = predict_action_batch

    orig_sac = MLPPolicy.sac_forward

    def sac_forward(self, obs, **kwargs):
        out = orig_sac(self, obs, **kwargs)
        feat = self.backbone(obs["states"])
        self._vr_anchor = torch.tanh(self.actor_mean(feat)).pow(2).sum(-1).mean()
        return out

    MLPPolicy.sac_forward = sac_forward


def _patch_sac_anchor():
    try:
        from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
    except Exception:  # pragma: no cover
        return
    if getattr(EmbodiedSACFSDPPolicy.forward_actor, "_vr", False):
        return
    orig = EmbodiedSACFSDPPolicy.forward_actor

    def forward_actor(self, batch):
        loss, entropy, metrics = orig(self, batch)
        lam = float(self.cfg.algorithm.get("vr_anchor", 0.0))
        if lam > 0:
            m = self.model.module if hasattr(self.model, "module") else self.model
            anc = getattr(m, "_vr_anchor", None)
            if anc is not None:
                loss = loss + lam * anc
                metrics["vr_anchor"] = float(anc.detach())
        return loss, entropy, metrics

    forward_actor._vr = True
    EmbodiedSACFSDPPolicy.forward_actor = forward_actor


def _patch_obs_keys():
    """EnvOutput.prepare_observations whitelists policy inputs; let the privileged expert label through."""
    from rlinf.data.schema.embodied_types import EnvOutput

    orig = EnvOutput.prepare_observations
    if getattr(orig, "_vr", False):
        return

    def prepare_observations(self, obs):
        out = orig(self, obs)
        if isinstance(obs, dict) and obs.get("expert_u") is not None:
            out["expert_u"] = obs["expert_u"]
        return out

    prepare_observations._vr = True
    EnvOutput.prepare_observations = prepare_observations


_COMMIT = {"update": 0}


def commit_k_from_mode(mode):
    """algorithm.advantage_mode = "commit_k=<int>" (the only algorithm key RLinf hands to advantage functions)."""
    assert isinstance(mode, str) and mode.startswith("commit_k="), (
        f"adv_type grpo_commit expects algorithm.advantage_mode='commit_k=<int>', found {mode!r}")
    k = int(mode.split("=", 1)[1])
    assert k >= 1, f"commit_k must be >= 1, found {k}"
    return k


def commit_mask_(loss_mask, loss_mask_sum, k):
    """T038: restrict the policy loss to sample decisions (step t with t % k == 0), in place.

    loss_mask [n_steps, bsz] must be a view of rollout_batch["loss_mask"] (num_action_chunks 1): row t = decision t
    because every train rollout epoch starts with a full reset (auto_reset False) and runs synchronized decisions.
    loss_mask_sum (the per-episode divisor of RLinf's masked_mean_ratio) becomes the number of sample decisions
    before the episode ended, so each episode keeps the same total loss weight as in E2. Returns stats.
    """
    import torch

    n_steps = loss_mask.shape[0]
    before = int(loss_mask.sum())
    assert loss_mask._is_view(), "loss_mask must be a view of rollout_batch['loss_mask'] to be edited in place"
    reuse = (torch.arange(n_steps, device=loss_mask.device) % k) != 0
    loss_mask[reuse] = False
    n_s = loss_mask.sum(0)  # [bsz] sample decisions per episode (>= 1: decision 0 always samples)
    tgt = n_s.view(1, -1, 1).to(loss_mask_sum.dtype)
    assert loss_mask_sum.dim() == 3 and loss_mask_sum.shape[1] == n_s.shape[0], (
        f"loss_mask_sum expected [n_chunk, bsz, 1], found {tuple(loss_mask_sum.shape)}")
    if loss_mask_sum.stride(0) == 0:  # RLinf builds it with expand_as: write the single underlying row
        loss_mask_sum[0:1].copy_(tgt)
    else:
        loss_mask_sum.copy_(tgt.expand_as(loss_mask_sum))
    assert bool((loss_mask_sum == tgt).all()), "loss_mask_sum was not updated everywhere"
    return dict(steps_before=before, steps_after=int(loss_mask.sum()), n_s_min=int(n_s.min()),
                n_s_mean=float(n_s.float().mean()))


def _patch_commit():
    """T038: advantage "grpo_commit" = RLinf GRPO with the loss restricted to sample decisions. k = 1 -> RLinf GRPO
    called with the untouched inputs. Inert unless algorithm.adv_type == grpo_commit."""
    from rlinf.algorithms import registry
    from rlinf.algorithms.advantages import compute_grpo_advantages

    if "grpo_commit" in registry.ADV_REGISTRY:
        return

    @registry.register_advantage("grpo_commit")
    def compute_grpo_commit_advantages(rewards, loss_mask, group_size, **kwargs):
        k = commit_k_from_mode(kwargs.get("advantage_mode"))
        if k > 1:
            st = commit_mask_(loss_mask, kwargs["loss_mask_sum"], k)
            print(f"[t038-commit] update {_COMMIT['update']} k={k}: loss steps {st['steps_before']} -> "
                  f"{st['steps_after']}, sample decisions/episode mean {st['n_s_mean']:.2f} min {st['n_s_min']}",
                  flush=True)
        _COMMIT["update"] += 1
        return compute_grpo_advantages(rewards=rewards, loss_mask=loss_mask, group_size=group_size, **kwargs)


def register():
    _patch_commit()
    _patch_obs_keys()
    _patch_env_cls()
    _patch_mlp()
    _patch_sac_anchor()

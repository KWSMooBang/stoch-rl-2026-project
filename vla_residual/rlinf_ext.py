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
  algorithm.adv_type: grpo_anchor + algorithm.advantage_mode: "anchor_nbase=<int>"
                                         -> T035 base-anchored GRPO (vla_residual.anchor): base members' loss_mask = 0,
                                            A = (R - mean base score) / std(residual scores). n_base 0 = RLinf GRPO.
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


_ANCHOR = {"update": 0}


def _anchor_n_base(mode):
    """algorithm.advantage_mode = "anchor_nbase=<int>" (the only algorithm key RLinf hands to advantage functions)."""
    assert isinstance(mode, str) and mode.startswith("anchor_nbase="), (
        f"adv_type grpo_anchor expects algorithm.advantage_mode='anchor_nbase=<int>', found {mode!r}")
    return int(mode.split("=", 1)[1])


def _patch_anchor():
    """T035: advantage "grpo_anchor" (base-anchored GRPO). Inert unless algorithm.adv_type == grpo_anchor.

    Everything happens inside the advantage function (patching the actor class from here does not reach the
    instance Ray builds). Base members leave the policy loss in two independent ways: their advantage is 0 (PPO
    surrogate and its gradient are exactly 0) and their loss_mask is cleared in place -- the [n_steps, bsz] mask
    RLinf passes is a view of rollout_batch["loss_mask"] (num_action_chunks 1), which the actor keeps using.
    loss_mask_sum (episode length, a divisor in the loss) is left as is: 0 would give 0/0 for base rows.
    """
    import torch
    from rlinf.algorithms import registry
    from rlinf.algorithms.advantages import compute_grpo_advantages

    from vla_residual import anchor as A

    if "grpo_anchor" in registry.ADV_REGISTRY:
        return

    @registry.register_advantage("grpo_anchor")
    def compute_grpo_anchor_advantages(rewards, loss_mask, group_size, **kwargs):
        n_base = _anchor_n_base(kwargs.get("advantage_mode"))
        if n_base == 0:  # no base members: exactly RLinf GRPO
            return compute_grpo_advantages(rewards=rewards, loss_mask=loss_mask, group_size=group_size, **kwargs)
        bc = A.base_columns(loss_mask.shape[1], group_size, n_base).to(loss_mask.device)
        base_steps = int(loss_mask[:, bc].sum())
        is_view = bool(loss_mask._is_view())
        loss_mask[:, bc] = False  # in place -> rollout_batch["loss_mask"]
        adv, stats = A.anchor_advantages(rewards, group_size, n_base)
        advantages = (torch.zeros_like(loss_mask) + adv.view(1, -1)) * loss_mask
        row = A.summarize(stats, loss_mask.sum(dim=0).cpu(), group_size, n_base)
        row.update(update=_ANCHOR["update"], base_steps_masked=base_steps, loss_mask_is_view=is_view)
        A.log_row(row)
        print(f"[t035-anchor] update {row['update']}: +{row['adv_pos']} -{row['adv_neg']} 0:{row['adv_zero']} "
              f"base {row['base_score_mean']:.3f} res {row['res_score_mean']:.3f} res_samples {row['res_samples']} "
              f"base_masked {base_steps} view {is_view}", flush=True)
        _ANCHOR["update"] += 1
        return advantages, None


def register():
    _patch_anchor()
    _patch_obs_keys()
    _patch_env_cls()
    _patch_mlp()
    _patch_sac_anchor()

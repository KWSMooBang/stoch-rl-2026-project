# vla_residual — T022 residual RL on frozen pi0.5 (RLinf PPO / GRPO / SAC / CrossQ / DAgger)

Design, validation and budget: `design_ko.md` (copy of `/outputs/StochRL/t022/design_ko.md`).

- `core.py` — `ResidualVLAEnv` (RLinf env interface; frozen pi0.5 + VLM-17 feature inside the env, 7-D residual per
  decision, sparse success reward, held-out / train episode splits), `ScriptedExpert` (privileged pick-and-place).
- `rlinf_ext.py` — loaded in every Ray worker via `RLINF_EXT_MODULE`: env class swap, MLP init (zero last layer,
  initial log-std), scripted-expert passthrough for DAgger, SAC/CrossQ actor anchor, obs whitelist for `expert_u`.
- `configs/` — `t022_{ppo,grpo,sac,crossq,dagger}.yaml` + shared env block.
- `run.sh <algo> <seed> [overrides]` — GPU 0/1 guard, dedicated Ray head, patch 0001, metrics.
- `validate.py --mode {b0,expert,upper,units}` — pre-training checks (single process).
- `submit_chain.sh` — harness submission only when GPU 0/1 is free; resubmits refused runs as `-rN`.

## T029 success-hold (branch exp/success-hold)
- `env.train.vla_residual.reward`: `first_success` (default, T022) | `hold` (H1: r_t = h_t - h_{t-1}, no success termination,
  return = success_at_end) | `hold_dense` (H2: h_t / T). Design/validation/budget: `design_t029_ko.md`.
- `run_t029.sh <ppo|grpo> <seed>` (GPU 0 only), `submit_t029.sh` (GPU-0 chain, budget guard 3.0), `validate_t029.py`,
  `gate_t029.py`. NOTE: `feat_stats_vlm17_img_mean.npz` is git-ignored; copy it into a new worktree.

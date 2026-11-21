# vla_residual — T022 residual RL on frozen pi0.5 (RLinf PPO / GRPO / SAC / CrossQ / DAgger)

Design, validation and budget: `design_ko.md` (copy of `/outputs/StochRL/t022/design_ko.md`).

- `core.py` — `ResidualVLAEnv` (RLinf env interface; frozen pi0.5 + VLM-17 feature inside the env, 7-D residual per
  decision, sparse success reward, held-out / train episode splits), `ScriptedExpert` (privileged pick-and-place).
- `rlinf_ext.py` — loaded in every Ray worker via `RLINF_EXT_MODULE`: env class swap, MLP init (zero last layer,
  initial log-std), scripted-expert passthrough for DAgger, SAC/CrossQ actor anchor, obs whitelist for `expert_u`.
- `configs/` — `t022_{ppo,grpo,sac,crossq,dagger}.yaml` + shared env block.
- `run.sh <algo> <seed> [overrides]` — GPU 0/1 guard, dedicated Ray head, patch 0001, metrics.
- `validate.py --mode {b0,expert,upper,units}` — pre-training checks (single process).
- `run_t030.sh <grpo|ppo|vla-b03> <seed>` / `submit_t030.sh` — T030 seed replication (GPU 0 only; delegates to
  `run.sh` / `run_t024.sh`, no training-code change; see `plan_t030_ko.md`, `t030_config_diff.txt`).
- `run_t032.sh e0 <seed>` / `submit_t032.sh` — T032 E0 (GRPO, `vla_residual.zero_vlm=True`: VLM feature block of the
  obs = 0; GPU 0 only). `validate_t032.py` unit test, `analyze_t032.py` paired preservation / recovery analysis;
  see `plan_t032_ko.md`, `t032_config_diff.txt`.
- `submit_chain.sh` — harness submission only when GPU 0/1 is free; resubmits refused runs as `-rN`.
- `run_t039.sh <seed>` / `submit_t039.sh` — T039 noise-space steering (GRPO, E2 obs; the 7-D policy output shifts
  pi0.5's initial flow noise, `vla_residual.noise_steer.mode=add, beta=2.0`: z = z_ref + beta*tanh(u) on model action
  dims 0..6 over the H=8 horizon, u = 0 -> B0 exactly; GPU 0 only; s0 first, s1/s2 only after L1's
  `/outputs/StochRL/t039/GO`). `validate_t039.py` units / sensitivity / held-out B0 checks; see `plan_t039_ko.md`.

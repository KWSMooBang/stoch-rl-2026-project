# repcritic — ManiSkill3 frozen π0.5 × representation-aware critic (T016~)

Single-process adapter around RLinf (`third_party/RLinf`, submodule) for the PoC in
`/outputs/StochRL/repcritic_docs/`. No training. Results: `/outputs/StochRL/t016/summary_ko.md`.

Setup (once):
```
git submodule update --init third_party/RLinf
cd third_party/RLinf && bash requirements/install.sh embodied --model openpi --env maniskill_libero --no-root --venv <repo>/.venv
hf download --repo-type dataset RLinf/maniskill_assets --local-dir /workspace/stochrl_t016/maniskill_assets
ln -s /workspace/stochrl_t016/maniskill_assets third_party/RLinf/rlinf/envs/sim/maniskill/assets
```
Run with `.venv/bin/python -m repcritic.<module>` (do NOT `source .venv/bin/activate`: it exports a
non-existent Vulkan ICD; `rc_common.setup_env_vars` sets `repcritic/vulkan/nvidia_icd.json`).
`CUDA_VISIBLE_DEVICES` must be 0 or 1 (enforced).

- `rc_common.py` — env/policy builders, `Policy.act` (explicit initial noise), snapshot/restore.
- `smoke.py` — PickCube smoke (`--pickcube`), pi0.5 load, frozen + hook invariance, 1 episode.
- `e01_baseline.py` — E01 (parallel envs, fixed episode_ids × noise seeds, videos, repro check).
- `oracle_diversity.py` — N=32 candidate diversity + Oracle-N (cand vs same-chunk ctrl arm), restore tests.
- `analyze_oracle.py` — tables (unbiased pass@N, episode bootstrap CIs).
- `diag_restore.py`, `probe_gpu.py` — restore fidelity / GPU placement diagnostics.

## E02 (T017) — `/outputs/StochRL/t017/summary_ko.md`, data `/workspace/stochrl_t017/`
- `e02_common.py` — FeatPolicy (VLM/SigLIP/action-expert hooks, ctx pass x_t=0), privileged labels, batched snapshots.
- `e02_rollout.py` — 500-episode buffer (70/15/15 by episode), features fp16, snapshots, label-only privileged files.
- `e02_branch.py` — candidate-level branch labels (16 cand) + multi-continuation oracle (8x4), packed env batches.
- `e02_analyze.py`, `e02_oracle_eb.py` — oracle E[max p]-E[p] estimators; `e02_card.py` — dataset card stats, feature index.

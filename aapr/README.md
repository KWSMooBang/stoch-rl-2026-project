# aapr — T014 Action-Attributed Progress Reward PoC (2D stochastic inertial pushing, SAC, CPU)

Design / Phase 0–2 results / gate: `/outputs/StochRL/t014/design_ko.md`.

| file | what |
|---|---|
| `env.py` | numpy Gymnasium env, wind presets (`iid`, `ar1`, `bias`, `ar1_hi`), full snapshot incl. RNG |
| `rewards.py` | shared reward terms, method shaping (M0–M6, AUG), `CounterfactualEvaluator` (separate env copy, paired/independent branches) |
| `sac.py` | minimal SAC (same net/hparams for every method) |
| `train.py` | one run → metrics (SR, AUC, sim transitions, wall-clock, attribution MAE/SPR, wind-helps SR) |
| `run_phase.py` | harness entry: whole phase in one CPU process pool (`p1`, `p3` with λ validation, `smoke`) |
| `phase2.py`, `report_phase2.py` | expectation check b(s), reward unit test, CRN variance |
| `report_runs.py` | markdown table from a run's `metrics.json` |

```
uv run pytest -q tests                                      # Phase 0 dynamics tests
uv run python -m aapr.phase2 OUT && uv run python -m aapr.report_phase2 OUT/phase2.json
RUN_HOURS=2.5 /harness/launch.sh t014-p3-main 1 bash scripts/run_harness.sh --phase p3 --workers 100 --steps 150000
uv run python -m aapr.report_runs /outputs/StochRL/runs/t014-p3-main main
```

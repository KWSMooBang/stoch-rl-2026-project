# T020 E06 closed-loop reranking — pre-registration

Committed before any closed-loop run on test episodes. The only earlier closed-loop run is the smoke run on 6 T017 TRAIN episodes, used to check code and timing.

## Episodes
- **150 test initial states**: `episode_id` = `randperm(total, seed=20261119)[500:]`, after removing the T016 ids. They are disjoint from every T017 episode (train, val and test), so no critic, probe or λ ever saw them.
- Run as 2 parts × 3 batches of 25 parallel envs on GPU 0 and GPU 1. Every condition runs on the same 150 ids, which gives a paired design.
- Size rationale: the smoke run projects about 1.3–1.5 GPU-h for 5 conditions × 150 episodes, which fits the 2 GPU-h cap.
- If the budget runs out, report only the batches that finished for all conditions. Conditions are interleaved within each batch.

## Conditions
All conditions use the same noise stream: generator seed `20200 + batch`, and every decision draws [n, 8] noises.

| cond | rule |
|---|---|
| B0 | frozen, N=1 (candidate 0) |
| Random8 | N=8, uniform pick (RandomState 777+batch) |
| ActOnly8 | N=8, argmax of T019 action-only within-state ridge (λ=0.01 chosen on val; reproduces test conc 0.543) |
| Critic8 | N=8, argmax of T019 `Geo_q_buf+br`. Ensemble = mean over 3 seeds of per-seed z-scored Q within the candidate set (offline test conc 0.554; per-seed mean 0.544) |
| CriticApp8 (optional) | Critic8 pick when the C-Phase probe (vlm9_txt_mean, argmax of 4 phase heads) says approach; otherwise candidate 0 |

- Receding horizon: execute the first K=5 of the chosen H=8 chunk, then resample. Episodes are 80 steps (16 decisions).
- Critic input z is the online T018 C-Geo probe on vlm9_img_grid4. It was verified against the T018 cache (max |err| 7e-9).
- VLA frozen.

## Metrics
- success_once and success_at_end per condition, with Wilson 95% CIs.
- Paired episode bootstrap of differences (10,000 resamples, seed 0), 95% percentile CI:
  - Critic8 − ActOnly8
  - Critic8 − B0
  - Critic8 − Random8
  - ActOnly8 − B0
  - Random8 − B0
  - CriticApp8 − B0
  - CriticApp8 − ActOnly8
- McNemar exact p-values are reported only as secondary statistics.
- Latency per decision:
  - batch=25 amortised ms per env-decision
  - batch-1 single-env benchmark (`--bench`)
- Diagnostics:
  - candidate diversity
  - agreement of Critic8 and ActOnly8 argmax on identical d=0 candidate sets. This is computed offline from logged scores where both exist; otherwise noted.
  - pick distribution, gate rate vs privileged phase

## Decision rule (primary endpoint: success_once)
- "The representation-conditioned critic improves control" is claimed **only if** both paired 95% CIs exclude 0 on the positive side:
  - Critic8 − ActOnly8
  - Critic8 − B0
- Otherwise the claim is not made. If Critic8 > B0 but Critic8 ≈ ActOnly8, the gain is attributed to a state-free action rule.
- Expected effect from offline pick gain is about +0.02. With n=150 the paired CI half-width is about ±0.10, so a null result here cannot rule out a small effect. That limitation will be stated.
- E07 is not judged No-Go on this basis (see T019).
- No parameter is tuned on these episodes.

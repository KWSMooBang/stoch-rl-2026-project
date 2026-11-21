# T038 — 시간적으로 commit하는 residual (설계 · 검증 · 체인 · 예산)

- 날짜 2026-11-21 / 브랜치 `exp/committed`(`exp/residual-input` 0b2d9ca에서 분기, E2 경로, gate 없음) / worktree `.claude/worktrees/committed`
- 질문: residual을 k 결정 동안 같은 값으로 유지하면(정책 출력과 탐색 noise 모두), 탐색이 base를 덜 망치고 π0.5에 흡수되지 않아서 E2보다 나은가.
- 근거
  - T031: expert 수정을 3 chunk 연속 적용하면 net +0.136 [+0.05, +0.24], 1 chunk만 적용하면 −0.076.
  - T026: 작은 수정을 한 번만 주면 π0.5가 되돌린다.
  - T035: 학습 rollout(탐색 noise)에서 residual 멤버 성공률 0.20, base 멤버 0.55.
- 선행연구(L1 조사, **원문은 확인하지 않았다**): WoVR+(arXiv 2610.17922, chunk 단위 noise), BORA(frozen VLA + chunk 단위 residual adapter), RTCF(arXiv 2608.04527, DCT 저주파 residual, 학습 없음).
  - 이 실험의 차별점: "frozen VLA가 단발 수정을 흡수한다"는 우리 측정(T026·T031)에 근거해, 한 chunk 안이 아니라 **여러 chunk(결정)에 걸쳐** 같은 residual을 commit한다.

## 1. 조건 (k = 3)
- 결정 d에서 `d % k == 0`이면 **샘플 결정**이다. 정책이 낸 u(평균 + 탐색 noise)를 실행하고 저장한다.
- 그 밖의 결정은 **재사용 결정**이다. 저장한 u를 그대로 실행한다. 정책이 그 결정에서 낸 u는 버린다.
  - 실행 행동은 `clip(a_base_new[j] + α·tanh(u_saved))`다. base chunk는 매 결정 새로 만든 π0.5 chunk이고, 그 위에 같은 residual을 더한다.
  - 16 결정 중 샘플 결정은 0, 3, 6, 9, 12, 15다. 마지막(15)은 commit이 1 결정으로 잘린다.
- 평가(결정적, 평균 행동)에도 같은 commit을 적용한다. 학습한 정책을 그대로 평가하기 위해서다.
- 관측 `states` 2093차원 = [E2의 2092차원 | **c_rem/k**].
  - c_rem = 이 결정에서 실행하는 residual이 이 결정을 포함해 앞으로 몇 결정 더 실행되는가 = `min(k − d % k, 16 − d)`.
  - 샘플 결정에서는 값이 1이고, 결정 15에서만 1/3이다. 정책 입력으로 의미가 있는 곳은 샘플 결정뿐이다.
- 그 밖은 E2와 같다: GRPO(group 8), α(0.02 m, 0.1 rad, 2.0 grip), sparse 보상(성공에서 종료), lr 1e-4, 27 epoch(1728 train episode), log-std 초기값 −1.5, 출력층 0 초기화, eval held-out 128(seed 1000), GPU 0.

### GRPO loss 구현 (`vla_residual/rlinf_ext.py`, advantage `grpo_commit`)
- RLinf 소스는 바꾸지 않는다. `RLINF_EXT_MODULE`에서 advantage 하나를 등록한다. config는 `algorithm.adv_type=grpo_commit`, `algorithm.advantage_mode='commit_k=3'`이다.
- 보상과 episode 점수는 그대로다. RLinf GRPO는 첫 done까지의 보상 합으로 episode 점수를 만들고 group 안에서 z-score한다. 재사용 결정의 보상도 이 합에 그대로 들어간다.
- loss_mask를 제자리에서 고친다. `[16, 배치]` loss_mask는 `rollout_batch["loss_mask"]`의 view이고(num_action_chunks 1), 행 t가 결정 t다.
  - 행 t = 결정 t가 성립하는 이유: 학습 rollout은 epoch마다 전체 reset으로 시작하고(auto_reset False) 결정이 동기화되어 있다.
  - `t % 3 != 0` 행을 0으로 만든다. 그래서 재사용 결정은 advantage가 0이고 로그확률 비율 항도 loss에 들어가지 않는다.
  - 로그확률은 모든 결정에서 계산되지만 loss에는 샘플 결정의 것만 쓰인다.
- loss_mask_sum(RLinf `masked_mean_ratio`가 episode 단위 가중치의 분모로 쓰는 episode 길이)을 **episode 안의 샘플 결정 수 n_s**로 바꾼다.
  - 그래서 episode 하나의 loss 가중치 합이 E2와 같은 16이다.
  - 샘플 결정 하나당 가중치는 E2보다 L/n_s배(약 3배)다. gradient 크기 차이는 Adam과 clip_grad 0.5가 대부분 흡수한다.
- k = 1이면 입력을 건드리지 않고 RLinf GRPO를 그대로 부른다.
- 학습 중 interruption(평가가 공유 scene을 쓴 뒤 학습 episode가 중간에 끊김)이 생기면 행 t = 결정 t가 깨진다. 그래서 `train_interrupted.jsonl` 수를 metrics.json의 `train_interruptions`에 기록한다. E2 run에서는 0이었고 smoke에서도 0이었다.

### E2와 달라지는 점 (commit 외)
- 관측 1열이 추가되어 MLP 첫 층 입력이 2093이다. 그래서 같은 seed라도 첫 층 초기 가중치가 E2와 다르다. 출력층은 0으로 초기화하므로 초기 정책(평균 0)은 같다.
- 탐색 noise는 샘플 결정에서만 새로 뽑힌다. episode당 독립 탐색이 16번에서 6번으로 줄고, 한 번 뽑힌 noise는 3 결정 동안 유지된다.

## 2. 학습 전 검증 (`/outputs/StochRL/t038/val/`, 모두 통과)

| 검사 | 결과 |
|---|---|
| C1 `grpo_commit`(k=1)을 RLinf `calculate_adv_and_returns`로 돌린 결과 = `grpo` (advantage, loss_mask, loss_mask_sum) | 비트 동일 (`cpu.json`) |
| C2 k=3: `rollout_batch` loss_mask가 제자리에서 `t%3≠0` 행만 0, loss_mask_sum = n_s, 샘플 행 advantage = grpo, episode 점수 불변 | 통과 |
| C3 RLinf actor loss(`compute_ppo_actor_loss`)의 gradient: 재사용 결정 0, \|A\|>0인 샘플 결정은 0이 아님, episode 가중치 합 16(E2와 같음) | 통과. 샘플 결정 gradient는 E2의 1–3배(=L/n_s) |
| C4 Hydra 구성 비교: k=1 대 E2는 `adv_type`, `advantage_mode`만 다름. k=3은 여기에 `commit_k`(train/eval), `obs_dim` 2093이 더해짐 | 통과 |
| G1 k=3, train(group 8, 성공 종료)·eval(held-out), 결정마다 무작위 u, 16 결정 × 16 env: 샘플 결정 = `d%3==0`, 재사용 결정의 실행 u = 직전 실행 u(비트 동일, 정책 u와는 다름), 모든 sim step의 실행 행동 = `clip(a_base[j] + α·tanh(u_exec))` 비트 동일, states 2093차원, 앞 2092차원 = E2 관측 구성 비트 동일, 마지막 열 = c_rem/3 | 통과 (`gpu.json`) |
| G2 k=1: 모든 결정에서 실행 u = 정책 u(비트 동일), 실행 행동 = E2 식, states 2092차원 = E2 관측(비트 동일) | 통과 |

- k=1 대 E2: 같은 상태와 같은 u에서 관측·실행 행동·advantage·loss_mask가 비트 동일하다(G2, C1). 별도 run끼리의 비트 비교는 하지 않았다. 같은 설정을 두 번 돌려도 PhysX·π0.5 비결정성 때문에 결정의 약 2–7%가 갈린다(T029·T037).
- **smoke** (`/outputs/StochRL/t038/smoke/`, GPU 0, harness 밖, 2 epoch = 128 train episode + eval 32 episode, 384 s): return code 0, metrics.json 생성.
  - env 로그에 `commit_k=3 commit_obs=True`가 train·eval 모두 찍혔다.
  - actor 로그: loss 대상 결정 1007 → 378, 978 → 367(약 3분의 1). episode당 샘플 결정 평균 5.9.
  - interruption 0, peak VRAM 17.0 GB.
  - eval once 0.688 / end 0.625(32 episode, 2 epoch라 판단 근거가 아니다).
  - 학습 rollout 성공률 0.070(128 episode; T037 smoke 0.109, E2 본 run 약 0.19). 2 epoch 값이라 판단하지 않는다. 본 run의 `train_success_rate`, `train_success_by_third`로 본다.

## 3. runs와 판정
| run_id | seed | 비교 대상 (GPU 0, 같은 seed) |
|---|---|---|
| t038-k3-s0 | 0 | E2 t022-grpo-s0 0.742/0.625 |
| t038-k3-s1 (GO일 때만) | 1 | t030-grpo-s1 0.820/0.758 |
| t038-k3-s2 (GO일 때만) | 2 | t030-grpo-s2 0.695/0.602 |

- GPU 0 B0 = 0.555/0.531(T034).
- **keep(GO) 기준 (s0, 사전 등록)**: once ≥ 0.72 **그리고** end ≥ 0.60(E2 s0보다 0.03 넘게 나빠지지 않음). 그 밖이면 STOP.
- 판정은 L1이 s0의 metrics.json으로 하고, `/outputs/StochRL/t038/GO` 또는 `STOP` 파일을 만든다.
- 함께 기록할 값: `train_success_rate`(E2 약 0.19), `train_success_by_third`, `train_interruptions`(0이어야 함), `aulc_once/end`.

## 4. 체인과 예산
- `vla_residual/submit_t038.sh 3` (setsid nohup, 로그 `/outputs/StochRL/t038/chain.log`)
  1. GPU 0이 비어 있고(< 1000 MiB) t036–t039 run이 queued/running이 아닐 때 s0를 제출한다.
  2. `run_t038.sh`는 GPU 0이 아니면 거부한다(exit 3). 그러면 `-rN`으로 재제출한다.
  3. s0가 끝나면 GO/STOP 파일을 기다린다. GO면 s1, s2를 차례로 제출하고, STOP이면 끝낸다.
- 상한 4.5 GPU-h. run 추정 1.3 h(E2 1.23–1.25 h, commit은 계산량이 같다).
  - harness 밖 사용분: 검증 + smoke ≈ 0.2 GPU-h(`T038_PRE_GPUH=0.2`).
  - 0.2 + 3 × 1.3 = 4.1 ≤ 4.5. 실제 사용량이 늘어 추정을 넘게 되면 체인이 s2부터 SKIP한다.
- BUDGET-CHECK는 보내지 않는다(사람 승인).

## 5. 코드
- `vla_residual/core.py`: `commit_k`, `commit_obs`, `commit_u()`, `commit_remaining()`
- `vla_residual/rlinf_ext.py`: advantage `grpo_commit`, `commit_mask_()`
- `vla_residual/configs/env/vla_residual_putonplate.yaml`: `commit_k: 1`, `commit_obs: null` 기본값(기존 run에는 영향 없음)
- `vla_residual/make_metrics.py`: `train_success_rate`, `train_success_by_third`, `train_interruptions` 추가
- `vla_residual/run_t038.sh`, `submit_t038.sh`, `validate_t038.py`, `val_t038.sh`

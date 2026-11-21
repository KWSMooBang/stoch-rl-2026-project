# T037 — 최소 정보 residual E0v (설계 · 검증 · 체인 · 예산)

- 날짜 2026-11-21 / 브랜치 `exp/min-vhat`(`exp/competence-gate` 737e220에서 분기) / worktree `.claude/worktrees/min-vhat`
- 질문: actor 입력의 VLA feature 2048차원을 "끝에 성공해 있을 확률" 스칼라 하나(V̂_end)로 바꿔도 되는가. E0에서 무너진 end(3 seed 평균 0.274)가 회복되는가.
- 비교 기준(모두 GPU 0, 같은 seed)

  | 조건 | s0 once/end | s1 | s2 |
  |---|---|---|---|
  | E2 (VLA feature 있음) | 0.742/0.625 | 0.820/0.758 | 0.695/0.602 |
  | E0 (VLA feature 0) | 0.758/0.477 | 0.664/0.172 | 0.648/0.172 |
  | B0 (residual 없음, T034) | 0.555/0.531 | | |

## 1. 조건 E0v
- 관측 `states` 2093차원 = [0×2048 | qpos 8 | base chunk 35 | t/16 | **V̂_end**].
  - 앞 2092차원은 E0와 같다(VLM feature 블록은 상수 0).
  - 2093번째 열은 T036의 V̂_end 예측기 값이다(VLM17 pooled feature → `success_at_end`, L2-logistic + Platt, `/outputs/StochRL/t036/predictors`). 값은 (0, 1) 확률 그대로 넣는다. 추가 표준화는 없다.
  - base chunk를 만든 π0.5 forward에서 이미 뽑은 feature로 계산한다. 추가 forward는 없다.
- 실행 행동은 E2·E0와 같은 `clip(a_base + α·tanh(u))`다. gate는 g = 1이다.
- actor는 RLinf 기본 MLP이고 `obs_dim`만 2093이다(파라미터 +256개, 첫 층의 V̂ 열).
- GRPO, α, sparse 보상, lr 1e-4, 27 epoch(1728 train episode), eval held-out 128(seed 1000)은 E2·E0와 같다.
- config: `t022_grpo.yaml` + override
  ```
  env.{train,eval}.vla_residual.zero_vlm=True
  env.{train,eval}.vla_residual.vhat_obs=True      # 새 플래그: states에 V̂ 열 추가
  env.{train,eval}.vla_residual.gate.mode=one      # g = 1. 예측기 적재와 결정별 V̂ 기록에만 쓴다
  env.{train,eval}.vla_residual.gate.vhat=end
  actor.model.obs_dim=2093
  ```
- 코드: `vla_residual/core.py`(`vhat_obs`), `run_t037.sh`, `submit_t037.sh`, `validate_t037.py`, `val_t037.sh`.

### E0와 달라지는 점 (관측 1열 외)
- MLP 첫 층의 입력 차원이 달라 같은 seed라도 초기 가중치가 E0와 다르다. 출력층은 0으로 초기화하므로 초기 정책(평균 0, log-std −1.5)은 같다.
- 학습은 성공에서 종료된다. 학습 rollout에는 성공 뒤 상태가 없으므로 정책이 "V̂_end가 높으면 residual을 줄인다"를 성공 뒤 상태에서 직접 배우지는 못한다. 성공 전의 높은 V̂ 상태에서 배운 것이 성공 뒤로 일반화되는지를 보는 실험이다.

## 2. 학습 전 검증 (`/outputs/StochRL/t037/val/`, 모두 통과)

| 검사 | 결과 |
|---|---|
| env가 쓰는 예측기 값 = T036 계산(`vhat_end_all.npz`, train+val 6800개 상태) | 최대 차이 0 (`cpu.json`) |
| 독립 numpy 재구현 = 같은 값 | 최대 차이 3e-8 |
| MLP 첫 층이 2093열을 읽음, 초기 평균 0, 파라미터 수 = E0 + 256 | 통과 |
| E0v 관측(결정마다, 34회): 2093차원, 앞 2048 = 0, 2048:2092 = E0 구성과 비트 동일, 2093번째 열 = 예측기 값과 비트 동일 | 통과 (`gpu.json`) |
| 같은 상태에서 `vhat_obs`를 끄면 E0 관측과 비트 동일, base chunk·feature 불변 | 통과 |
| 실행 행동 = `clip(a_base + α·tanh u)` 비트 비교(모든 sim step), g = 1 | 통과 |
| base chunk E0v 대 E0(별도 env, 같은 seed·같은 u, 순서 바꿔 반복) | 결정 0은 비트 동일. 전체 결정 일치율 0.938–0.949 |
| held-out 32 episode(residual 0)의 V̂ 열 대 T036 B0 rollout 기록(`b0_gates.parquet`) | 결정 0 최대 차이 0. 전체 결정의 90%가 1e-4 이내 |

- base chunk가 모든 결정에서 비트 동일하지는 않다. 같은 설정을 두 번 돌린 일치율이 0.925–0.971(PhysX·π0.5 비결정성)이고, E0v 대 E0의 0.938–0.949는 그 범위 안이다. 결정 0(reset 직후)은 모든 쌍에서 비트 동일하다.
- T036 기록과의 V̂ 차이도 같은 원인이다. 뒤 결정으로 갈수록 상태 자체가 갈린다(결정별 일치율 1.00 → 0.875).
- **smoke**(`/outputs/StochRL/t037/smoke/`, GPU 0, harness 밖, 2 epoch = 128 train episode + eval 32 episode): return code 0, metrics.json 생성.
  - env 로그에 `zero_vlm`, `vhat_obs`가 train·eval 모두 켜져 있다. g는 전부 1이다.
  - eval once 0.688 / end 0.656(2 epoch, 32 episode라 판단 근거가 아니다).
  - 평가에서 V̂ 열 평균: 성공 전 0.379, 성공 flag 켜진 상태 0.923. 정책이 볼 신호는 성공 뒤에 뚜렷이 높다.
  - 학습 rollout 성공률 0.109(128 episode).

## 3. runs와 판정
| run_id | seed | 비교 대상 |
|---|---|---|
| t037-e0v-s0 | 0 | E2 t022-grpo-s0, E0 t032-grpo-e0-s0 |
| t037-e0v-s1 (GO일 때만) | 1 | t030-grpo-s1, t032-grpo-e0-s1 |
| t037-e0v-s2 (GO일 때만) | 2 | t030-grpo-s2, t032-grpo-e0-s2 |

- keep 기준(s0, 사전 등록): **end ≥ 0.58 그리고 once ≥ 0.68**이면 GO, 아니면 STOP.
- 체인 `vla_residual/submit_t037.sh`, 로그 `/outputs/StochRL/t037/chain.log`
  1. T036 체인이 끝났고, t036–t039 run이 queued/running이 아니고, GPU 0이 비었을 때(< 1000 MiB) s0를 제출한다. GPU 0이 아니면 `run_t037.sh`가 REFUSE하고 `-rN`으로 다시 낸다.
  2. s0가 끝나면 `/outputs/StochRL/t037/GO` 또는 `STOP` 파일을 기다린다(L1이 만든다).
  3. GO면 s1, s2를 하나씩 제출한다. STOP이면 끝낸다. 끝날 때 `chain done`을 적는다.
- `metrics.json`은 T022 형식 + `gate` 블록(task = T037)이다. 평가마다 V̂ 분포(성공 전 / 성공 flag 켜짐), `loss_per_success_decision`, `s_abs_pre_success`, `s_abs_success_flag_on`(|tanh u| 평균)이 들어간다. 결정별 원자료는 `gate_decisions.jsonl`이다.

## 4. 예산 (상한 4.2 GPU-h, GPU 0만)
- 사전 사용(harness 밖): GPU 검증 437초 + smoke 402초 = **0.24 GPU-h**.
- 본 run 추정: 각 1.25 GPU-h(E2·E0·T036 G1이 1.20–1.25).
- 예상 합계: STOP이면 약 1.5, GO면 0.24 + 3 × 1.25 = **3.99 GPU-h**.
- 가드: `results.tsv`의 `t037-*` gpu_h 합 + 0.24 + 1.25가 4.2를 넘으면 그 run을 SKIP한다(s2부터 빠진다).

## 5. 분석 (run 종료 후, CPU)
- s0: `heldout_success_once`, `heldout_success_at_end`를 keep 기준과 비교한다.
- GO 뒤 3 seed: `analyze_t032.py` 방식으로 b0.json과 짝지어 once·end의 preservation, recovery, regression을 내고 E2·E0와 비교한다.
- 함께 볼 것
  - 성공 flag가 켜진 결정당 상실 확률(E2 0.021, B0 0.009)과 `s_abs_success_flag_on` 대 `s_abs_pre_success`: 정책이 V̂가 높을 때 residual을 줄였는가.
  - end가 회복되지 않으면, 성공 뒤 상태를 학습에서 못 본 탓인지(1절 주의) 스칼라 정보가 부족한 탓인지 이 실험만으로는 가를 수 없다.

# T036 — Competence-gated residual (설계 · gate 정의 · 검증 · 예산)

- 날짜 2026-11-21 / 브랜치 `exp/competence-gate`(`exp/residual-input` 0b2d9ca에서 분기, T035 코드와 독립) / worktree `.claude/worktrees/competence-gate`
- 질문: residual 개입 크기를 frozen VLA 자신의 성공 예측 V̂(s)로 줄이면(`a = a_base + α·g(V̂)·tanh(u)`), 기존 GRPO(E2)보다 regression이 줄고 end가 오르면서 once는 유지되는가.
- 비교 기준: E2 3 seed(t022-grpo-s0, t030-grpo-s1, t030-grpo-s2). once/end 0.752/0.662, end preservation 0.763, regression 0.237.
- T035 판정(L1): base-anchored는 discard. 그래서 T036은 **기존 GRPO(그룹 평균 baseline)에 gate만 더한다.**

## 0. 명세에서 바뀐 것 (L1 결정, 2026-11-21)
**gate용 예측기를 T023 V̂에서 V̂_end로 바꿨다.** 학습 run(G1·G2)은 모두 V̂_end gate를 쓴다.

- 발단: 학습 전 검증에서 T023 V̂ gate가 성공 뒤 상태에서 닫히지 않았다(held-out B0 성공 직후 g 평균 0.82).
  - T023 V̂는 성공 전 결정만으로 학습했다(그 MDP는 성공하면 끝난다). 물체가 접시에 놓인 성공 뒤 상태는 학습 분포 밖이고 V̂가 낮게 나온다(중앙값 0.11).
- L1 결정
  1. V̂_end를 새로 맞춘다. 입력은 같은 VLM17 pooled feature, 라벨은 `success_at_end`, 학습 상태는 T017 train split의 **성공 뒤 상태 포함 전체 결정**이다. val에서 Platt 보정한다. test·held-out은 쓰지 않는다.
  2. τ_g, T는 같은 규칙으로 train split에서 다시 정한다.
  3. held-out B0에서 g 분포를 다시 본다. 성공 뒤 g 평균이 0.5를 넘으면 학습 전에 멈춘다.
  4. 진단 평가는 V̂_end gate와 기존 T023 V̂ gate 둘 다 돈다.
- 결과: 성공 뒤 g 평균이 0.04–0.06으로 내려갔다(아래 표). 기준을 충족해 체인을 시작했다.

| held-out B0 rollout(128 episode, residual 0), 결정 시점 상태 | 결정 수 | **V̂_end gate** g 평균 | g<0.1 비율 | T023 V̂ gate g 평균 | g<0.1 비율 |
|---|---|---|---|---|---|
| 성공 전, 결국 성공하는 episode | 431 | 0.273 | 0.520 | 0.283 | 0.485 |
| 성공 전, 결국 실패하는 episode | 896 | 0.803 | 0.069 | 0.807 | 0.057 |
| 성공 직후(첫 성공 다음 결정) | 72 | **0.057** | 0.889 | 0.816 | 0.028 |
| 성공 flag가 켜진 상태 전체 | 707 | **0.043** | 0.929 | 0.845 | 0.004 |
| 성공을 잃은 뒤 | 14 | 0.452 | 0.500 | 0.676 | 0.143 |

- 두 gate는 같은 B0 rollout의 같은 상태에서 계산했다(`val/b0_gates.json`). 성공 전 상태에서는 거의 같고, 성공 뒤에서만 갈린다.
- 주의
  - **학습은 성공에서 종료된다.** 학습 rollout에는 성공 뒤 상태가 없으므로 두 gate의 차이는 학습 중에는 거의 없고, 16 결정을 다 도는 평가에서 드러난다.
  - G2(학습형)는 성공 뒤 상태를 겪지 못한다. 초기값(G1과 같음)의 `w_v`가 유지되는 한 성공 뒤에 닫히지만, 그 동작은 학습된 것이 아니다.
  - 성공을 잃은 뒤 상태(14개)에서는 g 평균 0.45로 반쯤 열린다. 표본이 작다.

## 1. FIND와의 차이
- FIND(arXiv 2609.32069): frozen π0.5 + residual, off-policy RL. gate는 **학습 critic**의 차이 `Q̄(x, a_comb) − Q̄(x, a_base) > δ`이고 0/1로 residual을 켜고 끈다.
- T036
  1. gate 신호는 **critic이 아니다.** frozen VLA의 VLM 17층 feature에 offline으로 맞춘 성공 예측(L2-logistic + Platt)이다. frozen base의 buffer만으로 맞추고, RL 학습 중에 바뀌지 않고, 사람 개입도 없다.
  2. 예측기는 행동을 입력으로 받지 않는다. "이 residual이 base보다 나은가"가 아니라 "base가 지금 잘 하고 있는가(끝에 성공 상태일 것인가)"만 본다. FIND의 gate는 행동별 판단이고 우리 것은 상태별 판단이다.
  3. gate는 연속값 g∈[0,1]이고 residual 크기에 곱한다(끄고 켜는 것이 아니다).
  4. on-policy GRPO와 결합한다. critic이 없다.
  5. 추가 forward가 없다. 입력은 base chunk를 만든 π0.5 forward에서 이미 뽑은 feature다.
- 한계: FIND식 critic gate와 직접 비교하지 않는다(우리 off-policy critic 경로는 붕괴 상태).

## 2. gate 정의
- 실행 행동: `a[k] = clip(a_base[k] + α · g · tanh(u))`, k = 0..4. g는 결정마다 하나이고 chunk 전체와 7개 성분(그리퍼 포함)에 같은 값을 곱한다.
- 탐색 noise도 `tanh(u)` 안에 있으므로 g가 함께 곱해진다. 예측기 값이 높은 곳에서는 탐색도 줄어든다.

### 예측기 V̂_end (`vla_residual/fit_vhat_end.py`, CPU, `/outputs/StochRL/t036/predictors/`)
- T023 `vla`와 같은 절차다: train 표준화 → L2-logistic(LBFGS), λ는 T023 격자에서 val log-loss로 고름, Platt (a, b)는 val.
- 다른 점은 둘이다: 라벨이 `success_at_end`이고, train split 350 episode의 16 결정 전부(5600개, 성공 뒤 2099개 포함)를 쓴다.
- 선택된 λ = 0.01, Platt (0.948, −0.469).
- val(75 episode): 전체 AUROC 0.939, log-loss 0.306. 성공 전 결정만 보면 AUROC 0.874(T023 V̂의 test 값은 0.900이었다. split과 라벨이 달라 직접 비교는 아니다).
- val에서 V̂_end 중앙값: 성공 뒤이고 끝에도 성공 0.980 / 성공 뒤인데 끝에는 실패 0.107(15개) / 성공 전, 끝에 성공 0.641 / 성공 전, 끝에 실패 0.083.
- test split 행은 읽는 즉시 버린다. test·held-out에서는 맞추지도, 고르지도, 보고하지도 않았다.
- env에서는 `Potential("vla", root=…/t036/predictors)`로 읽는다. 입력은 결정 시점 상태의 raw VLM17 image-token 평균(`env.feat_raw`)이다. `gate.vhat`이 `end`(기본) 또는 `t023`을 고른다.

### G1 고정형
- `g = σ((τ_g − V̂_end)/T)`, **τ_g = 0.41, T = 0.1047**. env 안에서만 계산한다. 정책·관측(2092)·loss·advantage는 E2와 같다.
- g의 범위는 약 [0.004, 0.980]이다. 1이 되지는 않는다.

### G2 학습형
- `g = σ(z)`, 정책이 z를 8번째 action 성분으로 낸다. z의 평균은 선형 gate `w_v·V̂_end + w_t·(t/16) + b`(파라미터 3개)이고, 초기값은 G1과 같다(`w_v = −1/T = −9.55, w_t = 0, b = τ_g/T = 3.92`).
- 왜 z를 확률적으로 두는가: GRPO는 likelihood ratio로 학습한다. gate가 상태의 결정적 함수이면 실행 행동 `g·tanh(u)`의 지지 집합이 gate 파라미터에 따라 바뀌어 ratio가 정의되지 않는다. z를 정책의 action 성분으로 두면 기존 GRPO loss가 그대로 gate를 학습한다(RLinf 수정 없음). 평가는 평균 z를 쓰므로 `g = σ(w·[V̂, t/16] + b)` 그대로다.
- 구현(`vla_residual/gate_model.py`): RLinf `MLPPolicy`의 `backbone`·`actor_mean`만 감싼다.
  - 7차원 residual 평균은 E2와 같은 MLP가 `states[:, :2092]`만 읽는다(같은 seed에서 E2 모델과 가중치·출력이 비트 동일).
  - V̂_end는 gate head 입력용으로 `states`의 2093번째 열에 붙인다. MLP에는 들어가지 않는다.
- 명세에 없어 여기서 정한 값 2개
  - **gate 계수 step 배율 S = 100**: 계수는 `S·p`로 쓰고 gradient는 p 기준으로 흘린다. lr 1e-4인 Adam이 계수를 step당 약 0.01 움직인다(검증에서 0.0100 확인). 배율이 없으면 432 step 동안 계수가 최대 0.04밖에 못 움직여 gate가 사실상 고정된다(`w_v ≈ −10` 대비). gradient clip에는 배율이 들어가지 않는다.
  - **z의 초기 log-std = 0.0**(σ = 1 logit). residual 성분은 E2와 같은 −1.5다. G1에서 g가 0.1→0.9로 바뀌는 폭이 4.4 logit이므로 σ = 0.22(log-std −1.5)로는 gate 탐색이 거의 없다.
- 붕괴 기록: 평가마다 g 분포(전체·성공 전·성공 flag 켜짐), `collapsed_to_0`(g<0.05가 95% 초과), `collapsed_to_1`(g>0.95가 95% 초과), 마지막 checkpoint의 `(w_v, w_t, b)`와 z log-std를 `metrics.json`의 `gate`에 적는다.

## 3. τ_g, T 선택 근거 (train split만 사용)
`vla_residual/gate.py end`, 결과 `/outputs/StochRL/t036/gate_select_end.json`(T023 V̂용은 `gate_select.json`).
- 데이터: T017 buffer의 **train split 350 episode**, 전체 결정 5600개, 라벨 `success_at_end`. val·test·held-out은 쓰지 않았다.
- 규칙(T023 V̂ 때 값을 보기 전에 정한 것과 같다): train에서 예측값 → 성공 라벨의 등위 회귀(PAV)를 맞춘다.
  - `v_lo` = 성공 확률 ≤ 0.1인 가장 큰 예측값 → **0.18**. 여기서 g = 0.9.
  - `v_hi` = 성공 확률 ≥ 0.9인 가장 작은 예측값 → **0.64**. 여기서 g = 0.1.
  - `τ_g = (v_lo + v_hi)/2 = 0.41`, `T = (v_hi − v_lo)/(2 ln 9) = 0.1047`.
- train에서의 g 분포
  | 상태 | 결정 수 | g 평균 | g<0.1 | g>0.9 |
  |---|---|---|---|---|
  | 성공 전 전체 | 3501 | 0.619 | 0.227 | 0.477 |
  | 성공 전, 끝에 성공하는 episode | 1305 | 0.184 | 0.591 | 0.016 |
  | 성공 전, 끝에 실패하는 episode | 2196 | 0.877 | 0.011 | 0.750 |
  | 성공 뒤 | 2099 | 0.043 | 0.948 | 0.028 |
  - "base 성공 확률이 높은 상태 상위 x%": V̂_end ≥ 0.64인 상태는 train 전체 결정의 **49.8%**이고(성공 뒤 상태가 대부분 여기에 든다) 그 안의 끝 성공률은 0.990이다. 이 구간에서 g < 0.1이다.
  - V̂_end ≤ 0.18인 상태는 30.8%이고 끝 성공률 0.013이다. 이 구간에서 g > 0.9이다.
  - phase별 g 평균: placed 0.01, transport 0.11, grasp_contact 0.63, approach 0.70, post_grasp_lost 0.84.
- 주의
  - 예측기를 train split으로 fit했으므로 train에서의 분리는 낙관적이다. held-out B0에서 다시 본 값이 0절 표다(같은 방향).
  - Platt 보정과 λ는 val split으로 정했다(L1 지시). τ_g와 T를 고르는 데에는 train split만 썼다.
- 참고: T023 V̂로 같은 규칙을 돌리면 `v_lo` 0.15, `v_hi` 0.59, τ_g 0.37, T 0.1001이다(성공 전 결정 3501개, 라벨 success_once). 진단 평가의 T023 gate는 이 값을 쓴다.

## 4. 학습 전 검증 (`/outputs/StochRL/t036/val/`)
`vla_residual/eval_t036.py`(`val_t036.sh cpu|unit|b0`). 모두 통과했다.

| 검사 | 결과 |
|---|---|
| g = 1 경로(`gate.mode=one`)의 실행 행동 = E2 식 `clip(a_base + α·tanh u)`, 모든 sim step 비트 비교(32 env × 16 결정, 임의 u) | 통과 |
| g = 1 rollout과 gate 없는 rollout의 관측 비트 일치(순서를 바꿔 none → one → none) | one vs 뒤 none 1.000, one vs 앞 none 0.982. 같은 설정 두 번의 일치도 0.982(PhysX 비결정성 바닥)와 같다 |
| G1 실행 행동 = `clip(a_base + α·σ((τ−V̂)/T)·tanh u)` 비트 비교, 관측 2092차원 | 통과 |
| G2 실행 행동 = `clip(a_base + α·σ(u₈)·tanh u₁..₇)` 비트 비교 | 통과 |
| G2 관측: 앞 2092열 = E2 관측 구성, 2093번째 열 = V̂, 2092번째 열 = t/16 | 통과(비트) |
| G2 모델: 7차원 평균이 같은 seed의 E2 MLP와 비트 동일, 초기 σ(z 평균) = G1 | 통과(g 차이 최대 2e-7) |
| G2 gate gradient = (ΣV̂, Σt/16, N), Adam 1 step 계수 변화 = S·lr | 통과(0.0100) |
| G2 평가 모드 action = 평균, log-prob 8차원(RLinf 기본 경로) | 통과 |
| held-out B0에서 g 분포(V̂_end gate) | 0절 표. 실패 상태에서 크다(0.80). 성공 직후 상태에서 작다(0.057) |
| env가 쓴 V̂_end = 평가 스크립트가 따로 계산한 V̂_end(B0 rollout 2048 결정) | 일치 |

- env 단위 검사(위 1–5행)는 예측기를 바꾸기 전, T023 V̂ gate로 돌렸다. 그 뒤 바뀐 것은 예측기 파일 경로와 (τ_g, T) 값뿐이라 다시 돌리지 않았다. V̂_end 경로는 B0 rollout과 smoke에서 확인했다.
- g = 1 비트 확인은 env 경로까지다. `gate.mode=none`(기본값)이면 gate 코드 분기를 타지 않는다. 같은 seed로 E2 학습을 다시 돌려 보지는 않았다.
- held-out B0(이번, GPU 0): once 0.563 / end 0.539. T034의 GPU 0 B0 0.555 / 0.531과 같은 수준이다.
- **smoke**(`/outputs/StochRL/t036/smoke/`, GPU 0, 2 epoch = 128 train episode + eval 32 episode, harness 밖). 3회 모두 return code 0, metrics.json 생성, peak VRAM 17.0 GB.

  | smoke | 학습 rollout 성공률 | 학습 중 g 평균 | 학습 중 \|tanh u\| → \|g·tanh u\| | eval once / end (32 episode) | approx_kl |
  |---|---|---|---|---|---|
  | G2, V̂_end gate | 0.375 | 0.577 | 0.172 → 0.100 | 0.750 / 0.688 | — |
  | G2, T023 V̂ gate | 0.359 | 0.593 | 0.171 → 0.102 | 0.781 / 0.688 | 0.011–0.015 |
  | G1, T023 V̂ gate | 0.359 | 0.625 | 0.174 → 0.108 | 0.750 / 0.719 | 0.009–0.011 |

  - **학습 rollout 성공률이 0.36–0.38이다.** E2는 처음부터 약 0.19였다(T035 진단: 탐색 noise가 base를 망친다). gate가 탐색 noise를 줄인 효과로 본다. 128 episode뿐이라 본 run에서 다시 본다.
  - G2 V̂_end smoke: 학습 중 g 평균이 성공 episode 0.27, 실패 episode 0.76. 평가에서 성공 flag 켜진 결정의 g 평균 0.006. 2 update 뒤 계수 `(w_v, w_t, b)` = (−9.57, −0.02, 3.86), 초기 (−9.55, 0, 3.92). z log-std −0.0006. 붕괴 flag 없음.
  - eval 값은 2 epoch, 32 episode라 판단 근거가 아니다.
  - G1의 V̂_end smoke는 따로 돌리지 않았다. G1과 G2의 env 경로 차이(gate mode)는 T023 gate smoke에서, 예측기 경로는 G2 V̂_end smoke에서 확인했다.

## 5. runs
| run_id | 내용 | seed | 비교 대상 |
|---|---|---|---|
| t036-diag-e2g1 | 진단 대조군(학습 없음): E2 checkpoint 3개를 ① V̂_end gate ② T023 V̂ gate ③ gate 없이 평가(9회) | — | 같은 process의 gate 없는 평가 |
| t036-g1-s0 / s1 / s2 | G1 학습(V̂_end gate) | 0 / 1 / 2 | t022-grpo-s0 / t030-grpo-s1 / t030-grpo-s2 |
| t036-g2-s0 / s1 / s2 | G2 학습(V̂_end gate) | 0 / 1 / 2 | 같음 |

- 실행: `vla_residual/run_t036.sh <g1|g2|diag> <seed>`. GPU 0이 아니면 REFUSE하고 체인이 `-rN`으로 다시 낸다.
- 체인: `vla_residual/submit_t036.sh`. GPU 0이 비고(< 1000 MiB) t035·t036 run이 없을 때 1개씩 낸다. 로그 `/outputs/StochRL/t036/chain.log`. T035 run 3개는 모두 끝난 뒤다.
- config는 `t022_grpo.yaml` + override다. 보상(sparse), α, lr 1e-4, update epoch 4, 27 epoch(1728 train episode), eval held-out 128(seed 1000)은 E2와 같다.
  ```
  G1: env.{train,eval}.vla_residual.gate.mode=fixed  gate.vhat=end  gate.tau=0.41  gate.temp=0.1047
  G2: env.{train,eval}.vla_residual.gate.mode=learned  (vhat, tau, temp 같음)
      actor.model.obs_dim=2093  actor.model.action_dim=8
      +actor.model.vr_gate={tau:0.41,temp:0.1047,scale:100,init_logstd:0.0}
  ```
- 진단 대조군은 세 조건을 같은 process에서 번갈아 돌린다(checkpoint마다 순서를 돌린다). 같은 설정을 다시 돌려도 episode의 8–22%가 달라지므로(T034), 저장된 E2 평가가 아니라 같은 process의 gate 없는 평가와 짝지어 비교한다. `metrics.json`의 맨 위 `heldout_success_*`는 V̂_end gate 조건의 3 checkpoint 평균이다.
- 학습 run의 `metrics.json`은 T022 형식 + `gate` 블록이다.
  - 학습 중: `train_success`, `train_success_by_third`, `train_success_first2_updates`, `train_g_mean`(+ by_third, 성공/실패 episode별), `train_s_abs`(|tanh u| 평균), `train_gs_abs`(|g·tanh u| 평균)
  - 평가마다: g 분포(전체 / 성공 전 / 성공 flag 켜짐 / 상실 뒤), `loss_per_success_decision`, `gs_abs_mean`, 붕괴 flag
  - G2: `learned.{w_v, w_t, b, logstd}`
  - 결정별 원자료: `gate_decisions.jsonl`(평가), `train_episodes.jsonl`

## 6. 예산 (상한 8.0 GPU-h, GPU 0만)
- 사전 사용(harness 밖): env 단위 검사 0.105 + B0 rollout 2회 0.079 + 0.078 + smoke 3회 약 0.33 = **0.60 GPU-h**. V̂_end fit과 τ 선택은 CPU다.
- 본 run 추정: 진단 평가 0.75(평가 9회 × 약 4.7분 + 모델 적재), 학습 run은 E2·E0·T035가 각 1.23–1.25 GPU-h였으므로 1.25씩.
- 가드: `results.tsv`의 `t036-*` gpu_h 합 + 0.60 + 그 run의 추정이 8.0을 넘으면 SKIP한다. 순서가 진단 → G1 s0·s1·s2 → G2 s0·s1·s2이므로 G2 s2, G2 s1 순으로 빠진다.
- 예상: 0.60 + 0.75 + 5 × 1.25 = 7.60. **G2 s2는 SKIP될 것이다**(넣으면 8.85).

## 7. 분석 (run 종료 후, CPU)
- `analyze_t032.py` / `analyze_t033.py` 방식으로 마지막 eval의 episode별 결과를 b0.json과 짝지어 once·end의 success, preservation, recovery, regression을 낸다. 비교는 G1 3 seed, G2 seed 대 E2 3 seed다.
- 판정 기준(사전 등록): end regression이 E2(0.237)보다 낮고, end가 E2(0.662)보다 높고, once가 E2(0.752) 이상인지 본다. seed 간 표준편차와 2단계 bootstrap CI를 함께 적는다.
- 함께 볼 것
  - 학습 중 성공률(E2 약 0.19, smoke 0.36–0.38)과 `train_g_mean`: gate가 탐색 noise를 줄여 학습 rollout 성공률을 올리는지, 그것이 held-out 개선으로 이어지는지.
  - 진단 대조군: 학습 없이 gate만 씌웠을 때 once·end 변화. V̂_end gate와 T023 V̂ gate의 차이가 "성공 뒤에 닫는 효과"다. 학습한 G1과 진단 V̂_end의 차이가 "gate 아래에서 학습한 효과"다.
  - 성공 flag가 켜진 결정당 상실 확률(E2 0.021, B0 0.009): V̂_end gate에서는 B0 쪽으로 내려갈 것으로 예상한다.
  - once 유지 여부: gate가 성공 전 base가 잘 가는 구간의 residual도 줄이므로, E2가 그 구간에서 얻던 개선이 있었다면 once가 내려갈 수 있다.
  - G2: 계수가 초기값에서 얼마나 움직였는지, 붕괴 여부.
- b0.json은 GPU 1에서 잰 값이다(GPU 0 B0는 0.555/0.531). preservation·regression 절대값은 조건들에 같이 치우치고, 조건 간 비교는 모두 GPU 0 run끼리다.

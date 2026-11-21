# T040 — Q-VGM/Q-Flow 관문: 사전 등록

- 날짜: 2026-11-21. 브랜치 `exp/qvgm-gate`(`exp/residual-input` 0b2d9ca에서 분기).
- 이 파일은 **critic test 평가와 closed-loop 실행 전에 커밋한다.** 아래 숫자(설정, 판정 규칙)는 결과를 보고 바꾸지 않는다. 바꿔야 하면 summary에 "사전 등록 이탈"로 따로 적는다.
- GPU: 물리 GPU 1만 쓴다(`CUDA_VISIBLE_DEVICES=1` 강제). 상한 1.0 GPU-h(critic 학습 + closed-loop + 메모리 측정 합계).

## 1. Critic

### 1.1 입력
- z: T017 VLM layer-17 image-token mean(2048). T022 env와 같은 표준화(`vla_residual/feat_stats_vlm17_img_mean.npz`, clip ±5)를 쓴다. closed-loop에서는 env가 계산하는 `env.feat`를 그대로 쓴다(같은 hook, 같은 표준화).
- **RL token autoencoder는 쓰지 않는다.** 이유:
  1. 우리 residual 정책의 관측과 T019 critic이 이미 이 feature를 쓴다. 같은 입력이어야 T019·T022와 비교된다.
  2. T018/T019에서 이 pooled feature는 같은 결정 index 성공 AUROC 0.81–0.85를 낸다. 상태 가치 정보는 충분하다. 병목은 행동 순위(0.54)였고, 입력 압축 방식이 그 병목을 바꿀 근거가 없다.
  3. AE 학습에는 frozen VLA prefix token 전체(256×2048/결정)가 필요하다. T017 캐시에는 pooled/grid4만 있어 새 forward가 필요하다(예산 밖).
- A: **실제로 실행한 5 step 행동** `exec_env`(5×7)를 π0.5 **모델 정규화 공간**(q01/q99 → [−1,1], 차원별 affine)으로 바꿔 35차원으로 편다.
  - affine 계수는 T026 후보 파일의 `exec_env`/`exec_model` 쌍에서 최소제곱으로 얻는다(잔차 < 1e-7, 즉 정확한 affine). 그리퍼는 항등.

### 1.2 구조 (Q-VGM과 같게)
- Q: 10-head ensemble. 각 head는 MLP widths [512, 512, 256], **매 hidden layer 입력에 A를 다시 붙인다**(h_{l+1} = f([h_l, A])). LayerNorm + ReLU. 출력 스칼라.
- V(z): MLP [512, 512, 256](IQL value).
- target Q: EMA(τ=0.005).

### 1.3 데이터와 목표
- **γ = 0.99 / 결정**(Q-VGM 값. 평가 지표가 성공 여부이고 horizon이 16 결정이라 V ≈ 성공 확률에 가깝다). T022의 0.96은 쓰지 않는다.
- **buffer TD(IQL)**: T017 500 ep 중 **train+val 425 ep**(아래 fold 제외 episode 빼고). episode마다 첫 성공 결정까지만 쓴다.
  - r = 1[이 결정 chunk 안에서 첫 성공], 종료 = 성공. truncation(결정 15, 미성공)은 bootstrap해야 하지만 다음 상태 z(결정 16)가 캐시에 없다 → **해당 transition은 제외한다**(개수 기록).
  - L_V = expectile_0.8(Q̄_target(z,A) − V(z)), L_Q = Σ_heads (Q_h(z,A) − [r + γ(1−term) V(z')])².
  - T017 test 75 ep는 critic 학습에 쓰지 않는다(상태 가치 보정 보고용).
- **branch MC**(T026 50 상태 × 12 후보, T031 33 상태 × 13 후보, 각 4 반복):
  - 라벨: 후보별 4 반복 평균 G = mean_rep[ γ^k · 1(success_once) ], k = ⌊(first_success_t − t)/5⌋(이 chunk 안 성공이면 0).
  - 손실: L_MC = Σ_heads (Q_h(z, A_cand) − G)². 미니배치마다 buffer 256 + branch 64 샘플, L = L_Q + L_MC(가중 1:1).
  - T031 `E3`(3 chunk 연속 expert)는 첫 chunk 뒤에도 π0.5가 아닌 행동이 들어가 Q(s,A)의 정의와 맞지 않는다 → **학습·평가 모두 제외**.
  - T031 `u0_ind`(base와 같은 행동, 독립 noise)는 학습에서는 base 행동의 추가 라벨로 쓰고, 평가 pair에서는 뺀다(행동이 같다).
  - T017 자체 branch(160 상태)는 쓰지 않는다(명세 범위).
- **split(상태/episode 단위 cross-fit)**: branch 83 상태(T026·T031 episode는 서로 겹치지 않음)를 episode 단위 **5-fold**로 나눈다(seed 0, T026/T031 비율 층화).
  - fold f critic은 fold f 상태의 branch 라벨과 **그 episode 전체의 buffer transition을 보지 않는다.**
  - 평가는 모두 out-of-fold 예측이다. 즉 각 branch test 상태는 자기를 보지 않은 critic으로만 점수 매긴다.
- 학습: AdamW 3e-4, wd 1e-4, 15000 step, 고정(early stop·선택 없음). seed 0, 1 두 개를 돌려 점수를 평균한다(seed 앙상블).

### 1.4 기준선
- (a) **action-only Q(A)**: 같은 구조(z 자리에 0), 같은 목표. TD 목표는 full critic의 target V(z')(detach)를 그대로 쓴다(같은 y, 상태 입력만 없음). MC 라벨도 같다.
- (b) **state-only V(z)**: IQL V. 후보 순위는 상수(0.5)이고 gradient는 0. 상태 가치·보정 보고용.
- (c) **noise ceiling**: T019의 0.81(단일 continuation)을 인용하고, 이번 데이터에서는 split-half oracle(반복 0–1의 p̂로 반복 2–3의 p̂ 순위를 맞추는 pairwise accuracy)을 같이 낸다.

### 1.5 평가 (out-of-fold, branch 83 상태, 상태 bootstrap 2000회 95% CI)
- **pairwise accuracy(1차)**: 같은 상태 안 후보 쌍(base 포함, u0_ind·E3 제외) 중 p̂(4 반복)가 다른 쌍만. 가중치 = 두 후보 반복 수의 최솟값(=4, 사실상 동일 가중). 정답 = sign(Q̄_i − Q̄_j) = sign(p̂_i − p̂_j).
- **within-state AUROC**: 상태 안 (후보, 반복) 결과(성공/실패)를 Q̄로 가르는 AUROC. 성공·실패가 섞인 상태만, 상태별 pos×neg 쌍 수로 가중 평균.
- **gradient 정렬(1차)**: 각 상태에서 g = ∇_A Q̄(z, A_base)(ensemble·seed 평균). 후보 c ≠ base마다 s_c = g·(A_c − A_base), Δ_c = mean_rep[1(succ_c,rep) − 1(succ_base,rep)](CRN paired). 모든 (상태, c)를 모아 Spearman ρ(s, Δ). 같은 것을 action-only critic으로도 계산한다.
  - 보조: 유한차분 ΔQ̄ = Q̄(A_c) − Q̄(A_base)와 Δ_c의 Spearman.
- **보정**: branch (상태, 후보)에서 Q̄ vs G(10 bin ECE, MSE), 상태 간 AUROC(Q̄ vs p̂ > 0.5 아님 — (후보,반복) 성공 이진). T017 test 75 ep에서 V(z)와 Q̄(z, A_logged)의 성공 AUROC(같은 결정 index 안).
- 모든 차이(full − action-only)는 **같은 bootstrap 표본에서 paired로** CI를 낸다.

## 2. Closed-loop Q-ascent (학습 없음)
- critic: 위 구조를 **branch 83 상태 전부 + T017 train+val buffer**로 seed 0 하나만 학습한 것(평가 episode는 T017과 겹치지 않는다).
- 평가: T022 held-out 128(`heldout_episodes`), env seed 1000, 16 결정, ignore_terminations, π0.5 noise = (eid, 결정)으로 고정, 32 slot × 4 block, RC_POLICY_MB 8. T034 `eval_t034.py`와 같은 절차. 모든 조건을 **한 프로세스, GPU 1**에서 돌리고 B0도 같이 돌린다(paired).
- 매 결정: base env chunk(5×7) → 정규화 a₀. Q̄ = 10-head 평균.
  - **QA(α)**: j = 1..3: a' = a + α·√35·g/‖g‖ (g = ∇_a Q̄(z,a), 좌표당 RMS 이동 = α). Q̄(a') > Q̄(a)이면 a ← a', 아니면 멈춘다(keep-best gate).
  - **RG(α)**: 같은 크기, 결정마다 고정한 무작위 단위 방향 r(seed = eid·64 + 결정)으로 같은 gate 루프.
  - **RN(α)**: 같은 r 방향으로 3 step 무조건(최대 QA와 같은 크기, gate 없음).
  - 최종 a → env 단위 → action bound clip → 실행. 다음 결정은 π0.5가 다시 만든다.
- 실행 순서(예산 초과가 예상되면 뒤에서부터 건너뛴다): **B0, QA0.05, QA0.1, RG0.1, RN0.1, RG0.05, RN0.05**.
- 지표: once, end, preservation = P(성공 | B0 성공), recovery = P(성공 | B0 실패), regression = 1 − preservation(once·end 각각). 결정당 채택률(≥1 step 채택), 평균 채택 step, ‖Δa‖(정규화 RMS, xyz mm/step), 채택 시 Q̄ 상승량.
- CI: episode bootstrap 5000회, B0와 paired.

## 3. 메모리 측정
- 같은 GPU 1 프로세스에서 π0.5를 로드하고 VLM(PaliGemma: vision + expert 0)은 frozen, **action expert(expert 1 + action/time proj)를 학습 가능**하게 둔다.
- flow-matching loss(`compute_loss`) forward + backward + Adam 1 step의 `torch.cuda.max_memory_allocated` peak. batch {4, 8} × {전체 학습, LoRA r=16(expert 1 attention·MLP linear)}. 모델 dtype은 로드한 그대로 기록.
- scene 동시 탑재 판단: peak + scene 5–6 GB(T022 memory) + 외부 점유를 24 GB와 비교. 가능하면 closed-loop 뒤 scene이 올라간 상태에서 nvidia-smi 합계도 기록.

## 4. 판정 (사전 등록)
- **조건 1(critic)**: full − action-only의 **pairwise accuracy 차이** 또는 **gradient Spearman 차이** 중 하나의 95% CI 하한 > 0.
- **조건 2(closed-loop)**: QA(α ∈ {0.05, 0.1} 중 once+end 평균이 높은 쪽; val 없음, 두 α 모두 보고)의 B0 대비 once 또는 end 차이 CI 하한 > 0, **그리고** 같은 α의 RG·RN(돌렸다면 둘 다)보다 같은 지표 점추정이 높다.
- **Go**: 1과 2 모두 → A 또는 B 구현 제안.
- **부분**: 1만 → critic은 행동을 구분하지만 작은 ascent로는 결과가 바뀌지 않는다(T026 흡수). ascent 크기와 반복 개입에 대해 서술.
- **No-Go**: 1을 만족하지 못하면(2와 무관) "우리 과제에서 chunk critic의 행동 gradient가 정책 개선 방향을 주지 못한다" → Q-Flow/Q-VGM 계열 구현을 권하지 않는다. 2만 만족하면 critic 신호와 무관한 효과 가능성(무작위 대조와 비교)으로 따로 서술한다.

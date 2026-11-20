# T035 — Base-anchored GRPO (설계 · CRN · 검증 · 예산)

- 날짜 2026-11-21 / 브랜치 `exp/base-anchored`(`exp/residual-input` 0b2d9ca에서 분기) / worktree `.claude/worktrees/base-anchored`
- 질문: GRPO group 안에 frozen π0.5(residual 0) rollout을 함께 돌리고 advantage를 "base보다 나았는가"로 계산하면, 그룹 평균 기준인 E2 GRPO보다 regression이 줄고 once·end가 유지되거나 오르는가.
- 비교 기준: E2 3 seed(t022-grpo-s0, t030-grpo-s1, t030-grpo-s2). once/end 0.752/0.662, end preservation 0.763, regression 0.237.

## 0. 먼저 볼 것 (학습 전 진단)
- **POMO 주의가 강하게 해당한다.** 학습 전 정책(mean 0, log-std −1.5)의 탐색 noise만으로 residual 멤버 성공률이 base 멤버보다 크게 낮다.
  | | base 멤버 성공 | residual 멤버 성공 | A>0 | A<0 | A=0 | A 평균 |
  |---|---|---|---|---|---|---|
  | 단독 검증(16 group, residual 96개) | 0.531 | 0.135 | 0.042 | 0.448 | 0.510 | −0.96 |
  | smoke 2 update(16 group, residual 96개) | 0.531 | 0.156 | 0.042 | 0.448 | 0.510 | −0.88 |
  | 참고: E2 GRPO(저장된 로그, 첫 2 update) | — | 0.15–0.19 | 0.15–0.19 | 0.45–0.60 | 0.25–0.38 | 0(정의상) |
  - E2도 학습 rollout 성공률은 처음부터 0.19 수준이다(train_curve). 탐색 noise가 base를 망가뜨리는 것은 기존과 같고, anchor가 그것을 advantage에 그대로 드러낸다.
  - 예상되는 동작: 신호 대부분이 "이 noise 때문에 base보다 못했다"이다. 탐색 폭(log-std)을 줄여 base로 수렴할 수 있다. 그러면 regression은 줄지만 개선(recovery)도 줄 수 있다. 이것이 이번 질문의 답이 되므로 사전 등록대로 진행한다.
  - 결과를 볼 때 `metrics.json`의 `anchor.adv_sign_*`, `residual_member_policy_u_abs`, held-out recovery를 함께 본다.
- **base 멤버 2개는 거의 같은 결과를 낸다.** CRN으로 noise까지 같아서 16 group 중 15개에서 두 base 결과가 같았다(다른 1개는 PhysX 비결정성). b는 사실상 {0, 1}이고 두 번째 base 멤버가 주는 정보가 작다. 사전 등록 설계(2 + 6, 같은 seed 공유)를 그대로 따랐고, 후속 변형 후보로만 적는다.

## 1. 설계
- **group 구성**: train env 32 = group 4개 × 8. group 안 slot 0·1이 base 멤버, slot 2–7이 residual 멤버다(`env.train.vla_residual.anchor_n_base=2`).
  - base 멤버는 env 안에서 실행 residual을 0으로 강제한다. 정책은 base 멤버에도 행동을 내지만 env가 무시한다. 평균 |tanh u|만 `train_episodes.jsonl`의 `u_abs`로 기록한다.
  - 관측은 base·residual 멤버 모두 그대로 정책에 들어간다(rollout 경로 변경 없음).
- **advantage**(`vla_residual/anchor.py`, RLinf advantage `grpo_anchor`):
  - R = RLinf GRPO episode score(첫 done까지 보상 합, sparse라 0 또는 1).
  - b = group 안 base 2개의 R 평균. σ_res = residual 6개 R의 표준편차(RLinf GRPO와 같은 불편 추정, n−1).
  - `A_i = (R_i − b) / (σ_res + 1e-6)`, residual 멤버의 유효 step 전체에 같은 값.
  - **σ_res = 0 처리**: σ_res를 `1/√6 = 0.408`로 바꾼다. 이진 보상 6개에서 0이 아닌 σ의 최솟값이다. 글자 그대로 ε만 더하면 "residual 6개가 모두 같고 base와 다른" group에서 |A| = 10⁶이 되어 update를 지배한다. 이렇게 두면 그 경우 |A| = 2.45이고, 1개만 다른 group(σ = 0.408)과 크기가 이어진다. residual과 base가 모두 같으면 A = 0이다.
  - smoke에서 σ_res = 0인 group은 16개 중 8개, 그중 A ≠ 0은 4개였다.
- **base 멤버 loss 제외**: 두 가지를 함께 건다.
  1. advantage 함수 안에서 loss_mask의 base 열을 제자리에서 0으로 만든다. RLinf가 넘기는 mask가 `rollout_batch["loss_mask"]`의 view라서 actor가 쓰는 mask가 바뀐다(run마다 `loss_mask_is_view`를 로그로 남긴다. smoke True).
  2. base 멤버의 advantage가 0이다. PPO surrogate와 gradient가 정확히 0이다.
  - `loss_mask_sum`(episode 길이, loss의 나눗수)은 건드리지 않는다. 0으로 두면 0/0이 된다.
  - RLinf loss는 minibatch 256개 전체로 평균한다. 그래서 residual 표본당 가중치는 E2와 같고, 유효 표본만 약 25% 줄어든다.
- **보상·나머지**: sparse(첫 성공 +1, 종료), α, 관측(VLA feature 포함), lr 1e-4, update epoch 4, 27 epoch, eval held-out 128(seed 1000)은 E2와 같다. config는 `t022_grpo.yaml` + override 4개다.
  ```
  env.train.vla_residual.anchor_n_base=2  env.train.vla_residual.anchor_crn=True
  algorithm.adv_type=grpo_anchor          +algorithm.advantage_mode='anchor_nbase=2'
  ```
- **같은 총 env step**: epoch 수가 같아 rollout 양이 같다(27 × 2 × 32 episode = 1728 episode, E2 실측 0.119–0.123M env step). base 멤버가 그중 25%(432 episode)를 쓴다. residual 학습 표본 수는 `anchor.residual_samples`에 따로 적는다.
- 변형 B(b = 8개 전체 평균)는 이번에 하지 않는다.

## 2. CRN 방식
- 가능하다. train split에서 π0.5 초기 noise를 `seed = episode_id × 64 + 결정 번호 + 1,000,003 × (1000 + env seed)`로 만든다. 같은 group(같은 episode id)의 8개 멤버가 결정마다 같은 noise를 받는다(`anchor_crn=True`).
- eval은 기존 그대로다(`episode_id × 64 + 결정 번호`). held-out id는 train에서 빠지므로 겹치지 않는다.
- 검증: 64개 결정 모두에서 group 안 noise가 비트 단위로 같았다.
- 한계
  - noise만 같고 관측은 residual 때문에 결정 1부터 갈라진다. base chunk가 같다는 뜻이 아니다.
  - 결정 0(관측·noise 동일)에서도 base chunk가 group 안에서 비트 단위로 같은 비율은 0.34였다. batch 위치에 따른 부동소수 차이로 본다(최대 차이 크기는 재지 않았다).
- E2는 멤버마다 독립 noise였다. 이번 비교는 "anchor advantage + CRN"을 묶어서 E2와 비교한다. 둘을 분리하지 않는다.

## 3. 학습 전 검증 (`/outputs/StochRL/t035/val/`)
`vla_residual/validate_t035.py`. cpu는 RLinf 코드 경로를 합성 데이터로, env는 GPU 0 단일 process(32 env, 4 rollout)로 돌렸다. 둘 다 통과했다.

| 검사 | 결과 |
|---|---|
| group 안 초기 상태(물체·접시 위치, episode id, 물체 종류) | 최대 차이 1.9e-6 m, id·물체 동일 |
| base 멤버 실행 행동 = clip(a_base), 모든 sim step 비트 비교(64 결정) | 통과. 이때 base 멤버의 정책 출력은 0이 아니다(평균 \|tanh u\| 0.17) |
| residual 멤버 실행 행동 ≠ a_base | 1.00 |
| base 멤버 loss_mask = 0, advantage = 0 (actor와 같은 RLinf 호출 순서) | 통과. residual 열 mask와 loss_mask_sum은 불변 |
| RLinf PPO actor loss에서 base 행의 gradient | 정확히 0, loss 유한 |
| advantage 공식(손계산 5개 group, σ = 0 세 경우 포함) | 최대 오차 < 1e-6 |
| RLinf 경로 결과 = 공식을 유효 step에 뿌린 값 | 비트 동일 |
| base 멤버 수 0 → RLinf `grpo`와 비교 | advantage·loss_mask 비트 동일 |

- "base 0 = 기존 GRPO"는 advantage 계산 경로만 확인했다. `anchor_n_base=0`, `anchor_crn=False`이면 env 코드도 기존 분기만 탄다(코드로 확인). 같은 seed로 E2를 다시 돌려 보지는 않았다.
- **smoke**(`/outputs/StochRL/t035/smoke/`, GPU 0, 2 epoch + eval 32 episode): return code 0, metrics.json 생성, peak VRAM 17.0 GB.
  - actor 로그: update 0 `+3 −16 0:29`, update 1 `+1 −27 0:20`, residual 표본 739·693, mask된 base step 198·144, view True.
  - approx_kl 0.04–0.05, clip fraction 0.09–0.10.
  - held-out 32 episode once 0.625 / end 0.563(2 epoch, 판단 근거 아님).
- 1차 smoke는 실패로 기록한다. actor class를 patch해 mask를 걸었는데 Ray가 만든 actor 인스턴스에 적용되지 않았다. 오류 없이 base 0 분기(기존 GRPO)로 돌았고 advantage 로그가 생기지 않아 발견했다. advantage 함수 안에서 모두 처리하도록 바꾼 뒤 2차 smoke에서 확인했다.

## 4. runs
| run_id | seed (actor.seed = env.train.seed) | 비교 대상 E2 |
|---|---|---|
| t035-anchor-s0 | 0 | t022-grpo-s0 (0.742/0.625) |
| t035-anchor-s1 | 1 | t030-grpo-s1 (0.820/0.758) |
| t035-anchor-s2 | 2 | t030-grpo-s2 (0.695/0.602) |

- 실행: `vla_residual/run_t035.sh anchor <seed>`. GPU 0이 아니면 REFUSE하고 체인이 `-rN`으로 다시 낸다.
- 체인: `vla_residual/submit_t035.sh`. GPU 0이 비고(< 1000 MiB) t035 run이 없을 때 1개씩 낸다. 로그 `/outputs/StochRL/t035/chain.log`.
- metrics.json은 T022 형식 + `anchor` 블록이다.
  - `adv_sign_episode`, `adv_sign_step`, `adv_sign_episode_by_third`, `adv_sign_first2_updates`, `adv_mean`
  - `base_member_train_success`(+ by_third), `residual_member_train_success`(+ by_third)
  - `residual_samples`, `base_samples_masked`, `residual_env_steps`, `base_env_steps`
  - `sigma_zero_group_frac`, `base_pair_disagree_group_frac`, `*_policy_u_abs`, `per_update`
- 주의: `train_curve`는 base 멤버를 포함한 전체 train episode 기준이다. E2의 train_curve와 직접 비교하지 말고 `residual_member_train_success`를 쓴다.

## 5. 예산 (상한 4.2 GPU-h)
- 사전 사용: env 검증 0.08 + smoke 2회 0.10 + 0.11 = **0.29 GPU-h**.
- 본 run: E2·E0가 각 1.23–1.25 GPU-h였다. 3 × 1.25 = 3.75 → 합계 ≈ **4.04 GPU-h**.
- 가드: results.tsv의 `t035-*` gpu_h 합 + 0.29 + 추정 1.25가 4.2를 넘으면 그 run을 SKIP한다. 앞 두 run 합이 2.66을 넘으면 s2가 빠진다.

## 6. 분석 (run 종료 후, CPU)
- `analyze_t032.py` / `analyze_t033.py` 방식으로 마지막 eval의 episode별 결과를 b0.json과 짝지어 once·end의 success, preservation, recovery, regression을 낸다. 비교는 anchor 3 seed 대 E2 3 seed다.
- 판정 기준(사전 등록): end regression이 E2(0.237)보다 낮고 once·end가 E2(0.752/0.662) 이상인지 본다. seed 3개라 seed 간 표준편차와 2단계 bootstrap CI를 함께 적는다.
- b0.json은 GPU 1에서 잰 값이다(GPU 0 B0는 0.555/0.531). preservation·regression 절대값은 두 조건에 같이 치우치고, 조건 간 비교는 모두 GPU 0 run끼리다.

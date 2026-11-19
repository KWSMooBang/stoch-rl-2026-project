# T029 — 성공 유지(success-hold) 보상으로 residual PPO·GRPO 재학습 (설계 · 검증 · 예산)

- 날짜 2026-11-20 / 브랜치 `exp/success-hold`(`exp/vla-residual` 47a4d76에서 분기) / worktree `/workspace/StochRL/.claude/worktrees/success-hold`
- 배경: T022(`/outputs/StochRL/t022/design_ko.md`), T024·T028 결과. RLinf `third_party/RLinf` @ b665e18 + T021 패치 0001(T022와 같음).
- 바뀐 코드(T022 위에 이것만 더함):
  - `vla_residual/core.py`: `cfg.vla_residual.reward` 스위치. 기본값 `first_success`는 T022 그대로다(§3.3에서 비트 단위로 확인). `hold`(H1)와 `hold_dense`(H2)는 성공해도 종료하지 않는다. train 로그에 `success_at_end`, `ret`, `n_lost`를 추가했다.
  - `vla_residual/make_metrics.py`: `aulc_once`, `aulc_end`(T024 식), `hold` 통계, train 곡선의 end/return/상실 횟수, `task` 이름을 run_id에서 읽는다.
  - `vla_residual/run_t029.sh`(GPU 0만 허용, 아니면 exit 3), `submit_t029.sh`(GPU 0 전용 순차 체인, 예산 가드 3.0), `validate_t029.py`, `gate_t029.py`(제출 전 자동 판정).

## 0. 요약
- **H1 채택**: r_t = h_t − h_{t−1}(h_t = 결정 끝 성공 상태). 성공해도 종료하지 않고 16 결정을 끝까지 돈다(eval과 같다). episode return = success_at_end가 **정확히** 성립한다(단위 테스트·실제 rollout·B0/상한 모두 확인). GRPO score도 = end다. H2(성공 결정마다 1/16)는 구현만 하고 쓰지 않았다(§2).
- 성공 flag는 보상에만 쓴다. 관측은 T022와 같은 2092차원이다.
- RLinf PPO GAE는 done에서 bootstrap하지 않는다. 16번째 truncation이 종료로 처리되어 H1 finite-horizon return과 맞는다(V_16=1000 테스트).
- 기본 경로는 T022 코드와 같다(r/term/trunc/id 전부 같음, states는 run 단위 비결정성 외에는 비트 동일).
- GRPO group 초기 상태 동일(group 안 최대 1.9e-6 m).
- H1 return으로 본 B0 / 상한(held-out 앞 64): B0 0.47(once 0.48), 상한 **0.42(once 0.80)**. 상한은 성공 후 절반 가까이 잃는다.
- **L1 확인 필요**: standalone 검증 process에서 B0가 T022 val보다 낮다(같은 64 id: 0.48–0.53 vs 0.61, 한 방향 뒤집힘). T029 vs T022 run 비교는 같은 GPU 0·같은 eval 경로라 영향이 없다(§3.2).
- runs: `t029-grpo-hold-s0`, `t029-ppo-hold-s0`(GPU 0 체인, 순서대로). 예산 0.48 + 2×1.25 ≈ 2.98 / 3.0 GPU-h.

## 1. 문제와 원인
- T022 train env는 처음 성공한 결정에서 `terminated=True`로 끝난다(+1). 그래서 성공 **이후** 상태는 train 데이터에 한 번도 나오지 않는다.
- eval은 `ignore_terminations=True`로 16 결정을 끝까지 돈다. 성공 이후의 residual은 학습되지 않은 외삽이다. 계속 움직이다가 물체를 다시 건드린다.
  - 결과: t022-ppo-s1은 once 0.703인데 end 0.305다. hand-coded 상한(expert−base)도 0.820인데 end 0.453이다(expert가 놓은 뒤 base가 계속 움직임).
- 수정: train env를 eval과 같게 만든다. 즉 16 결정을 끝까지 돌리고, 보상이 "끝까지 성공 상태"를 직접 가리키게 한다.

## 2. 보상 정의
성공 상태 h_t = 결정 t의 **마지막 sim step**에서 env `success`다(eval의 success_at_end와 같은 신호). h_0 = 0이다.
- **H1 `hold`(채택)**: r_t = h_t − h_{t−1}.
  - 처음 성공 상태가 된 결정에 +1, 성공 상태를 잃은 결정에 −1, 다시 얻으면 또 +1이다. 보상 값은 {−1, 0, +1}뿐이다.
  - Σ_t r_t = h_16 = **success_at_end**(telescoping, 정확히 같음).
  - 종료 없음: `terminated`는 항상 False다. 16번째 결정에서 `truncated=True`다.
- **H2 `hold_dense`(구현만, 실행 안 함)**: r_t = h_t / 16. 합 = 성공 상태였던 결정 비율.
- **H1을 고른 이유**
  1. 목표 지표와 정확히 같다. 특히 GRPO는 episode 보상을 할인 없이 더한 값을 group 안에서 z-score한다(T028 §1). 그래서 H1에서는 GRPO score가 **success_at_end 그 자체**다(§3.1 검증). H2의 score는 "성공 상태 비율"이다. 일찍 성공하면 end에서 실패해도 점수가 높을 수 있다(예: 4~12 결정 성공 후 상실 = 0.56 > 16번째에만 성공 = 0.06).
  2. 보상 크기가 T022와 같다(성공 1회 = +1). 그래서 lr, std, value clip 등 T022 설정을 그대로 쓸 수 있다. H2는 결정당 1/16이라 신호 크기가 달라진다.
  3. H2의 장점(성공 후에도 결정마다 양의 신호 → 유지 행동에 dense credit)은 PPO에서는 H1의 −1이 대신 준다. 상실한 결정에 직접 −1이 붙고, critic이 "성공 상태에서의 가치 ≈ 0, 움직이면 손해"를 배운다.
- **edge case(문서화)**
  - chunk 중간에만 잠깐 성공하고 chunk 끝에서 성공이 아니면 H1 보상은 0이다(eval success_once는 이것도 성공으로 센다). 빈도는 §3.2 `once_only_inside_chunk`로 쟀다.
  - PPO(γ=0.96)에서 t1에 성공하고 t2에 잃으면 할인 return이 γ^t1 − γ^t2 > 0이다(작은 양수). 끝까지 유지하면 γ^t1이다. 예를 들어 바로 다음 결정에 잃으면 유지 대비 4%만 남는다. GRPO에는 이 효과가 없다(무할인, 정확히 end).
- **특권 정보**: 성공 flag는 보상 계산에만 쓴다. 정책 관측 `states`는 T022 그대로다: [VLM17 feature 2048 | qpos 8 | base chunk 35 | t/T 1] = 2092. 성공 flag나 물체 pose는 없다.
- **truncation 처리(코드 확인)**: RLinf PPO GAE는 `δ_t = r_t + γV_{t+1}·(¬done_{t+1}) − V_t`다. done(terminated든 truncated든)에서 다음 가치를 쓰지 않는다. 그리고 embodied PPO에는 truncation bootstrap 경로가 없다(`bootstrap_type`은 SAC 계열에서만 쓰임). 그래서 16번째 truncation은 사실상 종료로 처리된다. 이것이 H1에서 맞는 처리다(episode가 16 결정에서 끝나고 eval도 그때 잰다). §3.1에서 V_16 = 1000을 넣어도 return에 새지 않는 것을 확인했다.
- `loss_mask`: 종료가 없으므로 모든 episode의 16 결정 전부가 학습에 들어간다(T022는 성공 후 step을 가렸다).

## 3. 학습 전 검증 (`/outputs/StochRL/t029/val/`, GPU 0)
### 3.1 단위 테스트 (`units.json`, CPU)
| 시나리오 h_1..h_16 | H1 보상 합 | success_at_end | 상실 횟수 | RLinf GRPO score | RLinf GAE return(γ, λ=1, V_16=1000) | H2 합 |
|---|---|---|---|---|---|---|
| never `0000000000000000` | 0 | 0 | 0 | 0 | -0.0000 (= Σγ^t r_t 0.0000) | 0.000 |
| success_at_16_only `0000000000000001` | 1 | 1 | 0 | 1 | 0.5421 (= Σγ^t r_t 0.5421) | 0.062 |
| hold_from_5 `0000111111111111` | 1 | 1 | 0 | 1 | 0.8493 (= Σγ^t r_t 0.8493) | 0.750 |
| success_then_lost `0001111000000000` | 0 | 0 | 1 | 0 | 0.1333 (= Σγ^t r_t 0.1333) | 0.250 |
| success_lost_regain `0001110000111111` | 1 | 1 | 1 | 1 | 0.7668 (= Σγ^t r_t 0.7668) | 0.562 |
| success_lost_regain_lost `0101100100000000` | 0 | 0 | 3 | 0 | 0.1378 (= Σγ^t r_t 0.1378) | 0.250 |
| flicker_end_success `1010101010101001` | 1 | 1 | 7 | 1 | 0.7642 (= Σγ^t r_t 0.7642) | 0.500 |
| success_from_start `1111111111111111` | 1 | 1 | 0 | 1 | 1.0000 (= Σγ^t r_t 1.0000) | 1.000 |

- 판정: h1_sum_eq_success_at_end **True**, h1_rewards_in_pm1 **True**, h2_sum_eq_frac_success **True**, grpo_score_eq_success_at_end **True**, gae_no_bootstrap_at_truncation **True**, loss_mask_all_16 **True**.
- `success_lost_regain`·`success_lost_regain_lost`·`flicker_end_success`가 명세의 성공 → 상실 → 재성공 시나리오다. 모든 경우 H1 합 = success_at_end다.
- RLinf 함수(`calculate_scores`, `calculate_adv_and_returns`)에 실제 RLinf 배치 형식으로 넣었다. GRPO score = end이고, PPO GAE는 truncation 뒤 가치(1000)를 쓰지 않는다. loss_mask는 16 결정 전부 1이다.

### 3.2 B0와 hand-coded 상한을 H1 return으로 (`eval.json`, held-out 128, T022 val과 같은 id·같은 π0.5 noise)
- **주의(검증 스크립트 버그, 발견 후 수정)**: `reset()`을 옵션 없이 부르면 eval id cursor가 0으로 돌아간다(RLinf eval 방식, T022 이후 코드). 그래서 이 검증의 batch 2(64개)는 **held-out 앞 64개 id를 한 번 더** 돌렸다. 아래 표는 고유한 64개(첫 batch) 기준이다. 두 번째 반복은 once가 62–64/64 같았다(재현성 참고). `validate_t029.py`는 고쳤다(batch b>0은 `env_idx`=전체로 reset). 예산 때문에 128개로 다시 돌리지 않았다.

| 조건 (held-out 앞 64, GPU 0) | once | end | **H1 return 평균** | once인데 end 실패 | 상실 있는 episode | 상실 후 재성공해 end 성공 | T022 val 같은 64 id (once / end, GPU 1) |
|---|---|---|---|---|---|---|---|
| B0 (MB 8 = run.sh와 같음) | 0.484 | 0.469 | **0.469** | 1 | 1 | 0 | 0.609 / 0.609 |
| B0 (MB 32) | 0.531 | 0.531 | **0.531** | 0 | 2 | 2 | 0.609 / 0.609 |
| hand-coded 상한 (MB 8) | 0.797 | 0.422 | **0.422** | 24 | 28 | 5 | 0.766 / 0.406 |

- **H1 return = success_at_end가 모든 episode에서 정확히 성립한다**(B0·상한 둘 다 `h1_return_eq_success_at_end` True, env 내부 return도 같음).
- 상한은 once 0.80인데 H1 return(=end)은 0.42다. 상한 episode의 절반 가까이가 성공 상태를 잃는다(같은 episode 안에서 재성공도 있음). 이것이 학습 가능한 개선 폭이다. B0는 성공하면 거의 유지한다(상실 2/64).
- chunk 안에서만 잠깐 성공하고 결정 경계에서는 성공이 아닌 경우(H1이 보상하지 않는 once): B0 0, 상한 2/128 행(= 1 id × 2). 무시할 수준이다.
- **B0 재현 차이(L1 확인 필요, T029 비교에는 영향 없음)**: 같은 64 id·같은 noise·같은 코드(§3.3)인데 GPU 0의 B0 once는 0.48(MB 8)–0.53(MB 32)이다. T022 val(GPU 1, 2026-11-19)은 0.61이다. 뒤집힌 방향이 한쪽으로 몰려 있다(MB 8: 성공→실패 9, 반대 0. MB 32: 8 vs 1). 상한은 0.80 vs 0.77로 차이가 없다.
  - 원인은 확인하지 못했다(GPU/날짜/MB 후보). 같은 process 안의 반복은 거의 같았다.
  - 영향: `paired_vs_b0`는 모든 run이 같은 `t022/val/b0.json`(GPU 1)을 기준으로 쓴다. 비교 대상 t022-grpo-s0·t022-ppo-s1·T024·T028 run은 모두 GPU 0에서 돌았고 T029도 GPU 0이다. 따라서 **T029 vs T022 run 비교는 같은 조건**이다. 다만 GPU 0 run의 "Δ vs B0"는 B0가 GPU 0에서 더 낮다면 과소추정이다. 별도 B0-on-RLinf-eval-path 확인을 권한다.

### 3.3 train split: GRPO group, 보상 경로, T022 기본 경로 비트 동일 (`train.json`)
- train split, 32 env, group 8(4 group), 행동 = 0 + 초기 탐색 noise N(0, e^-1.5²)(GRPO 첫 rollout과 같은 분포), seed 11.
- H1(`reward=hold`): 종료 없음 **True**, truncation은 16번째에만 **True**, 보상 합 = success_at_end **True**, env return = end **True**, RLinf GRPO score = end **True**, loss_mask 16 결정 전부 **True**. 보상 값 [-1.0, 0.0, 1.0], 상실(−1)이 있는 episode 5/32.
- **GRPO group 초기 상태 동일**: group 안 episode id 동일 True, 물체 종류 동일 True, group 안 물체·접시 위치 최대 차 **1.9e-06 m**(group 사이 평균 0.064 m). T022와 같다.
  - group별 once [0.25, 0.375, 0.125, 0.0] vs end [0.0, 0.125, 0.0, 0.0]: 초기 탐색 noise에서는 once인데 end가 아닌 경우가 많다. H1 GRPO에서는 이들이 score 0이다.
- 관측: 2092차원, `states` 구성 변경 없음(성공 flag 없음).
- **기본 경로(`first_success`) = T022 코드(47a4d76)**: 같은 process에서 old core.py(git show)와 새 core.py를 번갈아 4회 돌렸다(T022 → T029 → T022 → T029, 같은 seed·같은 행동 noise).
  - r, term, trunc, episode id: **모든 쌍에서 같다**.
  - states: T022①=T029②=T029④가 비트 단위로 같다(**코드가 다른 쌍 포함**). T022③만 결정 1.8%에서 다르다(feature 최대 2.0, base 0.74). 첫 진단에서도 한 run(T022 코드)만 달랐다. 즉 run 단위 비결정성(renderer/π0.5 수치)이지 코드 차이가 아니다.
  - 제출 gate는 "r/te/tr/eid 전 쌍 같음 + 코드가 다른 쌍 중 하나 이상 states 비트 동일"로 판정했다(`gate_t029.py`, PASS).

### 3.4 RLinf smoke (`/workspace/stochrl_t029/smoke/t029-smoke-ppo/`, GPU 0)
- PPO + H1, 2 epoch, eval 32 ep(held-out 앞 32), `runner.max_epochs=2 val_check_interval=2 save_interval=2 env.eval.rollout_epoch=1`.
- rc 0, 383 s, 전 GPU peak 16.6 GB, train 128 episode(모두 16 결정, env step 10,240).
- metrics.json에 `aulc_once`(0.668), `aulc_end`(0.645), `hold`가 있다. train return = success_at_end **True**.
- train(탐색 noise 포함): once 0.164, end 0.055, 상실 0.109/episode. noise가 있으면 성공을 잘 잃는다. 이것이 H1이 주는 학습 신호다.
- held-out 32: once 0.750 / end 0.719(2 epoch, smoke라 의미 없음). 같은 32 id의 T022 B0 0.781이다. RLinf 경로에서는 B0와 차이가 작다(§3.2 차이는 standalone 검증 process에서만 관찰).

## 4. runs
| run_id | 알고리즘 | 비교 대상 | 나머지 |
|---|---|---|---|
| `t029-grpo-hold-s0` | GRPO | t022-grpo-s0 (0.742 / 0.625) | `t022_grpo.yaml` 그대로 + `+env.train.vla_residual.reward=hold` |
| `t029-ppo-hold-s0` | PPO | t022-ppo-s0 (0.688 / 0.508), s1 (0.703 / 0.305) | `t022_ppo.yaml` 그대로 + 같은 override |
- 같은 것: α(0.02 m, 0.1 rad, 그리퍼 2.0), 관측 2092, lr(actor 1e-4, value 3e-4), log-std −1.5, γ 0.96, GAE λ 0.95, 27 epoch × 2 rollout × 32 env × 16 결정, eval 9·18·27 epoch에 held-out 128 paired, actor/train seed 0, eval seed 1000.
- **env step 수**: T022 train은 성공 시 종료했지만 `auto_reset=False`라 rollout 길이는 원래 16 결정 고정이었다. 그래서 sim 비용과 update당 transition 수(1024)가 같다. 다만 metrics의 `train_env_steps`는 episode 길이 합이라 T029가 더 크게 나온다(성공 episode도 16 결정). eval 곡선의 x축 비교는 epoch(= 같은 sim 비용) 기준으로 읽는다.
- metrics.json: T022 형식 + `aulc_once`, `aulc_end`, `hold`(train return = end 확인, 1/3 구간별 train once/end/상실 횟수).
- 순서: GRPO → PPO.

## 5. 실행 · 예산
- 제출: `gate_t029.py`가 PASS면 `nohup setsid vla_residual/submit_t029.sh "t029-grpo-hold-s0:grpo:0:1.25" "t029-ppo-hold-s0:ppo:0:1.25"`가 자동 실행된다. 로그는 `/workspace/stochrl_t029/chain.log`다.
  - GPU 0 전용(사람 지시, GPU 1·3 금지). GPU 0이 비고 우리 run이 queued/running이 아닐 때 1개씩 낸다.
  - harness가 다른 GPU를 주면 `run_t029.sh`가 exit 3으로 거부하고 `<run_id>-rN`으로 재제출한다.
  - T028 run은 시작 전에 이미 끝났다(06:24).
- RUN_HOURS 상한 = 1.25×1.4+0.2 ≈ 1.95 h.
- GPU-h(GPU 0, 시계 시간 기준):

| 항목 | GPU-h |
|---|---|
| 1차 검증 시도(feature 통계 npz가 git ignore라 새 worktree에 없어 시작 직후 실패 → 복사) | ≈0.03 |
| 검증 train + eval + PPO smoke (06:37–06:56) | 0.31 |
| 진단(B0 MB 32, bitwise 4회) | 0.14 |
| **제출 전 합계 (`T029_PRE_GPUH`)** | **0.48** |
| 본 run 2개 (t022-grpo-s0 실측 1.25, t028 1.24) | ≈2.5 |
| **합계** | **≈2.98** (상한 3.0) |
- 예산 가드: results.tsv의 `t029-*` gpu_h + 0.48 + 다음 run 추정 1.25 > 3.0이면 그 run을 SKIP한다. GRPO가 1.27 GPU-h를 넘으면 PPO가 SKIP될 수 있다. 그 경우 L1이 추가 승인 여부를 정한다.

## 6. 해석상 주의 (L1용)
- seed 1개씩이다. PPO s0/s1의 end 차이가 0.20(0.508 vs 0.305)이라 PPO end는 seed 잡음이 크다. GRPO는 T030에서 seed 잡음을 잰다.
- 주 지표는 **paired Δend vs 같은 알고리즘의 T022 run**과 Δonce다. "once를 잃지 않으면서 end를 올린다"가 목표다. `aulc_end`가 보조 지표다.
- H1은 성공 전 episode에서 T022와 보상이 같다(성공 전에는 둘 다 0). 차이는 성공 후 구간에서만 생긴다: T022는 학습에서 빠지고, H1은 잃으면 −1이다. 그래서 once의 학습 신호는 거의 같다. 다만 성공 후 step도 policy gradient에 들어가므로 update가 그쪽에 일부 쓰인다.
- GRPO에서는 "once 성공했지만 끝에 잃은" episode가 score 0이 되어, 실패 episode와 같은 취급을 받는다. 초반 train 성공률이 낮을 때(약 0.17) 모두 0인 group이 T022보다 늘 수 있다. train 곡선의 `success_at_end`와 group 통계를 본다.

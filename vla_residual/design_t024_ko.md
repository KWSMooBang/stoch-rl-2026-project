# T024 — frozen VLA V̂ potential PBRS로 residual PPO 학습이 빨라지는가 (설계 · 검증 · 예산)

- 날짜 2026-11-19 / 브랜치 `exp/vla-pbrs`(`exp/vla-residual` 47a4d76에서 분기) / worktree `/workspace/StochRL/.claude/worktrees/vla-pbrs`
- 설계서 B1–B7(Value-to-Control B안), 배경 T022(`/outputs/StochRL/t022/design_ko.md`), T023(`/outputs/StochRL/t023/summary_ko.md`).
- 코드(바뀐 것만):
  - `vla_residual/potential.py` — T023 고정 예측기(`/outputs/StochRL/t023/predictors`, Platt)의 torch 이식 `Potential(name)` + `shaped_reward()`.
  - `vla_residual/core.py` — train env에서만 켜지는 PBRS(`env.train.vla_residual.pbrs`), π0.5 feature의 표준화 전 값(`feat_raw`) 보관, train episode 로그에 shaping 통계 추가.
  - `vla_residual/run_t024.sh` — T022 `run.sh ppo`에 PBRS override 4개만 더한다. `submit_t024.sh` — GPU 0/1 전용 제출 체인.
  - `vla_residual/make_metrics.py` — `aulc_once`, `aulc_end`, `shaping` 추가(기존 키는 그대로).
  - `vla_residual/validate_t024.py` — 학습 전 검증.
- RLinf `third_party/RLinf` @ b665e18 + T021 패치 0001(T022와 같음).

## 0. 요약
- **알고리즘**: T022 PPO(GAE)와 같은 config(`t022_ppo.yaml`)에 **train 보상만** 바꾼다. 구조, α, 관측, 하이퍼파라미터, epoch 27(≈1.2e5 env step), eval(held-out 128, 같은 noise)이 모두 같다.
- **보상**: r′ = r_sparse + β(γΦ(s′) − Φ(s)), γ = 0.96/결정(= T022 `algorithm.gamma`). 성공 종료면 Φ(s′)=1, truncation이면 Φ(s′)=V̂(s′).
- **arms**(seed 0): B3a VLA β0.3, B3b VLA β1.0, B2 raw_q β1.0, B4 time β1.0. B0는 T022 `t022-ppo-s0/s1`를 재사용한다(같은 커밋 47a4d76, β=0 경로가 비트 단위로 같음을 확인). B5(shuffled)는 예산 때문에 하지 않는다.
- **검증 통과**: 예측기 이식 오차 ≤1e-6, T023 성공 chunk 순보상 표 재현(Φ_T=1, β=1 → 1.143), 실제 env transition에서 성공·truncation·β=0·종료 뒤 처리 5개 검사 통과, RLinf smoke 통과.
- **예산**: 검증·smoke ≈0.3 GPU-h + 본 run 4 × ≈1.3 = **≈5.5 GPU-h**(상한 6.5).

## 1. 왜 PPO(GAE)이고 GRPO가 아닌가
- GRPO advantage는 같은 초기 상태 group(8개) 안의 **episode return**을 정규화한 값이고, 그 episode의 모든 step에 같은 값이 붙는다.
- PBRS 항의 episode 합은 telescoping된다: Σ_t γ^t β(γΦ(s_{t+1}) − Φ(s_t)) = β(γ^T Φ(s_T) − Φ(s_0)). (RLinf `grpo` advantage는 episode 보상을 할인 없이 합해 group 평균·표준편차로 정규화한다. 이때 합은 β(Φ(s_T) − Φ(s_0)) + β(γ−1)ΣΦ(s_t)이다. 둘째 항(γ=0.96이면 −0.04βΣΦ)은 episode 전체에 고르게 퍼진 작은 상태-의존 항일 뿐이다. 이 항도 step별 credit을 주지 않는다.)
- Φ(s_0)는 group 안에서 같다(초기 상태가 같음). 그래서 group 정규화가 이를 지운다. 남는 것은 종단 Φ(s_T)뿐이다. 성공이면 Φ_T=1이라 상수다. 실패면 V̂(s_T)라서 "실패 episode끼리 끝 상태가 얼마나 좋은가"만 약하게 더해진다.
- 즉 GRPO에서는 shaping이 **step 단위 credit**으로 전달되지 않는다. PBRS가 학습 속도에 줄 수 있는 효과(중간 상태의 진척 신호)는 TD/GAE처럼 step마다 δ_t = r′_t + γV(s′) − V(s)를 쓰는 방법에서만 나타난다. 그래서 PPO(GAE, λ=0.95)를 쓴다.

## 2. 보상 정의
- 결정(chunk, K=5 sim step) 경계마다 Φ를 잰다. s = 정책이 본 관측 상태, s′ = 다음 결정 경계의 상태.
  - r′_t = r_t + β(γΦ(s_{t+1}) − Φ(s_t)), r_t = 그 결정 안에서 처음 성공하면 1(T022 그대로).
  - **성공 종료**: Φ(s_{t+1}) = 1(T023 권고). 성공 결정의 순보상 = 1 + β(0.96 − Φ(s_t)).
  - **truncation**(16번째 결정): Φ(s_16) = V̂(s_16)를 그대로 쓴다.
  - **episode가 끝난 뒤 step**(PPO는 `auto_reset=False`라 env가 16 결정까지 계속 돈다): shaping 0. 이 step은 RLinf `loss_mask`와 GAE의 done 마스크로 원래도 학습에 안 들어간다.
- **γ = 0.96/결정**: PBRS의 정책 불변성은 shaping의 γ가 RL의 할인율과 같을 때 성립한다. T022 PPO의 `algorithm.gamma`가 0.96/결정(≈0.992/step)이라 그 값을 쓴다. T023 표는 0.99/step(chunk Γ=0.99^K≈0.951)로 계산했다. 차이는 작다(β=1 성공 순보상 중앙값 1.143 vs 1.119, §4).
- **Φ 정의**(모두 T023 고정 예측기, Platt 보정, 학습 중 업데이트 없음):
  | arm | Φ | 입력 |
  |---|---|---|
  | VLA | `vla` L2-logistic | π0.5 VLM 17층 image token 평균(2048), **표준화·clip 전 값**(T018 캐시 정의). 정책 관측의 base chunk를 만드는 같은 forward의 hook 값을 그대로 쓴다(추가 forward 없음) |
  | raw_q | `raw_q` poly2 | wrapped env `states`(qpos 8D, T017 `state`와 같은 정의) |
  | time | `lookup.npz` time × Platt | 결정 index d(0–15). d=16(truncation 상태)은 d=15 값을 쓴다 |
  - V̂ 계산 비용: 2048차원 내적 1번(결정당)이라 무시할 수 있다. feature는 기존 경로(hook)를 그대로 재사용한다.
- **truncation과 bootstrap에 대한 주의**(RLinf 코드 확인):
  - T022 PPO는 `auto_reset=False`다. RLinf는 이때 truncation bootstrap을 하지 않는다(`rlinf/data/schema/embodied_types.py` 282행: bootstrap 값은 `auto_reset`일 때만 보상에 더한다). GAE는 `dones = term | trunc`로 끊는다. 즉 T022 PPO는 16 결정 timeout을 종단으로 다룬다(B0도 같다).
  - 그래서 truncation step의 βγV̂(s_16) 항은 "끝 상태 가치를 V̂로 bootstrap하는 근사"처럼 작동한다. 엄밀히는 16 결정 유한 horizon 목적에서는 Φ(s_16)이 정책에 따라 변하므로 정책 불변성이 정확하지 않다. 명세(truncation은 V̂(s′) 사용)를 따르고 여기에 기록만 한다.
- **β=0**: env가 sparse 보상 tensor를 그대로 돌려준다(β를 곱한 0도 더하지 않음). PBRS를 켜지 않은 config는 T022 코드 경로와 같다. 바뀐 점은 post-step `_make_obs` 호출 위치를 shaping 앞으로 옮긴 것뿐이다(로그 기록과 순서만 바뀌고 RNG 소비 순서는 같다).

## 3. arms와 run
| id | potential | β | run_id | 비고 |
|---|---|---|---|---|
| B0 | 없음 | 0 | `t022-ppo-s0`, `t022-ppo-s1` 재사용 | 둘 다 git 47a4d76(이 브랜치의 분기점), RLinf 패치 0001만. AULC once 0.691 / 0.659, end 0.591 / 0.496(아래 정의로 계산) |
| B3a | VLA Platt | 0.3 | `t024-vla-b03-s0` | |
| B3b | VLA Platt | 1.0 | `t024-vla-b10-s0` | |
| B2 | raw_q Platt | 1.0 | `t024-raw-b10-s0` | |
| B4 | time Platt | 1.0 | `t024-time-b10-s0` | |
| B5 | shuffled VLA | 1.0 | — | **하지 않음**: 5.5 + 1.3 > 6.5 GPU-h |
- 모두 seed 0(actor seed, train env seed 0; eval seed 1000), T022 `t022-ppo-s0`와 같은 seed다.
- 제출 순서: vla-b10 → time-b10 → vla-b03 → raw-b10(주 대조 B3b와 B4를 먼저).

## 4. 학습 전 검증 (`/outputs/StochRL/t024/val/{offline,online}.json`)
### 4.1 offline (CPU, T018 캐시 500 ep × 16 결정)
- **예측기 이식**: torch `Potential` vs T023 `vhat_all.npz`(Platt) 최대 오차 vla 6e-8, raw_q 1e-6, time 0.
- **성공 chunk 순보상**(T023 test 성공 41 ep):
  | 규약 | Φ_T | β=0.1 | β=0.3 | β=1.0 |
  |---|---|---|---|---|
  | T023(0.99/step, Γ=0.99^K_eff) | 0 | 0.908 | 0.744 | **0.153** (min −0.010) |
  | T023 | **1** | 1.006 | 1.033 | **1.143** (min 0.951) |
  | T024(0.96/결정, 결정 보상 1) | 1 | 1.012 | 1.036 | **1.119** (min 0.980) |
  - T023 표(Φ_T=0: 0.15, Φ_T=1: 1.14)를 그대로 재현한다.
- β=0이면 `shaped_reward`가 입력 보상과 `torch.equal`이다.

### 4.2 online (GPU 0, train split, 32 env × 4 rollout = 128 ep)
- 행동: zero mean + PPO 초기 탐색 noise(u ~ N(0, e^{−1.5}²)), 즉 학습 첫 rollout과 같은 분포. 앞 3 rollout은 β=1, 마지막은 β=0.
- 검사(모두 True):
  - `success_net_ok`: 모든 transition에서 r′ = r + β(0.96·[성공이면 1, 아니면 V̂(s′)] − Φ(s))(오차 1e-6).
  - `trunc_uses_vhat`: truncation step 80개에서 Φ(s′)=V̂(s′)(1로 바뀌지 않음). 평균 0.124, 최대 0.755.
  - `beta0_bitwise`: β=0 rollout에서 env 보상과 sparse 보상이 `torch.equal`.
  - `no_shaping_after_end`: 종료 뒤 step의 shaping은 정확히 0.
  - `sparse_eq_t022`: sparse 보상 = 첫 성공 결정의 terminated와 일치.
- **성공 step 순보상(β=1)**: 중앙값 1.24, 최솟값 1.01(n=16). 직전 Φ 중앙값이 0.72로 T023(≈0.84)보다 낮다. 탐색 noise 때문에 성공 직전 상태가 덜 확실하기 때문이다. 공식은 같다.
- **feature 일치**: env에서 잰 Φ_vla(s_0) 평균 0.457 vs T023 데이터 0.475. raw_q 0.452 vs 0.454, time 0.512 vs 0.512.
- **Φ·ΔΦ 분포 vs sparse**(활성 transition 1870개; ΔΦ = 0.96Φ(s′) − Φ(s), 비종단):
  | Φ | Φ 평균 [p5, p95] | \|ΔΦ\| 평균 | ΔΦ sd | ΔΦ [p5, p95] |
  |---|---|---|---|---|
  | VLA | 0.22 [0.004, 0.73] | 0.140 | 0.214 | [−0.41, +0.33] |
  | raw_q | 0.09 [0.000, 0.46] | 0.086 | 0.188 | [−0.43, +0.20] |
  | time | 0.22 [0.005, 0.51] | 0.040 | 0.060 | [−0.20, −0.0002] |
  | sparse r | 결정당 평균 **0.011**(train 성공률 ≈0.16) | | | |
  - β=1이면 결정당 평균 |βΔΦ|(VLA 0.14)가 sparse 보상 평균(0.011)의 약 13배다. β=0.3이면 약 4배다. PPO는 advantage를 정규화하므로 학습 신호의 대부분이 shaping에서 온다.
  - time potential의 ΔΦ는 거의 항상 음수다(시간이 갈수록 성공률이 떨어짐). 상태와 무관한 "빨리 끝내라" 압력이고, 정책이 바꿀 수 있는 부분은 성공 시점뿐이다.
  - raw_q는 초기 자세가 모두 같아 Φ(s_0)이 상수(0.452)다.
- train 성공률이 0.16으로 held-out B0(0.586)보다 낮은 것은 탐색 noise 때문이다. T022 PPO 학습 초기 train 곡선도 0.16–0.21이었다(같은 현상).

### 4.3 RLinf smoke (`/outputs/StochRL/t024/smoke/t024-smoke-vla.metrics.json`, GPU 0)
- VLA β=1, 2 epoch, eval 32 ep. rc=0, 376 s, 전 GPU peak 17.0 GB(T022와 같은 수준).
- `train_episodes.jsonl`에 shaping 필드가 기록되고, `metrics.json`에 `aulc_once`·`shaping`이 나온다.
- 성공 step 보상 평균 1.28 = 1 + (0.96 − 0.68)으로 공식과 같다.
- 확인한 버그 1건: 처음 구현에서 shaping이 post-step feature 갱신보다 먼저 실행되어 Φ(s′)에 이전 상태 값이 들어갔다. online 검사 `success_net_ok`/`trunc_uses_vhat`가 이를 잡았고, 갱신 뒤에 shaping하도록 고쳤다.

## 5. 지표 (`metrics.json`, T022 형식 + 추가)
- T022와 같은 키: `eval_curve`(env step, wall, 매 eval paired vs B0), `paired_vs_b0_once/end`, `best_heldout_success_once`, `peak_vram_*`, `train_curve`.
- **`aulc_once`**(primary): held-out success_once 곡선을 env step으로 사다리꼴 적분하고 마지막 env step으로 나눈 값. B0 점(0, 0.586)을 포함한다. `aulc_end`도 같다.
  - env step은 끝난 train episode의 결정 수 × 5다(T022 정의). 성공 시 episode가 일찍 끝나므로 run마다 마지막 env step이 조금 다르다(T022 PPO s0 123k, s1 120k). 정규화하므로 AULC는 "곡선의 평균 높이"다.
  - eval은 epoch 9·18·27의 3번뿐이라 점 4개짜리 곡선이다(T022와 같게 둔다).
- **`shaping`**: 결정당 sparse 보상과 |shaping| 크기, episode shaping 합(성공/실패별), Φ(s_0), 성공 step 보상과 직전 Φ, truncation Φ(s′), residual 크기 |tanh u|(전체·학습 3등분별), train 성공률(3등분별).
  - `train_episodes.jsonl`의 `phi_next_end`는 종료 override 전 V̂(s′)이다. 실제 보상은 `r_end`다.

## 6. 성공 후 end 하락 (T022 PPO: once 0.69–0.70인데 end 0.31–0.51)
- 학습 env는 성공한 결정에서 `terminated=True`로 끝난다(`ignore_terminations: False`). `auto_reset=False`라 env는 16 결정까지 계속 돌지만, 그 뒤 step은 `loss_mask`와 GAE done 마스크로 학습에서 빠진다. 평가 env는 `ignore_terminations: True`라 성공한 뒤에도 16 결정을 끝까지 돌고, 마지막 상태로 success_at_end를 잰다.
- 그래서 정책은 성공 뒤 상태에 대해 **한 번도 학습 신호를 받지 않는다**. residual은 성공 뒤에도 같은 크기로 계속 더해지고, 물체를 놓은 뒤 base+residual이 계속 움직여 접시에서 물체를 건드려도 학습에는 아무 비용이 없다. 성공을 더 일찍·더 자주 만드는 방향으로 residual이 커질수록 성공 뒤 교란도 커질 수 있다. hand-coded 상한(expert residual)도 once 0.82 / end 0.45로 같은 패턴이다(T022 §2).
- T024는 비교 가능성을 위해 평가 방식을 T022와 같게 둔다. Φ_T=1도 성공 이후 상태를 학습에 넣지 않으므로 이 현상에 직접 영향은 없다. 원인 확인(성공 뒤 residual 크기와 물체 변위)은 L1 분석 몫이다.

## 7. 실행 · 예산
- 제출: `nohup setsid vla_residual/submit_t024.sh "t024-vla-b10-s0:vla:1.0:0:1.3" "t024-time-b10-s0:time:1.0:0:1.3" "t024-vla-b03-s0:vla:0.3:0:1.3" "t024-raw-b10-s0:raw_q:1.0:0:1.3"`, 로그 `/workspace/stochrl_t024/chain.log`.
  - GPU 0 또는 1이 비었을 때만 harness에 낸다(동시 최대 2개). harness가 다른 GPU(예: 3)를 주면 `run.sh`가 exit 3으로 거부하고 체인이 `<run_id>-rN`으로 다시 낸다. **GPU 3은 쓰지 않는다.**
  - 예산 가드: results.tsv의 `t024-*` gpu_h + 0.3(검증) + 다음 run 추정 1.3 > 6.5이면 건너뛴다.
  - run당 RUN_HOURS 상한 = 1.3×1.4+0.2 ≈ 2.0 h.
- GPU-h:
  | 항목 | GPU-h |
  |---|---|
  | online 검증 3회(1회는 파일 누락으로 시작 직후 실패, 1회는 버그 발견) | ≈0.17 |
  | RLinf smoke | ≈0.11 |
  | 본 run 4개(T022 PPO 실측 1.29/run) | ≈5.2 |
  | **합계** | **≈5.5** (상한 6.5) |
- 주의: 같은 GPU에 외부 job이 올라오면 느려지거나 OOM이 날 수 있다(T022와 같은 위험). GPU 0/1이 차 있으면 run이 순차로 돌아 끝까지 5시간 이상 걸릴 수 있다.

## 8. 해석상 주의 (L1용)
- seed 1개씩이다. T022 PPO s0/s1의 AULC 차이(0.691 vs 0.659)와 마지막 once 차이(0.688 vs 0.703)가 seed 잡음의 크기를 보여 준다. B0는 두 seed를 다 쓸 수 있다.
- β=1이면 shaping이 학습 신호를 지배한다(§4.2). Φ의 성공 직전 dip(T023 §3)이 정책을 "내려놓기 직전 머뭇거림" 쪽으로 밀 수 있다. `shaping.success_step_phi_prev`와 성공 결정 분포를 보면 된다.
- V̂는 π0.5 behavior policy의 성공 확률이다. residual 정책이 바뀌면 calibration이 바뀐다(Φ는 고정이므로 PBRS 불변성 자체는 유지된다).

# T028 — GRPO + VLA V̂ potential PBRS (설계 · 검증 · 예산)

- 날짜 2026-11-20 / 브랜치 `exp/grpo-pbrs`(`exp/vla-pbrs` 59a6043에서 분기) / worktree `/workspace/StochRL/.claude/worktrees/grpo-pbrs`
- 배경: T022 GRPO(`t022-grpo-s0`, 47a4d76, once 0.742 / end 0.625), T024 PBRS(`/outputs/StochRL/t024/design_ko.md`), T023 예측기(`/outputs/StochRL/t023/predictors`, Platt).
- 바뀐 코드(T024 코드 위에 이것만 더함):
  - `vla_residual/run_t028.sh`: `run.sh grpo`(= `configs/t022_grpo.yaml`)에 T024 PBRS override 4개를 더한다. GPU 0이 아니면 exit 3으로 거부한다.
  - `vla_residual/submit_t028.sh`: T024 체인과 같다(GPU 0 전용, 한 번에 1개, 거부되면 `-rN`으로 재제출, 예산 가드).
  - `vla_residual/validate_t028.py`: 아래 §3 검증.
  - env·보상·metrics 코드는 T024와 같다(`core.py`, `potential.py`, `make_metrics.py` 변경 없음).
- RLinf `third_party/RLinf` @ b665e18 + T021 패치 0001(T022·T024와 같음).

## 0. 요약
- **RLinf GRPO advantage는 episode 단위다.** 할인하지 않은 episode 보상 합을 group(8) 안에서 z-score로 바꾸고, 그 값을 episode의 모든 step에 똑같이 붙인다. step 단위 return-to-go가 아니다.
- 그래서 shaping은 결정별 credit을 주지 않는다. episode score에 더해지는 것은 β(γΦ′_T − Φ_0) + β(γ−1)Σ_{t=1}^{T−1}Φ_t이다.
  - γ^T 꼴이 아니다. RLinf가 할인하지 않기 때문이다. 대신 경로 항이 붙는다.
  - 실제 rollout 64 episode에서 RLinf score와 이 식의 차이는 최대 3e-7이었다. 명세의 할인 telescoping 식 β(γ^TΦ_T − Φ_0)와는 최대 0.21(β=1)까지 차이가 났다.
- 실효 효과는 세 가지다.
  - ① 실패 episode끼리 마지막 상태 V̂로 순위가 매겨진다.
  - ② **모두 실패한 group**: sparse만 쓰면 advantage가 정확히 0이다. shaping을 넣으면 std 정규화 때문에 |A|≈0.85가 된다. 이 크기는 β와 무관하다.
  - ③ 일찍 성공할수록 경로 항이 덜 쌓여 score가 조금 높다.
- **β=0이면 T022 GRPO와 비트 단위로 같다.** 같은 seed·같은 행동으로 비교했을 때 보상, term, trunc, episode id, RLinf GRPO advantage가 모두 `torch.equal`이었다.
- smoke(2 epoch, β=1) 통과: rc 0, peak 17.0 GB. `metrics.json`에 `aulc_once`와 `shaping`이 있다.
- **runs**: `t028-grpo-vla-b03-s0`(β=0.3), `t028-grpo-vla-b10-s0`(β=1.0)이고 seed 0이다. 비교 대상은 `t022-grpo-s0`이다.
- **예산**: 검증·smoke ≈0.32 + run 2 × ≈1.25 = **≈2.82 GPU-h**. 승인 상한은 2.8이라 lead에 확인했다(§5).

## 1. RLinf GRPO advantage 계산 (코드 확인)
경로: `embodied_fsdp_actor_worker.compute_advantages_and_returns` → `algorithms/registry.calculate_adv_and_returns` → `utils.preprocess_embodied_advantages_inputs` → `utils.calculate_scores` → `advantages.compute_grpo_advantages`.
1. `_process_received_rollout_batch`: rollout epoch(2)를 batch 축에 쌓는다. 배치는 [16 결정, 2×32, 1]이다. 같은 episode id 8개가 연속 slot에 있으므로 `reshape(-1, 8)`이 곧 group이다. 실제 rollout에서 8 group 모두 확인했다.
   - `auto_reset=False`·`ignore_terminations=False`이면 `loss_mask = (dones.cumsum == 0)`이다. 즉 첫 done까지의 step만 학습한다.
2. `calculate_scores`(utils.py 182행):
   ```
   scores = 0
   for step in reversed(range(n_steps)):
       scores = scores * ~dones[step+1]
       scores += rewards[step]
   ```
   뒤에서부터 더하고 done에서 0으로 되돌린다. 그래서 최종값은 **첫 done까지 보상의 할인 없는 합**이다. `gamma`는 GRPO 경로에서 쓰이지 않는다.
   - 성공한 episode는 16 결정에서 trunc done이 한 번 더 나오지만, 그 뒤 꼬리는 첫 done에서 잘린다.
3. `compute_grpo_advantages`(advantages.py 210행): A = (score − group 평균) / (group std(불편) + 1e-6)이다. 이 값을 그 trajectory의 모든 step에 똑같이 넣고 `loss_mask`를 곱한다.
4. 결론: **episode return 방식이다(step 단위 return-to-go가 아니다).** advantage는 trajectory 안에서 상수다(실측 max range 0).

## 2. 이 구조에서 PBRS가 실제로 하는 일
r′_t = r_t + β(γΦ_{t+1} − Φ_t), γ=0.96/결정(T024와 같다). 성공 종료면 Φ_T=1이고, truncation이면 Φ_T = V̂(s_16)이다. 할인 없이 T개 결정을 더하면 아래 식이 된다.

  score = Σr + β(γ·Φ′_T − Φ_0) + β(γ−1)·Σ_{t=1}^{T−1} Φ_t

- **Φ_0**: group 안에서 거의 같다. 실측 group 안 sd 최대 0.008, group 사이 sd 0.21이다. 같은 초기 상태지만 렌더/feature가 1e-3 수준에서 다르다. score 분산 기여는 0.2%라 무시할 수 있다.
- **종단 항 βγΦ′_T**: 성공이면 상수 βγ, 실패면 βγV̂(s_T)이다.
  - 실패 episode끼리 "끝 상태가 얼마나 좋아 보이는가"로 순위가 매겨진다. group 안 실패 episode에서 score 순위와 V̂(s_T) 순위의 Spearman 평균은 0.93이었다(8 group).
- **경로 항 β(γ−1)ΣΦ_t = −0.04β·ΣΦ_t**: 할인하지 않아서 생기는 항이다. telescoping 식에서 이 부분이 남는다.
  - 높은 V̂ 상태에 오래 머문 episode가 오히려 조금 깎인다. 진척 방향과 반대인 작은 압력이다.
  - 일찍 성공하면 항이 덜 쌓이므로, 명세가 말한 "γ^T 차이"(조기 성공 이점)는 이 항을 통해 나타난다.
  - β=1 예시 group에서 경로 항은 −0.07~−0.15였다. 실패 episode 종단 항(0.003~0.17)과 크기가 비슷하다. 그래서 실패끼리의 순위가 V̂(s_T)만으로 정해지지는 않는다(Spearman 0.93 < 1).
- **결정별 신호는 없다.** 같은 episode 안의 모든 결정이 같은 A를 받는다. PBRS의 장점인 "중간 상태 진척을 그 결정에 귀속"하는 효과는 GRPO에서는 생기지 않는다(T024 §1과 같은 결론).
- **std 정규화와의 상호작용(가장 큰 실효 차이)**:
  - sparse만 쓰면 모두 실패한 group(실측 8개 중 3개, train 성공률 0.17)은 std가 0이라 A=0이다. 학습 신호가 없다.
  - shaping을 넣으면 그 group의 score sd는 0.15–0.31(β=1) 또는 0.04–0.09(β=0.3)이다. 이것이 1로 정규화되어 |A| 평균 0.83–0.88이 된다. **β=0.3과 β=1.0에서 값이 같다.**
  - 즉 모두 실패한 group에서는 β 크기와 상관없이 "V̂(s_T)가 높은 실패 쪽으로" 정책을 전력으로 민다. 모두 성공한 group도 같은 방식이다(경로 항, 곧 조기 성공).
  - β는 성공·실패가 섞인 group에서만 의미가 있다. 성공·실패 차이 대비 V̂ 차이의 상대 비중을 정한다. within-group score 분산 중 종단 Φ 몫은 β=1에서 49%, β=0.3에서 21%다.
  - 예시 group(성공 2/8)의 advantage는 β=1이면 실패 −0.45~−0.67, 성공 +1.61이다. β=0.3이면 실패 −0.50~−0.60이다. sparse만 쓰면 실패는 모두 같은 값이다.
- 해석상 주의: V̂는 π0.5 behavior의 성공 확률이다. 모두 실패한 group에서 "끝 상태 V̂ 최대화"가 실제 성공률을 올리는지는 V̂가 얼마나 정확한지에 달렸다. T023 기준 끝 상태 근처의 AUROC를 참고한다.

## 3. 학습 전 검증 (`/outputs/StochRL/t028/val/{grpo,beta0}.json`, GPU 0)
### 3.1 GRPO score 등식 (`--mode grpo`, train split, 32 env × 2 rollout = 8 group, β=1 기록 → β=0.3은 shp 선형으로 재계산)
- 행동: zero mean + 초기 탐색 noise u ~ N(0, e^{−1.5}²)이다. GRPO 첫 rollout과 같은 분포다.
- 실제 transition을 RLinf 함수(`calculate_scores`, `calculate_adv_and_returns`)에 그대로 넣었다.
| 검사 | β=1.0 | β=0.3 |
|---|---|---|
| RLinf score = 첫 done까지 shaped 보상의 할인 없는 합 (max err) | 2.6e-7 | 1.0e-7 |
| RLinf score = 닫힌 식 Σr + β(γΦ′_T − Φ_0) + β(γ−1)ΣΦ_t (max err) | 3.1e-7 | 1.2e-7 |
| RLinf score vs 할인 telescoping Σr + β(γ^TΦ′_T − Φ_0) (max / mean abs 차) | 0.21 / 0.083 | 0.064 / 0.025 |
| advantage가 trajectory 안에서 상수 (max range) | 0 | 0 |
| advantage = group z-score(불편 std) (max err) | 2.7e-7 | 2.5e-7 |
| within-group score 분산 몫: sparse / 종단 Φ / 경로 / Φ_0 | 0.52 / 0.49 / −0.008 / 0.002 | 0.79 / 0.21 / −0.005 / 0.001 |
| 실패끼리 score 순위 vs V̂(s_T) Spearman (8 group) | 0.93 | 0.93 |
| 모두 실패한 group(3개)의 \|A\| 평균 (sparse만이면 0) | 0.83 / 0.86 / 0.88 | 0.83 / 0.86 / 0.88 |
- 예시 group 0(β=1): 성공 2(T=5), 실패 6.
  - score [−0.567, −0.667, 1.225, −0.759, −0.566, −0.593, −0.709, 1.238]이고, 닫힌 식과 소수 넷째 자리까지 같다.
  - 할인 telescoping 식으로 계산하면 [−0.556, −0.566, 1.188, −0.632, …]이 되어 맞지 않는다.
- train 성공률 0.17, 성공 T 평균 7.7(최소 4), 메모리 peak 8.7 GB(torch).

### 3.2 β=0 비트 단위 (`--mode beta0`, 32 env, group 8, seed 11, 같은 행동 noise)
- 같은 프로세스에서 세 run을 차례로 돌렸다: T022 경로(pbrs cfg 없음) → pbrs β=0 → T022 경로 다시(대조).
| 비교 | 보상 | term | trunc | episode id | RLinf GRPO advantage |
|---|---|---|---|---|---|
| T022 vs T022(대조) | equal | equal | equal | equal | equal |
| T022 vs pbrs β=0 | **equal** | equal | equal | equal | **equal** |
- 코드상으로도 β=0이면 `_shape`가 sparse 보상 tensor 객체를 그대로 돌려준다(T024 §2).
- PBRS를 켜지 않은 경로는 T022 47a4d76과 같다(T024에서 확인). 그래서 t022-grpo-s0이 β=0 기준선이다.

### 3.3 RLinf smoke (`/workspace/stochrl_t028/smoke/t028-smoke-vla/`, GPU 0)
- VLA β=1, 2 epoch, eval 32 ep, `runner.save_interval=2`.
  - 첫 시도는 `save_interval=27 must be divisible by val_check_interval=2` assertion으로 실패했다. smoke 전용 override 문제이고 본 run은 T022 값(27/9)이라 해당 없다.
- rc 0, 388 s, 전 GPU peak 17.0 GB, train 128 episode.
- shaping 통계: 결정당 sparse 0.011, |shaping| 0.145(T024 PPO와 같은 수준). 성공 step 보상 평균 1.42, 실패 episode shaping 합 −0.40, 성공 episode +0.33.
- held-out 32 ep once 0.72(2 epoch 뒤, smoke라 의미 없음).

## 4. runs
| run_id | Φ | β | 나머지 |
|---|---|---|---|
| `t028-grpo-vla-b03-s0` | VLA Platt | 0.3 | t022-grpo-s0과 같다(`t022_grpo.yaml`): group 8, critic 없음, `auto_reset=False`, lr 1e-4, log-std −1.5, α, 관측 2092D, 27 epoch(≈0.12M env step), eval 9·18·27 epoch에 held-out 128 paired, actor seed 0, train seed 0, eval seed 1000 |
| `t028-grpo-vla-b10-s0` | VLA Platt | 1.0 | 같음 |
- 기준선 B0(β=0) = `t022-grpo-s0`(once 0.742 / end 0.625, best 0.773, 1.25 GPU-h)이다. §3.2에 따라 재사용한다.
- metrics.json은 T024와 같은 형식이다(`aulc_once`, `aulc_end`, `shaping`, eval_curve, paired_vs_b0).
  - t022-grpo-s0의 AULC는 같은 `make_metrics.py`로 다시 계산해 비교한다(L1).
- 제출 순서: b03 → b10.

## 5. 실행 · 예산
- 제출: `nohup setsid vla_residual/submit_t028.sh "t028-grpo-vla-b03-s0:vla:0.3:0:1.25" "t028-grpo-vla-b10-s0:vla:1.0:0:1.25"`, 로그 `/workspace/stochrl_t028/chain.log`.
  - GPU 0 전용이다(사람 지시, GPU 1·3 금지). GPU 0이 비어 있고 우리 run이 queued/running이 아닐 때만 1개씩 제출한다.
  - harness가 다른 GPU를 주면 `run_t028.sh`가 exit 3으로 거부하고 체인이 `<run_id>-rN`으로 다시 제출한다.
  - T024 체인(t024-raw-b10-s0)은 T028 검증 전에 이미 끝났다.
  - RUN_HOURS 상한 = 1.25×1.4+0.2 ≈ 1.95 h.
- GPU-h:
| 항목 | GPU-h |
|---|---|
| 검증 grpo 2회(2회째는 지표 추가) + beta0 1회 | ≈0.17 |
| smoke 2회(1회는 config assertion으로 시작 직후 실패) | ≈0.15 |
| 본 run 2개(t022-grpo-s0 실측 1.25/run) | ≈2.5 |
| **합계** | **≈2.82** (승인 상한 2.8, lead에 확인) |
- 예산 가드: results.tsv의 `t028-*` gpu_h + `T028_PRE_GPUH`(0.32) + 다음 run 추정 1.25가 `T028_BUDGET`을 넘으면 그 run을 건너뛴다.

## 6. 해석상 주의 (L1용)
- seed 1개씩이다. T022 GRPO는 s0만 있어 seed 잡음 크기를 모른다. 참고로 PPO s0/s1의 AULC 차이는 0.03이었다.
- §2에 따라 GRPO+PBRS의 차이는 "속도(결정별 credit)"가 아니다. 주로 **모두 실패하거나 모두 성공한 group에도 신호가 생긴다**는 데서 온다(train 성공률이 낮은 초반에 group 37%가 모두 실패).
  - 그 신호의 방향은 V̂(s_T)와 경로 항이 정한다. β=0.3과 β=1.0의 차이는 섞인 group에서만 생긴다.
- 경로 항(−0.04β·ΣΦ_t)은 높은 V̂ 상태에 오래 머무는 것을 벌한다. 성공 직전 머뭇거림(T023 dip)과 결합하면 해석이 복잡하다. `shaping.episode_shaping_sum_fail`과 성공 시점 분포를 본다.

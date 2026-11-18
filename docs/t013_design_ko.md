# T013 — DA-CGC gridworld PoC: 구현 정의, Phase 0–1 결과, 관문 판정

- 브랜치 `exp/dacgc` (local main 4eeb6b6에서 분기), 코드 `dacgc/`, 테스트 `tests/`
- 설계서: 01_Decision_Aware_Counterfactual_Goal_Consistency_RL.md (사람). 아래는 설계서 대비 **구현 정의와 바꾼 점**.
- 계산: 전부 CPU(torch 1 thread/run, run들은 process pool로 병렬). harness는 GPU 1개를 잡지만 쓰지 않는다(`dacgc/run_cpu.sh`가 GPU 3 배정 시 거부, `CUDA_VISIBLE_DEVICES=""`).

## 1. 환경 (설계서 4.1–4.2)

| 항목 | 구현 |
|---|---|
| 격자 | 8×8, object 3 (Bottle/Bowl/Box), target 3 (Rack/Plate/Stove), 매 episode 무작위 서로 다른 칸 |
| action | Up/Down/Left/Right/Pick/Drop (6) |
| dynamics | deterministic. 벽은 clip. Pick: 빈손 + 물체 칸이면 집음. Drop: 다른 물체가 없는 칸이면 내려놓음 |
| commit | target 칸에 Drop = commit → episode 종료. (goal object, goal target)이면 성공 r=1, 아니면 r=0 |
| outcome | success / wrong_target(지시 물체를 다른 target에) / wrong_object(다른 물체를 target에) / timeout |
| horizon | 50 (truncation, bootstrap) |
| 관측 | goal과 무관한 object-centric 특징 54차원: 절대 좌표(agent·object·target), 모든 쌍의 상대 벡터(agent→object, agent→target, object→target), 같은 칸 indicator, held one-hot. goal은 네트워크 입력으로 따로 |
| split | train 6 / held-out 3: Bottle→Stove, Bowl→Rack, Box→Plate (설계서 표 그대로) |
| skew | object별 선호 target: Bottle→Rack, Bowl→Plate, Box→Stove (각 target이 한 번씩 선호되도록). object는 균등, 선호 target 확률 = skew, 나머지 train target = 1−skew |
| Gymnasium | `GridPickPlaceEnv` (obs에 goal one-hot 포함), `clone_state()/restore_state()` |

SER 정의: held-out goal (o,t)에서 **지시 물체 o를 다른 target에 commit**한 비율. 이를 (i) o의 **자주 본 target**(PREF_TARGET[o])으로 간 것(`SER_unseen_freq`)과 (ii) 나머지 target(`unseen_wt_other`)으로 나누고, 다른 물체 commit(`unseen_wrong_obj`), 시간 초과(`unseen_timeout`)를 따로 센다. seen/unseen 평가는 **같은 100개 layout**에 goal만 바꿔 실행한다.

### 1.1 정답 Q* ("oracle")
보상이 성공 commit에서만 1이므로 Q*(s,a,g) = γ^{D(s',g)} (D = 성공 Drop까지 최소 action 수; commit이면 r). 이동은 막힘이 없으므로 D는 Manhattan 거리로 닫힌 형태다(다른 물체를 들고 있으면 놓을 빈 칸을 거치는 최소 경로). 이것이 value iteration의 고정점이며, `tests/test_env_oracle.py`에서 실제 simulator 위 brute-force BFS(단위 비용 VI)와 D가 일치함을 확인했다. oracle 정책 rollout은 모든 시작 상태에서 정확히 D step에 성공한다.

## 2. 학습기 (모든 방법 공통)

Goal-conditioned Double DQN, MLP 2×256 (goal = object one-hot ⊕ target one-hot concat; FiLM은 ablation), Adam 3e-4, batch 128, replay 200k, 4 env step당 1 update, target sync 1000 update, γ=0.95, 16 병렬 env, ε 1→0.05 (처음 250k step 선형), **같은 env step 예산**(Phase 0–1: 2M).

Phase 0 점검 중 바꾼 점(모든 방법에 동일 적용, 이유와 근거):

| 변경 | 이유 (측정) |
|---|---|
| 관측: one-hot plane(452차원) → object-centric 좌표 특징(54차원) | plane 입력에서는 Q* 직접 회귀조차 행동 순위를 못 맞춤 |
| 값 척도: 네트워크가 Q̃=(Q−1)/(1−γ)를 출력 | 0/1 보상의 Q*는 인접 행동 차이가 ~0.03이라 TD로 순위가 안 잡힘. Q̃는 Q의 affine 변환(= HER식 −1/step, 실패 commit은 −1/(1−γ) 흡수)이라 **최적 정책이 같고** action gap이 20배. 지도 회귀 sanity: MSE 5e-5에서야 SR 0.93 |
| PBRS shaping Φ = 0.1·([goal object를 듦] − [빈손]·d(agent, goal object)/14) | target 정보 없음 → shortcut이 생기는 target 선택은 여전히 sparse. 단 PBRS는 고정점에서 Q−Φ(s)라 행동 순위를 바꾸지 않으므로(Wiewiora 2003) 효과는 초기화 수준 |
| 학습 episode의 50%는 무작위 물체(goal과 무관)를 든 채 시작 | 무작위 정책 성공률 0.09%/episode → 0.4%. 평가는 항상 빈손 시작 |
| n-step=3 return | Rainbow 표준. 효과 작음 |
| ε 감쇠를 고정 250k step | 예산에 비례 감쇠(30%)면 학습 시작이 늦어짐 |

시도했지만 효과가 없어 쓰지 않은 것(600k step, balanced B0, SR_seen 0): lr 1e-3, target sync 250, n-step 6, batch 256/2 step당 update, γ=0.9, p_hold 0.8, hidden 512, horizon 80, εz-greedy(Dabney et al. 2021; 무작위 성공률을 오히려 낮춤), drop을 target 칸에서만 허용하는 dynamics 변형.
버그 확인: 한 goal·물체를 든 채 시작하는 쉬운 과제에서는 같은 코드가 125k step에 SR 1.0.

## 3. 방법 정의 (Phase 3에서 비교)

| ID | 정의 |
|---|---|
| B0 | 위 Double DQN |
| B1 | B0 + HER('final': commit한 (o',t')로 episode 전체 재레이블, **train goal일 때만**) |
| B2 | B0 + naive separation: replay 상태마다 두 train goal의 Boltzmann 정책 겹침 Σ_a p_i(a)p_j(a) 최소화(분기 여부 무시) |
| B3 | B0 + **gate 없는 정답 ranking**: B4와 같은 evaluator·같은 (s,g_i,g_j) 표본. 각 goal에 대해 positive = evaluator 최선, negative = 상대 goal 최선 행동이 이 goal에 엄격히 나쁘면 그것, 아니면 엄격히 나쁜 행동 중 무작위. **분기·공통 상태 모두** |
| **B4** | B0 + DA-CGC: 같은 evaluator, gate w=1[a_i≠a_j ∧ Δ_i>τ ∧ Δ_j>τ] 켜진 항목에만 [m−Q(s,a_i,g_i)+Q(s,a_j,g_i)]_+ + 대칭항 |
| B6 | B0 + **정답 Q* 회귀**: replay 상태마다 두 train goal, 6 action 모두 (Q̃_θ − Q̃*)² (shaping 반영한 Q*). 지도 정보량을 맞춘(오히려 더 많은) 대조군 |

공통: CGC 표본은 replay 상태 256개 × 무작위 train goal 쌍을 250 update마다 평가해 cache(16k)에 넣고, update마다 aux batch 128. B4는 gate-on 항목에서만 뽑는다(설계서 E[w(ℓ_i+ℓ_j)]의 조건부 형태; update당 ranking 항 수를 B3와 맞추기 위함). margin m=0.3(Q̃ 단위, 인접 행동 차이 ≈ 0.4–1.9). B2 온도 1.0(Q̃ 단위).

Evaluator J^{(K)}_{g,μ}(s,a): snapshot 복제 → a 실행 → μ로 K−1 step → V=0 bootstrap. 6 action 전부 후보. tie-break: 각 goal의 argmax 집합 안에서 상대 goal에 가장 나쁜 행동을 고르고, 두 goal 모두에 최적인 행동이 있으면 공유 행동으로 보고 gate off.
- μ = **oracle evaluator**(정답 정책; 주 조건). K≥남은 거리면 J=Q*.
- μ = learned(현재 online Q의 greedy; ablation). 같은 항목에 oracle gate도 계산해 recall/precision, "shortcut 상태"(oracle gate on인데 learned continuation이 한 goal에서 성공을 못 함)에서 gate가 켜지는 비율을 기록.
- branch simulator step은 `branch_steps`로 따로 센다(env step 예산에 포함하지 않음).

### 지표
SR seen/unseen(+seen 자주/드문), SER(위 정의, 자주 본 target vs 기타), RankAcc(정답 Q* 기준, eps-oracle rollout로 모은 독립 상태 1500개, goal 쌍 전부; 분기 상태 = 최적 행동 집합이 서로 소, 공통 상태 = 공유 최적 행동 존재; seen 쌍과 held-out 포함 쌍 분리; 분기는 빈손/들고 있음 분리), SAP(공통 상태에서 두 goal의 greedy 행동이 모두 최적인 비율)와 불필요한 divergence, goal-switch(같은 snapshot: t=0과 goal 물체를 집은 직후, target만 바꿔 commit 분포), 학습 곡선(250k마다 20 ep/goal). 평균 ± SE와 seed별 값.

### Leakage 검사 (자동 assert)
`assert_train_goals`가 (1) 학습 episode goal 샘플, (2) replay 삽입(RL 데이터와 HER relabel 모두), (3) CGC goal 쌍(evaluator 입력), (4) B2·B6 aux goal, (5) B6 Q* 회귀 대상, (6) biased-BC demo에서 held-out 3조합이면 `LeakageError`를 낸다. 호출 횟수는 run별 `leak_checks`로 기록. held-out goal은 평가(SR, RankAcc의 held-out 쌍, goal-switch)에서만 쓴다.

## 4. 사전 등록 (gate run 제출 전 커밋 deb77e6, `dacgc/aggregate.py`)
- Phase 0 Go: balanced B0의 SR_seen ≥ 0.7 (3 seed 평균).
- Phase 1 순서: balanced → skew 0.8 → skew 0.95 → skew 0.95 + biased BC(선호 goal 3개만의 oracle demo 2000개로 DQfD식 TD+margin 사전학습 후 같은 RL). 각 조건 B0 3 seed, 2M step.
- "의미 있는 SER": SER_unseen_freq ≥ 0.15 **그리고** SER_unseen_freq ≥ 2 × SER_unseen_other(무작위가 아니라 자주 본 target으로 향함) **그리고** SR_seen ≥ 0.5(기술은 있음). 순서상 처음 만족한 조건이 Phase 3 주 조건. 어느 것도 만족하지 않으면 Phase 2–3 중단.
- Phase 3 선택 규칙(held-out 미사용): validation seed(100, 101)에서 드문 **train** goal의 SR(seen_rare_sr) 최대, 0.02 이내 동률은 SR_seen, 다음은 작은 λ. 격자 B4 λ{0.3,1,3}×τ{0.005,0.02}×K{10,50}, B3 λ×K, B6 λ, B2 λ.

## 5. Phase 0–1 결과
harness run `t013-p01-gate` (git deb77e6, GPU 2 배정·미사용, CPU 21 run 병렬, **wall-clock 5074 s = 1.41 GPU-h**; B0/B1 run ≈ 2400 s, B6 run ≈ 5000 s). pytest 9/9 통과(clone/restore, dynamics, oracle=BFS, leakage). 실패 run 0. 2M env step, seed 0/1/2, 평가 goal당 100 episode(같은 layout). 값은 평균 ± SE (3 seed).

### 5.1 B0, 사전 등록 조건 (관문 판정 대상)

| 조건 | SR seen | seen 자주 | seen 드문 | SR unseen | SER freq | SER other | wrong obj | timeout (unseen) |
|---|---|---|---|---|---|---|---|---|
| balanced | 0.02 ± 0.01 (0.01, 0.01, 0.05) | 0.04 | 0.01 | 0.002 | 0.001 | 0.000 | 0.001 | **0.996** |
| skew 0.8 | 0.23 ± 0.04 (0.30, 0.23, 0.17) | 0.46 | 0.00 | 0.001 | 0.000 | 0.000 | 0.000 | **0.999** |
| skew 0.95 | 0.45 ± 0.03 (0.39, 0.49, 0.47) | 0.90 | 0.00 | 0.000 | 0.001 | 0.000 | 0.000 | **0.999** |
| skew 0.95 + biased BC | 0.49 ± 0.00 (0.49, 0.49, 0.49) | 0.99 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | **1.000** |

seen 드문 goal(학습 goal 중 빈도가 낮은 3개)의 실패도 자주 본 target으로의 commit이 아니다: `seen_rare_wt_freq` = 0.00 (모든 조건). 즉 **"자주 본 target으로 감"이 아니라 "아무 데도 놓지 않음, 시간 초과"**.

Goal-switch(같은 snapshot, target만 바꿈; 출발 rollout은 자주 본 goal):
- 물체를 집은 직후(carry) seen 요청 follow: skew 0.95 0.50, biased BC 0.50, skew 0.8 0.41, balanced 0.14. 0.50 = 자주 본 target 요청만 따르고 드문 target 요청은 실패한다는 뜻.
- carry에서 held-out 요청: follow ≤ 0.05, 자주 본 target으로 commit ≤ 0.03.
- t=0(approach)에서 held-out 요청 follow 0.00.

RankAcc(정답 Q* 기준, 분기/공통): skew 0.95 seen 0.77/0.76, held-out 쌍 0.62/0.68. balanced seen 0.70/0.71.

### 5.2 진단 (관문 판정에 쓰지 않음)

| 진단 | SR seen | SR unseen | SER | timeout unseen | RankAcc 분기 seen / held-out | SAP seen / held-out |
|---|---|---|---|---|---|---|
| B1 HER, balanced | 0.07 ± 0.03 | 0.00 | 0.00 | 1.00 | 0.81 / 0.73 | 0.69 / 0.53 |
| B6 Q* 회귀, balanced | **1.00** | **0.00** | 0.00 | **1.00** | 1.00 / 0.90 | 1.00 / 0.72 |
| B6 Q* 회귀, skew 0.95 | **1.00** | **0.00** | 0.00 | **1.00** | 1.00 / 0.88 | 1.00 / 0.68 |

- **Phase 0 실패 원인은 RL 학습기다.** 같은 네트워크·관측이 정답 Q* 보조 회귀(B6)로는 seen 6개를 모두 푼다(SR 1.00, 3/3 seed). 순수 RL(B0)은 2M step에서 빈도가 높은 goal만 배우고(skew), balanced에서는 거의 아무것도 못 배운다. HER도 0.07.
- **held-out 실패는 shortcut이 아니라 조합 일반화 실패다.** seen에서 완벽한 B6조차 held-out에서 SR 0, commit 0, timeout 100%. 상태별 행동 순위는 held-out 쌍에서도 0.88–0.90 맞지만, 틀린 10%가 궤적에서 누적돼(왕복 loop) 어디에도 놓지 못한다. concat goal 인코딩의 MLP는 처음 보는 (object, target) one-hot 조합을 "자주 본 target"으로 보내지 않고 정의되지 않은 행동으로 보낸다.

## 6. 관련 연구 비교

| 연구 | 무엇을 | 어디에 | 신호 출처 | 본 설계와 차이 |
|---|---|---|---|---|
| HER (Andrychowicz et al. 2017) | 달성 goal로 재레이블 | 모든 transition | 자기 경험 | 행동 간 비교를 강제하지 않음. B1 |
| **DQfD** (Hester et al., AAAI 2018) | large-margin supervised loss J_E = max_a[Q(s,a)+l(a_E,a)] − Q(s,a_E) + TD | **모든 demo 상태**, 전문가 행동 vs 나머지 전부 | 전문가 demo | goal 쌍·분기 선택 없음. B3(gate 없는 정답 ranking)이 "evaluator 라벨을 쓰는 DQfD식 margin"에 해당. B4는 같은 margin을 gate가 고른 상태와 상대 goal의 선호 행동에만 건다 |
| Action gap (Farahmand, NeurIPS 2011; Bellemare et al., AAAI 2016 advantage learning) | 최선과 차선의 Q 차이가 크면 근사 오차에 강함 / 연산자로 gap을 키움 | 모든 상태 | Bellman 연산자 | B4의 margin은 goal-조건 gap을 **분기 상태에서만** 키우는 국소판. 본 구현의 Q̃ 척도 변경도 gap 문제(0/1 보상에서 gap≈0.03) 때문 |
| PBRS (Ng et al. 1999; Wiewiora 2003) | potential shaping | 모든 상태 | 사람 설계 | 공통 학습기에만 사용(target 정보 없음) |
| CounterAlign (2026), CRAFT (2026) | instruction 교체·조합 일반화 | — | — | 설계서 1.3 참조. counterfactual 생성 자체는 새로움이 아님 |

## 7. 관문 판정
**No-Go (사전 등록 원칙에 따라 Phase 2–3 미진행).**

1. Phase 0 Go 조건(balanced B0 SR_seen ≥ 0.7) 불충족: 0.02 ± 0.01. 설계서 7장 규칙대로 "환경/기본 RL 구현을 먼저 점검"할 대상이다. 2절의 점검(관측·값 척도·shaping·탐색·n-step·하이퍼파라미터 10종)으로도 2M step 안에서는 해결되지 않았다.
2. Phase 1 "의미 있는 SER" 조건을 4개 사전 등록 조건 모두 불충족: SER_unseen_freq ≤ 0.001 (기준 0.15). held-out 실패는 **시간 초과 99.6–100%**이고, 자주 본 target으로 간 비율과 다른 target으로 간 비율이 모두 ≈ 0이다. seen 드문 goal의 실패도 같다.
3. 따라서 이 환경·학습기에서는 DA-CGC가 겨냥하는 goal confusion(H2)이 재현되지 않는다. held-out에서 실제로 일어나는 것은 skill이 아니라 조합 인코딩의 실패(설계서 1.2의 "skill failure" 쪽에 가까움)이고, 정답 Q*를 전 상태에 회귀시켜도(B6) 풀리지 않는다. 그러므로 ranking 보조 손실이 H3를 고칠 근거도 이 결과에서는 보이지 않는다.

사람이 결정할 다음 단계 후보(이번 task 범위 밖, 승인 필요):
- (a) Phase 0만 다시: 기본 RL을 바꿔 B0가 seen을 배우게 한다(예: 모든 방법에 같은 balanced demo warm start, 더 긴 예산 ≥ 5M step, 또는 작은 격자). 예산·설계 변경이 필요하다.
- (b) H3 질문을 먼저: goal 인코딩(FiLM 등)이 held-out 실패를 "시간 초과"에서 "다른 target commit"으로 바꾸는지 B6로 싸게 확인한다. shortcut이 생겨야 DA-CGC를 시험할 대상이 생긴다.
- (c) 방향 폐기: 이 gridworld에서는 설계서 8장의 "모든 방법 높은/낮은 SR, 실패 조건 없음" 경우에 해당한다고 보고 정리한다.

GPU-h 사용: 1.41 (상한 3). Phase 3 run은 제출하지 않았다.

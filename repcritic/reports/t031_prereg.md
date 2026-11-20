# T031 사전 등록 — Counterfactual P1 (intervention sensitivity), 범위 재설정판

작성: 2026-11-20. **후보 branch를 하나도 돌리기 전에** 커밋한다.
- 코드: `repcritic/t031_select.py`(상태 선택, CPU), `repcritic/t031_branch.py`(branch), `repcritic/t031_analyze.py`(분석, branch 뒤 작성. 아래 규칙만 구현한다).
- 설계서: "Counterfactual Advantage Learning for Frozen VLA Residual Control" §2–3, §6.2–6.3. **P1만 한다.** P2(critic)는 하지 않는다.

## 질문
같은 상태에서 한 chunk에만 residual을 주고, 이후는 frozen π0.5로 진행한다(one-shot, μ = π_VLA). 이때 base(u=0) 대비 결과(rescue/harm)를 **잡음보다 확실히** 바꾸는 수정이 있는가?

## 코드 출처
- 브랜치 `exp/counterfactual-p1`은 `exp/branch-adv`(4d98249)에서 분기했다.
- T022 expert·α 코드는 **파일 단위로 가져왔다**: `git checkout 47a4d76 -- vla_residual/core.py vla_residual/__init__.py`(47a4d76 = `exp/vla-residual` HEAD).
  - `ScriptedExpert`, `priv_state`만 import해서 쓴다. 수정은 없다.
  - `repcritic/rc_common.py`는 두 브랜치에서 같다.
- 렌더링 생략(chunk 안 앞 K−1 step은 `obs_mode="none"`)은 T022 `_sim_step` 방식을 그대로 가져왔다. 물리는 같다.

## 상태 (`/workspace/stochrl_t031/state_select.json`, seed 31031)
- 출처는 T017 buffer의 **train split**이다. episode당 1개, d ≤ 12, placed가 아닌 결정만 쓴다.
- T017 branch 집합과 T026에 쓴 episode는 제외했다.
- T022 held-out 128(perm[500:900])은 T017 buffer(perm[:500])와 구조적으로 겹치지 않는다. 코드에서 assert한다.
- 4층 × 12 = **48 상태**. 층마다 성공 episode 6개, 실패 episode 6개다.

| 층 | 정의 | d |
|---|---|---|
| `S0_init` | d = 0 | 0 |
| `S1_pre` | approach, d ≥ 1, gripper–물체 < 8 cm(접촉 직전) | 2–8 |
| `S2_post` | grasp_contact, 또는 이전 grasp 없는 첫 transport(접촉 직후) | 2–10 |
| `S3_lowV` | d ≥ 1, T023 V̂(`vla@platt`, = `load_potential("vla","platt")`) < τ = 0.082, 위 층에 속하지 않음 | 2–10 |

- S3 후보 pool에는 성공 episode가 7개뿐이다. 그중 6개를 썼다.
- 실행 순서는 층을 번갈아 섞었다. 예산 때문에 중간에 멈춰도 층 균형이 유지된다.

## 후보 (상태당 13개; α = T022: 위치 0.02 m/step, 회전 0.1 rad/step, 그리퍼 2.0)
- base chunk는 T017 episode에서 결정 d에 **실제로 실행된 chunk**다(K = 5 step, env 단위).
- 모든 후보는 env action bound로 clip하고, clip 비율을 기록한다.

| # | 이름 | 내용 |
|---|---|---|
| 0 | `base` | u = 0, CRN continuation noise(기준 branch R⁽⁰⁾) |
| 1 | `u0_ind` | u = 0, **독립 continuation noise**. **잡음 기준선**이다. 개입 없이 continuation 잡음만으로 생기는 discordance를 잰다 |
| 2 | `E1` | 양성 대조군(특권): Δ = clip(a_expert − mean_k base, ±α). **이 chunk 한 번만** 적용한다 |
| 3 | `E3` | 같은 expert residual을 **연속 3 chunk** 적용한다(다음 두 chunk는 VLA chunk + online expert Δ). 그 뒤 VLA |
| 4–7 | `T1_±u`, `T1_±v` | translation 1.0 α_pos(10 cm/chunk). u = root frame 수평 tcp→목표(물체, 잡고 있으면 접시), v = z × u |
| 8–9 | `T05_±u` | translation 0.5 α_pos(5 cm/chunk), ±u |
| 10–11 | `yaw_±` | 회전 dim 5(yaw) ± α_rot(0.5 rad/chunk) |
| 12 | `grip_flip` | 이 chunk 동안 그리퍼 반전: g′ = clip(g − 2·sign(g), −1, 1) |

- **expert 상태 초기화**(snapshot에는 expert phase 메모리가 없다):
  - 잡고 있지 않으면 APPROACH로 둔다. 같은 call 안에서 기하 조건에 따라 DESCEND/CLOSE로 넘어간다.
  - 잡고 있으면 LIFT로 둔다. cz0 = 물체 초기 높이, ez0 = 현재 ee z, offset = ee − 물체로 둔다.
  - E1·E3의 첫 chunk는 같다.
- **u=0 반복(같은 noise)**은 복원 테스트에서 잰다: 2 상태 × 4 반복 × base 13벌, 같은 CRN. 이것은 복제 잡음 하한이다. 본 실행의 잡음 기준선은 `u0_ind`다.

## branch
- snapshot 복원 → 후보 chunk → frozen π0.5로 t = 80(16 결정)까지 진행한다. **반복 4회**다.
- CRN: 결정 k의 continuation noise seed는 (상태, 반복, k)다. 13 후보에 같고, `u0_ind`만 독립 stream이다.
- success_once(주 지표)와 success_at_end를 모두 기록한다.
- 총 48 × 13 × 4 = 2496 branch, E = 52 env/batch(1 상태/batch). **GPU 1만 쓴다.**
- **예산**: 사람 상한은 0.6 GPU-h다. 본 실행 driver는 다음 상태의 예상 시간까지 더해 `--budget_h`를 넘으면 새 상태를 시작하지 않는다.
  - `--budget_h` = 0.6 − (복원 테스트 + 로드 시간) − 0.03.
  - 중간에 멈추면 실행 순서의 앞부분(층 균형)만 분석하고 상태 수를 보고한다. 32 상태 미만이면 "검정력 부족"이라 적는다.

## 측정 (success_once; 단위 = (상태, 반복) 쌍, 후보 c vs 같은 반복의 base)
- 정의:
  - rescue = P(R⁽⁰⁾ = 0, R⁽ᶜ⁾ = 1), harm = P(R⁽⁰⁾ = 1, R⁽ᶜ⁾ = 0)
  - **net = rescue − harm**, discordance = rescue + harm
- **후보군**: `NOISE` = u0_ind, `E1`, `E3`, `T1`(4개 묶음), `T05`(2), `YAW`(2), `GRIP`(1).
  - **비특권 후보군** = T1, T05, YAW, GRIP.
- CI: 상태 단위 군집 bootstrap(4000회, percentile 95%).
- **잡음 비교**: Δdisc = disc(후보군) − disc(u0_ind). 상태 단위 paired bootstrap CI로 잰다.
- 2차 지표:
  - 개별 후보 13개(탐색적, Bonferroni 보정 CI(12 비교, 99.58%)도 함께 보고)
  - 층별 표
  - 크기 비교(T1_±u vs T05_±u, paired)
  - **E3 − E1 net**(paired CI)
  - success_at_end
  - K step 뒤 gripper 변위·grasp 상태 차이(회복 동작 진단)
  - 복원 테스트의 같은-noise 일치율
- **split-half 선택 이득**(T026 방식):
  - 상태마다 반복 {0,1}로 후보 집합에서 p̂ 최대를 고른다. 동률이면 base를 고른다.
  - 고른 후보를 반복 {2,3}에서 평가하고, base의 {2,3} 값을 뺀다. 반대 방향도 해서 평균한다.
  - 집합: (a) base ∪ 비특권 9개, (b) (a) ∪ {E1, E3}.

## 판정 (사전 등록)
- **Go**: 비특권 후보군 중 하나 이상이 아래 두 조건을 모두 만족한다. 이때는 P2(critic) **제안만** 하고 실행하지 않는다.
  - net의 95% CI 하한 > 0
  - Δdisc(vs u0_ind)의 95% CI 하한 > 0
- **설정 No-Go**: E1 net의 95% CI 하한 ≤ 0(0을 포함하거나 음수).
  - 결론은 "one-shot + μ = π_VLA로는 ΔQ가 너무 작다"이다.
  - E3 net의 CI 하한 > 0이면 "반복 개입, 또는 μ = residual 정책이 필요하다"를 덧붙인다.
- **부분**: E1 net의 CI 하한 > 0이지만 비특권 후보군은 Go 조건을 만족하지 못한다.
  - 결론은 "rescue 가능한 상태는 있으나 후보 분포가 병목"이다.
- Go와 E1 No-Go가 동시에 나오면 Go를 우선하고, 둘 다 보고한다.
- 비특권 **개별 후보**만 보정 없는 CI로 통과하고 후보군은 통과하지 못하면, Go로 치지 않고 "탐색적 신호"로 적는다.

## 주의 (미리 적어 둠)
- expert는 episode 시작부터 돌 때(T022 상한 0.820) 검증된 것이다. snapshot에서 phase를 초기화한 one-shot 사용은 검증되지 않았다. E1이 약하면 E3와 K-step 진단으로 원인을 나눈다.
- 1.0 α translation(10 cm)과 0.5 rad yaw는 base 산포(T026: ≈ 2 cm)보다 훨씬 크다. harm이 큰 것은 예상된 결과다. 판정은 net 기준이다.
- `u0_ind`의 net 기댓값은 정확히 0이다. disc는 독립 잡음에서 나오는 상한 수준의 불일치다. 후보의 disc가 이를 넘으면 개입이 결과를 체계적으로 바꾼 것이다.

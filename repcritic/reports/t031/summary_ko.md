# T031 — Counterfactual P1(intervention sensitivity): one-shot residual이 base 결과를 잡음보다 확실히 바꾸는가

- 날짜: 2026-11-20
- 브랜치 `exp/counterfactual-p1`(`exp/branch-adv` 4d98249에서 분기)
- 코드: `repcritic/t031_{select,branch,analyze}.py`
- 사전 등록: `repcritic/reports/t031_prereg.md`. **후보 branch를 돌리기 전에 커밋했다**(b05ec15).
- T022 expert·α 코드는 **파일 단위 checkout**으로 가져왔다: `git checkout 47a4d76 -- vla_residual/core.py vla_residual/__init__.py`(47a4d76 = `exp/vla-residual` HEAD). `ScriptedExpert`, `priv_state`만 import했고 수정은 없다.
- 원자료: `/workspace/stochrl_t031/`
- **GPU 1만 사용했다**(시작 시 비어 있었다. 실행 중 GPU 1의 프로세스는 우리 것 하나뿐이었다). GPU 0·3은 쓰지 않았다.
  - **GPU-h ≈ 0.57**: 복원 테스트 2.3분 + 본 실행 31.8분(로드 포함). 상한은 0.6이다. 학습은 하지 않았다.
- **P1만 했다.** P2(critic 학습)는 하지 않았다.

## 0. 결론
1. **사전 등록 판정: 설정 No-Go(E3는 유의).**
   - E1(one-shot expert)의 net은 **−0.076 [−0.242, +0.091]**로 CI가 0을 포함한다. E3(연속 3 chunk)는 **+0.136 [+0.053, +0.242]**다.
   - 결론: **one-shot + μ = π_VLA로는 단일 chunk 개입의 ΔQ가 작거나, 부호가 상태에 따라 갈린다. 반복 개입, 또는 μ = residual 정책이 필요하다.**
   - E3 − E1 net(paired)은 **+0.212 [+0.061, +0.371]**이다.
2. **비특권 후보군 중 net > 0인 것은 없다.** 4개 후보군 모두 net 점추정이 음수다.
   - T1(10 cm): −0.189 [−0.312, −0.076]
   - T05(5 cm): −0.087 [−0.174, −0.011]
   - YAW: −0.023 [−0.098, +0.045]
   - GRIP: −0.144 [−0.288, −0.023]
3. **결과는 확실히 바뀐다. 다만 대부분 나쁜 쪽이다.**
   - 모든 후보군의 discordance가 잡음 기준선(u0_ind, 0.068)보다 확실히 높다(Δdisc CI 하한 > 0).
   - 큰 개입은 결과를 바꾼다. 이는 T026의 2.5 cm 결과(일치 0.87, 잡음과 비슷)와 다르다. 그러나 바뀐 결과는 harm이 rescue보다 많다(T1: rescue 0.034, harm 0.223).
   - frozen π0.5 continuation이 10 cm 변위를 흡수하지 못한다. 흡수하지 못한 경우 대부분 실패한다.
4. **E1은 one-shot 양성 대조군으로 약했다.** E3는 rescue를 유지하면서 harm을 없앤다(0.220 → 0.030).
   - 즉 expert의 단일 chunk는 그 자체로 좋은 수정이 아니다. 여러 chunk에 걸친 계획의 첫 조각이다. 아래 §4에 기제를 적었다.
   - 판정은 사전 등록 그대로 적되, 이 한계 때문에 "one-shot으로는 불가능하다"의 증거로는 약하다(§5).

## 1. 사전 등록 요약
- **상태**: T017 buffer train split에서 뽑았다.
  - episode당 1개, d ≤ 12, placed가 아닌 결정만 썼다. T017 branch와 T026의 episode는 제외했다.
  - T022 held-out 128과는 구조적으로 겹치지 않는다(assert).
  - 4층 × 12 = 48 상태를 계획했다. 층마다 성공 episode 6, 실패 episode 6이다.
    - `S0_init`: d = 0
    - `S1_pre`: approach, 물체 < 8 cm
    - `S2_post`: grasp_contact, 또는 첫 grasp 직후 transport
    - `S3_lowV`: T023 V̂(vla@platt) < 0.082
  - **예산 규칙에 따라 33 상태에서 멈췄다**(층 교차 순서: 9/8/8/8). 사전 등록 하한 32를 넘는다.
    - t=0 상태 하나에 약 71초가 걸렸다. 렌더링 생략도 정책 추론 시간에 가려 효과가 작았다.
- **후보 13개/상태**(α = T022: 0.02 m/step, 0.1 rad/step, 그리퍼 2.0. action bound clip):
  - `base`(u = 0, CRN)
  - `u0_ind`(u = 0, 독립 continuation noise = **잡음 기준선**)
  - `E1`(expert residual, 1 chunk)
  - `E3`(expert residual, 연속 3 chunk)
  - `T1_±u/±v`(1.0 α = 10 cm/chunk)
  - `T05_±u`(5 cm)
  - `yaw_±`(0.5 rad/chunk)
  - `grip_flip`
- **branch**: snapshot → 후보 chunk → frozen π0.5로 t = 80까지. 반복 4회, CRN(u0_ind만 독립)이다.
  - 총 33 × 13 × 4 = **1716 branch**다.
- **clip**: 비율은 E1 0.077, grip_flip 0.109(그리퍼 ±1 경계), translation·yaw 0이다. expert α-clip 비율은 0.039다.
- **판정 규칙**(사전 등록):
  - Go = 비특권 후보군이 net CI 하한 > 0, 그리고 Δdisc(vs u0_ind) CI 하한 > 0
  - 설정 No-Go = E1 net CI 하한 ≤ 0
  - 부분 = E1만 유의

## 2. 후보군 표 (success_once, 단위 = (상태, 반복) 쌍 vs 같은 반복의 base, 상태 군집 bootstrap 95% CI, n = 33)
- base 성공률은 0.455다.

| 후보군 | rescue | harm | **net [95% CI]** | discordance | Δdisc vs 잡음 [CI] |
|---|---|---|---|---|---|
| **잡음 기준선 u0_ind** | 0.038 | 0.030 | +0.008 [−0.015, +0.030] | **0.068** | — |
| E1 (특권, 1 chunk) | 0.144 | 0.220 | **−0.076 [−0.242, +0.091]** | 0.364 | +0.295 [+0.152, +0.439] |
| **E3 (특권, 3 chunk)** | 0.167 | 0.030 | **+0.136 [+0.053, +0.242]** | 0.197 | +0.129 [+0.030, +0.235] |
| T1 (10 cm, 4방향) | 0.034 | 0.223 | −0.189 [−0.312, −0.076] | 0.258 | +0.189 [+0.087, +0.307] |
| T05 (5 cm, ±u) | 0.057 | 0.144 | −0.087 [−0.174, −0.011] | 0.201 | +0.133 [+0.057, +0.220] |
| YAW (±0.5 rad) | 0.061 | 0.083 | −0.023 [−0.098, +0.045] | 0.144 | +0.076 [+0.008, +0.144] |
| GRIP (반전) | 0.061 | 0.205 | −0.144 [−0.288, −0.023] | 0.265 | +0.197 [+0.083, +0.333] |

- **복제 잡음 하한**(복원 테스트: u = 0, 같은 CRN noise, 2 상태 × 4 반복 × 13벌):
  - 다수결 일치율은 0.913이고, slot 0 대비 discordance는 0.094다.
  - d = 0 상태는 4/4 반복이 완전히 일치했다. t = 40 상태(성공이 t = 72–75로 마감 직전)는 2/4 반복이 4/13, 5/13으로 갈렸다.
  - GPU PhysX의 env slot 간 비결정성 때문으로 본다. memory의 "~2% obs 차이"와 같은 현상이다.
  - 마감 근처 상태에서는 같은 noise여도 결과가 갈린다. 모든 비교(u0_ind 포함)가 이 잡음을 안고 있다. 그래서 잡음 기준선은 같은 조건의 u0_ind로 잡았다.
- **개별 후보**(탐색적): 비특권 후보 9개 중 net CI 하한 > 0인 것은 없다.
  - 가장 높은 것은 `yaw_+` +0.038 [−0.045, +0.129]다.
  - `T1_−u`, `T1_±v`는 Bonferroni CI로도 0보다 작다.
- **success_at_end**도 같은 그림이다. E1 −0.083 [−0.250, +0.083], E3 +0.159 [+0.068, +0.273], T1 −0.184, YAW −0.019.
- **split-half 선택 이득**(T026 방식, base 대비):
  - base ∪ 비특권 9개: +0.068 [−0.023, +0.174]
  - ∪ {E1, E3}: +0.121 [+0.015, +0.242]
  - u0_ind만(잡음 대조): −0.023 [−0.053, 0]
  - 비특권 후보만으로는 고른 수정의 이득이 0과 구분되지 않는다.
- **크기**(±u, paired):
  - T1 − T05 net = −0.106 [−0.197, −0.023]. 클수록 나쁘다.
  - disc는 +0.098 [+0.027, +0.178]. 클수록 결과를 더 바꾼다.

## 3. 층별 (net [95% CI], n = 8–9 → CI가 넓다. 탐색적)

| 층 | n | p(base) | 잡음 | E1 | E3 | T1 | T05 | YAW | GRIP |
|---|---|---|---|---|---|---|---|---|---|
| S0_init (d=0) | 9 | 0.28 | +0.03 | +0.14 [−0.17, +0.44] | +0.11 [−0.06, +0.31] | −0.22 [−0.48, 0] | −0.03 | 0.00 | −0.06 |
| S1_pre (접촉 직전) | 8 | 0.66 | −0.03 | −0.19 [−0.44, 0] | +0.06 [0, +0.16] | −0.23 [−0.41, −0.07] | −0.11 | −0.13 | −0.13 |
| S2_post (접촉 직후) | 8 | 0.59 | 0.00 | −0.13 [−0.50, +0.31] | +0.16 [0, +0.41] | −0.08 | −0.06 | **+0.09 [+0.02, +0.22]** | −0.22 |
| S3_lowV (V̂ < 0.082) | 8 | 0.31 | +0.03 | −0.16 [−0.53, +0.16] | **+0.22 [+0.03, +0.47]** | −0.22 | −0.16 | −0.06 [−0.13, −0.02] | −0.19 |

- 비특권 후보 중 유일하게 CI가 0 위인 칸은 **S2_post의 YAW(+0.09)**다. 그러나 층 4 × 후보군 4 = 16 비교 중 하나이고, 보정이 없다. **탐색적 신호로만 적는다.**
  - T026의 "접촉 직후 차이는 위치가 아닌 회전·그리퍼 timing"이라는 관찰과는 방향이 맞는다.
- E3의 이득은 S3_lowV(V̂ 낮은 상태)와 S2_post에서 크다. S1_pre에서는 작다(base가 이미 0.66).
- **base 결과(로그 episode 성공)로 나누면**(설계서 §6.3의 base-success subgroup):
  - 성공 episode(n = 17, p(base) 0.84): E1 **−0.37** [−0.57, −0.16], E3 +0.06 [−0.02, +0.13], 비특권 모두 음수
  - 실패 episode(n = 16, p(base) 0.05): E1 **+0.23** [+0.08, +0.41], E3 +0.22 [+0.06, +0.41], T05 +0.05 [+0.01, +0.09], 나머지는 0 근처
  - 즉 **ΔQ의 부호가 상태에 따라 갈린다.** 무조건 적용하면 상쇄되거나 harm이 이긴다. "실패할 상태에서만 개입"하는 gate라면 E1도 이득이다.
  - 단 이 분할은 로그 결과(사후 정보)를 쓴다. 실제 gate 신호(V̂)로 고른 S3_lowV에서 E1은 −0.16이었다.

## 4. E1 vs E3 — 왜 one-shot expert는 실패하고 3-chunk는 성공하는가
- 첫 chunk는 같다. K step 뒤 gripper 변위는 base 대비 중앙값 4.7 cm이고, grasp 상태가 base와 다른 비율은 0.22다.
- **E3 − E1 net = +0.212 [+0.061, +0.371]**이다. 차이는 거의 harm에서 난다(0.220 → 0.030). rescue는 0.144 → 0.167로 비슷하다.
- 기제(사후 기술, `expert_first_chunk_diag.csv`):
  - expert phase 초기값은 33 중 29가 APPROACH였다. 특권 `is_grasping`이 거짓이라, 접촉 직후나 닫는 중인 상태에서도 APPROACH로 시작한다.
  - 그래서 **17/33 상태에서 E1은 base가 닫고 있던 그리퍼를 연다**(base grip ≈ −1 → expert +1).
  - 또 10 상태에서는 hover 높이(물체 위 7 cm)로 **올라간다**. 이 10 상태의 E1 net은 평균 −0.23이다.
  - E1만 주면 VLA가 반쯤 연 grasp나 올라간 그리퍼에서 다시 시작한다. 이 상태가 성공 경로를 망친다.
  - E3는 expert가 다시 내려가 다시 잡을 시간(DESCEND → CLOSE)을 준다. 그 뒤에는 VLA가 이어받아 끝낸다.
  - 연다/안 연다로 나누면 E1 net은 −0.12 / −0.03, E3 net은 +0.16 / +0.11이다. 그리퍼 개방만으로 다 설명되지는 않는다.
- 해석:
  - **수정은 효과가 남으려면 여러 chunk에 걸쳐 일관되게 적용돼야 한다.** T026의 "π0.5가 작은 offset을 회복한다"와 반대 방향의 같은 현상이다.
  - 작은 수정은 흡수되고, 큰 수정은 미완성 상태를 남겨 harm이 된다.

## 5. 판정과 제한
- **판정(사전 등록): 설정 No-Go.** E1 net CI가 0을 포함한다. **E3가 유의하므로 "반복 개입, 또는 μ = residual 정책이 필요하다."** 비특권 Go 후보군은 없다.
- 제한:
  1. **E1은 one-shot에 최적화된 수정이 아니다.** 멀티 step expert의 첫 조각이고, phase 초기화가 휴리스틱이다(사전 등록 §주의). 그래서 "어떤 one-shot 수정도 ΔQ > 0이 아니다"를 증명하지는 못한다. 확인한 것은 "이 후보 분포(특권 1-chunk 포함)에서는 없다"이다.
  2. 예산 때문에 33/48 상태만 돌렸다. 층별 n은 8–9이고 CI가 넓다.
  3. 복제 잡음(같은 noise)이 0.094로 작지 않다. 마감 근처 상태의 결과는 물리 비결정성만으로도 갈린다.
  4. T05는 ±u만, YAW는 크기 1개(0.5 rad)만 시험했다. 더 작은 회전과 그리퍼 timing(반전이 아닌 1–2 step 지연·앞당김)은 시험하지 않았다.

## 6. 다음 단계 제안 (실행하지 않음)
- P2(critic)를 **one-shot μ = π_VLA 설정으로 진행하는 것은 권하지 않는다.** 비특권 후보의 ΔQ가 0 이하이고 학습할 신호가 harm뿐이다.
- 대안 A(설계서 §2.2의 확장 순서를 앞당김): **개입 단위를 "n chunk 연속"(n = 3)으로 바꾸고 P1을 다시 한다.**
  - E3가 유의하므로 양성 대조군이 성립한다.
  - 비특권 후보(예: 3 chunk 연속 yaw, 0.5 α translation)가 net > 0인지 먼저 본다.
- 대안 B: μ = residual 정책(T022 학습 정책)으로 continuation을 바꾼 paired branch를 만든다.
- 어느 쪽이든 **gate가 핵심이다**(§3: 실패할 상태에서는 E1·E3 모두 +0.22–0.23, 성공할 상태에서는 E1 −0.37). base-relative critic의 가치는 "언제 개입하지 않을지"에서 나올 가능성이 크다.

## 7. 산출물
- `/outputs/StochRL/t031/`
  - `summary_ko.md`(이 파일)
  - `analysis.json`: 모든 추정치와 CI, 층별, base-success subgroup, success_at_end, 개별 후보(Bonferroni 포함), split-half, after-K 진단, 복제 하한, 판정
  - `branch_rollouts.parquet`: 1716 행. 상태, 층, V̂, 후보, 반복, success_once/at_end, 첫 성공 t, clip 비율, afterK 특권 라벨, E3 후속 Δ
  - `effects_by_candidate.csv`: 후보군·개별 후보·층별 rescue/harm/net/disc/Δdisc와 CI, 개별 후보 Bonferroni CI와 after-K 진단
  - `expert_first_chunk_diag.csv`: 상태별 expert 첫 chunk(Δz, Δxy, Δyaw, 그리퍼)와 E1/E3 net
  - `rescue_harm.png`
    - 왼쪽: 후보별 rescue(위)/harm(아래), net ± CI, 잡음 띠
    - 오른쪽: 층 × 후보군 net
- `/workspace/stochrl_t031/`
  - `branch_results_{main,restore_test}.jsonl`
  - `candidates/*.npz`: 실행 chunk 13개(env 단위), base, expert 행동·Δ·phase, u/v 방향, clip 비율
  - `state_select.json`, `logs/`

## 8. 재실행
```
PY=.venv/bin/python   # -> repcritic worktree venv; RC_RLINF=<populated RLinf checkout>
$PY -m repcritic.t031_select
CUDA_VISIBLE_DEVICES=1 RC_POLICY_MB=26 $PY -m repcritic.t031_branch --restore_test
CUDA_VISIBLE_DEVICES=1 RC_POLICY_MB=26 $PY -m repcritic.t031_branch --budget_h 0.52   # 재시작 가능(완료 상태 건너뜀)
$PY -m repcritic.t031_analyze --out /outputs/StochRL/t031
```

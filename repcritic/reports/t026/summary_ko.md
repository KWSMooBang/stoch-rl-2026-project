# T026 — branch-advantage mini (설계서 A안 축소판): perturbation이 oracle gap을 키우는가?

- 날짜: 2026-11-19 / 브랜치 `exp/branch-adv`(`exp/repcritic-e06` 973ee40에서 분기) / 코드 `repcritic/t026_*.py`
- 사전 등록: `repcritic/reports/t026_prereg.md`. **branch를 돌리기 전에 커밋했다**(35f3f96).
- 원자료: `/workspace/stochrl_t026/` / 요약: `/outputs/StochRL/t026/`
- GPU: **GPU 1만 사용**(T022가 GPU 0 사용). 우리 프로세스 peak 8.8 GB(torch). GPU 1 합계 ≤ 17 GB(≤ 22 GB 기준).
  - 첫 시도 두 번은 실패했다. E=96은 카메라 버퍼 초과였고, E=48 첫 시도는 다른 프로세스가 GPU 1을 잠깐 ~18 GB 쓰는 순간 정책 로드가 OOM이 났다.
  - 이후 여유 메모리를 기다렸다가 재시도하는 driver(`t026_run.sh`)로 돌렸다.
  - GPU 3은 쓰지 않았다(~12 MiB는 T016에 기록된 Vulkan ICD context).
- **GPU-h ≈ 0.81**(복원 검증 0.05 + 본 실행 0.74 + 실패 로드 ≈ 0.02). 상한 1.5. 학습은 하지 않았다.

## 0. 결론
1. **판정(사전 등록): perturbation은 gap을 키우지 않는다 → "후보 분포가 병목이 아니라 결과 잡음·회복 동작이 병목".**
   - 1차 지표는 후보 수·반복 수를 같게 맞춘 Δexcess(MIX8 − VLA8)다. 값은 **−0.017 [−0.048, +0.006]**이다.
   - Δ split-half 선택 값은 −0.025 [−0.09, +0.035]다.
   - 두 지표 모두 0 이하라서 **pairwise critic은 학습하지 않았다**(사전 등록 규칙).
2. **같은 수의 VLA 샘플을 하나 더 쓰는 것이 perturbation보다 낫거나 같다.**
   - noise 보정 gap(excess): VLA8 0.030 [0.006, 0.062] / MIX8 0.013 [−0.005, 0.031] / VLA4 0.015 [−0.001, 0.032]
   - perturbation 4개를 더해도(MIX8 − VLA4) excess는 −0.001 [−0.015, 0.012]이다. 반면 VLA 4개를 더하면(VLA8 − VLA4) +0.016 [−0.004, 0.046]이다.
3. **base보다 확실히 나은 perturbation은 없다.**
   - 방향별 p(pert) − p(base)는 −0.03 ~ −0.005이고, CI가 모두 0을 포함한다.
   - "pert 4/4 성공 & base 0/4" 상태는 800 pert branch 중 1개다(+v). 반대 경우는 3개다.
   - split-half로 고른 최선 pert − base는 +0.005 [−0.055, 0.06]이다. 같은 방식으로 VLA cand 1–7 중 최선을 고르면 +0.02 [−0.055, 0.105]다.
4. **회복 동작이 직접 보인다.**
   - perturbation은 K step 뒤 gripper를 base에서 **2.1 cm** 옮긴다(중앙값). VLA 샘플 간 차이 1.0 cm의 두 배다.
   - 그런데 같은 CRN 반복에서 결과가 base와 같을 확률은 pert 0.87, VLA 0.90으로 거의 같다.
   - frozen π0.5 continuation이 2–2.5 cm 수평 offset을 대부분 되돌린다. "더 넓은 후보"가 결과의 차이로 이어지지 않는다.

## 1. 설계 (사전 등록 요약)
- **상태 50개**
  - T017 buffer의 train split에서 episode당 1개를 뽑았다. T017 branch/oracle에 쓴 100 episode는 제외했다(T019 critic 학습 상태와 겹치지 않는다). 결정 d ≤ 12다.
  - phase 층화(특권 라벨, 라벨 전용):

    | 층 | 정의 | 상태 수 |
    |---|---|---|
    | `A0_init` | d=0 | 8 |
    | `A1_far` | approach, gripper–물체 ≥ 8 cm | 12 |
    | `A2_near` | approach, < 8 cm(접촉 직전) | 12 |
    | `C_contact` | grasp_contact 또는 첫 grasp 직후(접촉 직후) | 8 |
    | `L_lost` | post_grasp_lost | 6 |
    | `T_transport` | transport | 4 |

  - approach는 32/50이다. 로그 episode 성공은 25/50이다.
- **후보 12개/상태**
  - VLA 8개: cand 0은 로그에 실행된 chunk(=base), 1–7은 새 initial noise 샘플이다.
  - perturbation 4개: base의 K=5 step에 xyz offset 0.005 m/step을 더했다(목표 2.5 cm 이동, 실제 2.1–2.5 cm).
    - 방향 `±u`: root frame에서 tcp→목표(물체, 또는 잡고 있으면 접시)의 수평 방향
    - 방향 `±v`: 측면
    - 회전과 그리퍼는 그대로다. xyz는 norm stats [q01, q99](모델 정규화 [−1, 1])와 env bound로 clip했다.
  - 크기는 정책 action std(0.008–0.013/step)와 T016 후보 산포(≈2 cm)에 맞췄다.
- **branch**
  - snapshot 복원 → 후보 K=5 step → frozen π0.5로 t=80까지 진행했다.
  - **후보당 4 반복**이다(명세의 2보다 많다. 예산 안이고 split-half 2+2가 필요했다). 2-반복 부분집합도 아래에 보고한다.
  - continuation noise는 **CRN**이다: (상태, 반복, continuation 결정)으로 seed를 정해 12 후보에 같은 noise를 썼다.
  - 총 50 × 12 × 4 = **2400 branch**다.
- **집합**: `VLA4` = cand 0–3, `VLA8` = cand 0–7, **`MIX8` = cand 0–3 + pert 4**(VLA8과 N이 같다). 공정한 비교는 MIX8 vs VLA8이다.
  - 후보 수가 늘면 max−mean은 그 자체로 커진다. 그래서 VLA4 vs MIX8은 공정한 비교가 아니다.

### 복원 검증 (본 실행 전, `restore_test.json`)
- 2 상태 × 4 반복 × cand 0 12벌을 돌렸다. 같은 반복(같은 CRN noise) 안 12벌의 결과 일치율은 **1.00**이다(8 group 모두).
- 로그 noise로 cand 0을 재계산했을 때 차이는 최대 0.0036이다(bf16 batch 차이). cand 0은 로그의 실행 action을 그대로 실행하므로 영향이 없다.
- 복원 이미지의 평균 절대 차이는 ≈0.001이다(T016과 같은 수준).

## 2. gap 표 (success_once, 상태 bootstrap 95% CI, n=50)

| 집합 | E[p] | naive max−mean | perm null | **excess** | split-half 이득 | split-half 선택 값 | E[max p̂] | ICC | beta-binom E[max p]−E[p] |
|---|---|---|---|---|---|---|---|---|---|
| VLA4 | 0.476 | 0.084 | 0.069 | 0.015 [−0.001, 0.032] | 0.024 | 0.500 | 0.56 | 0.15 | 0.051 [0.018, 0.078] |
| VLA8 | 0.475 | 0.125 | 0.095 | **0.030 [0.006, 0.062]** | 0.045 | 0.520 | 0.60 | 0.20 | 0.085 [0.047, 0.121] |
| MIX8 | 0.471 | 0.119 | 0.106 | **0.013 [−0.005, 0.031]** | 0.024 | 0.495 | 0.59 | 0.16 | 0.085 [0.024, 0.125] |

| 대비 (paired) | Δexcess | Δsplit-half 선택 값 | Δ후보 간 분산(2-way) | ΔE[max p̂] |
|---|---|---|---|---|
| **MIX8 − VLA8 (1차)** | **−0.017 [−0.048, +0.006]** | −0.025 [−0.09, +0.035] | −0.003 [−0.013, 0.006] | −0.010 [−0.06, 0.03] |
| MIX8 − VLA4 | −0.001 [−0.015, 0.012] | −0.005 [−0.04, 0.03] | +0.002 [−0.004, 0.010] | +0.030 [0.005, 0.06] |
| VLA8 − VLA4 | +0.016 [−0.004, 0.046] | +0.020 [−0.04, 0.085] | +0.005 [−0.004, 0.013] | +0.040 [0.010, 0.085] |

- **민감도 1(CRN 보정 null)**: CRN 때문에 같은 반복 안의 결과가 후보 간에 상관된다(반복 평균의 분산 0.012, 독립이면 0.005).
  - 그래서 사전 등록한 null(반복 사이에서도 섞음)은 약간 보수적이다.
  - 반복 안에서만 후보 라벨을 섞는 null로 다시 계산해도 결론은 같다: excess_rep는 VLA8 0.035 / MIX8 0.018, Δ −0.018 [−0.049, +0.007].
  - 2-way 분산성분 ICC는 VLA8 0.24 / MIX8 0.19다.
- **민감도 2(2-반복 부분집합, 명세 그대로)**: excess는 VLA4 0.013 / VLA8 0.008 / MIX8 0.012, Δ(MIX8 − VLA8) +0.005 [−0.022, 0.030]이다. 판단할 수 없을 만큼 시끄럽고, 4 반복으로 늘린 것이 맞았다.
- **민감도 3(success_at_end)**: Δexcess(MIX8 − VLA8)는 −0.014 [−0.046, +0.009]다.
- **T017과의 비교**: VLA8의 beta-binomial 상한 0.085 [0.047, 0.121]와 ICC 0.20은 T017의 0.118 [0.074, 0.154] / ICC 0.21과 같은 범위다(상태 집합은 다름). VLA 후보 간 실제 차이는 재현됐다.

## 3. phase별 (`analysis.json` → by_stratum, CI는 상태 bootstrap)

| 층 | n | E[p] | excess VLA8 | excess MIX8 | Δexcess [CI] | Δsplit 선택 값 | pert agree w/ base | VLA agree w/ base |
|---|---|---|---|---|---|---|---|---|
| A0_init | 8 | 0.60 | 0.064 | −0.016 | **−0.079 [−0.153, −0.015]** | −0.19 | 0.68 | 0.73 |
| A1_far | 12 | 0.22 | 0.027 | 0.035 | +0.008 [−0.012, 0.030] | +0.06 | 0.88 | 0.90 |
| A2_near (접촉 직전) | 12 | 0.47 | 0.011 | 0.014 | +0.003 [−0.020, 0.028] | +0.02 | 0.91 | 0.93 |
| C_contact (접촉 직후) | 8 | 0.73 | 0.069 | −0.003 | −0.072 [−0.217, 0.000] | −0.09 | 0.99 | 0.93 |
| L_lost | 6 | 0.13 | 0.000 | 0.038 | +0.038 [0.000, 0.096] | 0.00 | 0.79 | 0.95 |
| T_transport | 4 | 1.00 | 0 | 0 | 0 | 0 | 0.98 | 1.00 |
| approach 전체 | 32 | 0.41 | 0.030 | 0.014 | −0.016 [−0.039, 0.008] | −0.02 | | |
| 그 외 | 18 | 0.59 | 0.031 | 0.011 | −0.020 [−0.096, 0.032] | −0.04 | | |

- 후보 간 실제 차이(VLA8 excess)는 **초기 상태(d=0)와 접촉 직후**에 몰린다. 두 곳 모두 **VLA 샘플 간 차이가 perturbation보다 크다**.
  - A0_init에서는 MIX8이 VLA8보다 확실히 작다.
  - 접촉 직후 perturbation은 결과를 거의 바꾸지 않는다(base와 일치 0.99).
  - 즉 이 구간에서 중요한 것은 수평 위치가 아니다. VLA 샘플이 다르게 내는 그리퍼 timing이나 회전처럼 위치 이외의 차이로 보인다. VLA 후보 간 K-step 뒤 grasp 상태가 다른 비율은 2.7%, pert는 0.1%다.
- perturbation이 차이를 조금이라도 만든 곳은 `A1_far`(+0.06 선택 값)와 `L_lost`(excess +0.038)다. 하지만 n이 6–12이고 CI가 0에 걸친다. `L_lost`에서 pert는 평균적으로 결과를 **나쁘게** 만든다(방향별 −0.08 ~ −0.17). gap 증가는 대부분 mean 하락에서 나온다.
- 방향별로 보면 `A0_init`에서 ±u, +v가 +0.09 ~ +0.13이다. 그러나 8 상태 × 4 반복이고, CI 계산 전 탐색값이며, 다중 비교 보정이 없다. 주장하지 않는다.

## 4. 해석
- **H-A1(action identifiability)**: VLA 후보 사이에는 작지만 실재하는 차이가 있다. VLA8 excess 0.030, CI가 0을 넘는다. T017을 재현한다.
- **"후보 분포가 좁아서 critic이 못 가렸다"는 가설은 지지되지 않는다.**
  - VLA 산포의 두 배인 수평 offset(2.1 cm)을 줘도 결과 일치율은 그대로다(0.87 vs 0.90).
  - 정책 continuation이 offset을 흡수한다. 단일 결정의 수평 위치 차이는 이 과제에서 결과를 거의 결정하지 않는다.
  - 후보 간 결과 분산은 주로 같은 상태의 VLA 샘플 간 차이(grasp timing 등)와 continuation 잡음에서 나온다.
- 따라서 T019 critic의 0.544가 낮은 원인은 다음 두 가지 쪽이다.
  - **결과 잡음**: ICC 0.2, 4 반복으로도 상태당 excess 0.03
  - **회복 동작**이 단일 chunk의 효과를 지운다는 점
  - 지원(support) 부족 쪽은 아니다.
- 이 결론은 다음 조건에 한정된다:
  - 이번 perturbation은 **수평 위치만, 2.5 cm 한 크기**다.
  - 회전·그리퍼 timing perturbation, 여러 결정에 걸친 perturbation은 시험하지 않았다.
  - A0_init과 C_contact에서 VLA 샘플이 이기는 것을 보면, 다음에 시험할 축은 위치보다 **그리퍼 timing / 회전**이다.

## 5. 권고
- 설계서 A7 기준으로 보면 이렇다.
  - No-Go-1("후보 분포 확장으로 oracle 상한이 늘지 않음")에 해당한다. **수평 위치 perturbation 기반 A안 확장(200 상태 × 8 × 2)은 하지 않기를 권한다.**
  - 차이가 있는 구간은 VLA 후보 자체에서 나온다. 그 구간에 대한 신호는 이미 T017/T019 데이터와 같은 규모이고, 천장은 낮다(VLA8 상한 ≈ 0.085/결정).
- A안을 계속한다면 범위를 좁힌다:
  - (a) **d=0과 접촉 직후 상태**만
  - (b) 후보 축은 **그리퍼 timing·회전**(별도 1–2개)
  - (c) 상태당 반복 ≥ 4
  - 비용은 이번과 같은 ≈0.8 GPU-h/50 상태다.
- B/C안(설계서)은 이 결과와 독립적이다. 오히려 "정책이 작은 offset을 회복한다"는 관찰은 **residual 보정이 매 step 반복 적용돼야 효과가 남는다**는 근거다(B4의 K=1 권고와 일치).

## 6. 산출물
- `/outputs/StochRL/t026/`
  - `summary_ko.md`(이 파일)
  - `analysis.json`: 모든 추정치, CI, phase별, 2-반복, success_at_end, 판정
  - `branch_rollouts.parquet`: 2400 행. 상태, 층, 후보, 종류, 방향, 반복, 결과, afterK 특권 라벨
  - `ranking_by_gap.csv`: 상태별 후보 p̂ 12개, 집합별 excess, MIX8 excess 내림차순
  - `oracle_headroom.png`
    - 왼쪽: 집합 × phase excess
    - 가운데: 방향별 p − p(base)
    - 오른쪽: 상태별 VLA8 vs MIX8 excess
  - `recovery_diag.json`: base와의 CRN 결과 일치율, K-step 뒤 변위
- `/workspace/stochrl_t026/`
  - `branch_results_{main,restore_test}.jsonl`
  - `candidates/*.npz`: 후보 action(env·모델 정규화), u/v 방향, 실제 shift, 복원 진단
  - `state_select.json`, `logs/`

## 7. 재실행
```
PY=.venv/bin/python   # repcritic worktree venv; RC_RLINF=<populated RLinf checkout>
$PY -m repcritic.t026_branch --select
bash repcritic/t026_run.sh                       # GPU 1: restore test(일치율 ≥ 0.9 확인) → 본 실행(재시작 가능)
$PY -m repcritic.t026_analyze --out /outputs/StochRL/t026
$PY -m repcritic.t026_recovery --out /outputs/StochRL/t026
```

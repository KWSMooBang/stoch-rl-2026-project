# T026 사전 등록 — branch-advantage mini (설계서 A안 축소판)

작성: 2026-11-19, **branch rollout을 하나도 돌리기 전에** 커밋한다. 코드 `repcritic/t026_branch.py`, 분석 `repcritic/t026_analyze.py`.

## 질문
T019 critic이 같은 상태 후보 순위를 못 가린 것(0.544 ≈ action-only 0.543)이 π0.5 후보 분포가 좁아서인가?
후보에 constrained perturbation을 섞으면 후보 간 실제 결과 차이(oracle gap)가 커지는가?

## 데이터
- 상태: T017 buffer **train split**, episode당 1개, T017 branch/oracle에 쓴 episode 100개는 제외(T019 critic 학습 상태와 겹치지 않음). 결정 d ≤ 12(이후 최소 20 step).
  phase 층화(특권 라벨, 라벨 전용), seed 2026:
  `A0_init`(d=0) 8 · `A1_far`(approach, gripper–물체 ≥ 8 cm) 12 · `A2_near`(approach, < 8 cm, 접촉 직전) 12 ·
  `C_contact`(grasp_contact 또는 grasp 직후 첫 transport 결정, 접촉 직후) 8 · `L_lost`(post_grasp_lost) 6 · `T_transport` 4. approach 32/50.
- 후보 12개/상태:
  - VLA 8개: cand 0 = 로그에 실행된 chunk(base), cand 1–7 = 새 initial noise 샘플.
  - perturbation 4개: base의 실행 K=5 step 각각에 xyz offset **0.005 m/step**(합계 2.5 cm 목표 이동), 방향 `+u, −u, +v, −v`.
    u = 로봇 root frame에서 tcp→목표(잡고 있지 않으면 물체, 잡고 있으면 접시)의 수평 단위벡터, v = z×u(측면). 수평 거리 < 5 mm면 u = root x축.
    회전·그리퍼는 그대로. xyz는 정책 norm stats [q01, q99](모델 정규화 [−1, 1])와 env action bound로 clip.
  - 크기 근거: 정책 action std(xyz) 0.008–0.013/step, T016 후보 간 K-step EE 변위 산포 ≈ 2 cm. 즉 perturbation은 VLA 산포와 같은 크기의 **다른 방향** 이동이다.
- branch: snapshot 복원 → 후보 K=5 step → frozen π0.5로 t=80까지. **후보당 4 반복**(과제 명세는 2; 예산 안이라 4로 늘렸고, 2-반복 부분집합 결과도 보고한다).
  continuation noise는 CRN: (상태, 반복, continuation 결정 index)로 seed를 정해 12 후보에 같은 noise.
- 총 50 × 12 × 4 = 2400 branch. GPU 1 하나(T022가 GPU 0 사용 중), 예상 ≈ 0.8 GPU-h.

## 지표 (success_once, 상태 bootstrap 95% CI, 2000회)
후보 집합 세 개를 같은 방식으로 잰다.
- `VLA4` = cand 0–3, `VLA8` = cand 0–7, **`MIX8` = cand 0–3 + pert 4개**(VLA8과 후보 수·반복 수가 같다).
- 상태별 p̂_i = 4 반복 평균. 집합별:
  - naive gap = max p̂ − mean p̂, permutation null(집합 안에서 결과를 섞음), **excess = naive − null**(T017 방식의 잡음 보정 gap)
  - split-half 값: 반복 {0,1}로 argmax를 고르고 {2,3}으로 평가(또 반대) − 집합 평균 → 실현 가능한 선택 이득
  - between-candidate 분산 s_b², ICC, beta-binomial 모형 E[max p] − E[p](T017 e02_oracle_eb 방식, 집합 크기 N에 맞춤)
  - oracle 값 E[max p̂](naive)와 split-half로 고른 후보의 p(편향 없음)
- **1차 지표(사전 등록)**: Δexcess = excess(MIX8) − excess(VLA8), 상태 단위 paired bootstrap CI.
- 2차: Δ split-half 선택 값(MIX8 − VLA8), Δ s_b², VLA4 대비 MIX8, phase별 표, perturbation 방향별 p(pert) − p(base) (CI), base보다 4/4 vs 0/4로 갈린 상태 수.

## 판정 규칙
- **"perturbation이 gap을 확실히 키운다"** = Δexcess의 95% CI 하한 > 0 **그리고** Δ split-half 선택 값 점추정 > 0.
  → 이 데이터로 pairwise advantage critic(A5 loss, 상태 단위 5-fold split)을 학습하고 T019 0.544 / action-only 0.543과 비교한다. 50 상태라 학습은 **탐색적**이라 적는다.
- 그렇지 않으면(CI가 0을 포함하거나 음수) → "후보 분포가 병목이 아니라 결과 잡음·회복 동작이 병목"으로 결론 내고 critic 학습은 하지 않는다.
- 주의: perturbation이 결과를 **나쁘게** 만들어도 gap(max − mean)은 커진다(mean이 내려감). 그래서 판정에 split-half 선택 값(= 고른 후보가 base 분포보다 실제로 나은가)을 함께 건다. "base보다 확실히 나은 perturbation"은 방향별 p(pert) − p(base) CI 하한 > 0으로 정의한다.

## 복원 검증 (본 실행 전)
`--restore_test`: 2 상태 × (4 반복 × 같은 cand 0 12벌). 같은 반복(같은 CRN noise) 안 12벌의 결과 일치율(시뮬레이터 잡음 바닥), cand 0 재계산 차이(로그 noise로 같은 chunk가 나오는지), 복원 이미지 차이를 확인한다. 일치율 < 90%면 본 실행을 멈추고 프로토콜을 고친다.

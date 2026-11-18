# T019 — E04 critic (frozen π0.5 표현 × BCE / Q / NCE)

- 날짜: 2026-11-19. 브랜치 `exp/repcritic-e04`(`exp/repcritic-e03` 1c445c7에서 분기).
- 코드:
  - `repcritic/e04_data.py`: 데이터 조립
  - `e04_lc.py`: 라벨 비용을 포함한 probe 재fit
  - `e04_critic.py`: 학습과 평가
  - `e04_report.py`: 집계, CI, 관문, 그림
  - `e04_actiononly.py`, `e04_afterk.py`: 원인 진단
  - 사전 등록: `repcritic/reports/t019_prereg.md`(test를 보기 전에 커밋했다)
- 데이터: T017 buffer와 branch, T018 `e04_inputs`(**z_crossfit**)만 썼다. 새 rollout은 없다. VLA는 로드하지 않았다(feature 캐시만 사용).
- **GPU 0·1만 썼고 ≈0.95 GPU-h를 썼다**(상한 3).
  - GPU 0·1에서 shard 4개, 2 GPU × 27.8분 + smoke ≈ 0.01. GPU 3은 쓰지 않았다.
  - 두 GPU는 다른 사용자 프로세스와 함께 쓰는 상태였다. 라벨 비용 probe 재fit은 CPU로 했다.
- 실행 수: 408회(3 seed). critic 하나의 학습 시간은 평균 15초, peak VRAM ≤ 2.0 GB.

## 0. 결론 (한 줄씩)
1. **관문(사전 등록): 형식상 경계선에서 통과.** val로 고른 최선 critic은 `Geo_q_buf+br`이다.
   - 같은 상태 후보 ranking test 일치도는 **0.544 [95% CI 0.501, 0.586]**. CI 하한이 0.5보다 0.001 크다.
   - test oracle p̂ 기준은 0.565 [0.45, 0.69](15 상태).
2. **실질 판정: E07 No-Go, E06은 action-only 대조군을 붙인 조건부(보조) 확인만.**
   - 상태를 전혀 보지 않는 **action-only 선형 모델**도 test 0.543 [0.502, 0.586]을 낸다. 최선 critic과 같다.
   - 즉 critic의 후보 구분은 "어떤 chunk가 대체로 좋은가"라는 상태 무관 규칙 수준이다. 표현을 조건으로 한 행동 가치는 보이지 않는다.
   - gradient도 E07에 쓸 만하지 않다.
     - 더 좋은 후보 방향과 정렬되는 비율이 test에서 0.49–0.61이다.
     - 20 step 상승만으로 모든 critic의 점수가 in-distribution 후보 최대치를 92–99% 넘는다(critic hacking).
3. **가설 1(Representation): 상태 가치 수준에서는 지지, 후보 ranking 수준에서는 판정할 수 없다.**
   - 같은 결정 index 안에서 비교한 성공 AUROC(test): VLA(Fusion/Full) **0.81–0.85** vs Raw CNN 0.60–0.68 / proprio+t 0.79–0.80.
   - 10% 데이터에서는 Fusion/Full 0.74–0.80 vs Raw 0.55–0.57이다.
   - 그러나 ranking 일치도는 모든 입력에서 0.53–0.59이고, 입력·데이터 양과 거의 무관하다.
4. **가설 2(Concept): 약한 부분 지지에 그친다. 라벨 비용을 넣으면 이점이 대부분 사라진다.**
   - Fusion(21차원)은 같은 차원의 RandomProj/PCA/Shuffle보다 상태 가치 AUROC가 높다(+0.04–0.06).
   - 하지만 probe도 같은 부분집합으로 다시 fit한 FusionLC는 10%에서 Full과 비슷하거나 낮다(bce 0.716 vs 0.744, q 0.761 vs 0.758).
   - ranking에서는 개념 입력과 대조군(RandomProj 0.565, Shuffle 0.563, PCA 0.573 / Fusion 0.568, Full 0.571; buf+br 세 objective 평균)의 차이가 없다.

## 1. 설정
- **critic**: `f(z(o,l), E_a(a_{t:t+8}) [, g])`
  - z 인코더: Linear→256. Raw는 작은 CNN(64×48 RGB) + proprio+t.
  - action 인코더: 2층 MLP(8×7 → 256). action은 **모델 정규화 공간 chunk_model[..., :7]**이고, flow sampler가 적분하는 공간과 같다.
  - trunk: 2층 256, dropout 0.1.
  - 모든 critic의 action/trunk 예산이 같다. 입력 투영층만 d_z에 비례한다.
- **objective**:
  - `bce`: success_once
  - `q`: semi-MDP Q^π. γ=0.98/step, Γ=γ^5, 보상은 첫 성공 시 γ^j. buffer는 EMA target과 로그의 다음 chunk로 TD(= frozen π의 SARSA), branch는 MC.
  - `nce`: InfoNCE.
    - positive g: k~Geom(0.7) 결정 뒤 미래 상태의 vlm17_img_mean.
    - 같은 episode의 negative는 mask했다.
    - G0 점수: 같은 instruction의 train 성공 상태 goal bank 64개에 대한 logmeanexp.
    - `nce-k1` 변형: k=1.
  - buf+br regime에서는 NCE에 branch BCE 항을 더했다(하이브리드).
  - 진단용 `pair`: bce + 같은 상태 쌍 logistic ranking loss.
- **데이터 regime**:
  - `buf`: buffer 성공 전 결정 3501개만. 상태당 행동 1개.
  - `buf+br`: buffer + train branch 160 상태 × 16 후보.
- **입력**:
  - 과제가 지정한 것: Raw(=C0), Full, Goal, Geo, Phase, Fusion, RandomProj
  - 추가: Proprio, PCA, Shuffle, Fusion+Full, AECand(후보 의존 ablation, E07에서는 쓸 수 없음)
  - "Q-Raw / Q-VLA"는 각각 `Raw_q`, `Full_q`다.
- **절차**:
  - 하이퍼파라미터는 고정했다(AdamW 3e-4, wd 1e-4, 3000 step, batch 256).
  - early stop은 val objective loss로 했다.
  - critic 선택은 val ranking으로 했다.
  - **test는 보고에만 썼다.**
- **ranking 지표**:
  - `conc_1c`: test 120 상태에서 결과가 다른 후보 쌍의 같은 상태 일치도. oracle 상태 cand 0–7은 p̂(4)를 라벨로 썼다.
  - `conc_or`: test oracle 15 상태 × 8 후보의 p̂(4 continuation) 기준.
  - CI는 상태 bootstrap 2000회이고, 3 seed 평균에 대해 구했다.
- **noise 천장**: 진짜 p_i를 아는 critic이라도 단일 continuation 라벨에 대해서는 **0.81 [0.79, 0.84]** 까지만 얻는다(beta-binomial, ICC 0.21, test 상태 평균). 최선 critic 0.544는 달성 가능한 신호의 약 14%다.

## 2. critic 표 (frac 100%, 3 seed 평균, test; val은 선택용) — 전체: `critic_eval.csv`

| critic | val conc | **test conc [95% CI]** | test oracle conc | pick gain(1c) | AUROC branch | AUROC buf | AUROC 같은 t |
|---|---|---|---|---|---|---|---|
| **Geo_q_buf+br (선택)** | **0.591** | **0.544 [0.501, 0.586]** | 0.565 | +0.019 | 0.835 | 0.851 | 0.810 |
| Fusion_q_buf+br | 0.585 | 0.564 [0.524, 0.602] | 0.569 | +0.015 | 0.852 | 0.865 | 0.815 |
| Geo_nce_buf+br | 0.573 | 0.588 [0.541, 0.635] | 0.586 | +0.011 | 0.785 | 0.788 | 0.750 |
| Fusion_nce_buf+br | 0.569 | 0.591 [0.543, 0.641] | 0.602 | +0.026 | 0.762 | 0.805 | 0.773 |
| Fusion_bce_buf+br | 0.558 | 0.549 [0.506, 0.591] | 0.521 | +0.002 | 0.861 | 0.891 | 0.831 |
| Full_q_buf+br (Q-VLA) | 0.544 | 0.582 [0.538, 0.623] | 0.583 | +0.033 | 0.833 | 0.866 | 0.817 |
| Full_bce_buf+br | 0.538 | 0.573 [0.527, 0.617] | 0.583 | +0.018 | 0.841 | 0.884 | 0.835 |
| Full_nce_buf+br | 0.551 | 0.560 [0.511, 0.606] | 0.502 | +0.018 | 0.690 | 0.745 | 0.709 |
| Goal_q_buf+br | 0.543 | 0.561 [0.512, 0.610] | 0.546 | +0.013 | 0.861 | 0.889 | 0.843 |
| Phase_bce_buf+br | 0.554 | 0.548 [0.505, 0.594] | 0.521 | +0.006 | 0.803 | 0.855 | 0.786 |
| RandomProj_q_buf+br | 0.546 | 0.559 [0.518, 0.598] | 0.618 | +0.006 | 0.784 | 0.830 | 0.761 |
| RandomProj_bce_buf+br | 0.525 | 0.573 [0.526, 0.621] | 0.576 | +0.040 | 0.798 | 0.838 | 0.765 |
| Raw_q_buf+br (Q-Raw) | 0.481 | 0.556 [0.503, 0.613] | 0.417 | +0.020 | 0.746 | 0.809 | 0.679 |
| Raw_bce_buf+br (C-Raw) | 0.491 | 0.537 [0.492, 0.587] | 0.509 | −0.010 | 0.739 | 0.811 | 0.603 |
| Raw_nce_buf+br | 0.541 | 0.526 [0.479, 0.576] | 0.523 | +0.002 | 0.722 | 0.740 | 0.591 |
| AECand_pair (후보 의존) | 0.582 | 0.542 [0.494, 0.593] | 0.498 | −0.001 | 0.774 | 0.845 | 0.780 |
| **action-only 선형(상태 없음)** | 0.568 | **0.543 [0.502, 0.586]** | – | – | – | – | – |
| noise 천장(진짜 p를 아는 critic) | 0.82 | 0.81 [0.79, 0.84] | – | +0.20 | – | – | – |

- pick gain: 상태마다 argmax 후보의 단일 continuation 성공에서 후보 평균을 뺀 값. 최선이 +0.02–0.04이고 천장은 +0.20이다.
- seed 앙상블(점수 z 평균)도 ranking을 바꾸지 않는다. 예: Geo_q 0.559, Fusion_nce 0.589.
- objective 비교:
  - 상태 가치 판별(AUROC 같은 t)은 bce ≈ q(0.79–0.85) > nce(0.71–0.77)다.
  - NCE는 buf만 쓰면 상태 가치도 거의 학습하지 못한다(0.61). goal 하나(G0)라서 미래 상태 판별이 성공 판별로 옮겨가지 않는다.
  - ranking은 nce가 근소하게 높다(test 0.56–0.59). 하지만 val 기준으로는 Q가 1위였다.
- counterfactual branch 데이터의 효과: buf(상태당 행동 1개) vs buf+br의 test ranking 차이가 ≤0.02다. 그래서 critic이 branch 대비를 활용하지 못한다고 본다.
  - within-state 점수 분산은 전체의 1–4%이고, AECand만 6–14%다.

## 3. 소량 데이터 (`fig/sample_efficiency.png`; buf+br, train episode 10/25/50/100%, 3 seed, test)

| objective · 입력 | conc 10% / 25% / 50% / 100% | AUROC 같은 t 10% / 100% |
|---|---|---|
| q · Full | 0.551 / 0.563 / 0.562 / 0.582 | 0.758 / 0.817 |
| q · Fusion | 0.560 / 0.565 / 0.570 / 0.564 | **0.802** / 0.815 |
| q · FusionLC(probe도 부분집합으로 다시 fit) | 0.568 / 0.576 / 0.568 / – | 0.761 / – |
| q · RandomProj | 0.540 / 0.548 / 0.548 / 0.559 | 0.730 / 0.761 |
| q · Raw | 0.532 / 0.535 / 0.546 / 0.556 | 0.565 / 0.679 |
| bce · Full / Fusion / FusionLC / Raw | 0.561 / 0.551 / 0.569 / 0.540 (10%) | 0.744 / 0.787 / 0.716 / 0.547 (10%) |
| nce · Full / Fusion / FusionLC / Raw | 0.557 / 0.549 / 0.579 / 0.524 (10%) | 0.611 / 0.661 / 0.657 / 0.554 (10%) |

- ranking은 데이터 양과 함께 오르지 않는다(10% → 100%에서 ±0.03). 상태 가치는 VLA 입력이 데이터 10%에서도 Raw의 100%보다 높다.

## 4. OOD 점수 부풀림 · calibration · gradient (test; `critic_eval.csv`)
- **OOD**: 같은 상태 16개 후보의 최대 점수를 OOD 행동이 넘는 비율. in-distribution 후보 하나라면 기대치는 ≈1/17 = 0.06이다.

| critic | gauss 0.5σ | 1σ | 2σ | uniform[-1,1] | 다른 상태 행동 | grad 상승 20 step(Δz, 승률) |
|---|---|---|---|---|---|---|
| Geo_q | 0.21 | 0.30 | 0.35 | 0.29 | 0.32 | +23σ, 0.99 |
| Fusion_q | 0.22 | 0.31 | 0.38 | 0.24 | 0.29 | +21σ, 0.99 |
| Fusion_nce | 0.18 | 0.25 | 0.29 | 0.32 | 0.37 | +17σ, 0.98 |
| Full_q | 0.22 | 0.34 | 0.49 | 0.30 | 0.37 | +20σ, 0.99 |
| Raw_q | 0.27 | 0.35 | 0.48 | 0.37 | 0.37 | +25σ, 0.98 |

  → 모든 critic이 정책에서 먼 행동에 높은 점수를 준다. 2σ에서 승률이 0.29–0.49다. E06에서 후보 밖 행동을 쓰거나 E07 guidance를 하려면 prior/trust-region이 필수다.
- **calibration**(bce, test branch / buffer): ECE 0.08 / 0.075(Fusion), 0.087 / 0.11(Full), 0.096 / 0.094(Raw). Brier는 0.15–0.20이다.
- **gradient(E07 준비)**:
  - 모두 action에 대해 미분할 수 있다. grad 유한 100%, finite-difference 상대오차 ≈0, 0.1σ 이웃에서 grad 방향 cos ≈1.0으로 국소적으로 거의 선형이다.
  - grad 크기 중앙값: q 0.15–0.41, bce 0.6–2.3, nce 5–6(normalized action 단위).
  - 에너지 비중: 회전 0.37–0.62, 병진 0.28–0.51, gripper 0.08–0.34.
  - **방향의 유용성은 낮다.** cos(∇f(a_j), a_i − a_j) > 0(i가 더 좋은 후보)인 비율:
    - test 단일 continuation 0.49–0.61, test oracle 0.43–0.59
    - 최선은 Fusion_nce 0.61 / 0.59, Geo_q 0.49 / 0.51
  - 상승 20 step(L2 0.05/step) 후 action이 ±1 밖으로 나가는 비율은 7–10%다.

## 5. 원인 분석 (관문은 경계선이었고, 실질 신호가 없다)
1. **상태 무관 규칙만 배운다.**
   - action-only 선형 모델(같은 상태 안에서 평균을 뺀 ridge, val로 λ 선택)이 0.543이다. 1특징 휴리스틱(gripper 평균, z 이동 등)은 0.50–0.53이다.
   - 표현, objective, 데이터 양을 바꿔도 ranking은 0.53–0.59 띠에 머문다.
2. **critic이 따르는 행동 신호는 "K step 뒤 grasp"다.** `afterK_concordance.json` 참고.
   - 같은 상태 안에서 critic 점수와 grasp_after의 일치도: Full_bce 0.84, Fusion_q 0.79, 나머지 0.62–0.67.
   - 그런데 grasp_after가 최종 성공을 가르는 정도는 0.53뿐이다(gripper→물체 거리 감소는 0.57).
   - 즉 critic은 상태 간 데이터에서 배운 "곧 잡으면 성공한다"는 상관을 후보 비교에 그대로 쓴다. 이 상관은 같은 상태 안에서는 거의 성립하지 않는다.
3. **counterfactual 라벨이 적고 잡음이 크다.**
   - train branch는 160 상태 × 16이고, 그중 결과가 섞인 상태는 약 70개다. 결과 분산의 79%는 continuation 운이다(ICC 0.21).
   - 단일 continuation 라벨로는 후보 간 신호가 학습 손실에서 거의 보이지 않는다. 그래서 early stopping이 상태 가치에 맞춰진다.
   - pair loss를 직접 넣어도(`pair`) 0.55–0.56으로 개선되지 않았다. 후보 의존 AE feature(AECand)도 0.54다.
4. 상태 가치 쪽은 T018 probe와 같은 결론이다. frozen 표현은 "이 상태가 성공할지"를 잘 담고, raw보다 데이터 효율도 좋다.

## 6. 권고
- **E07(guided flow): 지금 critic으로는 진행하지 않는다.**
  - gradient 방향이 더 좋은 후보 쪽과 정렬되는 비율이 chance 근처다.
  - 상승하면 점수가 +17–25σ로 폭주한다.
  - 그래도 진행한다면 val grad 정렬이 가장 높은 `Fusion_nce_buf+br`(val 0.71)을 쓰고, trust-region과 random-direction 대조를 반드시 함께 둔다.
- **E06(reranking): 보조 확인만 한다.**
  - d=0과 approach 상태에서 N=8, `Geo_q_buf+br` vs action-only 선형 vs 무작위로 비교한다.
  - 기대 이득은 오프라인 pick gain 기준 +0.02 내외다(천장 +0.20). action-only와 같으면 critic의 기여는 0으로 판정한다.
- **다음 단계로 필요한 것**(critic 재시도 조건):
  1. 후보 수준 다중 continuation 라벨(p̂, 상태당 후보 8 × continuation 4 이상, 수백 상태)
  2. 또는 continuation 운을 줄인 단기 목표(성공까지 남은 거리나 phase 전이를 k step 안에서 보는 라벨)
  3. 상태 가치를 빼는 advantage 형태의 학습(A = Q − V, 같은 상태 baseline)

## 7. 산출물
- 표:
  - `critic_eval.csv`: 변형별 3 seed 평균, val/test, CI, 앙상블, OOD, gradient, calibration
  - `critic_eval_seeds.csv`
  - `selection.json`: 관문, objective·입력별 최선, noise 천장
  - `action_only_baselines.json`, `afterK_concordance.json`
- 그림: `fig/sample_efficiency.png`, `fig/ranking_ci.png`
- **선택한 critic 체크포인트**: `/outputs/StochRL/t019/critic_ckpt/Geo_q_buf+br_f1.0/seed{0,1,2}.pt`
  - 내용: state_dict, cfg, dz, kind. 모델 클래스는 `repcritic.e04_critic.Critic`.
  - 입력은 `C-Geo.npz` z_crossfit을 train 기준으로 표준화한 것이고, action은 chunk_model[..., :7]이다.
  - objective별 최선(`Geo_bce_buf`, `Geo_nce_buf+br`, `Full_nce-k1_buf+br`, `Full_pair_buf+br`)도 같은 폴더에 있다.
  - 전체 체크포인트는 `/workspace/stochrl_t019/ckpt/`에 있다.

## 8. 재실행
```
PY=<repo>/.venv/bin/python
$PY -m repcritic.e04_data                         # 데이터 조립 (CPU, 수 초)
$PY -m repcritic.e04_lc                           # 라벨 비용 probe 재fit (CPU, 약 5분)
for s in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$((s%2)) $PY -m repcritic.e04_critic --plan all --shard $s --nshard 4 & done
$PY -m repcritic.e04_report --out /outputs/StochRL/t019 ; $PY -m repcritic.e04_actiononly ; $PY -m repcritic.e04_afterk
```

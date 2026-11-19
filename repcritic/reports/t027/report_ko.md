# Frozen π0.5 표현으로 critic을 만들면 제어가 좋아지는가 — ManiSkill3 PutOnPlate PoC 최종 보고서 (E09, T016–T020 정리)

- 작성: 2026-11-19, T027. 브랜치 `exp/repcritic-e09`(`exp/repcritic-e06` 973ee40에서 분기). 그림 코드: `repcritic/e09_report.py`, `repcritic/e09_hacking_case.py`.
- **새 실험은 하지 않았다**(사람 결정: 정리만). GPU 0 h, 모두 CPU. 숫자는 T016–T020 `summary_ko.md`와 원자료 csv·json에서 그대로 옮겼다. 새로 계산한 값은 두 가지이고, 출처 스크립트를 함께 적었다(§7.3).
- `/outputs/StochRL/results.tsv`에는 T016–T020 행이 없다(이 계열은 run 단위 등록 없이 요약만 남겼다). 그래서 GPU-h는 각 요약과 `EXPERIMENTS.md`에서 가져왔다.
- 산출물: `report_ko.md`(이 파일), `summary_results.csv`(108행, 행마다 출처와 사전 등록/탐색 표시), `figs/figA–E.png`, `hacking_case.json/.npz`, `holm_t020.json`.

---

## 0. 요약

frozen π0.5(RLinf SFT, ManiSkill3 `PutOnPlateInScene25Main-v3`)의 내부 표현을 입력으로 critic을 학습했다. 그 critic으로 행동 chunk를 다시 고르면 성공률이 오르는지 확인했다.

| # | 최종 연구 질문(09 문서) | 판정 | 핵심 근거 [95% CI] |
|---|---|---|---|
| Q1 | VLA 표현이 raw보다 critic의 sample efficiency를 높이는가 | **부분 지지** | 상태 가치는 그렇다. 같은 t AUROC, train 10%: Fusion 0.802 / Full 0.758 vs Raw CNN 0.565. 같은 상태 행동 ranking은 아니다. 입력과 데이터 양에 관계없이 0.53–0.59이고, 사전 등록 critic 0.544 [0.501, 0.586] ≈ action-only 0.543 [0.502, 0.586] |
| Q2 | 개념 subspace가 같은 차원 control보다 나은가 | **부분 지지(약함)** | probe 수준에서는 그렇다. C-Fusion(21) succ_pre AUC 0.897 vs RandomProj 0.694 / PCA 0.826. 하지만 라벨 비용을 넣으면(FusionLC 10%: q 0.761 vs Full 0.758) 차이가 없다. ranking도 대조군과 같다(Fusion 0.568 vs RandomProj 0.565 / PCA 0.573 / Shuffle 0.563) |
| Q3 | 좋은 offline critic이 online 성공으로 이어지는가 | **기각(사전 등록 기준 미충족)** | Critic8 − ActOnly8 = +0.007 [−0.087, +0.100]이고 Critic8 − B0 = +0.073 [−0.013, +0.167]이다. 두 CI 모두 0을 포함한다. 단, n=150에서 CI 반폭이 ±0.09라 작은 효과(+0.02)까지 배제하지는 못한다 |
| Q4 | reranking / guided flow / residual 중 무엇이 실용적인가 | **부분(비교 미완)** | reranking만 closed-loop로 돌렸다. 이득은 상태 무관 규칙과 같았고 batch-1 latency는 ×3.1(266 → 827 ms)이다. guided flow(E07)는 gradient 진단으로 No-Go를 내려 하지 않았다. residual(E08)은 후순위로 미뤄 하지 않았다 |

**한 문장 결론.** frozen VLA 표현에는 "이 상태가 성공할지"가 raw 입력보다 훨씬 잘, 더 적은 데이터로 담겨 있다. 그러나 "같은 상태에서 어느 chunk가 더 나은지"는 우리가 만든 critic들이 상태를 보지 않는 규칙 이상으로 읽어내지 못했다. 그래서 reranking 이득도 critic 몫이 0이다. 주된 원인은 두 가지로 보인다. 결과 분산의 79%가 continuation 운이라는 점(ICC 0.21), 그리고 π0.5 후보 사이의 실제 차이가 작다는 점(완벽한 critic의 상한이 결정당 +0.12)이다.

---

## 1. 설정

| 항목 | 값 (출처) |
|---|---|
| 과제 | RLinf `PutOnPlateInScene25Main-v3`, obj_set=train. goal은 G0 하나(`src_on_target`)다. instruction 16종은 물체명만 다르다 (T016, T017) |
| VLA | `RLinf/RLinf-Pi05-ManiSkill-25Main-SFT` rev `eb0e2c90726f7f0bdf99e8c9b6656917f0eb5030`, 3.35B, 모든 파라미터 `requires_grad=False`다. hook 유무에 따른 max \|Δa\| = 0.0 (T016) |
| RLinf | submodule @ `b665e1873b07501f779a5b11dc5bed5b6dcb2125`. Ray runner 없이 단일 프로세스 adapter로 돌렸다 (T016) |
| 제어 | H=8 생성, K=5 실행, denoise 4 step, 80 step = 16 결정, 5 Hz control. camera는 `3rd_view_camera` 640×480이다 (T016) |
| 데이터 | E02 500 episode, episode 단위 350/75/75(split seed 7). 후보 라벨 397 상태 × 16, 다중 continuation 50 × 8 × 4. 합계 7552 branch (T017) |
| 표현 | VLM(gemma 2B) 4/9/13/17층, SigLIP 8/17/26층, AE context, AE candidate. fp16 캐시 (T017) |
| 선택 규칙 | train으로 fit하고, val로 층·α·critic을 고르고, test는 보고에만 썼다. T019·T020은 test를 보기 전에 사전 등록을 커밋했다 |
| 평가 seed | E06은 T017 500 ep와 겹치지 않는 test 초기 상태 150개를 5개 조건이 공유한다(paired) (T020) |
| 시뮬레이터 잡음 | GPU PhysX는 bitwise 재현되지 않는다. 같은 정책을 다시 돌리면 episode의 ≈1–3%가 뒤집힌다 (T016, T020) |

---

## 2. 결과

### 2.1 Primary 결과표 (09 레이아웃)

SR은 success_once이고, n=150 paired, Wilson 95% CI다(T020 `rerank_metrics.csv`). OOD 평가는 하지 않았다.

| Method | Rep | Critic | SR ID (%) | SR OOD (%) | Decision ms (batch 1) | Critic data | Train env steps |
|---|---|---|---:|---:|---:|---|---|
| Frozen VLA (B0) | — | — | **54.7** [46.7, 62.4] | 미수행 | 266 | 0 | 0 |
| Random-N8 (대조) | — | — | 57.3 [49.3, 65.0] | 미수행 | 782 | 0 | 0 |
| ActOnly-N8 (대조, 상태 없음) | — | ridge | 61.3 [53.3, 68.8] | 미수행 | 788 | branch train 후보(상태 내 평균 제거) | 0 |
| **Q-Rerank N8** (`Geo_q_buf+br`) | concepts (C-Geo 12차원) | Q (semi-MDP SARSA) | **62.0** [54.0, 69.4] | 미수행 | 827 | buffer 3501 결정 + branch 160 상태 × 16 = 2560 후보 | 0 (critic 데이터는 아래 각주) |
| Q-Rerank N8, approach gate (CriticApp8, 탐색) | concepts | Q | 65.3 [57.4, 72.5] | 미수행 | ≈ Critic8 | 같음 | 0 |
| Q-Rerank N8 | full | Q | 미수행(오프라인만: conc 0.582) | 미수행 | — | 같음 | 0 |
| CRL-Rerank N8 | full | NCE | 미수행(오프라인만: conc 0.560) | 미수행 | — | 같음 | 0 |
| CRL-Rerank N8 | concepts (Fusion) | NCE | 미수행(오프라인만: conc 0.591) | 미수행 | — | 같음 | 0 |
| Q-Guided / CRL-Guided (E07) | full / concepts | Q / NCE | **미수행**(E07 No-Go, §3 Q4) | 미수행 | — | — | 0 |
| Residual SAC (E08) | concepts | RL | **미수행**(후순위; 별도 트랙 T022) | 미수행 | — | — | — |

- **Train env steps 각주**:
  - critic 데이터 생성: E02 rollout 500 ep × 80 step = 40,000 env step이고, 그중 train split은 350 ep = 28,000 step이다.
  - 후보 라벨: branch 7552회(T017). 각 branch는 결정 시점부터 80 step까지 진행한다. step 합계는 요약에 기록되지 않았다.
  - critic 학습은 offline이라 추가 env step이 0이다. residual 온라인 step은 해당이 없다.
- 오프라인 conc는 T019 test 같은 상태 일치도(3 seed 평균)다. closed-loop로 평가하지 않은 critic은 SR을 비워 두었다.
- 09 레이아웃의 "Q-Rerank full" 자리에는, 실제로 돌린 critic이 val 규칙으로 고른 concept-input `Geo_q_buf+br`라는 점을 표에 그대로 적었다.

### 2.2 critic 표현 × objective (T019, frac 100%, test, 3 seed 평균)

같은 상태 일치도(conc_1c)는 test 120 상태에서 결과가 다른 후보 쌍으로 쟀고, CI는 상태 bootstrap 2000회다.

| 입력 \ objective | bce | q | nce (buf+br는 BCE 하이브리드) | 같은 t AUROC (bce / q / nce) |
|---|---|---|---|---|
| Full (vlm17, 2048) | 0.573 [0.527, 0.617] | 0.582 [0.538, 0.623] | 0.560 [0.511, 0.606] | 0.835 / 0.817 / 0.709 |
| Fusion (21) | 0.549 [0.506, 0.591] | 0.564 [0.524, 0.602] | 0.591 [0.543, 0.641] | 0.831 / 0.815 / 0.773 |
| Geo (12) | — | **0.544 [0.501, 0.586]** (선택) | 0.588 [0.541, 0.635] | — / 0.810 / 0.750 |
| Goal (3) | — | 0.561 [0.512, 0.610] | — | — / 0.843 / — |
| Phase (6) | 0.548 [0.505, 0.594] | — | — | 0.786 / — / — |
| RandomProj (21) | 0.573 [0.526, 0.621] | 0.559 [0.518, 0.598] | — | 0.765 / 0.761 / — |
| Raw CNN | 0.537 [0.492, 0.587] | 0.556 [0.503, 0.613] | 0.526 [0.479, 0.576] | 0.603 / 0.679 / 0.591 |
| AECand (후보 의존, pair) | 0.542 [0.494, 0.593] | | | 0.780 |
| action-only ridge(상태 없음) | **0.543 [0.502, 0.586]** | | | — |
| noise 천장(진짜 p_i) | **0.81 [0.79, 0.84]** | | | — |

- 셀 값은 T019 요약 §2 표와 `critic_eval.csv`의 해당 행이다. "—"는 그 조합이 주 표에 없다는 뜻이다(전체 136행은 `critic_eval.csv`).
- objective 비교:
  - 상태 가치는 bce ≈ q(0.79–0.85) > nce(0.71–0.77)다.
  - ranking은 nce가 근소하게 높다(0.56–0.59). 하지만 val 1위는 Q였다.
  - branch(counterfactual) 데이터를 넣어도 ranking 변화는 ≤0.02다.

### 2.3 probing (T018, val로 고르고 test로 보고, episode bootstrap 1000회)

| 개념 · 지표 | 최선 feature | test [95% CI] | proprio+t | 이득 [95% CI] | 35 ep(10%): VLM / proprio+t |
|---|---|---|---|---|---|
| succ_pre · AUC | vlm17_img_mean | 0.894 [0.86, 0.93] | 0.817 | +0.077 [0.044, 0.110] | 0.82 / 0.76 |
| succ_pre · AUC(같은 t) | vlm17_img_mean | 0.800 [0.74, 0.86] | 0.646 | +0.154 [0.087, 0.222] | — |
| succk15_pre · AUC | vlm13_img_mean | 0.971 [0.96, 0.98] | 0.861 | +0.111 [0.069, 0.158] | 0.95 / 0.80 |
| value · R² | vlm13_img_mean | 0.731 [0.67, 0.78] | 0.142 | +0.589 [0.50, 0.68] | 0.63 / 0.07 |
| geo_rel · R² | vlm9_img_grid4 | 0.943 [0.93, 0.96] | 0.340 | +0.603 [0.57, 0.64] | 0.84 / 0.29 |
| phase4 · macro-F1 | vlm9_txt_mean | 0.695 [0.64, 0.74] | 0.413 | +0.282 [0.22, 0.35] | 0.61 / 0.41 |
| grasp · AUC | vlm9_img_mean | 0.989 [0.98, 0.99] | 0.874 | +0.114 [0.07, 0.17] | 0.97 / 0.86 |
| **후보 단위**: 같은 상태 16 후보 쌍 일치도 | ae_cand 4/9/13/17 | 0.51–0.58 | — | ≈ chance | — |

dimension-matched 대조(E04 입력 자체, test, cross-fit z):

| 입력(차원) | succ_pre AUC 10% / 100% | 같은 t AUC | geo R² 10% |
|---|---|---|---|
| C-Fusion (21) | 0.877 / 0.897 | 0.792 | 0.937 |
| C-PCA (21) | 0.794 / 0.826 | 0.718 | 0.593 |
| C-RandomProj (21) | 0.657 / 0.694 | 0.598 | 0.434 |
| C-Shuffle (21) | 0.764 / 0.820 | 0.660 | 0.416 |
| C-Full (2048) | 0.817 / 0.894 | 0.800 | 0.795 |

- label-shuffle 대조에서도 전체 succ_pre AUC는 0.68–0.75로 남는다(결정 index 사전분포). 그래서 성공 예측은 "같은 t" AUC로 판단한다(T018 §2.4).
- C-Fusion은 train 100% 라벨로 fit한 subspace다. 그래서 "10%" 열은 2차 probe의 데이터만 줄인 결과다(T018 §3 주의 1).

### 2.4 Oracle과 후보 다양성 (T016, T017)

| 진단 | 값 [95% CI] | 출처 |
|---|---|---|
| E01 frozen 성공률 | success_once **0.533** [0.431, 0.633], at_end 0.511 [0.410, 0.612] (90 ep) | T016 |
| 후보 다양성(성공 전 phase) | EE 변위 std/평균 0.53–0.64, 그리퍼 패턴 2.2–5.0개 / 32. placed는 거의 동일(1.1) | T016 §4 |
| OraclePass@16 − Pass@1 (성공 전 59 상태) | **+0.207** [0.15, 0.27] | T016 |
| 그중 첫 chunk 선택 몫(cand16 − ctrl16) | **+0.034** [0.00, 0.08] (≈17%) | T016 |
| ICC(결과 분산 중 후보 간 비율) | **0.21** | T017 |
| 완벽한 critic 상한 E[max p_i] − E[p_i] | **0.118** [0.074, 0.154] | T017 |
| noise null 대비 초과분 | 0.035 [0.013, 0.062], 순열 p = 0.0005 | T017 |
| split-half selector(후보당 2 rollout) | 0.039 [−0.015, 0.099] | T017 |
| Pass@1 → Pass@4 → Pass@16 (397 상태, 단일 continuation) | 0.553 → 0.689 → 0.748 | T017 |
| 상태의 42%는 32 branch 결과가 모두 같았다(all-0 18%, all-1 24%) | — | T017 |
| 부분집합 | 이득은 d=0·approach·실패 경로에 몰렸고 transport는 ≈0이었다 | T016, T017 |

### 2.5 reranking paired 차이 (T020, n=150, episode bootstrap 10,000)

| 비교 | Δ success_once [95% CI] | a/b-only | McNemar p | **Holm 보정 p (8개, 새로 계산)** | 구분 |
|---|---|---|---|---|---|
| **Critic8 − ActOnly8** | **+0.007 [−0.087, +0.100]** | 25/24 | 1.00 | 1.00 | 사전 등록 주 |
| **Critic8 − B0** | **+0.073 [−0.013, +0.167]** | 31/20 | 0.16 | 1.00 | 사전 등록 주 |
| Critic8 − Random8 | +0.047 [−0.047, +0.140] | 28/21 | 0.39 | 1.00 | 사전 등록 부차 |
| ActOnly8 − B0 | +0.067 [−0.013, +0.147] | 25/15 | 0.15 | 1.00 | 사전 등록 부차 |
| ActOnly8 − Random8 | +0.040 [−0.047, +0.127] | 24/18 | 0.44 | 1.00 | 부차(prereg 목록 밖, 보고서에 추가) |
| Random8 − B0 | +0.027 [−0.053, +0.107] | 22/18 | 0.64 | 1.00 | 사전 등록 부차 |
| CriticApp8 − B0 | +0.107 [+0.027, +0.187] | 29/13 | 0.02 | **0.156** | 사전 등록 부차(선택 조건) |
| CriticApp8 − ActOnly8 | +0.040 [−0.047, +0.127] | 26/20 | 0.46 | 1.00 | 사전 등록 부차 |

- success_at_end도 같은 결론이다. Critic8 − ActOnly8 = +0.027 [−0.067, +0.120] (T020 §2).
- d=0에서 같은 후보 집합을 두고 Critic8과 ActOnly8의 argmax가 일치한 비율은 0.29였다(우연 0.125). 둘은 다른 후보를 고르는데 성공률은 같다.

### 2.6 비용

| 단계 | GPU-h | 내용 (출처) |
|---|---:|---|
| T016 E01 + 다양성 + Oracle-N | ≈2.5 | GPU 0 1.1 + GPU 1 1.4 (T016) |
| T017 E02 rollout + branch 7552 | ≈3.2 | (T017 §6) |
| T018 E03 probing | 0 | CPU 약 25분 (T018) |
| T019 E04 critic 408 run | ≈0.95 | 2 GPU × 27.8분. critic 1개 평균 15 s, peak ≤2.0 GB (T019) |
| T020 E06 closed-loop | ≈1.5 | smoke 0.09 + 0.67 + 0.69 + bench 0.03 (T020) |
| T027 E09 정리 | 0 | CPU. hacking 재실행 약 2분 |
| **합계** | **≈8.2** | RESEARCH.md의 "사용 ≈8.2 GPU-h"와 같다 |

| latency (ms/결정) | B0 | Random8 | ActOnly8 | Critic8 |
|---|---:|---:|---:|---:|
| batch 1 (GPU 1, 공유 부하, `bench_latency.json`) | 266 ± 17 | 782 ± 19 | 788 ± 10 | 827 ± 22 |
| batch 25, env당 분할 | 55 | 369 | 364 | 367 |
| critic 점수 계산 자체 | — | — | — | 2.3 ms / batch 결정 |

- batch 1에서 비용은 거의 모두 N=8 sampling이 차지한다(×3.1). Geo feature 캡처를 포함한 critic 추가분은 +40 ms다.
- 참고: T016 batch-1 latency는 141 ms였다. T020의 266 ms는 공유 GPU 부하에서 잰 값이라 서로 비교하지 않는다.

### 2.7 그림 (`figs/`)

> 서버에 한글 글꼴이 없어 그림 라벨은 영어로 남겼다(코드는 글꼴이 있으면 한글로 그린다).

| 그림 | 내용 | 데이터 |
|---|---|---|
| `figA_sample_efficiency.png` | 표현별 (a) 같은 t AUROC, (b) 같은 상태 conc를 데이터 비율 10/25/50/100%에 대해 그렸다. Q objective, 3 seed | T019 `critic_eval.csv` |
| `figB_same_state_ranking.png` | 같은 상태 ranking: critic 10종, action-only, noise 천장 [CI] | T019 `critic_eval.csv`, `action_only_baselines.json`, `selection.json` |
| `figC_oracle_decomposition.png` | (a) Oracle-N cand vs ctrl arm, (b) chunk 선택 기여 분해: Oracle → 완벽한 critic → 실제 critic → closed loop | T016 json, T017 json, 요약 인용 |
| `figD_rerank_success.png` | (a) 조건별 성공률 [Wilson], (b) paired 차이 [bootstrap], McNemar p, Holm p | T020 `rerank_metrics.csv` |
| `figE_critic_hacking.png` | (a) gradient 상승 20 step 동안의 점수(3 seed), (b) test 상태 하나의 16개 후보 궤적, 성공/실패 색 구분, (c) Gaussian 섭동 σ별로 in-dist 최대 점수를 넘는 비율 | `e09_hacking_case.py`(새로 계산) |

---

## 3. 연구 질문별 판정

### Q1. 표현의 sample efficiency — **부분 지지**
- **지지하는 근거(상태 가치)**:
  - critic의 같은 t AUROC(Q, test)는 10%에서 Fusion 0.802 / Full 0.758 / RandomProj 0.730 / Raw 0.565이고, 100%에서 0.815 / 0.817 / 0.761 / 0.679다.
  - VLA 입력은 데이터 10%만으로 Raw의 100%를 넘는다(Fig A-a).
  - probe도 같은 결론이다. 35 ep의 VLM이 350 ep의 proprio+t를 넘고(value R² 0.63 vs 0.14, geo 0.84 vs 0.34), succ_pre만 비슷하다.
- **지지하지 않는 근거(행동 ranking, 09의 primary endpoint (b))**:
  - 같은 상태 일치도는 모든 입력에서 0.53–0.59이고, 10% → 100%에서 ±0.03만 변한다(Fig A-b).
  - 사전 등록 관문은 형식상 통과했다(0.544 [0.501, 0.586], 하한이 0.5보다 0.001 높음).
  - 하지만 action-only 0.543 [0.502, 0.586]과 같다. noise 천장 0.81의 약 14%다(Fig B).
- **해석**: frozen 표현의 데이터 효율 이점은 V(s) 수준에서만 확인됐다. Q(s,a) − V(s), 즉 행동 구분 신호에서는 확인되지 않았다.

### Q2. 개념 subspace — **부분 지지(약함)**
- **지지하는 근거**:
  - 21차원 C-Fusion은 같은 차원의 RandomProj, PCA, label-shuffle보다 정보가 많다(succ_pre AUC 0.897 vs 0.694 / 0.826 / 0.820, geo R² 10% 0.937 vs 0.434 / 0.593 / 0.416).
  - critic 상태 가치도 +0.04–0.06 높다.
- **약화하는 근거**:
  1. subspace를 train 100% 라벨로 만들었다. probe도 같은 부분집합으로 다시 fit한 FusionLC는 10%에서 Full과 비슷하거나 낮다(bce 0.716 vs 0.744, q 0.761 vs 0.758).
  2. C-Goal은 성공 라벨을 회귀한 것이라 부분적으로 순환적이다.
  3. C-Geo나 C-Phase 하나만으로는 succ_pre가 PCA21보다 낫지 않다.
  4. ranking에서는 개념 입력과 대조군의 차이가 없다(buf+br 세 objective 평균 Fusion 0.568, Full 0.571 vs RandomProj 0.565, Shuffle 0.563, PCA 0.573).
- **해석**: 개념 subspace는 "압축된 상태 요약"으로는 유효하다. 하지만 critic의 행동 구분 능력에는 기여하지 않았다.

### Q3. offline critic → online 성공 — **기각(사전 등록 기준 미충족)**
- **사전 등록 규칙**: Critic8 − ActOnly8과 Critic8 − B0가 둘 다 양의 방향으로 0을 배제해야 개선을 주장한다.
- **결과**: 둘 다 0을 포함한다(+0.007 [−0.087, +0.100], +0.073 [−0.013, +0.167]).
- N=8 재선택의 이득은 상태 무관 규칙(ActOnly8 − B0 = +0.067 [−0.013, +0.147])과 거의 같다. critic의 몫은 0에 가깝다(Fig D). 이는 오프라인 결론(critic 0.544 ≈ action-only 0.543)과 일치한다.
- **주의 1 — 검정력**: n=150에서 CI 반폭이 ±0.09이므로, 오프라인에서 기대한 이득(+0.02)은 검출할 수 없는 크기였다. "효과 없음"이 아니라 "개선을 주장할 근거 없음"으로 읽어야 한다.
- **주의 2 — CriticApp8**: approach에서만 critic을 쓴 CriticApp8은 B0 대비 +0.107 [+0.027, +0.187](McNemar p=0.02)이었다. 그러나 부차 비교 8개에 Holm 보정을 하면 p=0.156이라 유의하지 않다. ActOnly8 대비로도 +0.040 [−0.047, +0.127]이다. 탐색적 신호로만 남긴다.
- **원인 진단(T019 §5)**:
  - critic이 따르는 행동 신호는 "K step 뒤 grasp"다(같은 상태 일치도 0.62–0.84). 그런데 grasp_after가 최종 성공을 가르는 정도는 0.53뿐이다.
  - 상태 간 데이터에서 배운 "곧 잡으면 성공한다"는 상관이 같은 상태 안에서는 거의 성립하지 않는다.

### Q4. reranking vs guided flow vs residual — **부분(세 방법 비교는 미완)**
- **reranking(E06)만 수행했다**:
  - 같은 compute 조건에서 Critic8과 ActOnly8의 성공률은 같고(0.620 vs 0.613), batch-1 latency는 둘 다 B0의 약 3배(827 / 788 vs 266 ms)다.
  - 따라서 현재 critic을 쓰는 reranking의 실용적 가치는 "N개 sampling + 싼 상태 무관 규칙"과 구별되지 않는다.
- **guided flow(E07)는 하지 않았다. 이유: T019 gradient 진단에서 No-Go.**
  - gradient가 더 좋은 후보 방향과 정렬되는 비율은 test 0.49–0.61로 chance 근처다.
  - 20 step 상승만으로 모든 critic의 점수가 +17–25σ로 폭주하고, in-dist 후보 최대를 92–99% 넘는다.
  - 이 보고서에서 선택 critic을 CPU로 다시 돌려 같은 결과를 확인했다(Fig E, §7.3). 평균 +23.1σ, 승률 0.99다.
  - 같은 상태 16개 후보 중 branch 성공 후보와 실패 후보가 거의 평행하게 함께 올라간다(Fig E-b). 상승 방향이 성공과 무관하다는 뜻이다.
  - Gaussian 1σ 섭동만 줘도 후보 최대를 넘는 비율이 0.28–0.34로, in-dist 기대치 1/17의 약 5배다(Fig E-c).
- **residual RL(E08)은 하지 않았다. 이유:**
  - L1에서 후순위로 정했다(RESEARCH.md "E08 residual은 후순위").
  - 앞선 frozen-VLA residual SAC PoC(T008·T011)들이 실패한 이력이 있다.
  - 이 질문은 별도 트랙 T022(PPO, GRPO, SAC, CrossQ, DAgger 비교, 진행 중)로 옮겨졌다.
- 그래서 "같은 compute/interaction budget에서 어느 방법이 가장 실용적인가"에는 이 PoC로 답할 수 없다. 남은 질문은 T022–T025로 이어진다(§6).

---

## 4. 통계 점검

| 결과 | 단위 · n | CI 방법 | 다중 비교 | 사전 등록 |
|---|---|---|---|---|
| E01 성공률 (T016) | episode 90 | Wilson 95% | 없음 | 아니오(baseline) |
| Oracle-N, cand − ctrl (T016) | 상태 59 / episode 20 | episode bootstrap 95% | 없음 | 아니오(진단) |
| 다중 continuation Oracle (T017) | 상태 50 × 8 × 4 | 상태 bootstrap. 순열 null p = 0.0005. beta-binomial 모형 CI | 없음 | 아니오(진단) |
| probing (T018) | test episode 75 | episode bootstrap 1000회 | 보정 없음. 층·α·subspace는 val로 선택(37 feature × 다수 개념) | **아니오 — 탐색적**. 단 test는 선택에 쓰지 않았다 |
| critic 관문 (T019) | test 120 상태 | 상태 bootstrap 2000회, 3 seed 평균 | 단일 관문(val 1위 critic 하나만 test 검정) | **예** (`t019_prereg.md`) |
| critic 전체 표·sample efficiency (T019) | 136 변형 × 3 seed | 상태 bootstrap / seed sd | 보정 없음 | 아니오 — 탐색적(관문 critic 외) |
| OOD·gradient·hacking (T019, T027 재실행) | test branch 1920행 | 없음(3 seed 범위만) | — | 아니오(진단). E07 조건은 사전 등록에 "기록" 항목으로 있었다 |
| E06 주 비교 2개 (T020) | episode 150 paired | Wilson(조건별), paired episode bootstrap 10,000(차이) | **교집합 규칙**(두 CI 모두 양수여야 주장). 이 규칙은 그 자체로 보수적이라 α 보정이 필요 없다 | **예** (`t020_prereg.md`, 커밋 a400e45) |
| E06 부차 비교 (T020) | 같음 | 같음 + McNemar exact | **Holm**(8개): T020은 "0.02 > 0.05/8"로 판정했고, 보정 p를 이 보고서에서 새로 계산했다(CriticApp8 − B0 p_Holm = 0.156, 나머지 1.00) | 예(목록). ActOnly8 − Random8은 사전 등록 목록 밖이다 |

- 09 문서의 primary endpoint 두 개에 대해:
  - (a) N=8 reranking success delta: 사전 등록 결과 개선을 주장하지 않는다.
  - (b) critic 5k sample matched-state ranking: 관문은 경계선에서 통과했지만 action-only와 같다. 실질 판정은 미달이다.
- 시뮬레이터 바닥 잡음: GPU PhysX 비재현성(episode의 1–3%)이 paired 차이에 그대로 섞인다. Random8 − B0의 discordant 40쌍이 바닥 잡음과 sampling 분산을 합한 크기다(T020).
- seed 반복: closed-loop 조건은 noise seed를 batch당 1개 썼다. 09 문서의 "seed 3–5 반복"은 하지 않았다. critic은 학습 seed 3개다.

---

## 5. 논의

1. **"상태 가치는 쉽고, 행동 우위는 어렵다."**
   - 이 PoC의 모든 층위(probe, critic, closed loop)에서 같은 모양이 나왔다.
   - frozen π0.5 표현은 V(s)를 raw보다 훨씬 잘, 더 적은 데이터로 담는다.
   - 그러나 같은 상태 후보 간 A(s,a)는 선형 probe(0.51–0.58), 비선형 critic(0.53–0.59), closed-loop(critic 몫 +0.007) 어디에서도 상태 무관 규칙을 넘지 못했다.
2. **신호가 작고 잡음이 크다.**
   - 결과 분산의 79%가 continuation 운이다(ICC 0.21).
   - 완벽한 critic의 상한도 결정당 +0.12이고, 후보당 2 rollout selector 수준이면 +0.04다.
   - 단일 continuation 라벨로 학습하면 이 신호가 손실에서 거의 보이지 않는다. 그래서 early stopping도 상태 가치에 맞춰진다(T019 §5.3).
3. **critic은 "곧 잡는 chunk"를 고른다.**
   - Critic8과 ActOnly8 모두 평균 step이 줄었다(48.4 / 47.8 vs B0 51.4). approach 결정 비율도 낮다(0.33 vs 0.45).
   - "빨리 잡기"는 state-free 규칙으로도 얻을 수 있는 이득이고, 그것이 N=8 이득의 대부분이다.
4. **gradient를 이용한 정책 개선은 현재 critic으로는 위험하다.**
   - 점수가 정책에서 먼 행동에 높게 매겨진다(Fig E-c).
   - 상승 방향은 성공·실패 후보를 구분하지 않는다(Fig E-b).
   - guidance나 residual을 다시 시도하려면 trust-region/prior와 random-direction 대조가 필수다.

---

## 6. 한계와 다음 단계

### 한계
- **단일 goal(G0)**:
  - instruction 16종은 물체명만 다르고 "접시 위"라는 관계는 고정이다. 그래서 critic은 사실상 이진 성공 critic이다.
  - goal 교환이나 language composition에 대한 주장은 하지 않는다.
  - NCE의 goal 조건이 성공 판별로 옮겨가지 않은 것(buf만 쓰면 0.61)도 이 구조 탓일 수 있다.
- **후보 다양성**:
  - π0.5 후보는 성공 전 phase에서 서로 다르다(EE std/평균 0.5–0.65, 그리퍼 패턴 2–5개).
  - 하지만 성공 확률 차이는 작다(상한 +0.12, 상태의 42%는 32 branch 결과가 모두 같음).
  - 후보 분포가 병목인지는 T026에서 직접 검정한다.
- **라벨 잡음(ICC 0.21)**:
  - 후보 라벨 대부분이 continuation 1개짜리다.
  - 다중 continuation(p̂) 상태는 test 15개뿐이라 oracle 기준 일치도의 CI가 넓다(예: 0.565 [0.45, 0.69]).
- **분포와 일반화**:
  - 물체와 overlay 16종은 모든 split에 나온다. novel 물체, OOD, 다른 과제에 대한 일반화는 보지 않았다.
  - **OOD split(Init/Visual/Execution/Semantic)은 미수행이다.**
- **검정력**: E06 n=150, 조건당 noise seed 1개다. +0.02 수준의 효과는 검출할 수 없다.
- **privileged label**:
  - 개념 probe(C-Geo, C-Phase, C-Goal)는 sim 특권 라벨로 학습했다. 따라서 이들 critic은 "privileged supervision으로 만든 표현" 기준선이다.
  - 정책 입력에는 특권 정보를 넣지 않았다.
- **재현성**: GPU PhysX 비재현성 1–3%가 있다. T016·T020의 latency는 공유 GPU 부하에서 쟀다.
- **표현과 명칭**: Frozen VLA 본체는 고정했다. 하지만 probe, action encoder, critic은 학습했다. 정책 개선이 확인되지 않았으므로 이 작업은 "contrastive **critic learning**"이라고 부른다.

### 하지 않은 실험과 이유
| 실험 | 상태 | 이유 |
|---|---|---|
| E05 goal-swap 반사실 | 미수행 | 단일 goal(G0) 과제라 유효한 goal 교환이 없다 |
| E07 critic-guided flow | 미수행 (No-Go) | T019 gradient 진단: 정렬 0.49–0.61, 상승 시 +17–25σ 폭주 |
| E08 residual RL | 미수행 (후순위) | L1 결정. residual 알고리즘 비교는 T022로 분리했다 |
| OOD split (E09) | 미수행 | 사람 결정(정리만). π0.5 OOD config 호환성도 확인하지 않았다 |
| E06 approach-gated 확인 | 미수행 | T020 권고: CriticApp8 vs ActOnlyApp8 vs B0, n≥200, 사전 등록, ≈1 GPU-h |

### 진행 중인 과제와의 연결
- **T026 branch-advantage mini**(Value-to-Control A안 축소판, 진행 중, 결과 없음):
  - 이 보고서의 핵심 미해결 질문을 직접 다룬다. 질문은 "ranking 실패가 π0.5 후보 분포가 좁아서인가"이다.
  - 방법: 상태 50 × 후보 8(VLA 4 + constrained perturbation 4) × 반복 2로 oracle gap을 비교한다.
  - gap이 커지면 pairwise advantage critic(A = Q − V, 같은 상태 baseline)을 학습해 이 보고서의 0.544 / 0.543과 비교한다.
  - 이것은 T019 §6의 critic 재시도 조건 1·3(다중 continuation 라벨, advantage 형태 학습)과 같다.
  - gap이 커지지 않으면 결론은 "병목은 후보 분포가 아니라 결과 잡음과 회복 동작"이 된다.
- **T022–T025 value-to-control**:
  - 이 PoC에서 확실히 남은 자산은 **상태 가치 V̂(s)** 다(같은 t AUROC 0.81–0.85).
  - T023 value audit이 이를 이어받았다. 결과는 V̂ AUROC 0.90(같은 t 0.81)이고, PBRS 종료 Φ_T=1 권고, B·C 조건부 Go다.
  - 남은 단계:
    - T024: V̂를 residual RL(T022)의 potential shaping으로 쓴다.
    - T025: low-V gate로 개입 시점을 고른다.
  - 즉 "행동을 직접 고르는 critic" 경로는 여기서 닫고, "상태 가치를 RL의 보상과 gate로 쓰는" 경로로 넘어간다.

---

## 7. 재현

### 7.1 그림·표 재생성 (CPU)
```
PY=<repcritic venv>/bin/python     # 예: .claude/worktrees/repcritic-e04/.venv/bin/python (matplotlib 포함)
CUDA_VISIBLE_DEVICES="" $PY -m repcritic.e09_hacking_case     # ≈2분, /outputs/StochRL/t027/hacking_case.{json,npz}
$PY -m repcritic.e09_report --out /outputs/StochRL/t027       # figs/, summary_results.csv, holm_t020.json
```

### 7.2 입력 파일
- T016: `oracle_diversity_analysis.json`
- T017: `oracle_eb_main.json`
- T018: `small_data_curves.csv`
- T019: `critic_eval.csv`, `selection.json`, `action_only_baselines.json`, `critic_ckpt/Geo_q_buf+br_f1.0/seed{0,1,2}.pt`
- T020: `rerank_metrics.csv`, `bench_latency.json`
- critic 데이터: `/workspace/stochrl_t019/data.npz`, T018 `e04_inputs/C-Geo.npz`

### 7.3 이 보고서에서 새로 계산한 값
1. **Holm 보정 p**(T020 McNemar p 8개, success_once): `e09_report.py::holm` → `holm_t020.json`. CriticApp8 − B0는 0.156이고 나머지는 1.00이다.
2. **critic hacking 재실행**(`e09_hacking_case.py`):
   - 대상과 절차: `Geo_q_buf+br` 3 seed, test branch 1920행(120 상태). T019 `ood_and_grad`와 같은 프로토콜(정규화 상승 20 step, L2 0.05)을 쓰고, 궤적 전체를 기록했다.
   - 상승 20 step 뒤 평균 z: seed 0/1/2 = 22.7 / 21.2 / 25.3(평균 23.1). T019 csv는 +23σ다.
   - 승률: 0.999 / 0.981 / 0.994. T019는 0.99다.
   - Gaussian 1σ 승률: 0.28 / 0.34 / 0.28(평균 0.30). T019 csv는 0.30이다.
   - in-dist 후보 최대의 평균 z는 ≈1.8이다. 상승 후 ±1 밖 비율은 0.08–0.10이다(T019: 7–10%).
   - T019 값과 일치하므로 재현을 확인했다.

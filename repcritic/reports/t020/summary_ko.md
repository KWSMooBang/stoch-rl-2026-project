# T020 — E06 closed-loop action reranking (보조 확인)

- 날짜: 2026-11-19. 브랜치 `exp/repcritic-e06`(`exp/repcritic-e04` 69ac0bc에서 분기).
- 코드:
  - `repcritic/e06_scorers.py`: 온라인 Geo probe, phase gate, critic 앙상블, action-only
  - `e06_rerank.py`: closed-loop runner, latency bench
  - `e06_report.py`: 집계
- **사전 등록**: `repcritic/reports/t020_prereg.md`(= `prereg.md`)
  - 커밋 a400e45. test episode를 돌리기 전에 커밋했다.
  - smoke는 T017 **train** episode 6개로만 돌렸다.
- **GPU 0·1만 썼다. 합계 ≈1.5 GPU-h**(상한 2):
  - smoke 0.09
  - 본 실행 0.67(GPU1) + 0.69(GPU0)
  - batch-1 bench ≈0.03
  - GPU 3은 쓰지 않았다. 두 GPU는 다른 사용자 프로세스와 함께 쓰는 상태였다(이용률 ~95%).
- VLA는 고정했다. test episode로는 아무것도 튜닝하지 않았다.

## 0. 결론
1. **판정(사전 등록): "표현 조건 critic이 제어를 개선한다"고 말할 수 없다.**
   - Critic8 − ActOnly8 = **+0.007 [−0.087, +0.100]**
   - Critic8 − B0 = +0.073 [−0.013, +0.167]
   - 두 paired CI 모두 0을 포함한다.
2. N=8 재선택의 이득은 상태를 보지 않는 규칙으로도 거의 같다.
   - ActOnly8 − B0 = +0.067 [−0.013, +0.147]
   - Critic8 − B0 = +0.073
   - critic의 몫(Critic8 − ActOnly8)은 0에 가깝다. T019 오프라인 결론(critic ≈ action-only, 0.544 vs 0.543)과 일치한다.
3. (선택 조건) CriticApp8(approach로 판정된 결정에서만 critic 선택):
   - B0 대비 **+0.107 [+0.027, +0.187]**, McNemar p=0.02.
   - 그러나 이것은 사전 등록에서 **부차 비교 8개 중 하나**다. Holm 보정을 하면 유의하지 않다(0.02 > 0.05/8).
   - ActOnly8 대비는 +0.040 [−0.047, +0.127]이고 Critic8 대비도 +0.033이다. 탐색적 신호로만 기록한다.
   - T017에서 "Oracle 이득이 approach에 몰려 있다"는 관찰과 방향은 맞다. 확인하려면 별도로 사전 등록하고, approach-gated ActOnly 대조를 둔 반복 실험이 필요하다.
4. 검정력 한계: n=150 paired에서 CI 반폭이 ≈±0.09다.
   - 오프라인에서 기대한 이득(+0.02)은 이 실험으로 검출할 수 없다.
   - "작은 효과가 없다"는 결론도 아니다.
5. **E07은 이 결과를 근거로 No-Go 판정하지 않는다**(지시대로). E07 판단은 T019의 gradient 진단을 따른다.

## 1. 설정 (사전 등록 요약)
- **episode**:
  - test 초기 상태 **150개**. `randperm(total, 20261119)[500:]`에서 T016 id를 뺐다.
  - T017의 500 episode(train/val/test 전부)와 겹치지 않는다. critic·probe·λ 어느 것도 이 상태를 보지 않았다.
  - 5개 조건 모두 같은 150개 id에서 돌렸다(paired).
- **실행**:
  - 80 step, 결정 16회(결정마다 K=5를 실행). H=8 chunk에서 앞 5개를 실행하고 나머지는 버린 뒤 다시 sampling한다.
  - noise는 batch마다 seed `20200+batch` 하나다. 매 결정에서 [n, 8]개를 뽑는다.
  - 그래서 d=0 후보 집합은 N=8 조건 모두 같다. B0는 후보 0을 실행한다.

| 조건 | 선택 규칙 |
|---|---|
| B0 | frozen N=1 |
| Random8 | 후보 8개 중 무작위 |
| ActOnly8 | T019 action-only ridge(상태 없음, λ=0.01, val로 선택). 재현 test conc 0.543 |
| Critic8 | T019 `Geo_q_buf+br`, 3 seed 앙상블(후보 집합 안에서 z-score한 Q의 평균). 오프라인 test conc 0.554(seed별 평균 0.544) |
| CriticApp8 | C-Phase probe(vlm9_txt_mean)가 approach라고 하면 Critic8의 선택, 아니면 후보 0(= frozen sample) |

- critic 입력: 온라인 C-Geo probe(vlm9_img_grid4 → T018 ridge → T019 train 표준화). T018 cache와 비교한 오차 7e-9.

## 2. 결과 (n=150 paired) — `rerank_metrics.csv`, `rerank_episodes.csv`, `rerank_results.json`

| 조건 | success_once [Wilson 95%] | success_at_end [Wilson 95%] | 평균 step | ms/결정 (batch 25, env당 분할) | ms/결정 (batch 1) |
|---|---|---|---|---|---|
| B0 | 0.547 [0.467, 0.624] | 0.533 [0.454, 0.611] | 51.4 | 55 | 266 |
| Random8 | 0.573 [0.493, 0.650] | 0.560 [0.480, 0.637] | 51.1 | 369 | 782 |
| ActOnly8 | 0.613 [0.533, 0.688] | 0.567 [0.487, 0.643] | 47.8 | 364 | 788 |
| **Critic8** | **0.620** [0.540, 0.694] | **0.593** [0.513, 0.669] | 48.4 | 367 | 827 |
| CriticApp8 | 0.653 [0.574, 0.725] | 0.620 [0.540, 0.694] | 46.5 | 366 | ≈ Critic8 |

- latency:
  - batch 25: 결정당 B0 1.38 s, N=8 조건 9.1–9.2 s(공유 GPU).
  - critic 점수 계산 자체는 batch 결정당 2.3 ms이고, Geo feature 캡처를 포함한 batch-1 추가분은 +40 ms(827 vs 788)다.
  - 비용은 거의 모두 N=8 sampling에서 나온다(batch-1에서 ≈3배, batch 25에서 ≈6.6배).
  - batch-1 bench는 GPU 1에서 다른 사용자 부하가 있는 상태로 쟀다.

### paired 차이 (episode bootstrap 10,000, 95% CI; McNemar exact p; a/b-only = 한쪽만 성공한 episode 수)

| 비교 | Δ success_once [CI] | a/b-only, p | Δ success_at_end [CI] |
|---|---|---|---|
| **Critic8 − ActOnly8** (주) | **+0.007 [−0.087, +0.100]** | 25/24, p=1.00 | +0.027 [−0.067, +0.120] |
| **Critic8 − B0** (주) | **+0.073 [−0.013, +0.167]** | 31/20, p=0.16 | +0.060 [−0.033, +0.153] |
| Critic8 − Random8 | +0.047 [−0.047, +0.140] | 28/21, p=0.39 | +0.033 [−0.060, +0.127] |
| ActOnly8 − B0 | +0.067 [−0.013, +0.147] | 25/15, p=0.15 | +0.033 [−0.047, +0.120] |
| ActOnly8 − Random8 | +0.040 [−0.047, +0.127] | 24/18, p=0.44 | +0.007 [−0.080, +0.100] |
| Random8 − B0 (sampling 분산) | +0.027 [−0.053, +0.107] | 22/18, p=0.64 | +0.027 [−0.060, +0.113] |
| CriticApp8 − B0 | +0.107 [+0.027, +0.187] | 29/13, p=0.02 (Holm 보정 후 유의하지 않음) | +0.087 [0.000, +0.173] |
| CriticApp8 − ActOnly8 | +0.040 [−0.047, +0.127] | 26/20, p=0.46 | +0.053 [−0.040, +0.147] |

- 주의: GPU PhysX는 run마다 bitwise 재현되지 않는다(T016 기준 episode의 ≈1–3%가 뒤집힌다). 같은 정책끼리 비교해도 차이가 0이 되지는 않는다.
  - Random8 − B0의 discordant 40쌍이 이 바닥 잡음과 sampling 분산을 합한 크기다.

## 3. 진단
- **선택 분포**:
  - Critic8과 ActOnly8은 후보 index를 거의 균등하게 고른다. 특정 noise 슬롯에 대한 편향은 없다.
  - CriticApp8은 결정의 64%에서 후보 0을 골랐다(gate 밖).
- **d=0(같은 후보 집합)에서 argmax 일치율**: Critic8 vs ActOnly8 = **0.29**(우연 0.125).
  - 둘은 서로 다른 후보를 고르는 경우가 많지만 성공률은 같다. 오프라인 test 16후보에서의 일치율도 0.19였다.
  - 이것은 "critic의 상태 의존 부분이 성공에 기여하지 않는다"는 T019 해석과 맞는다.
- **후보 다양성**(후보 쌍의 model-space chunk L2 평균): 0.90–1.03. 후보가 동일해서 생기는 문제는 아니다.
- **clip 비율**: 0.038–0.043으로 모든 조건이 같다. 후보 안에서만 고르므로 OOD 행동은 없다.
- **phase gate**(특권 라벨은 분석에만 썼다): approach에서 gate 발동률 0.92, approach가 아닌 결정에서 0.17.
- **phase 분포**: N=8 critic 조건에서 approach 결정 비율이 낮고(0.33 vs B0 0.45) placed 비율이 높다(0.35–0.39 vs 0.33). 더 빨리 잡는 경향과 일치한다(평균 step 46.5–48.4 vs 51.4).
  - ActOnly8도 같은 방향이다(47.8 step). T019의 "critic ≈ grasp_after 추종" 진단과 맞는다.

## 4. E09 최종 정리용 표

| E06 closed-loop (N=8, n=150 paired, success_once) | 값 [95% CI] | 해석 |
|---|---|---|
| B0 (frozen N=1) | 0.547 [0.467, 0.624] | 기준 |
| Random8 | 0.573 (Δ +0.027 [−0.053, +0.107]) | sampling 분산만 |
| ActOnly8 (상태 없음) | 0.613 (Δ +0.067 [−0.013, +0.147]) | 상태 무관 chunk 규칙 |
| Critic8 (`Geo_q_buf+br`) | 0.620 (Δ +0.073 [−0.013, +0.167]) | — |
| **Critic8 − ActOnly8** | **+0.007 [−0.087, +0.100]** | **표현 조건 critic의 기여 = 0 (판정: 개선을 주장하지 않음)** |
| CriticApp8 (approach에서만) | 0.653 (Δ vs B0 +0.107 [+0.027, +0.187]; vs ActOnly8 +0.040 [−0.047, +0.127]) | 탐색적. 다중 비교 보정 후 유의하지 않음 |
| 비용 | ms/결정 batch1 266 → 827 (×3.1); critic 자체 +2 ms | sampling이 지배 |

## 5. 권고
- E06 critic 경로는 여기서 닫는다. 현재 critic으로 reranking을 늘려도(N=16/32) critic 몫이 action-only와 구별되지 않을 것으로 본다. 오프라인 conc 0.54가 천장 0.81의 14%이기 때문이다.
- approach 한정 신호를 확인하려면 두 가지를 하나의 사전 등록 비교로 만든다(≈1 GPU-h, n≥200):
  - CriticApp8 vs **ActOnlyApp8**(같은 gate)
  - CriticApp8 vs B0
- critic 재시도 조건은 T019 §6과 같다:
  - 후보 수준 다중 continuation 라벨
  - 또는 advantage 형태의 학습

## 6. 재실행
```
PY=<repcritic worktree>/.venv/bin/python; export RC_RLINF=<populated RLinf checkout>
$PY -m repcritic.e06_scorers                                    # 오프라인 재현 (CPU)
for p in 0 1; do CUDA_VISIBLE_DEVICES=$p RC_POLICY_MB=40 $PY -m repcritic.e06_rerank --part $p --nparts 2 --n_total 150 --batch 25 & done
CUDA_VISIBLE_DEVICES=1 $PY -m repcritic.e06_rerank --bench
$PY -m repcritic.e06_report --out /outputs/StochRL/t020
```
- 원자료: `/workspace/stochrl_t020/`
  - `episodes_p*.jsonl`
  - `decisions/*.npz`: 결정별 선택, 점수, gate, 특권 phase, 다양성
  - `logs/`

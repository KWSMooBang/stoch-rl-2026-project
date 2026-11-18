# T017 — E02 롤아웃 데이터 · 후보 단위 라벨 · 다중 continuation Oracle

- 날짜: 2026-11-19 / 브랜치 `exp/repcritic-e02`(exp/repcritic d3d51bd에서 분기) / 코드 `repcritic/e02_*.py`
- 데이터: `/workspace/stochrl_t017/` / 요약 파일: `/outputs/StochRL/t017/`
- GPU 0·1만 사용했다(GPU 3의 ~10–20 MiB는 T016에서 알려진 Vulkan ICD context이고, 계산은 하지 않았다). 학습은 없다. **총 ≈3.2 GPU-h**(상한 5).
- GPU 0·1은 다른 사용자의 프로세스(~2.3 GB, util ~93%)와 함께 쓰는 상태였다.

## 1. 한 일
1. **rollout 500 episode**:
   - 설정: frozen π0.5, obj_set=train, H=8, K=5, 80 step(16 결정) → 8000 결정.
   - 분할: episode 단위 70/15/15 = 350/75/75.
     - episode seed: `randperm(seed=20261119)[:500]`. 분할 seed: `RandomState(7)`.
   - 결정마다 저장한 것:
     - 입력: obs(RGB 640×480, qpos)
     - 행동: 초기 noise, raw chunk H=8(env와 정규화 공간), 실행 action K=5
     - 전이: step 보상·success·terminated·truncated, next obs
     - 라벨: G0, **라벨 전용 특권 파일**(pose, contact force, grasp, phase, on_target)
     - 기타: fp16 feature, 전체 sim snapshot
2. **후보 단위 라벨**:
   - 100 episode(시간이 남아 50에서 확장), **397 상태 × 후보 16**이다.
     - cand 0은 로그에 실제 실행한 chunk, 나머지 15개는 새 샘플이다.
   - snapshot branch로 끝까지 rollout했다.
   - 50개 상태는 cand 0–7에 continuation 4개씩 = **다중 continuation Oracle**(상태 50 × 후보 8 × continuation 4).
   - 합계 7552 branch.
3. **데이터 카드**: `dataset_card.md`(분할, 성공·phase 분포, feature 층·크기, 누수 검사, 라벨 권한).

## 2. 데이터 카드 요약 (상세: `dataset_card.md`)
| split | ep | 결정 | success_once | success_at_end | 성공 전 결정 |
|---|---|---|---|---|---|
| train | 350 | 5600 | 0.614 | 0.594 | 3501 |
| val | 75 | 1200 | 0.493 | 0.480 | 835 |
| test | 75 | 1200 | 0.547 | 0.533 | 795 |

- 전체 success_once는 0.586이다(T016 E01 0.533 [0.43, 0.63]과 같은 범위, episode 집합은 다름). 성공 293 / 실패 207 episode.
- phase(결정 8000개, 특권 라벨): approach 3462, placed 2757, post_grasp_lost 817, transport 771, grasp_contact 193.
- **누수 검사**:
  - split 간 episode_id 교집합은 0이다.
  - 한 episode의 결정과 branch 상태는 모두 같은 split에 있다.
  - 단, 물체 16종과 overlay 16종은 모든 split에 나온다(같은 분포의 다른 episode). novel 물체 일반화는 obj_set=test를 따로 모아야 한다.
- 전이 정합성: 50 ep에서 reward 합, success 플래그, exec=clip(chunk[:5])가 모두 일치했다.
- **G0**: env의 `src_on_target` 술어. instruction은 16종(물체명만 다르고, 접시 1개와 "on plate" 관계는 고정)이므로 사실상 이진 성공 critic이다.

### feature (결정당 0.50 MB, 총 4.0 GB, fp16)
hook을 붙여도 행동이 바뀌지 않는다: 같은 noise에서 **max |Δa| = 0.0**(hook 유무, context pass 유무 모두).

| 그룹 | 층 | 풀링 | 구분 |
|---|---|---|---|
| VLM prefix (gemma_2b, 18층) | 4, 9, 13, 17 | img 평균, img 4×4 grid, prompt 평균, 마지막 prompt token | **후보 무관 z(o,l)** |
| SigLIP (27층) | 8, 17, 26 | 평균, 4×4 grid | 후보 무관 z(o) |
| AE context (gemma_300m) | 4, 9, 13, 17 | x_t=0, t=1로 suffix pass 한 번 더: token 평균 + 17층 8 token | **후보 무관 z(o,l)** |
| AE candidate | 4, 9, 13, 17 × denoise 4 step | token 평균 + 마지막 step 17층 8 token | **후보 의존 z(o,l,a)** |

층 선택 이유:
- VLM은 early/mid/mid-late/last 4개로 03의 early/mid/late를 덮는다.
- SigLIP은 언어가 섞이기 전 시각 기준선이다.
- 공간 grid는 기하 probe용, txt 풀링은 언어 조건을 분리하기 위해 넣었다.
- AE는 x_t=0 pass로 후보 무관 context를 만들어 roadmap 3절의 분리 규칙을 지킨다.
- 전체 token은 결정·층당 4 MB라서 저장하지 않았다. 필요하면 snapshot과 noise로 재계산할 수 있다.

## 3. 다중 continuation Oracle (상태 50 × 후보 8 × continuation 4)
- 상태 선택: 100 episode 중 처음 50 episode에서 episode당 1개 상태를 무작위로 뽑았다(성공 전 결정).
- 기준 지표: success_once. success_at_end도 거의 같다(`analysis_main.json`).

| 추정량 | 값 [95% CI, 상태 bootstrap] | 의미 |
|---|---|---|
| E[p_i] | 0.586 | 후보 평균 성공확률 |
| E[max p̂_i] − E[p̂_i] (naive) | 0.164 [0.11, 0.22] | p̂가 4표본이라 이항 잡음으로 **위로 편향** |
| 같은 값의 순수 잡음 null(후보 간 차이 0, 순열) | 0.128 | |
| naive − null (excess) | **0.035 [0.013, 0.062]**, 순열 p = 0.0005 | 후보 간 차이는 **실재**한다 |
| split-half(2 continuation으로 고르고 나머지 2개로 평가) | **0.039 [−0.015, 0.099]** | 현실적인 noisy selector의 이득 |
| 분산성분 ICC(결과 분산 중 후보 간 비율) | **0.21** | 79%는 continuation 운이다 |
| **E[max p_i] − E[p_i]** (beta-binomial 모형, ICC 0.21) | **0.118 [0.074, 0.154]** | p_i를 **완벽히** 아는 critic의 상한 |
| 같은 양, 상태별 Gaussian 근사 | 0.103 | 확인용 |

부분집합별 (success_once):

| 부분집합 | n | E[p] | excess | split-half |
|---|---|---|---|---|
| d=0 | 16 | 0.59 | 0.089 | 0.070 |
| d>0 | 34 | 0.59 | 0.010 | 0.053 |
| 로그 episode가 실패 | 20 | 0.24 | 0.050 | 0.072 |
| approach | 26 | 0.52 | 0.058 | 0.103 |
| transport | 16 | 0.90 | −0.002 | −0.008 |
| post_grasp_lost | 7 | 0.20 | 0.042 | 0.089 |

- 상태의 42%는 32 branch 모두 같은 결과였다(all-0 18%, all-1 24%).

후보 단위 라벨 전체(397 상태, 후보 16 × continuation 1):
- Pass@1 0.553 → Pass@4 0.689 → Pass@16 0.748.
- 결과가 섞인 상태는 44%, 같은 상태 안의 성공×실패 쌍은 7383개다.

## 4. E06 판정
- **결정 하나에서 chunk를 재선택할 때의 상한**: p_i를 완벽히 알 때 ≈ **+0.12**(CI 0.07–0.15)다.
  - T016의 ctrl 분해(+0.03)보다 크다. T016 값은 continuation 1표본에 묶인 하한이었다.
- **현실적인 이득**: 후보마다 2 rollout으로 추정하는 selector 수준이면 ≈ +0.04다.
- 성공 결과 분산의 79%가 continuation에서 나온다. 그래서 후보 하나에 붙은 단일 성공 라벨은 대부분 잡음이다.
  - critic이 상한 0.12에 다가가려면, 단일 rollout 라벨에서 후보 간 ICC 0.21짜리 신호를 뽑아내야 한다.
- **판정: E06은 "조건부 Go, 보조 경로 유지"** 다.
  - 상한이 0은 아니다(+0.12/결정이고, 매 결정에 적용하면 누적될 여지가 있다). 그러나 현실적인 critic으로는 결정당 +0.03–0.05가 예상치다.
  - 이득은 초기 결정(d=0)과 approach / 실패 경로 상태에 몰려 있고, transport에서는 0이다.
  - E06은 E04 critic이 생긴 뒤 **d=0과 approach 상태에서 N=8 closed-loop**로 싸게 확인한다. 주 경로는 여전히 E03·E04 + E07이다.

## 5. E03 권고
1. **입력**: 위 4그룹 feature를 같은 split으로 비교한다. 후보 무관(VLM, SigLIP, ae_ctx)과 후보 의존(ae_cand)은 별도 ablation으로 둔다.
   - normalizer와 probe는 train(350 ep)으로 fit하고, 층·α는 val로 고르고, test는 한 번만 본다.
2. **probe 라벨**(`privileged/`):
   - 기하: `dec_gripper_carrot_rel`, `dec_carrot_plate_rel`, 각 거리
   - grasp/contact: `dec_grasping_now`, `dec_contact_any_finger`
   - phase: 5종. grasp_contact는 2.4%로 드물어 transport와 합치거나 macro-F1로 본다.
   - 미래 성공: 결정 행의 success_once / success_at_end, 또는 k-step `success_in_chunk`
   - confound 대조군: 결정 index(t)만 쓰는 probe, qpos만 쓰는 probe.
3. **성공 예측 probe는 time-to-go confound가 크다**(placed 상태가 34%). "성공 전 결정"(5131개)으로 제한한 결과를 따로 보고한다.
4. **E04로 이어지는 평가**:
   - probe 점수 외에 `branch/`의 같은 상태 후보 쌍(7383쌍; test는 120 상태)으로 ranking AUC를 잰다.
   - 기준은 Oracle 상태의 p̂_i(4 continuation)로 한다. 단일 continuation 라벨은 이론적 상한 자체가 낮다(ICC 0.21).
   - G0 critic의 학습 라벨로는 단일 branch 결과보다 rollout buffer 전체의 결정 단위 결과 + afterK 단기 라벨(거리 변화, grasp)이 덜 시끄럽다.
5. G1(기하 목표)은 특권 라벨에서 만들 수 있다. 정책 입력에는 넣지 않는다.

## 6. GPU 시간
| 항목 | GPU | GPU-h |
|---|---|---|
| smoke(4 ep, hook 검사) + 성능 프로파일 | 0, 1 | ≈0.20 |
| rollout 500 ep(250 + 250, env 50개 병렬) | 0, 1 | 0.19 + 0.15 + 로드 ≈ 0.37 |
| branch smoke(4 ep) | 1 | 0.12 |
| branch main(197 상태, 4352 branch) | 0, 1 | 0.73 + 0.70 |
| branch ext(200 상태, 3200 branch) | 0, 1 | 0.51 + 0.53 |
| 모델 로드 등 | | ≈0.1 |
| **합계** | | **≈3.2** |

- 효율: GPU PhysX(500 Hz/5 Hz = control step당 100 substep)는 env 수와 거의 무관하게 ~0.5–1 s/step이다.
  - 그래서 여러 상태의 branch를 env 64개 batch 하나에 묶었다. 상태마다 t가 달라도 env별 시계로 기록을 멈춘다.
  - 덕분에 T016 추정(100 ep ≈ 5 GPU-h)보다 싸게 100 episode와 Oracle을 함께 끝냈다.

## 7. 재실행
```
RC_POLICY_MB=8 CUDA_VISIBLE_DEVICES={0,1} .venv/bin/python -m repcritic.e02_rollout --start {0,250} --n 250 --batch 50 --noise_seed 1000 [--check_hook]
.venv/bin/python -m repcritic.e02_branch --select                      # state_select.json (seed 31)
.venv/bin/python -m repcritic.e02_branch --select --sel state_select_ext --sel_seed 32 --n_oracle 0 --exclude_from state_select
RC_POLICY_MB=8 CUDA_VISIBLE_DEVICES={0,1} .venv/bin/python -m repcritic.e02_branch --sel {state_select,state_select_ext} --shard {0,1} --nshard 2 --E 64
.venv/bin/python -m repcritic.e02_analyze ; .venv/bin/python -m repcritic.e02_oracle_eb analysis_main.json ; .venv/bin/python -m repcritic.e02_card
```

- GPU PhysX는 bitwise로 재현되지 않는다(T016: ~1%).
- branch 결과는 다음 위치에 있다:
  - `branch/branch_results_{sel}_shard{0,1}.jsonl`
  - 후보 데이터: `branch/candidates/`
  - 로그: `logs/`

# E02 rollout buffer v1 — 데이터 카드 (T017)

- 위치: `/workspace/stochrl_t017/` · 코드: 브랜치 `exp/repcritic-e02`, `repcritic/e02_*.py`
- 정책: frozen π0.5 SFT `RLinf/RLinf-Pi05-ManiSkill-25Main-SFT@eb0e2c9`. 학습 없음, `requires_grad=False`, eval.
- 환경: `PutOnPlateInScene25Main-v3`, obj_set=train (물체 16종 × 배경 overlay 16종 × 초기 pose 3136).
  - 제어 `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`, sim 500 Hz / control 5 Hz.
  - 카메라 `3rd_view_camera` RGB 640×480. state = qpos 8D.
- 행동: H=8 생성, K=5 실행, denoise 4 step. 정책의 확률성은 초기 noise뿐이다(저장함).
- episode는 항상 80 step(16 결정)이다. `ignore_terminations=True`라서 terminated는 늘 False이고 t=80에서 truncated다.

## 1. 목표 정의
- **G0 = task success**: env 술어 `src_on_target`. xy·z bbox 조건과 물체–접시 접촉력 > 0.03을 함께 만족해야 한다.
  - `success_once`: 80 step 안에 한 번이라도 성공. `success_at_end`: t=80에서 성공.
  - `goal_type="G0_task_success"`, `target_source="env success predicate"`, `goal_ref=null`.
- 언어 목표는 `put {물체명} on plate`로, train obj_set에서 **16가지 instruction**이 있다. 다만 접시는 1종이고 관계("on plate")는 고정이다.
  - 그래서 G0는 사실상 이진 성공 critic이다. 언어 goal-swap(E05)의 근거로 쓰지 않는다.
- G1(기하 목표) 진단용 참값은 `privileged/`에 있다: `dec_carrot_plate_rel`, `dec_pos_plate` 등. **라벨 전용**이다.

## 2. 분할 (episode 단위, 누수 없음)
- episode 집합: `randperm(802816, seed=20261119)[:500]`. 분할: `RandomState(7).permutation` → 350 / 75 / 75.
  - 파일: `split_manifest.json`, `split_{train,val,test}.txt`.
- **누수 검사**:
  - episode_id 교집합 train∩val = train∩test = val∩test = **0**.
  - 한 episode의 16개 결정은 모두 같은 split에 있다(검사 통과).
  - branch 상태와 후보도 소속 episode의 split을 그대로 따른다(검사 통과).
- **주의(OOD 아님)**: 물체 16종과 overlay 16종은 세 split에 모두 나온다. 초기 pose index는 split 사이에 3–7개 겹친다.
  - 즉 이 분할은 *같은 분포의 다른 episode*를 일반화하는지 보는 것이다. 새 물체로 일반화하는지는 보지 않는다.
  - novel 물체 평가가 필요하면 obj_set=test를 따로 모아야 한다.

| split | episodes | decisions | success_once | success_at_end | 성공/실패 ep | 성공 전 결정 |
|---|---|---|---|---|---|---|
| train | 350 | 5600 | 0.614 | 0.594 | 215 / 135 | 3501 |
| val | 75 | 1200 | 0.493 | 0.480 | 37 / 38 | 835 |
| test | 75 | 1200 | 0.547 | 0.533 | 41 / 34 | 795 |
| 전체 | 500 | 8000 | 0.586 | 0.568 | 293 / 207 | 5131 |

## 3. phase 분포 (결정 시점, 특권 라벨)
휴리스틱은 다음 우선순위로 정한다: placed(src_on_target) > transport(지금 grasp) > grasp_contact(손가락–물체 접촉, grasp 아님) > post_grasp_lost(전에 grasp했음) > approach.

| split | approach | grasp_contact | transport | post_grasp_lost | placed |
|---|---|---|---|---|---|
| train | 2349 | 127 | 563 | 546 | 2015 |
| val | 587 | 34 | 102 | 128 | 349 |
| test | 526 | 32 | 106 | 143 | 393 |

- grasp_contact가 적다(전체의 2.4%). E03의 phase probe에서는 grasp_contact를 transport와 합치거나, class-balanced 지표를 쓰기를 권한다.

## 4. 파일과 스키마
- `decisions_{0,250}.jsonl`: 결정 하나가 한 행이다(8000행). 02 문서 스키마를 따른다.
  - 주요 필드: `episode_id, split, decision, t_env, language_goal, goal_type, H_model=8, K_executed=5`
  - ref 필드: `obs_ref / next_obs_ref / action_candidate_ref / action_executed_ref / feature_ref / sim_labels_ref / snapshot_ref`
  - 전이 필드: `rewards[5], reward_sum, success_in_chunk, success_before, terminated, truncated, done, success_once, success_at_end`
  - `next_obs`는 `rgb[d+1]`이다(마지막은 t=80 관측).
- `episodes/epXXXXXX.npz`(정책 입력·행동·전이, 압축):
  - 관측: rgb [17,480,640,3] u8, state [17,8]
  - 행동: noise [16,8,32], chunk_env [16,8,7], chunk_model [16,8,32](정규화 공간), exec_env [16,5,7](env 범위로 clip)
  - step 단위: step_reward/success/terminated/truncated [80], step_qpos [80,8]
- reward는 RLinf 상대 shaped reward다(grasp 0.1 + 연속 grasp 0.1 + 성공 1.0 potential의 차분).
  - semi-MDP: R^(K)=Σ_{j<5} γ^j r_{t+j}, Γ=γ^5. γ는 사용자가 정한다. step 보상을 모두 저장했다.
  - 검사: 50 episode에서 reward_sum = step 합, success 플래그 일치, exec = clip(chunk[:5]) 모두 통과.
- `features/epXXXXXX.npz`: fp16, 결정당 ≈ 0.50 MB, 26개 키(`features_schema.json`). 색인은 `feature_index.csv`.
- `privileged/epXXXXXX.npz`: **라벨 전용**. 결정 [17]과 step [80].
  - pose: 물체·접시·그리퍼 pose, 상대벡터, 거리
  - grasp/contact: ever/now grasp, 연속 grasp count, 손가락–물체·물체–접시 접촉력, contact 여부
  - 그 밖: on_target, phase, carrot_id, overlay_id
- `snapshots/epXXXXXX.pt`: 결정 0..16의 전체 복원 상태다(physx state, controller target, grasp counter, episode_stats, elapsed).
  - E06/E07의 offline branch에 다시 쓸 수 있다.
- 디스크: episodes 3.6 GB, features 4.0 GB, snapshots 0.09 GB, privileged 6 MB.

## 5. feature 층 선택과 이유
hook은 관측만 하고 출력을 바꾸지 않는다. 같은 noise에서 hook이 있을 때와 없을 때, 그리고 context pass가 있을 때와 없을 때 행동 차이는 **max |Δa| = 0.0**이다(smoke run 4 ep, d=0·8: `smoke/logs/hook_check_0.json`, 본 rollout은 같은 코드 경로).

| 키 | 층 | 풀링 | 후보 의존성 | 이유 |
|---|---|---|---|---|
| `vlm{4,9,13,17}_img_mean / img_grid4 / txt_mean / txt_last` | PaliGemma gemma_2b(18층) prefix | 이미지 256 token 평균 · 4×4 공간 grid · prompt 평균 · 마지막 prompt token | **context z(o,l)** | early/mid/mid-late/last 4개다. 03의 VLM early/mid/late를 덮고, 기하(공간 grid)와 언어 조건(txt)을 분리한다. |
| `sig{8,17,26}_mean / grid4` | SigLIP So400m(27층) | 평균 · 4×4 grid | context z(o) | 언어가 섞이기 전 순수 시각 기준선이다(P0/C0 대비 "VLM이 더한 것" 분리). |
| `ae_ctx_mean [4층,1024]`, `ae_ctx_tok_last [8,1024]` | action expert gemma_300m 4·9·13·17층 | 8 action token 평균 / 17층 전체 token | **context z(o,l), 후보 독립** | x_t=0(초기 noise 분포의 평균), t=1로 suffix pass를 한 번 더 한다. (o,l)의 결정적 함수라서 roadmap 3절의 "AE 후보 독립 feature"에 해당한다. |
| `ae_cand_mean [4 step,4층,1024]`, `ae_cand_tok_last [8,1024]` | 같은 층, 실제 Euler 4 step | 같음 | **후보 의존 z(o,l,a)** | 실행한 후보의 noise 경로에 따른 hidden이다. C2 vs 후보-의존 ablation에 쓴다(03 공정성 규칙). |

- 이미지 slot: π0.5 prefix는 3 camera slot × 256 + prompt 200 = 968 token이다. 유효 slot은 0번 하나다(mask로 확인).
- 전체 token은 저장하지 않았다(층당 결정당 4 MB). 필요하면 snapshot과 `noise`로 재계산할 수 있다.
- normalizer는 저장하지 않았다. E03/E04에서 **train split으로만** fit한다.

## 6. 라벨 권한 (입력 vs 라벨)
| 구분 | 필드 | critic / probe **입력** 허용 |
|---|---|---|
| 정책 관측 | rgb, state(qpos), language_goal | 허용 |
| frozen feature | `features/*` | 허용(frozen, VLA backward 없음) |
| 행동 | chunk_env, chunk_model, exec_env, noise | 허용(후보 행동) |
| 결과 라벨 | success_*, rewards, branch 결과 | **라벨 전용** |
| 특권 sim | `privileged/*` 전부(pose, contact, grasp, phase, on_target, object/overlay id) | **라벨 전용**. G1 진단에서만 goal 입력으로 쓰며, 그때는 별도 조건으로 명시한다. |
| snapshot | `snapshots/*` | branch 생성용이다. 입력으로 쓰지 않는다. |

## 7. 후보 단위 branch 라벨 (`branch/`)
- 상태 선택(`state_select*.json`): split별로 episode를 뽑는다(train 20 / val 15 / test 15).
  - 결정은 "그 시점 전에 성공하지 않은" 결정 중에서 고른다. grid {0,3,6,9,12}에서 4개를 뽑고, 모자라면 성공 전의 다른 결정(≤12)으로 채운다.
- 후보 16개를 둔다. cand 0은 **로그에 실제로 실행한 chunk**(저장된 noise와 실행 action 그대로), cand 1–15는 새 noise로 뽑은 frozen 샘플이다.
  - 후보 chunk·noise·`ae_cand_*`는 `branch/candidates/<state>.npz`에 있다.
- branch 절차: snapshot 복원 → 후보의 첫 K=5 step 실행 → frozen 정책 continuation(env마다 새 noise) → t=80.
  - 기록 항목(`branch_results_*.jsonl`): success_once, success_at_end, 첫 성공 t, return, K step 뒤 특권 라벨(afterK: 거리, grasp, on_target, success_within_K).
- **main**(`state_select`): 50 ep, 197 상태(train 80 / val 57 / test 60). 후보마다 continuation 1개 → 3152 branch.
  - Oracle 상태 50개(episode당 1개)는 cand 0–7에 continuation 4개씩 → 총 **4352 branch**.
- **ext**(`state_select_ext`): 다른 50 ep, 200 상태(80/60/60), 후보 16 × continuation 1 = 3200 branch.
- 복원 fidelity(397 상태):
  - 복원한 관측과 로그 관측의 평균 |Δpixel|은 0.0045/255(최대 0.016)다.
  - 같은 noise로 cand 0을 다시 계산한 차이의 중앙값은 0.0025다. 최대 0.98은 그리퍼 dim이 뒤집힌 드문 경우다(픽셀 잡음에 민감). 실행은 로그 action 그대로 한다.
  - cand 0 branch와 로그 episode의 결과 일치율은 0.88이다. continuation noise가 다르므로 기대되는 수준이다.
- 주의: 라벨은 "이 chunk를 실행한 뒤 frozen 정책으로 끝까지 갔을 때"의 결과다. 후보의 고유 가치에 continuation 운이 섞여 있다(아래 Oracle의 ICC 참고).
  - ranking 평가는 Oracle 상태의 p̂_i(4 continuation)나, 같은 상태 안 pairwise 비교로 해야 한다.
- 합계: **397 상태, 7552 branch**(train 160 / val 117 / test 120 상태).
  - 후보 16개의 결과가 성공·실패로 섞인 상태는 44%다. 같은 상태 안의 성공×실패 쌍은 7383개다(E04 matched-state ranking용).

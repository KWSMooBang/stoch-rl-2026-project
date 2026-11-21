# T039 — π0.5 초기 noise 조종(noise-space steering) 계획

질문: 행동에 residual을 더하는 대신 π0.5 flow의 초기 noise를 조종하면 덜 흡수되고 더 나은가. 같은 관측·같은 GRPO·같은 seed·같은 평가에서 개입 위치만 바꿔 비교한다. GPU 0만 쓰고, 상한은 5.0 GPU-h(검증 포함)다.

## 1. 구현 방식: 우리 wrapper (RLinf DSRL 구현은 쓰지 않음)
- 브랜치 `exp/noise-steer`(`exp/residual-input` 0b2d9ca에서 분기). RLinf submodule b665e18 + T021 patch 0001은 그대로다.
- RLinf DSRL(`libero_spatial_dsrl_openpi_pi05.yaml`, `tasks/dsrl.py`)은 다음과 같이 구현되어 있다.
  - **SAC**로 학습한다. noise 정책은 64×64 이미지 encoder + state encoder 위의 Gaussian MLP다.
  - 출력은 32차원 noise 전체를 **대체**한다. 따라서 정책 출력 0은 "noise 0"이고 B0가 아니다.
  - env·평가 경로가 LIBERO용이다.
  - 이 구현을 쓰면 알고리즘(SAC vs GRPO), 관측(E2 VLM feature가 아님), 초기 상태·평가 paired 구조가 모두 달라진다. 그러면 "어디에 개입하나"를 비교할 수 없다.
- 그래서 우리 `ResidualVLAEnv`에서 noise 입력만 바꿨다(명세 기본값). 플래그는 `vla_residual.noise_steer.{mode,beta,dims}`이고 기본값은 `none`이다(기존 run 동작 불변).
- **매핑**(`mode=add`):
  - z = z_ref + β·tanh(u). model action 차원 0..6에 적용하고, H=8 horizon 전체에 같은 값을 broadcast한다. 패딩 차원 7..31은 z_ref 그대로다.
  - u는 GRPO MLP의 7차원 출력(raw Gaussian)이다. tanh로 bound하므로 |shift| ≤ β다.
  - z_ref는 env가 원래 그 결정에 쓰는 π0.5 noise다. 즉 B0의 noise이고, eval에서는 (episode id, 결정)으로 고정된다.
- **0 출력의 의미**:
  - add에서 u=0이면 z=z_ref다. 이 경우 실행 chunk는 B0 chunk 객체 그대로다(추가 forward 없음) → **B0를 bit 단위로 재현**한다.
  - actor mean 마지막 층이 0으로 초기화되어 있다(E2와 같음). 그래서 초기 결정적 정책 = B0다.
  - DSRL식 대체(`mode=replace`, 검증 전용)에서 u=0은 "noise 0"이다. 이것은 B0가 아니다(아래 §3: 0.594/0.555, B0와 episode 일치 0.74).
- **관측은 E2와 같다**: [VLM 17층 feature 2048 | qpos 8 | base chunk 5×7 | t/16] = 2092차원. base chunk는 z_ref로 만든 chunk(= u=0이면 실행될 행동)다.
- **계산**:
  - 관측용 π0.5 forward의 prefix KV cache(`build_prefix_cache` 출력)를 보관한다.
  - 정책이 u를 내면 action expert Euler 4 step만 steered noise로 다시 돈다(`VLABase.denoise`). VLM prefix는 다시 계산하지 않는다.
  - prefix cache는 Euler step 사이에 읽기만 하므로 재사용해도 안전하다.
- 실행: 관측 chunk 대신 steered chunk의 앞 K=5 step을 실행한다. action residual δ=0이다. 보상·종료·α(사용 안 함)·eval은 E2와 같다.
- β 외에는 GRPO 설정(`t022_grpo.yaml`: 27 epoch, lr 1e-4, log-std −1.5, group 8, eval 9/18/27 epoch)을 그대로 쓴다.

## 2. β = 2.0 선택 (`/outputs/StochRL/t039/val/sens_t039.json`)
held-out 32 env의 B0 rollout에서 결정 0, 3, 6, 9, 12의 상태를 모았다. 각 상태에서 shift를 바꿨을 때 실행 chunk가 얼마나 변하는지 쟀다. "sign"은 s ∈ {±1}^7 방향 8개, "explore"는 s = tanh(0.22·N(0,1))(초기 GRPO 탐색 크기)다.

| β | sign: Δpos (m/step) | sign: Δpos/α | sign: 그리퍼 부호 뒤집힘 | explore: Δpos/α | explore: 그리퍼 뒤집힘 |
|---|---|---|---|---|---|
| 0.5 | 0.0010 | 0.05 | 0.07 | 0.010 | 0.013 |
| 1.0 | 0.0025 | 0.13 | 0.12 | 0.019 | 0.027 |
| 1.5 | 0.0083 | 0.42 | 0.23 | 0.029 | 0.041 |
| **2.0** | **0.0237** | **1.18** | **0.42** | 0.039 | 0.052 |
| 3.0 | 0.0655 | 3.27 | 0.49 | 0.062 | 0.080 |

- 반응은 강하게 비선형이다. |shift|가 1σ 이하면 행동이 거의 그대로이고, 2σ를 넘으면 급격히 커진다(분포 밖).
- β=2에서 최대 조종 폭은 위치 기준으로 E2의 α(0.02 m/step)와 비슷하다(1.18α). 그리퍼는 절반 가까이 뒤집을 수 있다. 명세의 "표준정규 범위"(±2σ)에도 맞는다.
- β=3은 최대 폭이 3.3α여서 base를 너무 쉽게 망가뜨린다.

## 3. 검증 (GPU 0)
**단위 테스트** (`val/units_t039.json`, batch 16, train(GRPO group 8)·eval 각 6결정): 모두 통과.
- `denoise(z_ref)` vs π0.5 `sample_actions`: 최대 차이 **0.0**(bit 동일), 모든 결정.
- add, u=0 → 반환 chunk가 B0 chunk 객체 그대로다.
- u≠0 → 모든 env의 chunk가 바뀐다.
- obs 차원 2092. prefix cache 소유자 확인.
- replace, u=0 = noise 0 chunk(bit 동일). B0와의 평균 위치 차이는 1.9 mm/step이다.

**held-out 128 episode** (eval split, 결정적 u, 순서대로 실행, `val/eval_*_b2.0.json`):

| 조건 | once | end | B0(none_2)와 episode 일치(once) |
|---|---|---|---|
| none (B0, 첫 실행) | 0.563 | 0.539 | 0.992 (run 간 PhysX 차이) |
| **add, u=0 (초기 정책)** | **0.555** | **0.531** | **1.000** |
| none_2 (B0 반복) | 0.555 | 0.531 | — |
| replace, u=0 ("noise 0") | 0.594 | 0.555 | 0.742 |
| explore (add, u~N(0,0.22²), β=2) | 0.516 | 0.484 | 0.711 |

- 초기 정책 = B0다. T034 GPU 0 B0 0.555/0.531과 숫자까지 같다.
- 탐색 noise가 base를 덜 망가뜨린다. noise 조종의 초기 탐색에서 held-out 0.516이다. 행동 residual GRPO는 같은 log-std에서 train rollout 성공이 ≈0.15–0.2였다(T035). 그만큼 학습 신호(group 안 차이)는 작다(실행 행동 변화 0.7 mm/step).
- "noise 0"(z=0)이 B0보다 약간 높다(+0.04, 2–3 episode 차이 수준). 참고만 하고 판단에 쓰지 않는다.

## 4. smoke (`/outputs/StochRL/t039/smoke/`, GPU 0, 통과)
- `run_t039.sh 0`에 2 epoch과 eval 32 episode를 덧붙여 돌렸다. Ray 전체 경로(학습·평가·metrics.json)가 return code 0으로 끝났다.
- env worker 로그에 `noise_steer mode=add beta=2.0`이 train·eval 모두 찍혔다.
- peak VRAM 17.6 GB. 382 s로 T032 smoke(384 s)와 같다. 추가 denoise 비용은 무시할 만하다.
- train rollout(탐색 포함) success_once는 0.50(128 episode)이다. 평균 |tanh u| 0.17, 실행 행동 변화 1.0 mm/step.
- held-out 32 episode once/end 0.688/0.688(2 epoch만 학습, 판단 근거 아님).

## 5. runs · 판정
| run_id | seed | 비교 E2 (GPU 0) |
|---|---|---|
| t039-noise-s0 | 0 | t022-grpo-s0 0.742/0.625 |
| (GO일 때) t039-noise-s1 | 1 | t030-grpo-s1 0.820/0.758 |
| (GO일 때) t039-noise-s2 | 2 | t030-grpo-s2 0.695/0.602 |

- GPU 0 B0는 0.555/0.531이다.
- **keep(GO, s0, 사전 등록)**: 마지막 eval의 held-out once ≥ 0.68 **그리고** end ≥ 0.58. 그 밖이면 STOP이다.
- 판정은 L1이 s0의 `metrics.json`으로 하고 `/outputs/StochRL/t039/GO` 또는 `STOP` 파일을 만든다.
- 체인 `vla_residual/submit_t039.sh`(log `/outputs/StochRL/t039/chain.log`):
  - s0만 제출한다. s0가 끝나고 GO/STOP 파일이 생길 때까지 기다린다. GO면 s1·s2를 순서대로 제출하고, STOP이면 끝낸다.
  - GPU 0이 비어 있고(< 1000 MiB) t036–t039 run이 queued/running이 아닐 때만 제출한다.
  - GPU 0이 아니면 `run_t039.sh`가 REFUSE하고 `-rN`으로 다시 제출한다.

## 6. 예산 (상한 5.0 GPU-h)
- 사전 사용(GPU 0, harness 밖): 단위 테스트 0.02 + 민감도 0.06 + held-out 5조건 0.37 + smoke ≈0.12 = **≈0.57 GPU-h**.
- 본 run 추정 1.35 GPU-h/run. E2는 1.23–1.25였고, 여기에 action expert 재실행(4 Euler step, prefix 제외)이 더해진다.
- s0만 하면 ≈1.9, 3 seed면 ≈4.6이다.
- 가드: results.tsv의 `t039-*` gpu_h + 사전 사용 + 1.35가 5.0을 넘으면 그 run을 SKIP한다(s2부터). BUDGET-CHECK는 보내지 않는다.

## 7. 분석 메모
- train/eval episode 로그에 `ns_sabs`(평균 |tanh u|), `ns_dact`(실행 vs base chunk 위치 차이, m/step), `ns_gflip`(그리퍼 부호가 바뀐 결정 수)을 남긴다.
- 이 값으로 "조종이 실제로 행동을 바꿨는가 / 흡수됐는가"를 E2의 residual 크기와 비교할 수 있다.

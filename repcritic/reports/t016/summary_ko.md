# T016 — ManiSkill3 Frozen π0.5: 환경 구축 · E01 baseline · 후보 다양성 · Oracle-N 관문

- 날짜: 2026-11-19 / 브랜치 `exp/repcritic` / 코드 `repcritic/` / 큰 파일 `/workspace/stochrl_t016/`
- GPU: **0, 1만 사용**(GPU 3 사용 안 함. 단 아래 "GPU 3 메모" 참고). 총 **≈2.5 GPU-h**(상한 4), 학습 없음.

## 1. 설치 · 버전 · 커밋

| 항목 | 값 |
|---|---|
| RLinf | submodule `third_party/RLinf` @ `b665e1873b07501f779a5b11dc5bed5b6dcb2125` (2026-11-18) |
| 설치 명령 | `cd third_party/RLinf && bash requirements/install.sh embodied --model openpi --env maniskill_libero --no-root --venv <repo>/.venv` (UV_TORCH_BACKEND=cu126). apt 불필요(`--no-root`, 문제 없음) |
| flash-attn | Dao-AILab 공식 wheel 없음(torch 2.11) → install.sh가 RLinf 미러 wheel `2.8.3+cu12torch2.11`로 자동 대체 |
| 버전 | torch 2.11.0+cu126, ManiSkill 3.0.0b22, SAPIEN 3.0.1, rlinf-openpi 0.1.1, transformers 4.53.2(openpi fork 파일; dist metadata는 4.57.6), jax 0.5.3, gymnasium 0.29.1, numpy 1.26.4 |
| HW | RTX 4090 24GB, driver 550.120 (CUDA 12.4) |
| 체크포인트 | `RLinf/RLinf-Pi05-ManiSkill-25Main-SFT` revision **`eb0e2c90726f7f0bdf99e8c9b6656917f0eb5030`** (model.safetensors 7.47GB, 3.35B params) |
| 환경 asset | HF dataset `RLinf/maniskill_assets` @ `c23fc1880ed7861686d4f995360101eaee4d18a0` → `/workspace/stochrl_t016/maniskill_assets`, `rlinf/envs/sim/maniskill/assets`에 symlink(RLinf 문서 방식). ManiSkill 기본 asset은 `~/.maniskill`, openpi tokenizer는 `~/.cache/openpi` |

**실행 방식**: RLinf의 Ray/학습 runner는 쓰지 않고, 단일 프로세스 adapter(`repcritic/rc_common.py`)로 RLinf의 `ManiskillEnv`와 `get_model(task="eval")`(Pi0Eval)를 직접 호출한다. config는 `maniskill_ppo_openpi_pi05.yaml`의 rollout 부분과 같다(아래 표).

### Vulkan / EGL / GPU 배치
- 이 서버에는 NVIDIA Vulkan ICD json이 없다. 그래서 repo에 `repcritic/vulkan/nvidia_icd.json`(library `libGLX_nvidia.so.0`, api 1.3.277)을 두고 `VK_ICD_FILENAMES`와 `VK_DRIVER_FILES`를 **강제로** 설정한다(`rc_common.setup_env_vars`).
  - install.sh가 venv `activate`에 `/etc/vulkan/icd.d/nvidia_icd.json`(이 서버에 없음)을 export하므로 activate하지 말고 `.venv/bin/python`을 직접 쓴다.
- `setup_env_vars`는 `CUDA_VISIBLE_DEVICES`가 0이나 1이 아니면 실행을 거부한다. SAPIEN 렌더링·PhysX는 CUDA 장치를 따라간다.
  - probe 측정: env 생성 시 GPU 0만 +2.86GB, 나머지 GPU는 +1~10MiB.
  - E01 중 GPU 0은 20.5GB였다(정책+렌더).
- EGL은 쓰지 않는다(SAPIEN Vulkan 경로). EGL↔CUDA 번호 재측정은 필요 없었다.
- **GPU 3 메모**: NVIDIA Vulkan ICD는 instance 생성 때 모든 `/dev/nvidia*`를 열어 GPU마다 ~1–10MiB 드라이버 context를 만든다(GPU 3: 1→11MiB). 계산과 렌더링은 GPU 0/1에서만 돈다.
  - 사용자 공간에서는 막을 수 없었다(MESA device-select layer는 열린 뒤에 걸러서 소용없음). 완전히 막으려면 컨테이너/cgroup 수준의 장치 격리가 필요하다.
- GPU 0을 외부 프로세스(~7GB, util ~96%)가 같이 쓰는 시간대가 있었다. 그래서 첫 Oracle run이 OOM으로 중단됐다. 정책 추론을 micro-batch 8로 바꾸고(`RC_POLICY_MB=8`) 재실행했다. peak 8.7GB.

### Frozen 확인 (smoke)
- `PickCube-v1` 5 step: 정상(rgb 128², action Box(8)).
- π0.5 로드: `requires_grad=False` 전부, `eval()`.
- `llm.layers.{9,17}`, `img.encoder.layers.{9,17}`에 forward hook을 붙인 상태 / 뗀 상태로 같은 obs·noise를 넣었을 때 행동 차이 **max |Δa| = 0.0**. feature shape: VLM `[B,968,2048]`, SigLIP `[B,256,1152]`.
- `PutOnPlateInScene25Main-v3` 1 episode(2 env): 둘 다 성공.

## 2. E01 baseline (frozen SFT π0.5, obj_set=train)

설정: H(생성 horizon)=**8**, K(실행 길이)=**5**, denoise 4 step(Euler ODE, 확률성은 초기 noise뿐), max 80 step → 16 decisions.
- control `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`, sim 500Hz / control 5Hz.
- camera: `3rd_view_camera` RGB 640×480(green-screen overlay), state = qpos 8D.
- normalization: openpi `pi05_maniskill` norm_stats(체크포인트 내 `physical-intelligence/maniskill/norm_stats.json`), stateless prompt(tokenizer transform에 state를 넣지 않음).
- 초기 상태는 `episode_id`가 완전히 정한다(802,816개 중 고정 30개, `randperm(seed=12345)`). policy noise seed는 {0,1,2}다. 30 episodes × seed 3 = 90.

| seed | success_once | success_at_end | 평균 steps(첫 성공) | 평균 return |
|---|---|---|---|---|
| 0 | 0.467 | 0.467 | 55.4 | 0.56 |
| 1 | 0.667 | 0.633 | 47.4 | 0.78 |
| 2 | 0.467 | 0.433 | 55.3 | 0.54 |
| **전체(90)** | **0.533** [Wilson 0.431, 0.633] | **0.511** [0.410, 0.612] | | |

- 같은 초기 상태 기준: 3 seed 중 하나라도 성공한 초기 상태는 73%다. 공개값(RLinf 문서 SFT 40.1%)과 같은 수준이다.
- latency: batch 1 **141 ms/decision**, batch 30 1.37–1.65 s/decision. peak VRAM 12.4GB(30 env + 정책).
- action 범위: env action xyz ±0.036, rot −0.22~0.26, gripper −1.06~1.05(env의 [−1,1]로 clip). 정규화 action은 대부분 [−1.3, 1.3]이고 dim 3(roll)에 이상치 4.4가 있다.
- **재현성**: seed 0을 다시 돌리면 30개 중 **1개** episode에서 success_once가 달랐다. GPU PhysX는 run 사이에 bitwise 재현되지 않는다.
- 영상: `/workspace/stochrl_t016/videos/e01/`에 성공 15·실패 15. 추가로 `videos/oracle/`에 main trajectory 성공 10·실패 10.
- 성공률이 20–70% 안이라 01의 대응 절차(perturb / embodiment 점검)는 하지 않았다.
- 로그 컬럼: `baseline_metrics.csv`, `episode_manifest.jsonl`(01의 필수 컬럼 전부 + instruction, batch).

## 3. Snapshot 복원 검증 (Oracle 전제)

복원 대상은 다음과 같다(`rc_common.snapshot/restore_all`).
- `get_state_dict()`(actor·articulation pose/vel, qpos/qvel)
- controller target pose(**`set_state_dict`가 복원하지 않아서 직접 `set_controller_state`로 설정**)
- task 카운터(`consecutive_grasp`, `episode_stats`), `elapsed_steps`, reward 상태

| 검사 | 결과 |
|---|---|
| 복원 직후 상태 vs 저장 상태 | 최대 차이 1e-7(float 반올림). 이미지 평균 차 4e-4(가장자리 픽셀 0.0007%) |
| 같은 snapshot에서 두 번 복원 → 같은 action 열 | 같은 env index에서 **bitwise 동일**(Δq=0) |
| 복원 없이 이어간 경우 vs 복원 후 | 1e-6에서 시작해 20 step 뒤 1e-2까지 벌어짐. PhysX contact warm-start가 state에 없기 때문이고, 병렬 env끼리 생기는 차이와 같은 크기 |
| **결과 수준**(closed-loop, 모든 env 같은 noise, 6 episode × snapshot 2개 × 8 env) | env0 natural = restore1 = restore2 **12/12**. env별 일치 **95/96**(1개 뒤집힘) → 시뮬레이터 잡음 바닥 ≈1% |

결론: 저장한 상태로 정확히 돌아가고, 결과(성공 여부)도 ~99% 재현된다. 남는 잡음은 아래 Oracle의 control arm으로 직접 잰다.

## 4. 후보 다양성 (06 "후보 다양성 진단")

20 episodes × 16 decisions = **320 결정 상태**, 상태마다 같은 (obs, instruction)에서 N=32를 샘플했다.
- 거리는 후보 쌍 평균 L2를 step당(√step 수로 나눔)으로 쟀다.
- EE 변위는 xyz delta를 K/H step 동안 누적한 값이다.
- 그리퍼 패턴은 step별 open/close를 이진화해 셌다.

| phase(특권 라벨) | n | L2 env K / H (step당) | L2 정규화 K / H | EE 변위 평균 크기 K (m) | EE 변위 std/평균 K | 그리퍼 패턴 수 K / H (32개 중) | 최다 패턴 비율 H |
|---|---|---|---|---|---|---|---|
| 전체 | 320 | 0.186 / 0.219 | 0.333 / 0.361 | 0.033 | 0.62 | 2.2 / 3.3 | 0.83 |
| approach | 139 | 0.225 / 0.261 | 0.409 / 0.449 | 0.042 | 0.53 | 2.5 / 3.7 | 0.80 |
| transport | 21 | 0.215 / 0.342 | 0.503 / 0.570 | 0.081 | 0.26 | 2.3 / 4.0 | 0.69 |
| post_grasp_lost | 76 | 0.289 / 0.327 | 0.471 / 0.498 | 0.036 | 0.64 | 3.0 / 5.0 | 0.74 |
| placed(성공 후) | 84 | 0.021 / 0.020 | 0.039 / 0.038 | 0.001 | — | 1.1 / 1.1 | 1.00 |

- 성공 전 phase에서는 후보가 **서로 같지 않다**.
  - EE 예상 변위의 표준편차가 변위 크기의 50–65%다.
  - 그리퍼 열림/닫힘 timing도 2–5 패턴으로 갈린다.
- 성공 후(placed)에는 거의 같다.
- 전체 H의 거리가 첫 K보다 10–60% 크다(transport에서 가장 큼).

## 5. Oracle-N (test-only 진단)

**설계** (`repcritic/oracle_diversity.py`):
- main(baseline) trajectory의 결정 d ∈ {0,4,8,12}(20 episode × 4 = **80 상태**, 그중 main이 이미 성공하기 전 상태 59)에서 상태를 32 env로 복원한다.
- **cand arm**: env i(<16)가 후보 i의 첫 K=5 step을 실행한 뒤 frozen 정책(새 noise)으로 80 step까지 rollout한다.
- **ctrl arm**: 16 env가 같은 후보 0을 실행하고 continuation noise만 다르다.
- 결정 단위는 "끝까지 rollout"으로 정했다. K-step 진척 대리지표 대신 실제 성공을 보려는 선택이다.
  - 비용 때문에 결정 4개마다 하나만 Oracle을 했다(every-2는 episode당 610s로 상한 초과).
- pass@N은 16개 branch에서 무편향 추정 1−C(n−c,N)/C(n,N)으로 계산했다.
- CI는 episode bootstrap 95%다.
- N=1 = baseline 정책 자체(후보 평균 성공률).

| 상태 집합 | n | **cand** N=1 | 4 | 8 | **16** | ctrl N=16 | **cand16 − ctrl16** |
|---|---|---|---|---|---|---|---|
| 성공 전 결정(전체) | 59 | 0.404 | 0.567 | 0.600 | **0.610** [0.45,0.78] | 0.576 | **+0.034** [0.00, 0.08] |
| 초기 상태 d=0 | 20 | 0.647 | 0.918 | 0.945 | **0.950** | 0.950 | 0.000 |
| main 실패 episode의 결정 | 40 | 0.194 | 0.363 | 0.410 | **0.425** [0.28,0.60] | 0.375 | +0.050 [0.00, 0.12] |
| approach | 37 | 0.438 | 0.640 | 0.670 | 0.676 | 0.649 | +0.027 |
| post_grasp_lost | 17 | 0.154 | 0.279 | 0.331 | 0.353 | 0.294 | +0.059 [0.00, 0.20] |
| transport | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |

- success_at_end 기준도 비슷하다: 성공 전 결정에서 N=1 0.370 → N=16 0.593.
- OraclePass@16 − Pass@1(성공 전 결정) = **+0.207 [0.15, 0.27]**.
- episode 단위:
  - main 성공률 10/20 = 0.50(E01과 같은 수준).
  - 초기 상태 OraclePass@16 = 0.95(ctrl도 0.95).
  - main이 실패한 10개 중 9개에서 어떤 결정 시점에는 성공하는 후보가 있었다.

**해석.** OraclePass@N은 N=1보다 확실히 높다(+21pp, CI가 0을 넘음). 그러나 **같은 첫 chunk에 continuation noise만 바꾼 ctrl arm이 그 차이의 대부분을 낸다(성공 전 결정에서 cand 20.6pp 중 ctrl이 17.2pp, ≈83%).**
- 첫 chunk를 고르는 것 자체의 기여는 +3.4pp [0.0, 8.1]로 작고, 0과 구분되지 않는다.
- 이 Oracle은 단일 결정 개입 + 확률적 base 정책 continuation 1 sample이라는 가정에 종속된다(06에서 명시한 대로 절대적인 상한이 아니다).
- 따라서 "어떤 후보가 성공했나"의 대부분은 그 뒤의 운(continuation)이다.
- 후보마다 기대 성공확률 p_i가 다른 정도(재선택 critic이 활용할 신호)는 작다. 다만 failed-episode와 post_grasp_lost 상태에서 +5–6pp로 조금 크다.

## 6. 관문 판정

1. **(i) baseline 범위**: success_once 53.3% [43.1, 63.3], success_at_end 51.1%. 문서 기준 20–70% 안이고 성공·실패가 고르게 섞여 있다. 초기 상태의 73%가 3 seed 중 한 번 이상 성공하므로 정책이 "할 수는 있는데 자주 놓치는" 영역이다. → **Go**(critic 개선 여지 있음).
2. **(ii) Oracle-N vs N=1**:
   - 문자 기준으로는 OraclePass@16(0.61) > N=1(0.40)이고 차이 +0.21 [0.15, 0.27]이 확실하다 → 형식상 Go.
   - **그러나 control 분해로는 차이의 대부분이 continuation 확률성에서 나오고, 첫 chunk 선택의 순수 기여는 +0.03 [0.00, 0.08]이다.**
   - → **E06 reranking은 "약한/조건부 Go"로 판정한다.**
     - 단일 결정 K=5 chunk 재선택만으로 얻을 상한은 작다(결정당 몇 pp).
     - 매 결정에서 누적하면 커질 수 있으나 이 진단으로는 확인하지 못했다.
   - 권고:
     - E06을 주 경로로 삼지 말고 **E03·E04(표현·critic) + E07(guidance)을 주 경로**로 한다.
     - E06은 E04 critic이 생기면 N=4/8 closed-loop로 싸게 확인하는 보조 실험으로 둔다.
   - 결정에 더 확실한 근거가 필요하면 아래 "다중 continuation Oracle"을 먼저 한다(≈0.7 GPU-h).
     - 상태 50개 × 후보 8 × continuation 4로 후보별 p_i를 추정해 E[max p_i] − E[p_i]를 직접 잰다.
3. **(iii) 다음 단계(E02) 권고와 GPU-h**:
   - E02 rollout 데이터: frozen 정책으로 500 episode(train obj_set, episode_id 기준 70/15/15 분할), 결정마다 obs·raw chunk(H=8)·실행 action·phase/특권 라벨을 저장한다.
     - VLM/SigLIP hidden(위 hook, 영향 없음 확인)은 fp16으로 캐시하고, 층은 E03용으로 몇 개 고른다.
     - 병렬 30 env 기준 ~90 episode/4분 → **≈0.5–0.8 GPU-h**(feature 저장 I/O 포함).
   - 후보 단위 라벨(E04 ranking/E06 offline용, 같은 상태 후보 N개의 실제 결과): 100 episode × 결정 4 × 16 branch ≈ 190s/episode → **≈5 GPU-h**. 우선 50 episode(≈2.6 GPU-h)로 시작하길 권한다.
   - 위 다중 continuation Oracle(≈0.7 GPU-h)을 E02 후보 라벨 수집과 합쳐서 하면 비용이 거의 늘지 않는다(같은 branch 인프라).
   - 합계 권고: **E02 ≈ 3.5–4 GPU-h** (GPU 0/1).

## 7. 산출물 위치
- 요약/표 데이터: `/outputs/StochRL/t016/`
  - `e01_summary.json`, `baseline_metrics.csv`, `episode_manifest.jsonl`
  - `oracle_diversity_analysis.json`, `decisions_{0,10}.jsonl`(결정별 다양성·branch 결과)
  - `restore_test_E8.json`, `smoke*.json`
- 영상: `/workspace/stochrl_t016/videos/{e01,oracle}/`
- 로그: `/workspace/stochrl_t016/logs/`
- 재실행 명령(repo root):
  - `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m repcritic.smoke [--pickcube]`
  - `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m repcritic.e01_baseline --n_episodes 30 --seeds 0 1 2 --repro_check`
  - `RC_POLICY_MB=8 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m repcritic.oracle_diversity --test_restore --test_E 8 --n_episodes 6`
  - `RC_POLICY_MB=8 CUDA_VISIBLE_DEVICES={0,1} .venv/bin/python -m repcritic.oracle_diversity --n_episodes 10 --start {0,10} --oracle_every 4 --noise_seed {100,200}`
  - `.venv/bin/python -m repcritic.analyze_oracle "/workspace/stochrl_t016/oracle/decisions_*.jsonl" out.json`
- 중단된 run(기록용): `oracle/aborted_every2_decisions_0.jsonl`(every-2, 1 episode), `logs/restore_test2_E32_partial.log`(32 env 복원 테스트 1 episode, 결과 일치).

## GPU 시간
GPU 0 ≈ 1.1 h(smoke, E01, OOM으로 중단된 Oracle, Oracle 0–9), GPU 1 ≈ 1.4 h(복원 진단, 복원 테스트, Oracle 10–19). **합계 ≈ 2.5 GPU-h**.

# T032 — residual 입력 E0 (VLA feature 0) 계획

사람 설계서 "What Should a Residual Policy See?" Phase A의 최소 버전이다. 질문(H1): residual actor 입력의 VLA VLM feature가 correction에 실제로 기여하는가. 기존 GRPO residual(E2)에서 VLM feature만 지운 E0을 같은 조건으로 3 seed 학습해 비교한다. GPU 0만 쓰고 상한은 4.0 GPU-h(검증 포함)다.

## 1. 변경 내용
- 브랜치 `exp/residual-input` (`exp/residual-seeds` b60b623에서 분기). RLinf submodule b665e18은 그대로다.
- **플래그 하나**: `vla_residual.zero_vlm` (기본 False, `configs/env/vla_residual_putonplate.yaml`).
  - `core.py`의 `_obs()`에서만 읽는다. True면 `states`의 앞 2048차원을 상수 0으로 채운다.
  - π0.5 forward, base chunk, 내부 `self.feat`·`feat_raw`는 건드리지 않는다.
  - 관측 차원 2092, MLP 구조, qpos 8 · base chunk 35 · t/16, α, 보상, lr, env step, eval은 E2와 같다(설계서 §5.2: 빈 자리에 zero vector).
- 실행 도구(추가만):
  - `vla_residual/run_t032.sh e0 <seed>`: GPU 0 가드 후 `run.sh grpo <seed> env.train.vla_residual.zero_vlm=True env.eval.vla_residual.zero_vlm=True`.
  - `vla_residual/submit_t032.sh`: GPU 0 전용 순차 체인(T030 체인에서 T028/T029 관문만 뺌).
  - `vla_residual/validate_t032.py`: 단위 테스트. `vla_residual/analyze_t032.py`: §9.1 paired 지표(CPU).
- **config diff** (`vla_residual/t032_config_diff.txt`): E2 run의 저장된 override를 b60b623 config로, E0 override를 이 브랜치 config로 각각 compose해 비교했다. 다른 줄은 `zero_vlm: true`(train·eval)와 `log_path`, `experiment_name`뿐이다. seed는 E2와 같다.

## 2. 단위 테스트 (`/outputs/StochRL/t032/val/units_t032.json`, GPU 0, 통과)
한 프로세스에서 scene 하나를 공유하고 batch 16, seed 0으로 돌렸다. 모든 rollout에 같은 residual 수열을 넣었다.

| 검사 | train split (GRPO 설정, 16 결정) | eval split (held-out, 6 결정) |
|---|---|---|
| A. 같은 env 상태에서 플래그만 바꿔 obs 생성: E0 앞 2048 = 0, 나머지 44차원 비트 동일, 내부 상태 불변 | 17/17 통과 | 7/7 통과 |
| B. 별도 env: E0 `states` 앞 2048차원이 모두 0 | 통과 | 통과 |
| B. E2 `states` 앞 2048차원 = 내부 feature, 0 아닌 비율 | 같음, 1.00 | 같음, 1.00 |
| B. 결정 0(reset 직후) base chunk · 나머지 44차원 · episode id가 E0 = E2 (비트) | 통과 | 통과 |
| B. E0 vs E2b, E0b vs E2b: 모든 결정의 base chunk 비트 동일 비율 | 1.00, 1.00 | 1.00, 1.00 |
| B. E0 vs E0b (반복) | 1.00 | 1.00 |
| B. E0 vs E2(첫 rollout), E2 vs E2b(반복) | 0.941, 0.941 | 0.893, 0.893 |

- 실행 순서는 E2 → E0 → E2b → E0b다. E0, E2b, E0b는 모든 결정에서 base chunk와 내부 VLM feature가 비트 단위로 같다.
- scene을 만든 뒤 첫 rollout(E2)만 16 env 중 1개(train), 1→3개(eval)에서 결정 1 이후가 달랐다. 같은 조건 반복(E2 vs E2b)도 똑같이 다르므로 플래그 때문이 아니라 T029에서 본 run 간 PhysX/π0.5 비결정성이다.
- 결론: 플래그는 `states` 앞 2048차원만 바꾼다.

## 3. smoke (`/outputs/StochRL/t032/smoke/`, GPU 0, 통과)
- `run_t032.sh e0 0` + 2 epoch, eval 32 episode. Ray 전체 경로(학습·평가·metrics.json)가 return code 0으로 끝났다.
- env worker 로그에 `zero_vlm=True`가 train·eval 모두 찍혔다. peak VRAM 17.0 GB.
- held-out 32 episode once 0.781(2 epoch만 학습, 판단 근거 아님).

## 4. runs
| run_id | seed (actor.seed = env.train.seed) | 비교 대상 E2 (GPU 0) |
|---|---|---|
| t032-grpo-e0-s0 | 0 | t022-grpo-s0 (0.742/0.625) |
| t032-grpo-e0-s1 | 1 | t030-grpo-s1 (0.820/0.758) |
| t032-grpo-e0-s2 | 2 | t030-grpo-s2 (0.695/0.602) |

- eval seed 1000, held-out 128은 그대로다. config는 `t022_grpo.yaml`(27 epoch, eval 3회).
- 체인: GPU 0이 비고(< 1000 MiB) t032 run이 없을 때만 1개씩 제출한다. 다른 GPU가 잡히면 `run_t032.sh`가 REFUSE하고 `-rN`으로 다시 제출한다. 로그는 `/outputs/StochRL/t032/chain.log`.

## 5. 예산 (상한 4.0 GPU-h)
- 사전 사용: 단위 테스트 0.08 + smoke 0.11 = **0.19 GPU-h**.
- 본 run: E2 3개가 각 1.23–1.25 GPU-h였다. 3 × 1.23 ≈ 3.69 → 합계 ≈ **3.88 GPU-h**.
- 가드: results.tsv의 `t032-*` gpu_h 합 + 0.19 + 추정 1.25가 4.0을 넘으면 그 run을 SKIP한다. s2가 마지막이므로 초과 시 s2부터 빠진다. 앞 두 run 합이 2.56을 넘으면(각 1.28 초과) s2가 빠진다.

## 6. 분석 (run 종료 후, CPU)
- `python vla_residual/analyze_t032.py --out /outputs/StochRL/t032/analysis` (venv: `.claude/worktrees/repcritic/.venv`).
- 마지막 eval의 `final_eval_episodes`와 `t022/val/b0.json`을 episode id로 짝지어 once·end 각각 success, preservation, recovery, regression을 계산한다.
- 조건별 3 seed 평균, seed 간 표준편차, E0−E2 차이를 내고, CI는 seed(조건 안)와 episode id(모든 run 공유)를 함께 resample하는 2단계 bootstrap이다.
- 판정: E2−E0 success(once 또는 end)의 CI 하한 > 0이면 A2 진행 권고, 아니면 "GRPO residual 이득은 VLA feature 덕분이 아니다".
- E2 3 run만으로 미리 돌린 값: once success 0.753(seed SD 0.063), preservation 0.853, recovery 0.610 / end success 0.661(0.084), preservation 0.763, recovery 0.527.
- 주의: b0.json은 GPU 1에서 잰 값이라 GPU 0 B0보다 높을 수 있다(T029). preservation·recovery의 절대값은 두 조건에 똑같이 치우치고, E0−E2 비교는 모두 GPU 0 run끼리다.

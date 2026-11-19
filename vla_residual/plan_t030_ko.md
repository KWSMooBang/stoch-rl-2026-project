# T030 — residual RL 주요 결과 seed 재현 (계획)

사람 지시(2026-11-20): 코드는 바꾸지 않고, 기존 커밋 config에서 seed만 바꿔 돌린다. GPU 0만 쓰고, 예산 상한은 6.5 GPU-h, 순서는 T028 → T029 → T030.

## 커밋 / 브랜치
- 브랜치 `exp/residual-seeds`: `exp/vla-pbrs` 59a6043에서 분기했다. 59a6043은 47a4d76(T022) 위에 T024 PBRS를 얹은 커밋이다.
- 추가 파일(실행 도구만 추가, 학습 코드는 그대로):
  - `vla_residual/run_t030.sh <grpo|ppo|vla-b03> <seed>`: GPU 0 가드(그 밖의 GPU는 exit 3)를 건 뒤 기존 도구에 그대로 넘긴다.
    - `grpo` / `ppo` → `run.sh <algo> <seed>` (T022 도구)
    - `vla-b03` → `run_t024.sh vla 0.3 <seed>` (T024 도구)
  - `vla_residual/submit_t030.sh`: GPU 0 전용 순차 제출 체인.
  - `vla_residual/t030_config_diff.txt`: 아래 config diff 결과.
- RLinf submodule은 b665e18이고 기존과 같다. 패치 0001은 `run.sh`가 실행 때 적용하며, 이것도 기존과 같다.

## 코드 경로가 기존과 같은지 확인 (같은 seed로 다시 돌리지 않음)
1. **config diff.** 기존 run의 `t022_overrides.txt`(t022-grpo-s0, t022-ppo-s1, t024-vla-b03-s0)와 T030 run이 넘길 override로 Hydra config 전체를 compose해 비교했다.
   - 다른 줄은 `env.train.seed`, `actor.seed`, `runner.logger.{log_path,experiment_name}` 네 개뿐이다(`t030_config_diff.txt`).
   - `env.eval.seed=1000`, eval `total_num_envs 32 × rollout_epoch 4`(held-out 128)는 그대로다.
   - `configs/` 디렉터리는 47a4d76과 59a6043에서 byte 단위로 같다.
2. **GRPO / PPO β=0을 59a6043 코드로 돌릴 때.** 47a4d76 대비 `core.py`에서 바뀐 것은 세 가지다.
   - (a) `last_feat_raw` / `feat_raw` 저장: 값을 보관만 하고 읽지 않는다.
   - (b) PBRS 블록: `vla_residual.pbrs`가 없으면 `self.pbrs=None`이라 실행되지 않는다.
   - (c) `chunk_step`에서 `_make_obs(wobs)` 호출이 `infos`·로그 작성 앞으로 옮겨졌다. `infos`와 로그는 `success_once`, `returns`, `dec`만 읽고, `_make_obs`는 이 값들을 바꾸지 않는다. 학습 noise 생성기(`self.gen`) 호출 순서와 횟수도 같다.
   - 따라서 pbrs가 없으면 env 동작은 47a4d76과 같다.
   - `make_metrics.py`는 사후 지표(aulc, shaping)만 추가했고 학습에는 영향이 없다.

## seed
| 조건 | 기존 | T030 run_id | 도구 → config |
|---|---|---|---|
| GRPO sparse | t022-grpo-s0 | t030-grpo-s1, t030-grpo-s2 | run.sh grpo → t022_grpo.yaml |
| PPO + VLA Φ β=0.3 | t024-vla-b03-s0 | t030-vla-b03-s1, t030-vla-b03-s2 | run_t024.sh vla 0.3 → t022_ppo.yaml + pbrs(γ 0.96, Φ_T 1) |
| PPO β=0 | t022-ppo-s0, s1 | t030-ppo-s2 | run.sh ppo → t022_ppo.yaml |

## 체인 순서 (GPU 0 전용, 한 번에 1개)
1. **시작 조건.** 아래 두 조건이 모두 맞아야 하고, 10분 뒤 다시 확인해도 맞아야 시작한다.
   - t028-* / t029-*가 harness에서 queued/running이 아니다.
   - REFUSE를 빼고 t029-* run이 2개 이상 끝났다. 또는 해제 파일 `/outputs/StochRL/t030/GO`가 있다. T029가 SKIP되거나 줄어들면 lead가 `touch`로 해제한다.
2. **제출 순서.** `t030-grpo-s1` → `t030-grpo-s2` → `t030-vla-b03-s1` → `t030-vla-b03-s2` → `t030-ppo-s2`.
   - 예산 가드는 `t030-*`의 results.tsv gpu_h 합 + 사전 사용분(smoke) + 추정 1.3이 6.5를 넘으면 SKIP한다.
   - ppo-s2를 마지막에 두었으므로 상한을 넘으면 ppo-s2부터 SKIP된다.
3. **GPU 확인.** GPU 0이 비어 있을 때(< 1000 MiB)만 제출한다. 다른 GPU가 잡히면 `run_t030.sh`가 REFUSE(exit 3)하고, 체인이 `<run_id>-rN`으로 다시 제출한다.
4. **시간 상한.** RUN_HOURS = 1.3 × 1.4 + 0.2 = 2.02 h.

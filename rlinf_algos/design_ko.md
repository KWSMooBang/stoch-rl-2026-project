# T021 — RLinf 알고리즘(PPO · GRPO · SAC · CrossQ · DAgger) 구현 요약과 smoke

- 날짜: 2026-11-19 / 브랜치 `exp/rlinf-algos`(`exp/repcritic` d3d51bd에서 분기) / 코드 `rlinf_algos/`
- RLinf `third_party/RLinf` @ `b665e18`. venv는 T016 것(`.claude/worktrees/repcritic/.venv`, torch 2.11, ManiSkill 3.0.0b22, Ray 2.60)을 그대로 썼다.
- **범위 변경(lead 지시, T022로 이관):** 5개 알고리즘의 smoke와 이 문서(RLinf 구현 요약)까지만 했다. **본 run은 제출하지 않았다.** 사용한 GPU는 0과 1뿐이며 총 ≈0.6 GPU-h다.
- 과제: ManiSkill `PickCube-v1`(state obs 42D, `panda-ee-dpos` action 4D, 50 step, sim 100Hz / control 20Hz), MLP 정책.

## 0. 결론 먼저

| 알고리즘 | smoke | RLinf 그대로 돌아가나 | 비고 |
|---|---|---|---|
| PPO (`maniskill_ppo_mlp`) | ✅ 40 epoch(2.05M step) 250 s. train return 2.7→18.1, eval success_once 0→3.1% | 예 | 기준 |
| GRPO (MLP, 신규 config) | ✅ 40 epoch 253 s. return 2.6→8.9, eval 0.8% | **config 2줄을 추가하면 돈다. 다만 stock PickCube에서는 group이 같은 초기 상태를 공유하지 않는다**(§2.2) | 고친 task(`PickCubeGrouped-v1`)를 만들고 단독으로 검증했다. 학습에는 아직 쓰지 않았다 |
| SAC (`maniskill_sac_mlp`) | ✅ 300 epoch(19.2k step) 448 s. Q 0→0.72, α 0.01→0.0018, eval 0% | 예 | 19k step으로는 성공이 아직 없다 |
| CrossQ (`maniskill_crossq_mlp`) | ✅ 300 epoch 379 s. Q 0.13→0.59, eval 0% | 예 | target net을 만들고 갱신은 하지만 쓰지 않는다(§2.4) |
| DAgger (`maniskill_dagger_mlp`) | ✅(패치 후) 20 epoch 104 s. BC loss 0.287→0.023 | **아니오.** MLP의 `sft_forward`가 reduction 없는 tensor를 반환해 backward에서 죽는다. 1줄 패치 + expert 로드 override 1개가 필요하다 | expert는 smoke PPO-40 ckpt(성공률 3%)다. pipeline 확인 용도뿐이다 |

**이 저장소에서 RLinf를 쓸 때 지켜야 할 것(T022 필독):**
1. **GPU 고정.** RLinf는 `cluster.component_placement`의 번호를 **물리 GPU 번호로** 쓴다. 바깥의 `CUDA_VISIBLE_DEVICES`는 무시한다.
   - 측정 결과: `CUDA_VISIBLE_DEVICES=1` + placement `0`이면 부하가 **물리 GPU 0**에 걸렸다.
   - 그래서 wrapper가 placement를 `$CUDA_VISIBLE_DEVICES`(0 또는 1)로 덮어쓴다(`~cluster.component_placement '+cluster.component_placement={actor:G,env:G,rollout:G}'`). 덮어쓴 뒤 nvidia-smi로 GPU 1에 걸리는 것을 확인했다.
   - GPU 3에는 Vulkan ICD가 driver context(+13 MiB)만 만들었다. T016과 같은 현상이고 계산은 걸리지 않는다.
2. **Ray 격리.** RLinf는 `ray.init(address="auto")`를 쓴다. Ray는 이때 **머신에서 보이는 아무 `gcs_server`에나 붙는다**(legacy fallback).
   - 측정 결과: GRPO run 하나가 동시에 돌던 CrossQ run의 Ray에 붙었고, CrossQ가 끝나자 "Owner's node has crashed"로 같이 죽었다.
   - 그래서 wrapper가 run마다 전용 head(`ray start --head --block --port=<free>`)를 띄우고 `RAY_ADDRESS`로 고정한다.
   - `ray start`는 token auth를 기본으로 켜므로 driver도 `RAY_AUTH_MODE=token`으로 돌린다(token은 `~/.ray/auth_token`).
   - 동시 run 2개(GPU 0/1)가 각자의 head를 쓰고, 한쪽이 끝나도 다른 쪽이 살아남는 것을 확인했다.
   - **`ray stop`은 절대 쓰지 말 것.** 머신의 모든 Ray, 즉 다른 세션의 것까지 죽인다.

## 1. 공통 구조 (`examples/embodiment/train_embodied_agent.py`)

- **Runner:** 다섯 알고리즘 모두 `EmbodiedRunner`(`train_embodied_agent.py:187`)다. 한 global step = rollout(env↔rollout worker) → advantage/return 계산 → actor 학습 → weight sync → (주기마다) eval.
- **Actor worker**는 `algorithm.loss_type`으로 고른다(`train_embodied_agent.py:51-110`).
  - `embodied_sac` → `EmbodiedSACFSDPPolicy`(SAC와 CrossQ 공용)
  - `embodied_dagger` → `EmbodiedDAGGERFSDPPolicy`
  - 그 외(`actor_critic`, `actor`) → `EmbodiedFSDPActor`(PPO/GRPO)
- **Rollout worker**는 `MultiStepRolloutWorker`(huggingface backend), **Env worker**는 `EnvWorker`이고 env 구현은 `rlinf/envs/sim/maniskill/maniskill_env.py`다.
- **Model:** `MLPPolicy`(`rlinf/models/embodiment/mlp_policy/mlp_policy.py`). 3×256 MLP backbone + actor mean + 선택적 head다.
  - `add_value_head`(PPO), `add_q_head`(SAC/CrossQ, Q 2개) 중 하나만 고를 수 있다(`:53`).
  - 행동 분포:
    - PPO/GRPO: state와 무관한 log-std(초기 −0.5), tanh 없음, raw Gaussian log-prob(`:202-236`)
    - SAC/CrossQ: tanh-squashed Gaussian, state 의존 log-std [−5, 2](`:179-200`)
- **로그:** tensorboard `<log_path>/tensorboard/`
  - eval: `eval/success_once`, `eval/success_at_end`, `eval/return`, `eval/episode_len`
  - train: `env/*`, `train/*`, `time/*`
  - x축은 global step(=epoch)이다. env step 카운터는 없어서 `total_num_envs × max_steps_per_rollout_epoch × rollout_epoch`로 계산한다.
- **체크포인트:** `runner.save_interval > 0`일 때만 저장된다(−1이면 마지막에도 저장하지 않는다). eval을 켜면 `val_check_interval`의 배수여야 한다(`runner_utils.py:31-62`).
  - 경로: `<log_path>/<experiment_name>/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt`(전체 state dict)

## 2. 알고리즘별 구현

### 2.1 PPO — `adv_type: gae`, `loss_type: actor_critic`
- **Loss**(`rlinf/algorithms/losses.py:300-322, 424-475, 515-543`):
  - policy: `max(−A·r, −A·clip(r, 0.8, 1.2))`, r = exp(logp − logp_old)
  - value: `max(huber(R−V), huber(R−V_clip))`(huber δ=10, value_clip 1.0)
  - 총 loss = 둘의 합
  - dual-clip(`clip_ratio_c`)은 GR00T/PI0_FAST/LingBot에만 넘어가서 MLP에는 적용되지 않는다(`embodied_fsdp_actor_worker.py:828-843`).
  - entropy는 `entropy_bonus>0`일 때만 들어간다(기본 0). **`kl_beta`는 embodied actor에서 쓰이지 않는다.**
- **Critic:** MLP `ValueHead`(3×256)가 actor와 같은 모델 안에 있고 lr은 `value_lr`로 따로 준다. config 검증이 `add_value_head`를 요구한다(`config.py:1497-1506`).
- **Advantage:** GAE(γ=0.8, λ=0.9; `advantages.py:24-86`) 후 정규화한다.
- **Buffer:** on-policy다. 한 step에 1024 env × 50 step = 51,200 transition을 모으고, `update_epoch=8` × (51,200/6,400 = 8 minibatch) = **64 optimizer step/global step**이다.
- **smoke:** median 3.8 s/epoch(eval epoch 7 s), GPU 부하 ≈3.5 GB(1024 env + MLP).

### 2.2 GRPO — `adv_type: grpo`, `loss_type: actor` (MLP config 신규)
- **Loss:** PPO와 같은 clipped actor loss이고 **critic이 없다**(`losses.py:627-654`). value head는 `add_value_head: False`로 끈다.
- **Advantage**(`algorithms/utils.py:182-200`, `advantages.py:209-241`):
  - trajectory 점수 = 첫 done까지의 reward 합
  - 연속된 `algorithm.group_size`개 env를 한 group으로 묶어 `(r − mean)/(std + 1e-6)`를 구하고, 모든 step × `loss_mask`에 broadcast한다.
- **RLinf에서 group은 이렇게 만든다:** env가 `env.train.group_size`마다 `reset_state_ids`를 하나 뽑아 repeat_interleave하고, `use_fixed_reset_state_ids=True`이면 `options["episode_id"]`로 넘긴다(`maniskill_env.py:57-67, 121-135, 284-285, 383-384`).
- **우리 config:** `rlinf_algos/configs/maniskill_grpo_mlp.yaml`. `maniskill_ppo_mlp.yaml`에서 바꾼 것만 적는다(파일에서는 `# T021`로 표시).
  - `algorithm.group_size: 8`, `adv_type: grpo`, `loss_type: actor`
  - `actor.model.add_value_head: False`
  - `env.train.use_fixed_reset_state_ids: True`, `env.train.group_size: ${algorithm.group_size}`
  - **`env.train.auto_reset: False`**(+ `ignore_terminations: False`를 명시). `loss_mask`는 `auto_reset=False`이고 `ignore_terminations=False`일 때만 만들어진다(`embodied_fsdp_actor_worker.py:241-258`).
    - 이 값이 없으면 `compute_grpo_advantages`의 `torch.zeros_like(loss_mask=None)`에서 TypeError가 난다. smoke 1차에서 실제로 났다.
    - LIBERO GRPO env 기본값도 같다.
  - 나머지(lr 3e-4, clip 0.2, update_epoch 8, env 1024 = 128 group, γ 등)는 PPO 기본값과 같다. LIBERO GRPO MLP의 `reward_coef=5`·`filter_rewards`·`clip_ratio_high=0.28`·token-level logprob은 가져오지 않았다(MLP/ManiSkill에는 의미가 약하다).
- **⚠ Stock PickCube의 문제:** `PickCube-v1._initialize_episode`는 cube/goal 위치를 **전역 `torch.rand`**로 뽑고 `episode_id`를 읽지 않는다(ManiSkill `pick_cube.py`). per-env seed 리스트도 소용없다.
  - `episode_id`를 읽는 것은 RLinf의 자체 task(`tasks/put_on_in_scene_multi.py` 등)뿐이다.
  - 그래서 stock PickCube에서 "group" 8개는 **서로 다른 초기 상태**이고, advantage는 사실상 "batch 정규화한 REINFORCE"가 된다.
  - smoke는 이 상태로 돌렸다. 학습은 진행되지만(return 2.6→8.9) 같은 조건의 PPO(2.7→18.1)보다 느리다.
- **가장 가까운 올바른 방법(제안, 구현과 단독 검증 완료):** `rlinf_algos/pickcube_grouped.py`의 `PickCubeGrouped-v1`
  - PickCube와 같은 분포(cube xy, yaw, goal xyz)를 쓰되 **`episode_id`로 seed한 per-env generator**로 뽑는다.
  - 검증(GPU 1, 16 env, G=4): group 안 cube/goal 최대 차이 **4.8e-7**, group 사이 0.10 m, partial reset 뒤에도 같은 상태로 복원된다(차이 0).
  - 학습에 쓰려면 RLinf가 env worker 안에서 `rlinf/envs/sim/maniskill/tasks/*`를 자동 import하므로 이 파일을 거기에 복사(또는 symlink)하고 `env.train.init_params.id=PickCubeGrouped-v1`로 바꾼다. T021에서는 범위가 줄어 이 학습을 돌리지 않았다.
  - 다른 대안은 `group_size`를 1로 두고 RLOO/batch baseline으로 보는 것이다. 이는 GRPO가 아니다.

### 2.3 SAC — `adv_type/loss_type: embodied_sac`
- **Critic loss**(`fsdp_sac_policy_worker.py:355-482`):
  - target y = Σ_chunk r + γ·(agg_q(Q_targ(s′, a′)) − α·logπ(a′|s′)), a′ ~ π(s′)
  - `agg_q: min`(Q 2개), `bootstrap_type: always`는 termination을 무시하고 항상 bootstrap한다.
  - loss = MSE(Q1, y) + MSE(Q2, y)
- **Actor loss**(`:484-535`): mean(α·logπ(a|s) − agg_q(Q(s, a))), a = π(s)(reparam).
- **α loss**(`:537-557`): −α·(mean logπ + target_entropy)이고 target_entropy = −4, α는 softplus 파라미터화(초기 0.01)다.
- **Target net:** `tau=0.01` soft update, `target_update_freq: 1`(`:273-320, 664-669`). optimizer는 actor, critic(q_head), α 셋으로 나뉜다(`:123-136`).
- **Buffer:** `TrajectoryReplayBuffer`(`rlinf/data/storage/replay/buffer.py`).
  - "trajectory" 하나 = env rank 하나가 한 step에 낸 [T, B] chunk다(여기서는 2 step × 32 env).
  - `sample_window_size`(=10,000 trajectory)는 최근 N개에서 transition을 균등하게 뽑는다(`:584-625`). `auto_save` off이면 `cache_size`도 window로 고정된다(`:260-268`).
  - `min_buffer_size: 0`이라 첫 step부터 학습한다.
- **Update 비율:** 한 global step에 64 transition을 모으고 **gradient 32번**(각 1,024 샘플; `update_epoch`)을 돈다(`:744-748`). env step당 0.5 update다.
- **smoke:** median 1.19 s/epoch(eval epoch 4.8 s), GPU ≈11.6 GB(대부분 torch.compile과 Q batch 1024 × 32의 캐시로 보인다; whole-GPU 측정).
  - 300 epoch(19.2k env step)에서 Q 0→0.72, α 0.01→0.0018, train return 2.3→9.6이지만 **eval 성공은 아직 0**이다.

### 2.4 CrossQ — SAC worker + `q_head_type: crossq`
- **설정:** `q_head_type`을 `algorithm`(worker 분기)과 `actor.model`(구조) **두 곳 모두**에 넣어야 한다. 기본 config는 그렇게 되어 있다.
- **BatchNorm:** Q head(`MultiCrossQHead`)는 모든 Linear 앞에 **BatchRenorm**(momentum 0.999, r_max 3, d_max 5; `batch_renorm.py:25`, `q_head.py:167-250`)을 둔다.
- **Joint forward:** critic update에서 `[s; s′]`, `[a; π(s′)]`를 **batch 축으로 붙여 한 번에** forward하고(BRN train mode라 두 분포가 같은 batch 통계를 공유) 결과를 나눈다(`q_head.py:304-326`).
  - next-Q는 detach하고, a′는 online policy에서 no_grad로 뽑는다.
- **Target 처리:** CrossQ 경로에서는 target network를 **쓰지 않는다**(`fsdp_sac_policy_worker.py:392, 439` `if not use_crossq` 분기).
  - 다만 target model은 여전히 만들어지고 `tau`로 soft update된다(낭비지만 결과에는 영향 없음). 즉 `tau`는 사실상 무시된다.
- **Actor 쪽 Q forward:** next obs 없이 호출되어 BRN이 running stats(eval 경로)를 쓴다(`:512-521`).
- **`critic_actor_ratio: 2`:** critic은 매 update, actor와 α는 `update_step % 2 == 0`일 때만 갱신한다(`:607`).
- **나머지:** buffer, update_epoch 32, γ, α 설정은 SAC와 같다.
- **smoke:** median 1.11 s/epoch, GPU ≈4.2 GB. 300 epoch에서 Q 0.13→0.59, train return 2.4→7.9, eval 0%.

### 2.5 DAgger — `loss_type: embodied_dagger`
- **Expert 로드:** rollout worker가 `runner.expert_ckpt_path`를 `torch.load`하고 `actor.model` 설정 위에 `rollout.expert_model`을 덮은 구조로 **strict** `load_state_dict`를 한다(`huggingface_worker.py:191-201`, `algorithms/expert.py:21-47`). 형식은 순수 state-dict `.pt`(위 `full_weights.pt`)다.
  - PPO ckpt에는 `value_head.*`가 있고 DAgger config는 `add_value_head: False`다. 그래서 **`+rollout.expert_model.add_value_head=True`**를 줘야 strict 로드가 통과한다.
- **β 스케줄:** `init_beta 1.0`에서 global step마다 β ← max(0.05, β·0.98)로 줄이고, 각 rollout epoch 전에 적용한다(첫 epoch부터 0.98; `huggingface_worker.py:503-516, 942`).
  - 동전은 **step마다 env batch 전체에 하나**다(`:566`). env별 혼합이 아니다.
  - DAgger 모드에서는 student와 expert 모두 `mode="eval"`(평균 행동, 결정적)로 행동한다(`:528-551`).
- **Relabel(`only_save_expert: False`, classic DAgger):** student가 행동한 step에서도 **student가 방문한 obs에 expert를 질의**해, 그 평균 행동으로 `action`/`model_action`을 덮고 flag를 단다(`:586-608, 636-647`). actor는 flag된 step만 학습에 쓴다(`fsdp_dagger_policy_worker.py:299-307`).
  - `only_save_expert: True`이면 expert가 실제로 행동한 step만 저장한다(HG-DAgger).
- **Loss:** `MSE(actor_mean(s), a_expert)`(pre-tanh mean; `mlp_policy.py:162-177`)이고 critic은 없다. buffer는 SAC와 같은 `TrajectoryReplayBuffer`(cache 20,000 trajectory, `min_buffer_size 8`)다.
- **⚠ 버그(RLinf b665e18):** `MLPPolicy.sft_forward`가 `F.mse_loss(..., reduction="none")`을 그대로 반환하고, DAgger worker가 거기에 `.backward()`와 `.item()`을 부른다(`fsdp_dagger_policy_worker.py:499-503`).
  - smoke에서 실제로 `RuntimeError: grad can be implicitly created only for scalar outputs`로 죽었다.
  - 패치 `rlinf_algos/patches/0001-mlp-dagger-mse-mean.patch`(`.mean()` 추가)를 `run_algo.sh`가 idempotent하게 적용하고, run dir에 `rlinf_patch.diff`를 남긴다.
- **smoke**(expert = smoke PPO-40, 성공률 3%):
  - 20 epoch 104 s, median 3.5 s/epoch, GPU ≈4.7 GB
  - BC loss 0.287→0.023, train return ≈20.4로 expert 수준이다.
  - 학습 신호가 맞게 흐르는 것만 확인했고, 성능 비교로 의미 있는 값은 아니다.

## 3. 실행 도구 (`rlinf_algos/`)

- `run_algo.sh <ppo|grpo|sac|crossq|dagger> <seed> [hydra overrides]`
  - GPU guard: `CUDA_VISIBLE_DEVICES`가 0이나 1이 아니면 exit 3으로 거부한다(T008 방식). harness가 다른 GPU를 잡으면 실행되지 않는다.
  - placement를 물리 GPU로 고정하고, 전용 Ray head를 띄우고, 패치를 적용한다.
  - 공통 override:
    - `actor.seed=env.train.seed=<seed>`
    - **`env.eval.seed=1000`, `env.eval.total_num_envs=128` 고정**(공정성: 모든 알고리즘이 같은 128개 eval 초기 상태, `use_fixed_reset_state_ids=True`)
    - video off
  - whole-GPU memory.used를 5 s마다 기록한다.
- `make_metrics.py`: tensorboard → `metrics.json`
  - eval/train 성공률 곡선(epoch, env step, wall-clock), 최종/최고 성공률, 학습 시간, peak VRAM, config override 목록
  - peak VRAM은 whole-GPU 값이라, 같은 GPU를 쓰는 외부 프로세스(GPU 0/1에 ~2.4 GB짜리가 간헐적으로 있다)가 섞일 수 있다.
- `configs/maniskill_grpo_mlp.yaml`, `pickcube_grouped.py`, `patches/`.
- harness로 낼 때: `/harness/launch.sh t0xx-<algo>-s<seed> 1 rlinf_algos/run_algo.sh <algo> <seed> ...`. harness는 메모리가 빈 GPU 중 번호가 낮은 것을 잡으므로 GPU 2·3이 비어 있으면 거부(exit 3)될 수 있다. T008처럼 하나씩 내는 chain이 필요하다.

## 4. smoke 결과와 예산

smoke 상세는 `/outputs/StochRL/t021/smoke/*.metrics.json`에 있다(원본 로그는 `/workspace/stochrl_t021/scratch/`).

| run | GPU | epoch | env step | 시간 | s/epoch(median) | GPU 부하 | eval success_once(마지막) |
|---|---|---|---|---|---|---|---|
| PPO-40 (ckpt 20/40 저장) | 0* | 40 | 2.05M | 250 s | 3.8 | ≈3.5 GB | 0.031 |
| GRPO (stock PickCube) | 1 | 40 | 2.05M | 253 s | ≈3.6 | ≈3.4 GB | 0.008 |
| SAC | 0 | 300 | 19.2k | 448 s | 1.19 | ≈11.6 GB | 0.0 |
| CrossQ | 0 | 300 | 19.2k | 379 s | 1.11 | ≈4.2 GB | 0.0 |
| DAgger (패치) | 1 | 20 | 1.02M | 104 s | 3.5 | ≈4.7 GB | 0.016 |
| PPO-iso (Ray 격리 확인) | 0 | 4 | 0.2M | 54 s | — | — | 0.0 |

\* 이 run은 placement 버그를 고치기 전에 `CUDA_VISIBLE_DEVICES=1`로 냈다. 실제로는 물리 GPU 0에서 돌았다(허용 범위 안). VRAM 칸은 수정 후 probe 값이다.

실패/진단 run(기록용): `smoke-grpo-fail1`(loss_mask None), `smoke-grpo-fail2-rayjoin`(다른 run의 Ray에 붙었다가 같이 죽음), `smoke-dagger-stock`(MSE 버그), `probe-gpu`(placement 측정).

**본 run 예산(T022 참고용, 실행하지 않음):**
- PPO: epoch당 51.2k step, ≈3.8 s(eval 포함 평균 ≈4.3 s).
  - smoke 40 epoch에서 eval 3%였고 return이 아직 오르는 중이었다. 수렴(대략 성공률 >90%)까지 몇 epoch인지는 이 smoke로 정하지 못했다.
  - 300–500 epoch(15–26M step)를 잡으면 **≈20–36 min/run**이다.
- GRPO: PPO와 같은 비용(≈4 s/epoch)이다. 같은 epoch 수면 ≈20–36 min/run.
- SAC/CrossQ: ≈1.1–1.2 s/epoch(64 env step, 32 update). 기본 `max_epochs 8000` = 512k step이면 **≈2.5–2.7 h/run**이다.
  - 6 GPU-h 상한에서 seed 2개를 돌리려면 epoch를 2,000–3,000(128–192k step, ≈40–60 min)으로 줄여야 한다.
  - off-policy는 env step 효율이 높다. 그래서 x축을 env step과 wall-clock 둘 다로 보면 이 차이가 드러난다.
- DAgger: ≈3.5 s/epoch. 기본 200 epoch ≈12 min/run이다. 좋은 PPO expert ckpt(`save_interval`을 켠 본 PPO run)가 먼저 있어야 한다.
- 합계(5 알고리즘 × seed 2, PPO/GRPO 400 epoch, SAC/CrossQ 2,500 epoch, DAgger 200 epoch): 2×(0.5+0.5+0.85+0.75+0.2) ≈ **5.6 GPU-h**. eval과 시작 오버헤드(run당 ≈1 min)를 더하면 6 GPU-h 상한에 아슬아슬하다.

## 5. GPU 사용

GPU 0과 1만 썼다(smoke·probe·검증 합 ≈1,850 s의 run 시간 + 시작/실패 오버헤드). **합계 ≈0.6 GPU-h.** GPU 3에서는 계산이 돌지 않았다(Vulkan driver context +13 MiB만).

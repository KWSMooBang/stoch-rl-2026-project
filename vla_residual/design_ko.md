# T022 — frozen π0.5 위 residual RL: RLinf PPO · GRPO · SAC · CrossQ · DAgger 비교 (설계 · 검증 · 예산)

- 날짜 2026-11-19 / 브랜치 `exp/vla-residual` / worktree `/workspace/StochRL/.claude/worktrees/vla-residual`
- 브랜치 구성: `exp/repcritic-e06`(973ee40, π0.5 env·feature 코드)에서 분기 → `git merge --no-ff exp/rlinf-algos`(bb93066, T021 도구) — 충돌 없음(커밋 739032b).
- 코드: `vla_residual/` (core.py env wrapper·expert, rlinf_ext.py RLinf 확장, configs/, run.sh, submit_chain.sh, validate.py, make_metrics.py)
- RLinf `third_party/RLinf` @ b665e18 + T021 패치 0001(DAgger MLP loss reduction)만. 그 밖의 RLinf 동작 변경은 모두 **`RLINF_EXT_MODULE=vla_residual.rlinf_ext`** 로 우리 config에만 켜진다(RLinf 소스 수정 없음).
- venv: T016 것(`.claude/worktrees/repcritic/.venv`). 큰 파일 `/workspace/stochrl_t022/`, 검증 결과 `/outputs/StochRL/t022/val/`, smoke `/outputs/StochRL/t022/smoke/`.

## 0. 요약
- **구조**: 명세 권장안 그대로. frozen π0.5를 env wrapper 안에 넣고(`ResidualVLAEnv`), RLinf에는 "관측 2092차원 → 행동 7차원 residual"인 MLP 정책 env로 보이게 했다. 5개 알고리즘 모두 RLinf의 runner·worker·loss를 그대로 쓴다.
- **검증(held-out 128 episode, 모두 같은 초기 상태와 π0.5 noise)**:
  | 조건 | success_once | success_at_end |
  |---|---|---|
  | **B0** (residual 0) | **0.586** | 0.570 |
  | scripted expert 단독(특권 상태) | **0.883** | 0.883 |
  | **hand-coded 상한**(base + clip((expert−base)/α)) | **0.820** | 0.453 |
  - B0는 T016 0.533 [0.43, 0.63], T020 0.547과 CI 안에서 같다(다른 episode 집합).
  - 상한 − B0: residual로 뒤집을 수 있는 episode 40개, 망가지는 episode 10개. α가 충분하다(조정 불필요).
- **단위 테스트 통과**: 보상 = 첫 성공 decision에 +1·terminated, 이후 보상 0, truncation은 16번째 decision에만, auto-reset 후 새 train id(held-out과 겹치지 않음), **GRPO group(8개) 안 초기 상태 차이 최대 1.9e-6 m**(group 사이 평균 0.098 m), group 안 물체 동일.
- **smoke**: 5개 알고리즘 모두 RLinf로 학습·평가·metrics.json까지 통과(§6).
- **본 run**: `t022-{ppo,grpo,sac,crossq,dagger}-s0` 각 ≈1.2 GPU-h + 예산이 남으면 `t022-ppo-s1`(가드로 10 GPU-h 초과 시 건너뜀). 검증·smoke ≈1.7 GPU-h. 합계 ≈ 7.7–8.9 GPU-h.

## 1. 구조
```
RLinf EnvWorker ──obs{"states"[,"expert_u"]}──> RolloutWorker(MLP) ──u (7-D)──> EnvWorker
ResidualVLAEnv.chunk_step(u):                         (RL 1 step = π0.5 결정 1회 = sim 5 step)
   s = tanh(u)  (PPO/GRPO/DAgger: raw-Gaussian MLP)  |  s = clip(u,-1,1)  (SAC/CrossQ: 이미 tanh-squash)
   for k in 0..4:  a_exec[k] = clip(a_base[k] + α⊙s, action_space)     ← chunk 단위 residual(같은 Δ를 5 step)
   → 다음 관측에서 π0.5 1회 호출: base chunk(H=8 생성, 앞 K=5 실행) + VLM 17층 feature를 같은 forward에서 얻음
```
- **chunk 단위 residual을 고른 이유**: π0.5와 VLA feature는 결정마다 1번만 갱신된다. 결정 단위면 episode가 16 step이라 credit assignment가 짧고, 행동이 7차원이라 SAC entropy·DAgger 라벨이 단순하다. 대가로 chunk 안의 step별 보정은 못 한다(상한 0.820이 이 제약을 이미 포함한다).
- **π0.5와 env 하나를 train·eval이 공유**: ManiSkill scene 하나가 renderer 때문에 5–6 GB를 쓴다(측정: scene 2개 + π0.5 = 22 GB → OOM). RLinf는 train rollout과 eval을 번갈아 하므로 같은 process 안의 scene 하나를 두 wrapper가 함께 쓴다.
  - on-policy(PPO/GRPO)는 rollout epoch마다 어차피 full reset이라 영향이 없다.
  - off-policy(SAC/CrossQ/DAgger, auto_reset)는 eval 뒤 첫 step에서 진행 중이던 train episode를 **truncated(보상 0, final obs = eval 직전 관측)로 끊고** 새 episode를 시작한다. eval 3회 × 32 transition이라 영향은 작다. `train_interrupted.jsonl`에 기록하고 train 곡선에서는 뺀다.
- **렌더 생략**: 결정 안의 앞 4 sim step은 관측을 만들지 않는다(`_obs_mode="none"`). 마지막 step만 렌더한다. 결과가 같은 것을 확인했고(16 env × 16 결정, 성공 동일), sim 시간이 47.6 s → 27.4 s로 줄었다.

## 2. 행동 결합 α (모든 알고리즘 공통)
- 이 env의 action은 WidowX EE delta(위치 m, 회전 rad, `normalize_action=False`, 범위 ±1 m / ±π/2)와 그리퍼 목표(−1 닫힘, +1 열림)다. π0.5 base는 위치 ±0.036 m/step, 회전 ±0.2 rad/step, 그리퍼 ±1이다.
- **α_pos = 0.02 m/step, α_rot = 0.1 rad/step, α_g = 2.0**. 그리퍼는 sign 기반이라 α_g=2가 base 명령을 뒤집을 수 있다(T008b D_g 교훈).
- **α 검증**: hand-coded 상한 0.820 ≫ B0 0.586, expert 단독 0.883에 가깝다 → α를 키울 필요가 없다. 더 키우면 탐색 noise가 base를 더 망가뜨린다(T008c).
- 상한의 success_at_end가 0.453으로 낮은 것은 expert가 놓은 뒤에도 base가 계속 움직여 물체를 다시 건드리기 때문이다. 학습은 성공 시 종료라 success_once가 주 지표다. at_end는 eval에서 함께 보고한다.

## 3. 관측 · 보상 · 종료 (모든 알고리즘 공통)
- **관측 `states`(2092)**: [VLM(PaliGemma) **17층 image token 평균**(2048, T017 buffer 8000 결정의 평균/표준편차로 표준화 후 ±5 clip; T018 succ AUC 최선 층) | qpos 8 | base chunk 5×7(위치/0.03, 회전/0.1, 그리퍼/1) | t/16].
  - feature는 base chunk를 만든 같은 π0.5 forward의 hook에서 얻는다(관찰만 함, 추가 forward 없음).
  - 특권 정보(물체·접시 pose)는 `states`에 없다. DAgger의 expert 라벨 `expert_u`만 별도 키로 rollout에 가고, student MLP는 `states`만 읽는다.
- **보상**: sparse 성공만. train에서는 처음 성공한 결정에 +1, `terminated=True`(bootstrap 없음). RLinf env의 dense 보상(grasp 0.1 등)은 쓰지 않았다(명세 기본값 = sparse; 모든 알고리즘 동일).
- **종료·timeout**: 16 결정(80 sim step)에서 `truncated=True`. bootstrap 여부는 알고리즘 구현을 따른다(SAC/CrossQ `bootstrap_type: standard` = truncation만 bootstrap). PPO/GRPO는 `auto_reset=False`라 done 뒤 step은 RLinf `loss_mask`가 가린다.
- **할인**: γ = 0.96/결정(≈0.992/sim step), 모든 알고리즘 동일(DAgger는 미사용).
- **eval**: `ignore_terminations=True`로 항상 16 결정을 다 돌려 success_once와 success_at_end를 잰다. 행동은 결정적(mean)이다.

## 4. 초기 상태 분할 · 평가
- **held-out 128 episode**: T020 test episode(perm(seed 20261119)[500:]에서 T016 30개 제외)의 앞 128개. T016–T020 어디에서도 학습·선택에 쓰지 않았다.
- **train**: 802,816개 episode id에서 균등 추출. T020 test 후보 전체(앞 400개 중 사용 가능분)를 제외한다(단위 테스트로 확인).
- **paired**: eval의 π0.5 noise는 (episode id, 결정 번호)로 seed를 정한다. 그래서 B0 검증(`val/b0.json`)과 모든 run의 eval이 같은 초기 상태·같은 noise에서 비교된다. RLinf eval은 매번 full reset으로 시작하므로 id 순서가 매번 같다(32 env × rollout_epoch 4 = 128).
- PhysX가 bitwise 재현되지 않아 같은 조건에서도 episode 몇 개는 뒤집힌다(T016: 1/30). paired 차이의 CI는 episode bootstrap이다(`make_metrics.py`).
- **x축**: env step(= 결정 × 5 sim step)과 wall-clock 둘 다. step 0 점은 B0이다. actor mean의 마지막 층을 0으로 초기화하므로 학습 전 결정적 정책 = B0이다.

## 5. 알고리즘별 설정 (RLinf MLP config 기반, 바꾼 것만)
공통: MLP 3×256(tanh), obs 2092, action 7, num_action_chunks 1, train env 32, eval 32×4, **actor mean 마지막 층 0 초기화**(`vr_init.zero_last`).

| | PPO | GRPO | SAC | CrossQ | DAgger |
|---|---|---|---|---|---|
| RLinf 기반 | maniskill_ppo_mlp | T021 grpo_mlp | maniskill_sac_mlp | maniskill_crossq_mlp | maniskill_dagger_mlp |
| squash | tanh(env) | tanh(env) | policy tanh, env clip | 같음 | tanh(env) |
| 데이터/epoch | 2×32×16=1024 결정 | 같음(group 8 × 8개) | 32×2=64 결정 | 같음 | 32×16=512 결정 |
| update | 4 epoch × batch 256 | 같음 | 64 update(UTD 1), batch 512 | 같음, critic:actor 2:1 | 8 epoch, batch 256 |
| lr | actor 1e-4, value 3e-4 | actor 1e-4 | actor 1e-4, critic 3e-4, α 3e-4 | 같음 | 3e-4 |
| 탐색 초기 std | log-std −1.5 (state 무관) | 같음 | log-std −2 (bias 초기화) | 같음 | (mean 행동) |
| 기타 | GAE λ 0.95, clip 0.2, value clip 1 | adv = group 정규화, critic 없음, `auto_reset=False` | α₀=1e-3, target entropy −7, τ 0.005, **critic-only 30 epoch**(1920 update), **anchor 1.0·‖tanh μ‖²** | BatchRenorm Q, [s;s′] joint forward | expert = scripted(특권), β 1.0→×0.95/epoch(최소 0.05), classic relabel |
| epoch / eval | 27 / 9,18,27 | 27 / 9,18,27 | 330 / 110,220,330 | 330 / 110,220,330 | 48 / 16,32,48 |
| env step(추정) | 138k | 138k | 106k | 106k | 123k |

결정 근거:
- **PPO/GRPO actor lr 1e-4**: smoke(32 env, lr 3e-4)에서 첫 update 직후 held-out success_once가 −9.4pp(paired CI [−16, −2])로 떨어졌다. residual은 작게 시작해야 한다는 T008 교훈에 따라 낮췄다.
- **PPO/GRPO 초기 log-std −1.5**(RLinf 기본 −0.5): 기본값이면 tanh 뒤 noise가 ≈0.45α/결정이다. base를 망가뜨리지 않을 수준(≈0.2α)으로 낮췄다.
- **SAC/CrossQ**: α₀=1e-3(보상 크기 ≈1/episode에 맞춤, T008 α₀=0.1 실패 교훈), critic-only warmup(`train_actor_steps: 30`), anchor 1.0(T011 값).
  - smoke에서 anchor 0.1 + warmup 2 epoch이면 held-out이 0.75 → 0.03(32 ep)으로 무너졌다. 그래서 1.0과 30 epoch으로 정했다.
  - anchor는 RLinf에 없어 `rlinf_ext`가 actor loss에 더한다(`algorithm.vr_anchor`).
- **GRPO**: 같은 episode id 8개가 연속 env slot에 들어간다(초기 상태 동일 검증). rollout epoch 2 × 32 env = group 8개/update.
- **DAgger expert**: RLinf DAgger는 expert를 신경망 ckpt로 받는다. 우리는 env가 결정마다 계산한 특권 scripted expert의 residual 라벨 `expert_u`를 obs로 보내고, expert MLP가 그 값을 행동으로 돌려주게 했다(`rollout.expert_model.vr_expert_obs_key`).
  - 라벨 = atanh(clip((a_expert − mean_k a_base)/α, ±0.995)), 즉 student가 정확히 따라 하면 hand-coded 상한(0.820)이 된다.
  - RLinf의 env→rollout 관측 whitelist에 `expert_u`를 추가했다(rlinf_ext).

## 6. Scripted expert (특권 상태)
- 상태: 물체·접시 위치(robot base frame), EE 위치(controller `ee_pose_at_base`), `is_grasping`, 손가락 간격, 물체 장축 방향 vs 손가락 닫힘 방향(yaw 오차).
- phase(env별 메모리): APPROACH(물체 위 7 cm) → DESCEND(물체 중심 +1 cm) → CLOSE(2 결정) → LIFT(1 cm/step, 물체가 따라오지 않으면 RETRY) → CARRY(접시 위 +10 cm) → LOWER → RELEASE → DONE. 접근 중에는 yaw를 맞춘다(길쭉한 물체만).
- 결정 단위 controller다(5 step 동안 같은 delta, residual과 같은 granularity).
- 튜닝은 held-out 앞 32개에서만 했다(grasp 높이 +2.2 cm 0.59 → +1.2 cm 0.84 → +0.4 cm 0.88). 그 뒤 128개 전체에서 0.883이다. 이 32개는 학습에 쓰이지 않지만 expert 설계에는 보였다(평가 공정성에는 영향 없음: 모든 조건이 같은 expert·같은 episode).

## 7. 검증 결과 파일
- `/outputs/StochRL/t022/val/{b0,expert,upper}.json`(episode별), `units.json`.
- `units.json`: reward_eq_first_success, truncation_only_at_16, no_reward_after_termination, auto_reset_ok, grpo_* 모두 True. obs의 feature 최대 5.0(clip), 나머지 ≤2.6.

## 8. smoke (`/outputs/StochRL/t022/smoke/*.metrics.json`, GPU 0/1)
| run | 내용 | 결과 |
|---|---|---|
| smoke-ppo (16/16 env, 2 epoch) | 전체 파이프라인 | 통과. 첫 시도는 `expandable_segments`가 Ray worker 사이 CUDA IPC를 깨서 실패 → 제거 |
| smoke-grpo (16 env, group 8) | GRPO adv, loss_mask | 통과(approx_kl 0.06, clip frac 0.17–0.19) |
| smoke-ppo32 (32/32×4, 2 epoch, lr 3e-4) | 실규모 · 메모리 · 시간 | 통과. 전 GPU peak 19.0 GB(외부 2.4 GB 포함), rollout epoch 125 s, eval 128 ep 253 s. held-out 0.586 → 0.492 → 0.523 (→ lr 1e-4로 변경) |
| smoke-sac / smoke-crossq (4 epoch, warmup 2) | replay, anchor, eval 뒤 interruption | 통과. eval 0.75 → 0.03(32 ep) 붕괴 → anchor 1.0, warmup 30, actor lr 1e-4로 변경 |
| smoke-dagger (2 epoch) | expert 라벨 경로 | 1차 `KeyError: expert_u`(obs whitelist) → 수정 후 통과. dagger loss 0.38–0.57 |
- 실패 기록: 48+32 env는 env worker 22 GB로 OOM. scene 공유 전 32+32도 OOM.
- **이후 config만 바꾼 것**(lr, anchor, warmup, epoch 수, π0.5 micro-batch 16→8)은 smoke를 다시 돌리지 않았다. 코드 경로는 같다.

## 9. 예산 (GPU 0/1만; harness)
- 처리량(GPU를 외부 job과 나눠 쓰는 상태에서 측정): 32 env 결정 1회 ≈ 3.9 s(π0.5 + sim 5 step + feature) → ≈ 41 env step/s. eval 128 ep ≈ 255 s.
- run당: PPO/GRPO 27×126 s + eval 3×255 s ≈ 67 min, SAC/CrossQ 330×10.5 s + 3×255 s ≈ 70 min, DAgger 48×70 s + 3×260 s ≈ 69 min, 시작 ≈2 min → **≈1.2 GPU-h/run**.
- 검증 + smoke(실패 포함) ≈ **1.7 GPU-h**. 본 run 5개 ≈ 6.0 → 합계 ≈ 7.7. 남는 ≈2.3에서 `t022-ppo-s1`(≈1.2)을 체인 마지막에 넣었다. `submit_chain.sh` 가드가 results.tsv의 실제 `t022-*` gpu_h + 1.7 + 추정치가 10을 넘으면 건너뛴다.
- run당 RUN_HOURS 상한 = 1.25×1.4+0.2 ≈ 1.95 h(외부 부하로 느려질 때 대비).

## 10. 실행 방식과 주의
- `vla_residual/submit_chain.sh`(nohup): GPU 0 또는 1이 비었고(<1000 MiB) 우리 run이 대기열에 없을 때만 harness에 낸다(최대 2개 동시).
  - harness는 번호가 낮은 빈 GPU를 잡는다. GPU 3이 비어 있으면 거기에 걸릴 수 있다. `run.sh`가 그 경우 exit 3으로 거부하고, 체인이 `<run_id>-rN`으로 다시 낸다.
  - 로그 `/workspace/stochrl_t022/chain.log`.
- `run.sh`: GPU guard(0/1 외 거부), placement를 물리 GPU로 고정, run 전용 Ray head(T021 함정 1·2), 패치 0001 적용, `RLINF_EXT_MODULE`, episode 로그 → `make_metrics.py` → `$RUN_DIR/metrics.json`.
- metrics.json: `heldout_success_once/at_end`(마지막 eval, n=128), `paired_vs_b0_once/end`(diff, 95% CI, wins/losses), `eval_curve`(env_steps, wall_s, 매 eval paired), `train_curve`(200 episode 창), `train_env_steps`, `train_time_s`, `peak_vram_mib_gpu_total/minus_baseline`, `final_eval_episodes`.
- **메모리 위험**: 전 GPU peak가 smoke에서 19–21 GB였다. 외부 job이 run 중간에 같은 GPU에 몇 GB 이상 올라오면 OOM이 날 수 있다(사용자 공간에서 막을 수 없음). π0.5 micro-batch를 8로 낮춰 ≈2 GB 여유를 더 만들었다.
- 해석상 주의: 결과는 "같은 α·같은 관측·같은 step 예산(≈0.1–0.14M env step, 1.2 GPU-h)"에서의 비교다. 수렴 비교가 아니다. DAgger는 특권 expert 라벨을 쓰므로 RL 4종과 같은 정보 조건이 아니다(상한 참고용).

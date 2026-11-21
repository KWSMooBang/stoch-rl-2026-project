# T040 — Q-VGM/Q-Flow 관문: chunk critic의 행동 gradient가 더 나은 행동을 가리키는가

- 날짜: 2026-11-21. 브랜치 `exp/qvgm-gate`(`exp/residual-input` 0b2d9ca에서 분기). 코드 `qvgm/`.
- **사전 등록**: `qvgm/plan_t040_ko.md`. critic 평가와 closed-loop 실행 전에 커밋했다(717cba8).
- **GPU 1만 썼다.** 합계 **≈ 0.92 GPU-h**(상한 1.0). GPU 0·3은 쓰지 않았다. 정책 학습은 하지 않았다.
  - critic 학습 0.30: run 11개를 병렬로 돌렸고 wall 1074 s, smoke 15 s.
  - closed-loop 1차 0.47: 1680 s, 첫 memory probe 포함.
  - closed-loop 2차 0.15: 541 s, RG/RN α=0.05와 KV 경로 memory probe.
  - 시작할 때 GPU 1에 외부 프로세스(4.9–5.4 GB, util 100%)가 있었다. 이 때문에 B0가 608 s 걸렸다(나머지 조건은 각 약 252 s). 외부 프로세스는 B0 도중에 빠졌다.

## 0. 결론
1. **사전 등록 판정: "부분"(조건 1 충족, 조건 2 불충족).** 다만 실질 권고는 **Q-VGM/Q-Flow 계열(Experiment A·B)을 지금 구현하지 않는 것**이다. 이유는 아래 2–4다.
2. **critic은 같은 상태의 후보를 action-only보다 잘 줄 세운다. 하지만 그 이득은 큰 행동 변화에서만 나온다.**
   - pairwise accuracy(out-of-fold, 83 상태)는 full **0.634** vs action-only 0.548이다. 차이 **+0.086 [+0.013, +0.163]**로 조건 1을 충족한다.
   - 이득은 큰 변화(10 cm 이동, 그리퍼 반전, expert)가 낀 쌍에 몰려 있다: 0.74 vs 0.62, 차이 CI [+0.04, +0.21].
   - 작은 변화끼리의 쌍(T026 VLA 샘플, 2.5 cm, 5 cm, yaw)에서는 0.57 vs 0.52이고 CI가 0을 포함한다. VLA 샘플끼리는 0.58 vs 0.50(n.s.), 2.5 cm vs base는 0.50 vs 0.50이다.
3. **gradient 정렬은 약하고 action-only와 구분되지 않는다.**
   - Spearman ρ(∇_A Q̄·(A_c − A_base), paired 성공 차이)는 **0.081 [0.013, 0.143]**이다. action-only는 0.041이고, 차이 +0.040 [−0.055, +0.134]다.
4. **closed-loop Q-ascent는 결과를 확실히 나쁘게 만든다. critic 방향은 같은 크기의 무작위 방향보다 낫지 않다.**
   - B0 once 0.562 / end 0.539 대비 QA α=0.05는 once **−0.109 [−0.211, 0.000]**, end **−0.234 [−0.328, −0.133]**다. α=0.1은 once −0.281, end −0.383이다.
   - ascent는 거의 모든 결정에서 3 step을 모두 채택한다(채택률 1.00, 평균 2.99 step). 그때마다 Q̄가 +0.22(α=0.05), +0.36(α=0.1) 오른다. **critic hacking이다**(T019 §4와 같은 현상).
   - 피해는 수정 크기를 따라간다.
     - 크기를 맞춘 무조건 무작위(RN)와 비교하면 QA는 비슷하거나 조금 낫다. QA − RN은 α=0.05에서 once +0.06 [−0.04, +0.17], α=0.1에서 end +0.08 [+0.01, +0.15]다.
     - 그러나 Q gate를 붙인 무작위(RG)보다는 확실히 나쁘다. QA − RG는 α=0.05에서 end −0.21 [−0.32, −0.10], α=0.1에서 once −0.19 [−0.29, −0.09]다.
     - RG가 덜 해로운 이유는 gate가 수정 크기를 줄이기 때문이다(채택 약 0.46, xyz 3–5 mm/step).
5. **메모리**: VLM은 frozen KV cache로 두고 action expert(430 M param)만 학습하는 경우를 쟀다. scene과 frozen π0.5가 이미 올라간 같은 프로세스에서 batch 4–8의 peak allocated는 11.1 GB, nvidia-smi 합계는 17.4–17.5 GB다. **24 GB 4090에 함께 올라간다.** LoRA r=16에서는 7.7–8.3 GB, 합계 14.3–15.5 GB다.
   - 단, 측정은 bf16 파라미터와 bf16 Adam 기준이다. fp32 master weight(약 +5 GB)를 쓰거나 GPU에 외부 점유가 있으면 full 학습은 빠듯하다.
   - prefix와 suffix를 한 번에 forward하는 경로(RLinf `compute_loss`)는 frozen VLM의 activation까지 저장해서 4개 설정 모두 OOM이 났다(≥17.3 GB allocated).

## 1. 설정 (사전 등록 요약, 상세는 plan)
- **z**: VLM layer-17 image-token mean(2048). T022 env와 같은 표준화를 쓴다.
  - **RL token AE는 쓰지 않았다.** 이유는 세 가지다.
    1. 우리 residual 관측과 T019 critic이 이미 이 feature를 쓴다.
    2. 이 feature로도 상태 가치 AUROC가 0.73–0.85 나온다. 병목은 행동 순위다.
    3. AE를 학습하려면 prefix token 전체를 다시 forward해야 해서 예산 밖이다.
- **A**: 실제로 실행한 5×7 행동을 π0.5 모델 정규화 공간(q01/q99 affine, 오차 < 1e-7)으로 옮긴 35차원 벡터다.
- **구조**: Q-VGM과 같다. 10-head ensemble, [512, 512, 256], 매 hidden layer에 A를 다시 넣는다, LayerNorm. IQL V(z)는 expectile 0.8이고 EMA target을 쓴다. **γ = 0.99/결정**(T022의 0.96 대신)으로 정했다.
- **데이터**:
  - T017 train+val buffer의 transition 4163개. 첫 성공까지만 썼다. 다음 z가 없는 결정 15 truncation 207개는 제외했다.
  - T026(50 × 12)과 T031(33 × 12, E3 제외) branch의 MC 라벨 G = 4 반복 평균 γ^k·성공. 미니배치는 buffer 256 + branch 64, 손실은 1:1이다.
- **split**: branch 83 상태를 episode 단위 5-fold로 cross-fit했다. 각 fold critic은 그 fold 상태의 branch 라벨과 그 episode의 buffer transition을 보지 않는다. 평가는 모두 out-of-fold 점수다(seed 0·1 평균). T017 test 75 ep는 학습에 쓰지 않았다.
- **action-only**: 같은 구조에서 z를 뺐다. 목표는 full critic의 target V(z′)로 만든 같은 y와 같은 MC 라벨이다.

## 2. Critic 표 (out-of-fold, 83 상태, 상태 bootstrap 2000회) — `critic_eval.csv`

| 지표 | full Q(z,A) | action-only Q(A) | state-only V(z) | 무작위 | full − action-only |
|---|---|---|---|---|---|
| **pairwise acc (once, 1차)** | **0.634 [0.574, 0.693]** | 0.548 [0.484, 0.609] | 0.500 | 0.526 | **+0.086 [+0.013, +0.163]** |
| pairwise acc (end) | 0.621 [0.563, 0.681] | 0.534 | 0.500 | 0.518 | +0.087 [+0.018, +0.163] |
| within-state AUROC | 0.638 [0.579, 0.699] | 0.565 | 0.500 | 0.515 | +0.073 [+0.009, +0.144] |
| **gradient 정렬 Spearman (1차)** | **0.081 [0.013, 0.143]** | 0.041 [−0.031, 0.111] | — (grad 0) | — | **+0.040 [−0.055, +0.134]** |
| gradient 정렬 (end) | 0.082 [0.012, 0.149] | 0.025 | — | — | — |
| 유한차분 ΔQ̄ Spearman | 0.154 [0.042, 0.268] | 0.129 [0.002, 0.247] | — | — | +0.026 [−0.092, +0.147] |

- pair 1538개(p̂가 다른 쌍), 정렬 행 880개(상태 × base가 아닌 후보).
- **천장**: split-half oracle(반복 2개로 나머지 2개 순위 맞추기)은 0.719 [0.661, 0.775]다. T019 단일 continuation 천장은 0.81이다. full critic은 0.5에서 split-half 천장까지 거리의 약 61%((0.634−0.5)/(0.719−0.5))를 메운다.
- **출처별(탐색적)**: T026(작은 변화)에서는 full 0.555 vs action-only 0.513, T031(큰 변화)에서는 0.723 vs 0.588이다. gradient 정렬은 T026에서 0.002, T031에서 0.138이다.
- **크기별(탐색적, `critic_explore.json`)**:

| 쌍 종류 | n pair | full | action-only | 차이 CI |
|---|---|---|---|---|
| 큰 변화 하나 포함(T1 10 cm / grip_flip / E1) | 407 | 0.742 | 0.619 | [+0.04, +0.21] |
| 둘 다 작은 변화 | 921 | 0.568 | 0.524 | [−0.05, +0.15] |
| T026 VLA vs VLA | 318 | 0.585 | 0.503 | [−0.06, +0.23] |
| T026 2.5 cm vs base | 44 | 0.500 | 0.500 | [−0.23, +0.20] |

- **보정**:
  - branch에서 Q̄ − G 평균은 **+0.18**(과대평가, expectile 0.8의 낙관)이고 ECE는 0.185다.
  - 상태 간 AUROC는 full 0.80, V 0.75, action-only 0.64다.
  - T017 test(같은 결정 index 안 성공 AUROC)는 V 0.735, Q̄(z, 로그 행동) 0.747이다.
- **규모**: Q̄의 상태 안 표준편차 0.084, 상태 간 0.30, ensemble 표준편차 평균 0.081이다. 상태 안 차이가 ensemble 불확실성과 비슷한 크기다.

## 3. Closed-loop 표 (held-out 128, B0와 paired, episode bootstrap 5000회) — `closed_loop.json`

| 조건 | once | end | Δonce vs B0 [CI] | Δend vs B0 [CI] | pres(once) | rec(once) | 채택률 | 평균 step | 수정 RMS | xyz mm/step | ΔQ̄ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 0.562 | 0.539 | — | — | 1 | 0 | — | — | 0 | 0 | — |
| **QA 0.05** | 0.453 | 0.305 | −0.109 [−0.211, 0.000] | **−0.234 [−0.328, −0.133]** | 0.58 | 0.29 | 1.00 | 2.99 | 0.142 | 9.1 | +0.224 |
| **QA 0.1** | 0.281 | 0.156 | −0.281 [−0.383, −0.180] | −0.383 [−0.477, −0.281] | 0.38 | 0.16 | 0.99 | 2.91 | 0.260 | 16.6 | +0.364 |
| RG 0.05 (무작위 + gate) | 0.555 | 0.516 | −0.008 [−0.086, +0.071] | −0.023 [−0.102, +0.055] | 0.79 | 0.25 | 0.47 | 1.26 | 0.062 | 3.1 | +0.014 |
| RG 0.1 | 0.469 | 0.359 | −0.094 [−0.172, −0.016] | −0.180 [−0.266, −0.094] | 0.71 | 0.16 | 0.45 | 1.08 | 0.104 | 5.2 | +0.022 |
| RN 0.05 (무작위, gate 없음) | 0.391 | 0.266 | −0.172 [−0.273, −0.063] | −0.273 [−0.375, −0.172] | 0.50 | 0.25 | 1 | 3 | 0.145 | 7.4 | −0.013 |
| RN 0.1 | 0.242 | 0.078 | −0.320 [−0.422, −0.219] | −0.461 [−0.547, −0.367] | 0.29 | 0.18 | 1 | 3 | 0.289 | 14.8 | −0.043 |

- pres = P(once 성공 | B0 once 성공), rec = P(once 성공 | B0 once 실패), regression = 1 − pres. end 기준 값은 json에 있다.
- 이번 B0는 GPU 1에서 0.562 / 0.539이다. T034(GPU 0)의 0.555 / 0.531과 같은 수준이다.
- **QA − 무작위(같은 α, paired)**:

| | Δonce | Δend |
|---|---|---|
| QA0.05 − RN0.05 | +0.063 [−0.039, +0.172] | +0.039 [−0.070, +0.148] |
| QA0.05 − RG0.05 | −0.102 [−0.211, +0.008] | −0.211 [−0.320, −0.102] |
| QA0.1 − RN0.1 | +0.039 [−0.063, +0.141] | +0.078 [+0.008, +0.148] |
| QA0.1 − RG0.1 | −0.188 [−0.289, −0.086] | −0.203 [−0.297, −0.109] |

- **크기 대비 해석**:
  - 성공률은 수정 크기(RMS)에 거의 단조로 떨어진다. RG0.05(0.06) → RG0.1(0.10) → QA0.05·RN0.05(0.14) → QA0.1(0.26) → RN0.1(0.29) 순이다.
  - 크기를 맞춘 비교(QA vs RN)에서 critic 방향의 이점은 end 기준 +0.04–0.08이다. 하나만 CI 하한이 0을 겨우 넘는다. B0 대비 피해(−0.23 ~ −0.38)를 상쇄하기에는 훨씬 작다.
  - 정규화 ascent는 gradient가 몰리는 xyz로 거의 모든 이동을 보낸다(α=0.05에서 9 mm/step, 즉 chunk당 4.5 cm). T031에서 5 cm 이동이 net −0.09였던 것과 같은 영역이다.
- 사전 등록 이탈: 1차 실행의 예산 guard가 B0의 느린 시간(608 s)으로 다음 조건 시간을 추정해서 RG/RN 0.05를 건너뛰었다. 이 둘은 같은 스크립트, 같은 설정, 같은 GPU로 2차 프로세스에서 따로 돌렸다. B0와의 paired 비교는 episode id와 noise로 맞춘다. 프로세스가 다르면 PhysX 비결정성이 조금 더해질 수 있다.

## 4. 메모리 (RTX 4090 24 GB, GPU 1) — `memory_probe.json`
- 측정 조건은 closed-loop 프로세스 안이다. frozen π0.5(bf16, 6.9 GB allocated), ManiSkill scene(32 env), critic이 이미 올라가 있다. 측정 시 GPU 1에 다른 프로세스는 없었고, 시작 전 smi는 15.7 GB였다.
- 경로는 Q-VGM 방식이다. VLM prefix KV cache는 no_grad로 만들고, action-expert suffix만 grad를 갖는다. 이 상태에서 flow-matching loss forward + backward + Adam 1 step을 쟀다.

| 설정 | 학습 param | peak allocated | 증가분(VLM 위) | Adam state | nvidia-smi 합계(scene 포함) |
|---|---|---|---|---|---|
| full, batch 4 | 430 M | 11.06 GB | +4.18 GB | 2.04 GB | 17.4 GB |
| full, batch 8 | 430 M | 11.14 GB | +4.17 GB | 2.04 GB | 17.5 GB |
| LoRA r=16, batch 4 | 6.9 M | 7.70 GB | +0.64 GB | 0.03 GB | 14.3 GB |
| LoRA r=16, batch 8 | 6.9 M | 8.26 GB | +1.26 GB | 0.03 GB | 15.5 GB |
| (joint forward `compute_loss`, 4개 설정) | — | ≥ 17.3 GB에서 **OOM** | ≥ +10.4 GB | — | 24.1 GB |

- 판단:
  - **scene + π0.5 + action expert 학습(bf16)은 4090 한 장에 올라간다.** 여유는 full 약 6.5 GB, LoRA 8–10 GB다.
  - 다음 경우에는 full 학습이 빠듯하거나 넘친다.
    - fp32 master weight와 Adam(약 +5 GB)
    - Q-VGM의 look-forward denoising 경로 activation과 critic backward
    - GPU 1에 흔히 있는 외부 점유(5–17 GB)
  - 그러면 LoRA가 현실적이다.
  - prefix를 KV cache로 분리하지 않으면(RLinf `compute_loss` 그대로) batch 4에서도 OOM이다.

## 5. 판정 (사전 등록 규칙)
- 조건 1 충족: pairwise accuracy 차이 CI 하한 +0.013 > 0. gradient 정렬 차이는 CI가 0을 포함한다.
- 조건 2 불충족: 최선 QA(α=0.05)의 B0 대비 once·end CI 상한이 각각 0.000, −0.133이다.
- → **"부분"**. 사전 등록 문구("critic은 행동을 구분하지만 작은 ascent로는 결과가 바뀌지 않는다")와 실제는 다르다.
  - 결과는 **바뀐다**. Q-VGM 크기(α=0.05, J=3)의 ascent는 chunk당 수 cm 수정이고, frozen π0.5 continuation이 흡수하지 못하는 크기다(T031 영역). 그래서 결과가 나빠진다.
  - 작게 하면(RG0.05, 3 mm/step) B0와 같아진다. 이것이 T026의 흡수 현상이다.
  - 즉 "흡수되는 작은 수정"과 "해가 되는 큰 수정" 사이에서 critic gradient가 이득을 주는 구간을 찾지 못했다.

## 6. A/B 권고
- **Experiment A(action expert Q-Flow/Q-VGM)와 B(residual Q-Flow) 모두 지금은 구현하지 않기를 권한다.** 근거는 세 가지다.
  1. 정책 개선 방향을 주는 것은 **행동 gradient**다. 그런데 그 정렬(ρ 0.08)이 action-only와 구분되지 않는다. critic의 순위 이점은 큰 수정(대부분 해로운 것)을 알아보는 데서 나온다.
  2. test-time ascent(Q-VGM이 학습으로 근사하려는 목표)가 B0보다 확실히 나쁘다. 같은 크기의 무작위 방향보다 낫다는 증거도 약하다(end +0.04–0.08).
  3. ascent마다 Q̄가 +0.2–0.4 오른다. 그런데 실제 성공은 떨어진다. offline critic의 행동 gradient가 buffer support 밖에서 믿을 수 없다는 Q-VGM 저자의 한계가 우리 과제에서 그대로 나타난다.
- 그래도 진행한다면 최소 조건은 다음과 같다(실행하지 않음).
  - (a) trust region을 행동 support 기준으로 둔다. 예: π0.5 샘플 산포 ≈ 1 cm 이내, 또는 BC/flow prior를 정규화항으로 둔다.
  - (b) critic을 online TD로 보강한다(Q-VGM 98.4%는 online 단계 결과다).
  - (c) 가장 싼 확인부터 한다. 학습 없이 "π0.5 후보 N개 중 Q̄ argmax"(T020 E06 재판)를 이번 critic으로 closed-loop에서 1 조건 돌리는 것이다. support 안에서만 고르기 때문에 hacking이 없다. 오프라인 VLA vs VLA 순위는 0.585로 n.s.라서 기대 이득은 작다.
- 메모리는 걸림돌이 아니다(LoRA면 넉넉하다). 병목은 critic 신호다.

## 7. 산출물
- `/outputs/StochRL/t040/`:
  - `summary_ko.md`
  - `critic_eval.csv`
  - `critic_explore.json`(탐색적 크기별)
  - `critic_gate.json`
  - `closed_loop.json`
  - `memory_probe.json`
- `/workspace/stochrl_t040/`:
  - `data.npz/json`: 조립 데이터
  - `critic/`: fold × seed checkpoint와 oof 예측. closed-loop critic은 `all_s0.pt`
  - `closed_loop{,2}/`: 결정 단위 parquet, eval.log, gpu_trace
  - `logs/`
- 코드(`qvgm/`):
  - `plan_t040_ko.md`
  - `data.py`
  - `critic.py`
  - `analyze_critic.py`
  - `explore_critic.py`
  - `eval_qascent.py`: closed-loop와 memory probe
  - `memprobe.py`: scene 없는 probe, 쓰지 않음
  - `analyze_closed_loop.py`

## 8. 재실행
```
PY=/workspace/StochRL/.claude/worktrees/repcritic/.venv/bin/python   # worktree: submodule init + assets symlink + feat_stats 복사 필요
$PY -m qvgm.data
CUDA_VISIBLE_DEVICES=1 $PY -m qvgm.critic --folds 0,1,2,3,4,all --seeds 0,1
$PY -m qvgm.analyze_critic ; $PY -m qvgm.explore_critic
bash /workspace/stochrl_t040/run_closed_loop.sh                # B0, QA, RG/RN 0.1 + memory probe(GPU 1)
bash /workspace/stochrl_t040/run_closed_loop2.sh --conds rg0.05,rn0.05
$PY -m qvgm.analyze_closed_loop --raw /workspace/stochrl_t040/closed_loop
```

# StochRL — RESEARCH

## 현재 방향 2 (2026-11-18): DA-CGC (Decision-Aware Counterfactual Goal-Consistency RL, 기말 프로젝트)
gridworld PoC, goal-conditioned Double DQN + gated counterfactual Q-ranking. T013: Phase 0–1(shortcut 재현) 관문 후 Phase 3 비교. CPU 위주, GPU 0 ≤1 GB, 상한 3 GPU-h.
L1 수정: μ(oracle vs learned), B3 = gate 없는 정답 ranking, B6 = Q* 회귀 대조군, goal 인코딩 ablation, DQfD 비교.

## 현재 방향 3 (2026-11-18): AAPR (Action-Attributed Progress Reward, 기말 프로젝트)
2D stochastic inertial pushing + SAC. T014: 기대값 점검(AAPR ≈ progress − b(s)) + 관문 후 M4–M6 비교. 주 조건 = 지속·치우친 바람, 계산량 맞춘 augmentation 기준선 필수. CPU 위주, 상한 3 GPU-h.

## 폐기 (2026-11-18, 사람 결정): Compositional Contrastive RL
이유: T012에서 shortcut은 transport 방향이고 목적지 waypoint로 해결(H-dest 50/50), 후보 coverage 낮아 critic 재선택 No-Go.
~~T012 진단~~ (완료: shortcut은 transport 방향, H-dest 50/50, coverage 낮음 → critic은 목적지 추론 역할로 한정, VLM grounding과 비교 필요). 다음 단계 사람 결정 대기.
원래 T012 질문: LIBERO-Goal 새 조합에서 목적지 shortcut의 phase(handover)와 후보 coverage. VLA-shortcut-exps goal15 결과·코드 읽기 재사용. GPU 2, 상한 3 GPU-h.

## 현재 방향 (2026-11-18): Attention-Grounded RL for Shortcut-Resistant VLA (사람 설계서 v0.1)
L1 검토: P1(frozen π0.5 + residual SAC + attention 보상)은 T008c 근거로 실패 가능성 높아 보류. P0 진단만 먼저.
P0 질문: LIBERO-Object swap에서 π0.5 action expert→image attention이 (a) 지시 물체 j (b) 기억된 자리의 물체 (c) 원래 물체 i 중 어디를 따르고, 행동과 일치하는가.
주의: 같은 서버의 VLA-shortcut-exps 연구(π0.5 LIBERO shortcut, 위치 암기 H4, LIBERO-PRO, SAE)와 주제가 겹침 — 관계는 사람 결정 대기, 코드는 읽기만.
백로그: ~~T010 P0~~ (완료: attention은 기억 자리의 물체를 따름) → T011 P1 (사람 지시로 3 run 후 중단: 쌍 A E1 4/20, E3 6/20, E4 4/20, E0 5/20) → T015 P2 취소 (2026-11-18 사람 지시로 attention 실험 전체 중지). worktree·exp/attn-p0·exp/attn-rl·/workspace/stochrl_t010–t011 삭제, 요약은 /outputs에 남김. t011-A-E1-s1은 의도적 중지(stale, 재제출 안 함) → T011 P1 residual SAC + attention 보상 E0–E4 (사람 지시 "RL 효과까지 검증", 상한 10 GPU-h, T010 뒤 착수) → 결과에 따라 P2(adapter).

## 폐기 (2026-11-18, 사람 결정): 가상 손 trace 보상
이유: T009에서 손 trace 보상이 물체 궤적 보상보다는 낫지만, 같은 trace를 따라가는 controller를 RL이 넘지 못함.

## 이전 방향 (2026-11-17, 오후, 폐기)
Fully-simulated human hand trace reward for robot/VLA RL (사람 설계서 v1.0). ManiSkill 3, state PPO, Stage 0부터.
L1 검토로 고친 점: 주 비교 = 손+물체(S5) vs 물체만(X3, HuDOR arXiv 2410.23289), trace 추종 controller C0 기준선 필수,
PushCube는 smoke용, 주 과제는 접촉 전략이 중요한 과제, 할인 PBRS 주 보상 금지(T008c 교훈), 보상 단위 테스트 선행.
백로그: ~~T009 Stage 0~~ (완료: 손 > 물체 궤적, RL < trace 추종 controller) → Stage 0.5(embodiment) 여부 사람 결정 → Stage 1(GT+noise, 손 메시 필요).

## 이전 방향 (중단, 2026-11-17 오전)
Point cloud 기반 gripper goal + frozen VLA(π0.5, lerobot/pi05_libero_finetuned) residual RL. 아이디어 노트(사람, 2026-11-17).
차별점 후보: T004에서 frozen VLA가 instruction을 무시함(LIBERO-Object swap follow 2%) → instruction에 맞는 3D 기하 목표로 shortcut을 고칠 수 있는가.
유사 연구: Object-Centric Residual RL (Microsoft 2026; frozen VLA + 물체 6-DoF pose residual, 목표 물체는 사람이 지정).

## 이전 목표 (보류, 기록용)
고정 WAM(UVA)의 상상을 residual SAC의 보상에 쓰는 두 방식을 비교한다.
- 시간 기준 tracking 비용 (기존 제안서)
- 상상 경로 cost-to-go potential (새 방식, 최적 정책 보존)

설계 문서: https://claude.ai/artifact/YCViUHYmUBFwhwrM2y21Eg (PA-WFRL 보상 설계 비교)

## 가설 (문서 7절)
- H-a: 계획 품질이 높고 shift가 작으면 tracking ≥ potential
- H-b: shift가 크거나 계획이 자주 틀리면 potential > tracking (최종 성공률)
- H-c: 부호 있는 gap vₜ 입력이 두 보상 모두에 도움
- H-d: potential + 작은 tracking 혼합이 두 조건에서 안정적

## 적용 범위
> 2026-11-16 전환: UVA/PushT 계획은 보류. LeRobot + Cosmos3 Edge(LIBERO)로 옮긴다. 아래 PushT 범위는 기록용.

- 주 실험: PushT 물리 shift (마찰·질량·PD 이득 중 pilot으로 선택)
- 2축: PushT 물체 위치 OOD
- 선택: PushT 목표 이어 붙이기 (pusht_multitask)
- LIBERO-10: 향후 과제. 무게·마찰 shift 반응만 pilot에서 확인

## 백로그
1. ~~T001 환경 설정~~ (완료, exp/env-setup)
2. ~~T002 pilot~~ (완료, exp/pilot)
3. ~~T003 UVA 코드 제거 + LeRobot 서브모듈 + Cosmos3 Edge 환경~~ (완료, exp/lerobot-setup)
4. ~~T004 LIBERO-Object instruction swap~~ (완료: 가설 기각, 영상도 shortcut)
   가설(사람): LIBERO-Object는 instruction↔scene 1:1이라 shortcut 위험. 영상은 instruction을 따르고 행동은 scene shortcut을 따른다.
5. ~~T005 base Cosmos3-Edge world model swap~~ (중지: base 모델이 LIBERO 장면에서 생성 안 됨, 사람 판단)
6. ~~T006 handover PoC~~ (완료: shortcut은 위치, handover로 해결, 남은 실패는 place)
7. ~~T007 GraspGen~~ (완료: 필터 top-k lift 87.2%)
8. T008 residual RL PoC → T008b(그리퍼 수정) → T008c(디버그): RL 최선 1/20, oracle controller 20/20. 다음 방향 사람 결정 대기
9. (T008 뒤) SF critic
10. 보류: UVA 상상 보상 비교 (문서 9절)

## 결정
- 2026-11-15: PushT를 주 환경으로. 기본 damping=0이라 damping으로는 마찰을 키울 수 없음 → env 수정 필요
- 2026-11-15: r_task는 성공 시 +1 sparse, 모든 변형에 같은 입력
- 2026-11-15: pilot은 GPU 0에서 진행 (사용자 지시)
- 2026-11-16 (T001): UVA 상상 = t+4/8/12/16 4프레임, 1회 0.53 s. 에이전트는 상상에서 흐려짐 → z는 T 포즈만.
- 2026-11-16 (T002): 주 shift 축 = block_mass_scale(agent_mass=10, 기준선 dyn_m10), 단계 ×3/×5/×10.
  위치 OOD는 벽에서 떨어진 드문 위치만 범위 안, 벽·모서리(방문 밀도 하위 5%)는 범위 밖 → 계획별 방문 밀도로 층화.
  E = 색 마스크 + T 템플릿 맞춤 (0.9 px / 0.37°), 학습 불필요. LIBERO 무게 shift는 효과 미미 → 제외.
- 2026-11-16: UVA 사용 중단 (사람 결정). 이유: 상상이 짧고(1.6 s) PushT 질량 shift에서는 상상 방향 ≈ 목표 방향이라 Φ_img가 Φ_goal과 구분되지 않을 위험, 모델이 작고 오래됨.
  → LeRobot 최신 버전 서브모듈 + Cosmos3 Edge(4.6B, 상상 영상 출력, LIBERO-10 95–97%). Nano(16.5B)는 4090 24GB에 안 들어감.
  UVA pilot 코드는 exp/env-setup, exp/pilot 브랜치에 보존(main 미병합).
- 2026-11-17: 방향 전환 → point cloud gripper goal + frozen VLA residual RL. 먼저 RL 없이 oracle 목표 handover PoC(T006).
  grasp 추정기 = GraspGen(PTv3 encoder, diffusion, Franka Panda ckpt, 물체 point cloud 입력). 대안 M2T2. 언어 grounding은 당분간 sim segmentation.
  SF critic은 residual RL이 동작한 뒤로 미룸.
- 2026-11-17: frozen VLA = π0.5 (사람 지시, Cosmos3 Edge 대신). T006은 π0.5로 H0 shortcut 기준선부터 다시 잰다.

## 정리 (2026-11-17)
- 사람 판단: 지금까지 PoC가 모두 실패 → 중단.
- 삭제함: 실험 worktree 10개(venv 포함), exp/* 브랜치(로컬·origin), /workspace/stochrl_t001–t008c 원자료, /cache/graspgen.
  위 기록과 EXPERIMENTS.md에 나오는 브랜치 이름은 이제 존재하지 않는다.
- 남김: /outputs/StochRL(요약·그림·runs·results.tsv·task 명세), 아티팩트(연구 노트, PushT OOD 결과, PA-WFRL 설계 문서), 이 파일과 EXPERIMENTS.md.
- main에는 여전히 UVA 원본 코드가 있다(UVA 제거·LeRobot 서브모듈은 exp/lerobot-setup에만 있었고 병합 전에 삭제됨).

## GPU 예산
- 사용 가능: GPU 0 (RTX 4090). GPU 1·2는 다른 사용자 사용 중
- 2026-11-18: GPU 0이 다른 사용자로 점유 → 사람 결정으로 T010·T011은 GPU 2. GPU 3은 VRAM이 일정 이상이면 문제가 있어 사용 금지.
- 총 예산: 미정 — T001의 추론 시간 측정 후 본 실험 규모와 함께 정한다
- pilot(T001+T002): 상한 10 GPU-h → 사용 3.9 GPU-h
- (보류) UVA 본 실험 추정 400–700 GPU-h
- 2026-11-17: T006 상한 5, T007 상한 2, T008 RL PoC 상한 10 GPU-h (GPU 0). 그 이상 본 실험 예산은 PoC 뒤 사람과 결정

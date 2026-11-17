# StochRL — RESEARCH

## 현재 방향 (2026-11-17, 오후)
Fully-simulated human hand trace reward for robot/VLA RL (사람 설계서 v1.0). ManiSkill 3, state PPO, Stage 0부터.
L1 검토로 고친 점: 주 비교 = 손+물체(S5) vs 물체만(X3, HuDOR arXiv 2410.23289), trace 추종 controller C0 기준선 필수,
PushCube는 smoke용, 주 과제는 접촉 전략이 중요한 과제, 할인 PBRS 주 보상 금지(T008c 교훈), 보상 단위 테스트 선행.
백로그: T009 Stage 0 (환경 + 시연 + 보상 + PPO 비교, GPU 0, 상한 6 GPU-h) → Go면 Stage 0.5(embodiment) → Stage 1(GT+noise, 손 메시 필요).

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
- 총 예산: 미정 — T001의 추론 시간 측정 후 본 실험 규모와 함께 정한다
- pilot(T001+T002): 상한 10 GPU-h → 사용 3.9 GPU-h
- (보류) UVA 본 실험 추정 400–700 GPU-h
- 2026-11-17: T006 상한 5, T007 상한 2, T008 RL PoC 상한 10 GPU-h (GPU 0). 그 이상 본 실험 예산은 PoC 뒤 사람과 결정

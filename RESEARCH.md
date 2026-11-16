# StochRL — RESEARCH

## 현재 방향 (2026-11-17)
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
6. T006 handover PoC: oracle 목표로 controller가 j 앞까지 → frozen VLA에 넘김 (진행 중)
7. T007 GraspGen 설치 + LIBERO point cloud grasp 추정·sim lift 검증 (진행 중)
8. (T006 결과에 따라) residual SAC: oracle 목표 → 예측 목표(GraspGen) → SF critic
9. 보류: UVA 상상 보상 비교 (문서 9절)

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

## GPU 예산
- 사용 가능: GPU 0 (RTX 4090). GPU 1·2는 다른 사용자 사용 중
- 총 예산: 미정 — T001의 추론 시간 측정 후 본 실험 규모와 함께 정한다
- pilot(T001+T002): 상한 10 GPU-h → 사용 3.9 GPU-h
- 본 실험: 상상 보상 RL run 1개(20만 step) ≈ 2.3–3.8 GPU-h. 문서 계획(약 180 run) 그대로면 400–700 GPU-h → 사람 결정 대기

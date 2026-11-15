# StochRL — RESEARCH

## 목표
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
- 주 실험: PushT 물리 shift (마찰·질량·PD 이득 중 pilot으로 선택)
- 2축: PushT 물체 위치 OOD
- 선택: PushT 목표 이어 붙이기 (pusht_multitask)
- LIBERO-10: 향후 과제. 무게·마찰 shift 반응만 pilot에서 확인

## 백로그
1. ~~T001 환경 설정~~ (완료, exp/env-setup)
2. ~~T002 pilot~~ (완료, exp/pilot)
3. residual SAC + 보상 R0–R5 구현 (다음, 예산 결정 후)
4. λ 튜닝 → 마찰 주 실험 → 틀린 계획 주입 → OOD 확장 (문서 9절, 약 180 run)

## 결정
- 2026-11-15: PushT를 주 환경으로. 기본 damping=0이라 damping으로는 마찰을 키울 수 없음 → env 수정 필요
- 2026-11-15: r_task는 성공 시 +1 sparse, 모든 변형에 같은 입력
- 2026-11-15: pilot은 GPU 0에서 진행 (사용자 지시)
- 2026-11-16 (T001): UVA 상상 = t+4/8/12/16 4프레임, 1회 0.53 s. 에이전트는 상상에서 흐려짐 → z는 T 포즈만.
- 2026-11-16 (T002): 주 shift 축 = block_mass_scale(agent_mass=10, 기준선 dyn_m10), 단계 ×3/×5/×10.
  위치 OOD는 벽에서 떨어진 드문 위치만 범위 안, 벽·모서리(방문 밀도 하위 5%)는 범위 밖 → 계획별 방문 밀도로 층화.
  E = 색 마스크 + T 템플릿 맞춤 (0.9 px / 0.37°), 학습 불필요. LIBERO 무게 shift는 효과 미미 → 제외.

## GPU 예산
- 사용 가능: GPU 0 (RTX 4090). GPU 1·2는 다른 사용자 사용 중
- 총 예산: 미정 — T001의 추론 시간 측정 후 본 실험 규모와 함께 정한다
- pilot(T001+T002): 상한 10 GPU-h → 사용 3.9 GPU-h
- 본 실험: 상상 보상 RL run 1개(20만 step) ≈ 2.3–3.8 GPU-h. 문서 계획(약 180 run) 그대로면 400–700 GPU-h → 사람 결정 대기

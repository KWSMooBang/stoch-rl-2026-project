# StochRL — EXPERIMENTS

학습 run은 아직 없음. 아래는 평가·분석 task 결과 (GPU 0).

| task | 날짜 | 내용 | 결과 | 판단 | GPU-h |
|---|---|---|---|---|---|
| T001 | 2026-11-16 | UVA 환경·공식 ckpt 재현 | PushT 0.993 / PushT-M 0.856 / LIBERO-10 0.90 (mean score) | keep → 이후 UVA 중단 | 0.4 |
| T002 | 2026-11-16 | UVA PushT 물리 shift·위치 OOD pilot | 질량 ×5/10/20 성공률 0.54/0.16/0.10, 위치 OOD는 벽·모서리만 상상 붕괴 | 기록용 (UVA 중단) | 3.5 |
| T003 | 2026-11-16 | LeRobot 서브모듈 + Cosmos3 Edge 환경 | LIBERO-10 47/50 = 94.0% (문서 95.2%), chunk b1 0.93 s | keep | 0.7 |
| T004 | 2026-11-16 | LIBERO-Object instruction swap (Cosmos3-Edge-Policy-LIBERO) | swap 91–94%가 원래 물체 i, follow 2%. 같은 샘플 영상≠행동 0.3%. t=0 민감도 행동 1.08 / 영상 1.11 | 가설(영상은 shortcut 안 빠짐) 기각 | 2.8 |
| T005 | 2026-11-17 | base Cosmos3-Edge world model로 swap 영상 | 생성 실패(사람 판단) | discard, 중지 | ≈1 (중지) |
| T006 | 2026-11-17 | π0.5 handover PoC (oracle pre-grasp) | H0 follow 25.6% / H1 follow 55.6%, partial 39.2%, return 0% / C 100% | keep: shortcut은 위치, handover로 해결. 남은 실패는 place | 2.4 |
| T007 | 2026-11-17 | GraspGen + LIBERO point cloud lift | 원본 top-1 24.4% / 필터 top-k 87.2% / oracle top-down 98.9% | keep (필터 필수) | 0.5 |

## T004 메모
- 영상과 행동은 같은 denoising에서 함께 샘플되므로 일치는 구조상 예상 가능. 남은 질문: 정책 미세조정 전 base world model(Cosmos3-Edge)의 영상은 instruction을 따르는가.
- 4 suite 체크포인트(LIBERO-Goal 포함)인데도 shortcut이 강함.
## T006 메모
- π0.5 H0 shortcut = "j가 자기 scene에서 놓이던 자리(A/B)"의 물체를 집음 (193/250). j가 그 자리면 follow 50/50, 아니면 7%. Cosmos3도 같은 분할에서 5/50 vs 0/200.
- 그리퍼를 j 앞에 두면 instruction이 i여도(H2) j를 집음 → 그 지점에서 언어는 대상을 정하지 않음.

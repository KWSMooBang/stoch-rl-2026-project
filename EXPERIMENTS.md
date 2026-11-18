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
| T008 | 2026-11-17 | residual RL PoC 첫 시도 (frozen π0.5 + SAC) | t008-A-R2-s0 0/20 (results.tsv final_success 0.0) | discard: 그리퍼 채널 D 결함 | ≈2.1 |
| T008b | 2026-11-17 | 그리퍼 분리(D_g=2) 후 재제출 | t008b-A-R2-s0 0/20, t008b-A-R3-s0 0/20 (hand-coded 상한 10/20, 20/20) | discard: 학습 설정 문제, 나머지 7 run 취소 | ≈1.8 |
| T008c | 2026-11-17 | 원인 디버그 + 40k 확인 학습 | c1 0/20(EE–j 5 cm) / c3 0/20 / c2b(RLPD 데모) 1/20, grasp j 50% | 원인 확인. 데모 없이 성공 0 | 1.75 |
| T009 | 2026-11-17 | 가상 손 trace 보상 Stage 0 (ManiSkill 3 state PPO, 1.0M step, seed 3) | PullCube: X3 0%, S3 81.8%, S5 53.0%, S1 82.9%, S6 0%; C0 σ10 87.1% vs S5n10 51.2%. PushCube: X3 0%, S5 81.5%, S1 82.0% | keep(부분 지지): 손 > 물체, RL < controller | ≈1.3 |

## T004 메모
- 영상과 행동은 같은 denoising에서 함께 샘플되므로 일치는 구조상 예상 가능. 남은 질문: 정책 미세조정 전 base world model(Cosmos3-Edge)의 영상은 instruction을 따르는가.
- 4 suite 체크포인트(LIBERO-Goal 포함)인데도 shortcut이 강함.
## T006 메모
- π0.5 H0 shortcut = "j가 자기 scene에서 놓이던 자리(A/B)"의 물체를 집음 (193/250). j가 그 자리면 follow 50/50, 아니면 7%. Cosmos3도 같은 분할에서 5/50 vs 0/200.
- 그리퍼를 j 앞에 두면 instruction이 i여도(H2) j를 집음 → 그 지점에서 언어는 대상을 정하지 않음.

## T008 메모
- 할인 PBRS(Φ(terminal)=0)는 에피소드 합이 정책과 무관하게 ≈ −Φ(s0)라 reach 신호를 주지 못했다. 성공 종료 step 순보상 +0.10 < shortcut 종료 +0.50.
- SAC α0=0.1은 step 보상(~0.01)보다 훨씬 커서 탐색 noise가 frozen π0.5를 망가뜨렸다. α0=1e-3, target entropy −4로 고치면 손이 j 근처(5 cm)까지 간다.
- 같은 조건에서 oracle goal hand-coded controller는 20/20. RL(최선 1/20)은 아직 단순 기하 controller보다 훨씬 낮다.

## T009 메모
- 물체 궤적만 쓰는 보상(X3, HuDOR 방식)은 두 과제 모두 1.0M step에서 0% — 접촉 전 신호가 없음. 단 X3는 정지에도 정답 보상의 54%를 받는 약점이 있다.
- 같은 trace를 따라가는 controller(C0)가 깨끗한 trace 100%, σ=10 mm 87–91%. RL(51–82%)은 어느 조건에서도 넘지 못함 — T008c와 같은 결론.
- S5(손+물체+접촉)가 PullCube에서 seed 1개씩 0%로 붕괴. 손 trace만(S3)이 더 안정.

## T010 (2026-11-18) attention 진단 P0 — keep
| T010 | 2026-11-18 | π0.5 action expert→image attention, LIBERO-Object swap | attention 1위: off-spot 80개 중 spot obj 55 / j 7, 행동 일치 77%, dissociation 3%. spot obj masking Δpos 0.29–0.40 | keep: attention은 위치 shortcut을 그대로 반영. P1 No-Go 권고, 사람 지시로 T011 진행 | ≈1.6 |

## T012 (2026-11-18) LIBERO-Goal 목적지 handover 진단 — keep
| T012 | 2026-11-18 | π0.5, LIBERO-Goal 새 조합 5개(goal15) handover·coverage | H-grasp 15/50(=base), **H-dest 50/50**, H-dest-wrong 학습 목적지 46/50, coverage best-of-16 29% vs 23% | keep: shortcut은 transport 방향, 목적지 waypoint로 해결, 후보 재선택 No-Go | ≈1.6 |

## T013 (2026-11-18) DA-CGC gridworld 관문 — No-Go
| T013 | 2026-11-18 | 8×8 gridworld goal-conditioned DQN, shortcut 재현 관문 | balanced B0 SR_seen 0.02, SER ≈0(held-out 실패는 전부 시간 초과), B6 seen 1.00/unseen 0.00 | 관문 No-Go: shortcut 재현 안 됨, 조합 일반화 실패 | 1.41 |

## T014 (2026-11-18) AAPR 2D pushing — 부분 지지
| T014 | 2026-11-18 | 2D 관성 pushing + SAC, 방법 10 × 바람 4 × seed 5 | ar1 AUC: aapr1 0.81, mh 0.75, M0 0.69, prog 0.65, aug 0.87 / ar1_hi SR: aapr1 0.64, M0 0.28, prog 0.20, aug 0.98 / bias·iid SR ≈1.00 | 부분 지지: one-step AAPR > progress·M0(지속 바람), 그러나 multi-horizon < one-step, 계산량 맞춘 aug가 전 조건 최고 | 1.91 |

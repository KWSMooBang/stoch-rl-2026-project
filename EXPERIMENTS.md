# StochRL — EXPERIMENTS

학습 run은 아직 없음. 아래는 평가·분석 task 결과 (GPU 0).

| task | 날짜 | 내용 | 결과 | 판단 | GPU-h |
|---|---|---|---|---|---|
| T001 | 2026-11-16 | UVA 환경·공식 ckpt 재현 | PushT 0.993 / PushT-M 0.856 / LIBERO-10 0.90 (mean score) | keep → 이후 UVA 중단 | 0.4 |
| T002 | 2026-11-16 | UVA PushT 물리 shift·위치 OOD pilot | 질량 ×5/10/20 성공률 0.54/0.16/0.10, 위치 OOD는 벽·모서리만 상상 붕괴 | 기록용 (UVA 중단) | 3.5 |
| T003 | 2026-11-16 | LeRobot 서브모듈 + Cosmos3 Edge 환경 | LIBERO-10 47/50 = 94.0% (문서 95.2%), chunk b1 0.93 s | keep | 0.7 |
| T004 | 2026-11-16 | LIBERO-Object instruction swap (Cosmos3-Edge-Policy-LIBERO) | swap 91–94%가 원래 물체 i, follow 2%. 같은 샘플 영상≠행동 0.3%. t=0 민감도 행동 1.08 / 영상 1.11 | 가설(영상은 shortcut 안 빠짐) 기각 | 2.8 |

## T004 메모
- 영상과 행동은 같은 denoising에서 함께 샘플되므로 일치는 구조상 예상 가능. 남은 질문: 정책 미세조정 전 base world model(Cosmos3-Edge)의 영상은 instruction을 따르는가.
- 4 suite 체크포인트(LIBERO-Goal 포함)인데도 shortcut이 강함.

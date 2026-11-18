# T019 E04 — 사전 등록 (test를 보기 전에 고정, 2026-11-19)

- 선택: frac=1.0, 후보 무관 입력(E07에서 쓸 수 있는 것; AECand 제외) critic 중 **val 같은 상태 ranking 일치도(conc_1c, 3 seed 평균)** 최고. 동률이면 val conc_or.
  objective별 최선, 입력별 최선도 같은 규칙(val)으로 고른다.
- 관문(E06·E07 사용 조건): 선택된 critic의 **test conc_1c(3 seed 평균) state-bootstrap 95% CI 하한 > 0.5**.
  보조: test oracle p̂ 일치도(15 상태)가 0.5 위인지(방향만), 3 seed 모두 test conc_1c > 0.5인지.
- E07 추가 조건(보고): grad 유한 100%, val grad_align_or > 0.5, ascent 시 score 폭주 정도(ascent_dz) 기록.
- 하이퍼파라미터: 고정(hidden 256, AdamW 3e-4, wd 1e-4, dropout 0.1, 3000 step, batch 256, early stop = val objective loss). test로 바꾸지 않는다.

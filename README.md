# AI에게 투자를 그냥 맡기면 벌어 줄까?

상용 AI와 과거를 학습한 소형 AI 모델 비교. 제1회 라이너 스콜라 연구공모전 출품 논문의 소스코드와 검증 자료다.

## 요약

암호화폐 무기한선물 740종목(상장폐지 71종목 포함)에서 같은 규칙(상·하위 10% 롱숏, 14일 보유, 왕복 비용 0.2%)으로 AI들을 채점했다.

| 지도의 칸 | 모델·입력 | 합산 Sharpe | 누적수익률 |
|---|---|---:|---:|
| 차트만 · 그냥 맡김 | Claude Haiku 5.5 · 차트 이미지(3회 반복) | −0.57~−0.22 | −7.63~−3.49% |
| 지표와 맥락 · 그냥 맡김 | Claude Haiku 5.5 · 지표 37개(3회 반복) | −0.86~−0.75 | −12.59~−11.28% |
| 맥락 그림 · 그냥 맡김 | Claude Haiku 5.5 · 열지도(3회 반복) | −2.96~−2.27 | −13.52~−11.27% |
| 지표와 맥락 · 학습·검증 | LightGBM · 같은 지표 37개 | 2.30 | +48.23% |
| 맥락 그림 · 학습·검증 | 열지도 CNN(10개 초기화 결합) | 3.53 | +62.57% |

설정 확정 뒤 남겨 둔 2026년 3~5월에도 열지도 CNN 2.49, LightGBM 1.84였다(펀딩 반영 2.00, 1.43). 무작위로 고른 대조는 누적수익률 −4.55%·−4.88%였다. 모의 운영은 https://autocrypto.ai-ve.uk/ 에서 공개로 이어 가고 있다.

AI에게 투자를 그냥 맡기면 잃었다. 과거 자료로 학습·검증하는 모델에게 종목 사이의 비교 기준을 담은 값을 줄 때 벌 기회가 생겼다.

## 구성

| 경로 | 내용 |
|---|---|
| `code/hyfe/` | 피처·열지도 생성, CNN 학습(`train_cnn.py`), LightGBM(`pilot_gbm.py`), 앙상블(`zavg.py`), 코호트 백테스트(`cohort.py`), 통계(`perf.py`, `dsr.py`), 모의 운영(`live*.py`, `paper_*.py`) |
| `code/experiments/llm_indicators/` | 지표 27·37개 표 실험: 입력 생성, 지시문(`runner.py`의 `SYSTEM`), 출력 검사, Batch 실행(`run.py`), 평가, 실행 전 계획서 |
| `code/experiments/llm_chart/` | 차트 이미지 실험: 차트 그리기, Batch 실행, 평가, 계획서 |
| `code/experiments/llm_heatmap/` | 열지도 실험: CNN 입력과 같은 열지도 재현(`build_heat.py`, 재현 상관 0.9997은 `verify.json`), 그리기, Batch 실행, 평가, 보조 분석, 계획서, 중단·재개 기록 |
| `code/experiments/holdout_lightgbm/` | 별도 기간 LightGBM 학습과 CNN·LightGBM 공통 평가 |
| `code/experiments/checks_*/` | 심사 검토 때 원자료로 다시 계산한 스크립트와 출력(무작위 대조 비교, 동점 비율, 점수 뒤집기 점검 등) |
| `code/experiments/figures/` | 논문 그림 5·1·3·9를 원자료로 다시 그리는 코드 |
| `results/*/evaluation/` | 조건별 성과(`performance.csv`), LightGBM·CNN 대비 블록 부트스트랩(`paired_sharpe_intervals.csv`), 지표 의존(`reliance*.csv`), 출력 품질, 일별 순자산 |
| `results/scores/` | 종목·판단 시각별 점수 전부. Haiku 네 조건(반복마다, 구간마다), 학습 모델 4구간, 별도 기간 |
| `results/holdout/` | 별도 기간 평가표와 LightGBM 구성원 학습 기록 |
| `results/hyfe_tables/` | 펀딩 반영표, DSR 표, 캔들 CNN 등 시험한 설정 전체 성과 |
| `results/live/` | 모의 운영 장부 2026-10-29 17:00 KST 스냅샷과 GBM 장부 시작 시각 통일 계산 |
| `results/universe/` | 종목표(740종목, 제외 사유, 미학습 종목 표시) |

## 논문 수치와 파일

| 논문 | 파일 |
|---|---|
| 표 6, 표 8, 그림 5·8의 Haiku·LightGBM·CNN 성과 | `results/llm_indicators/evaluation/performance.csv`, `results/llm_chart/...`, `results/llm_heatmap/...` |
| 무작위 대조 대비 차이, 연 변동성, 열지도 점수와 CNN 점수 상관 | `results/llm_heatmap/analysis_out.txt`, `code/experiments/checks_20261028/diff_random_out.txt` |
| 별도 기간 2.49·1.84, 펀딩 반영 2.00·1.43 | `results/holdout/holdout_eval.csv` |
| 펀딩 반영 2.95·1.99, DSR 0.26~0.35 | `results/hyfe_tables/funding_table.csv`, `results/hyfe_tables/dsr_table.csv` |
| 표 11 모의 운영(모든 장부 2026-09-08 13:00 KST 시작으로 통일) | `results/live/live_snapshot_20261029_1700.json`, `results/live/gbm_rebase.json`, `code/experiments/holdout_lightgbm/gbm_rebase.py` |

## 다시 돌리기

- 파이썬 3.12, numpy·pandas·lightgbm·torch·pillow·anthropic.
- 시세는 바이낸스·바이비트 등 거래소의 무기한선물 1분봉을 모아 종목별 4시간봉으로 묶었다(`code/hyfe/bars.py`). 원시 시세는 용량 때문에 넣지 않았다.
- Haiku 실험은 Anthropic Batch API로 실행했다. `run.py`는 API 키를 저장소 밖 파일에서 읽으므로 경로를 자기 환경에 맞게 바꾼다. 비용은 지표 실험 약 21달러, 차트 약 35달러, 열지도 약 83달러였다.
- CNN 학습은 대여 GPU에서 `code/hyfe/train_cnn.py`(별도 기간은 `hold.sh`)로 했다.

## 넣지 않은 것

- 원시 시세와 4시간봉 캐시, 학습된 모델 가중치, 예측 원본(npz): 용량과 형식 때문이다. 예측 점수는 `results/scores/`에 CSV로 모두 풀어 두었다.
- Haiku에게 보낸 이미지(차트·열지도 각 399,057장): 위 코드로 다시 그릴 수 있다.
- 응답 원본 데이터베이스: 점수·상태·시도 횟수를 `results/scores/`에 CSV로 옮겼다.
- API 키와 개인 설정.

저자: 김현호

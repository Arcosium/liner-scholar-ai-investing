"""논문 표 6의 Haiku 세 조건·LightGBM·열지도 CNN 성과를 이 저장소의 구간별 일별 순자산만으로 다시 계산한다.
python3 code/reproduce_main_results.py      (저장소 맨 위에서 실행, numpy·pandas 만 필요)
Sharpe = 일별 순수익률 평균 / 표본 표준편차 × √365, 누적수익률 = 일별 수익 복리, MDD = 이전 고점 대비 최대 하락. 네 구간(A→D)을 이어 붙인 409일.
"""
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[1] / 'results'
PAPER = {  # 논문 표 6 값(Sharpe, 누적수익률 %, MDD %)
    'Haiku 차트 이미지': ('llm_chart/evaluation/LLM_image_r{r}_s{s}_daily_nav.csv', [(-0.22, -3.49, -11.68), (-0.57, -7.63, -13.12), (-0.53, -7.13, -13.76)]),
    'Haiku 지표 37개': ('llm_indicators/evaluation/LLM_context_r{r}_s{s}_daily_nav.csv', [(-0.75, -11.28, -15.39), (-0.86, -12.54, -15.80), (-0.86, -12.59, -15.53)]),
    'Haiku 열지도': ('llm_heatmap/evaluation/LLM_heat_r{r}_s{s}_daily_nav.csv', [(-2.96, -13.52, -14.09), (-2.35, -11.27, -11.56), (-2.27, -11.65, -11.84)]),
    'LightGBM': ('llm_indicators/evaluation/GBM_s{s}_daily_nav.csv', [(2.30, 48.23, -6.69)]),
    '열지도 CNN': ('llm_indicators/evaluation/CNN_s{s}_daily_nav.csv', [(3.53, 62.57, -4.24)]),
}


def pooled(pattern, rep):
    r = pd.concat([pd.read_csv(R / pattern.format(r=rep, s=s), index_col=0)['nav'].pct_change().dropna() for s in (3, 2, 1, 0)])
    eq = (1 + r).cumprod()
    return round(r.mean() / r.std(ddof=1) * np.sqrt(365), 2), round((eq.iloc[-1] - 1) * 100, 2), round((eq / eq.cummax() - 1).min() * 100, 2), len(r)


ok = True
print(f"{'모델':<14}{'반복':>4}{'Sharpe':>9}{'누적%':>9}{'MDD%':>9}{'일수':>6}  논문과 일치")
for name, (pattern, paper) in PAPER.items():
    for rep, want in enumerate(paper):
        got = pooled(pattern, rep)
        same = all(abs(a - b) < 0.006 for a, b in zip(got[:3], want)); ok &= same
        print(f"{name:<14}{rep:>4}{got[0]:>9}{got[1]:>9}{got[2]:>9}{got[3]:>6}  {'예' if same else '아니오 ' + str(want)}")
print('전부 일치' if ok else '불일치 있음')

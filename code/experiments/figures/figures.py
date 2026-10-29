"""v3 원고 그림: 평가 구간별 Sharpe(Haiku·단순 모멘텀 규칙·GBM·CNN). 기존 그림과 같은 글꼴·색 규칙."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

W = Path(__file__).resolve().parents[1]
A = W.parent / 'submissions/assets'
font_manager.fontManager.addfont('/home/arcosium/.local/share/fonts/NotoSansKR-Regular.otf')
plt.rcParams.update({'font.family': 'Noto Sans KR', 'axes.unicode_minus': False, 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 220})
FOLDS = {'3': 'A', '2': 'B', '1': 'C', '0': 'D'}


def fold_sharpe(perf, prefix):
    p = perf[perf.fold.isin(FOLDS) & perf.variant.str.startswith(prefix)]
    return p.groupby('fold').sharpe.mean().rename(index=FOLDS).reindex(list('ABCD'))


haiku = pd.read_csv(W / 'haiku_full_20261028/evaluation/performance.csv')
series = []
image = W / 'haiku_image_20261028/evaluation/performance.csv'
if image.exists():
    series.append(('Haiku · 차트 이미지', fold_sharpe(pd.read_csv(image), 'LLM_'), '#a83b32', '////'))
series.append(('Haiku · 지표 37개', fold_sharpe(haiku, 'LLM_'), '#a83b32', ''))
series.append(('Haiku · 열지도', fold_sharpe(pd.read_csv(W / 'haiku_heatmap_20261029/evaluation/performance.csv'), 'LLM_'), '#a83b32', '....'))
rule = pd.read_csv(W / 'paper_v2_20261028/rule_eval/rule_performance.csv')
series.append(('최근 가격 위치 순 규칙', rule[(rule.rule == 'z_W') & (rule.fold != 'pooled')].set_index('fold').sharpe, '#cccccc', ''))
series += [('LightGBM', fold_sharpe(haiku, 'GBM'), '#888888', ''), ('열지도 CNN', fold_sharpe(haiku, 'CNN'), '#222222', '')]

fig, ax = plt.subplots(figsize=(6.3, 3.1))  # 본문 폭 158mm 와 거의 같은 인쇄 크기
x = np.arange(4); w = 0.8 / len(series)
plt.rcParams['hatch.color'] = 'white'
for i, (name, s, color, hatch) in enumerate(series):
    ax.bar(x + (i - (len(series) - 1) / 2) * w, s.to_numpy(), w * 0.92, label=name, color=color, hatch=hatch, linewidth=0)
ax.axhline(0, color='#555555', lw=.8)
ax.set_xticks(x, ['A 2024.6~8', 'B 2024.12~2025.2', 'C 2025.9~11', 'D 2025.12~2026.2'], fontsize=8)
ax.set_ylabel('연율 Sharpe'); ax.grid(axis='y', color='#dddddd', lw=.5)
ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.3), columnspacing=1.2, handlelength=1.6)
fig.tight_layout(); fig.savefig(A / 'llm_fold_sharpe_v3.png'); plt.close(fig)
print('saved', A / 'llm_fold_sharpe_v3.png', [n for n, *_ in series])

# 그림 1: Haiku 가 본 차트 두 장(같은 판단 시각, 최근 10일 수익률 상위·하위 종목)
from PIL import Image
import json
t = next(t for t in json.loads((W / 'llm_full_20261019/tasks.json').read_text()) if t['arm'] == 'context' and t['repeat'] == 0 and t['fold'] == 1)
x = json.loads((W / 'llm_full_20261019/inputs' / t['input']).read_text())
r = pd.DataFrame(x['rows'], columns=x['columns']).set_index('id').ret_W.rename(index=dict(zip(t['ids'], t['bases'])))
pair = [Image.open(W / f"haiku_image_20261028/images/{t['ts']}/{b}.png").convert('RGB') for b in (r.idxmax(), r.idxmin())]
gap = 24; sheet = Image.new('RGB', (pair[0].width * 2 + gap, pair[0].height), 'white')
sheet.paste(pair[0], (0, 0)); sheet.paste(pair[1], (pair[0].width + gap, 0))
sheet.save(A / 'haiku_chart_examples.png', dpi=(220, 220)); print('saved', A / 'haiku_chart_examples.png', t['ts'])

# 그림 3·8 다시 그리기(원 논문 그림에서 내부 코드명·폴드 번호를 뺀 판). 원본 PNG는 그대로 둔다.
H = Path('/home/arcosium/projects/HYFE_QTPA')
INK, GRAY, RED = '#222222', '#888888', '#a83b32'
neg = lambda v: f'{v:+.2f}'.replace('-', '−')

# 그림 3: 일봉 캔들 CNN(20일 창·20일 보유)의 모멘텀 진단. 원 그림 폴드 0~3 = 구간 D·C·B·A
corr = {'A': -0.72, 'B': -0.44, 'C': -0.44, 'D': 0.32}
spread = {'A': 0.08, 'B': 0.17, 'C': 0.09, 'D': -0.19}
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
for ax, vals, title, lim in ((axes[0], corr, '캔들 CNN 점수와 직전 20일 수익률의 상관', (-.9, .5)),
                             (axes[1], spread, '20일 모멘텀 자체의 상대방향 신호', (-.3, .3))):
    v = np.array(list(vals.values()))
    ax.bar(range(4), v, .58, color=[INK if x > 0 else GRAY for x in v], linewidth=0)
    for i, x in enumerate(v):
        ax.text(i, x + (.03 if x > 0 else -.03) * (lim[1] - lim[0]), neg(x), ha='center', va='bottom' if x > 0 else 'top', fontsize=8)
    ax.axhline(0, color='#555555', lw=.8); ax.set_ylim(*lim)
    ax.set_xticks(range(4), [f'구간 {k}' for k in vals], fontsize=8)
    ax.set_title(title, fontsize=9, pad=10); ax.tick_params(axis='y', labelsize=8)
fig.tight_layout(w_pad=3); fig.savefig(A / 'candle_momentum_v3.png'); plt.close(fig)
print('saved', A / 'candle_momentum_v3.png')

# 그림 8: 열지도 CNN 10개 결합 롱숏과 BTC 단순 보유(4시간봉의 일별 마지막 종가)
btc = pd.read_parquet(H / 'work/bars_hold/4h/BTC.parquet')
btc = btc.assign(date=pd.to_datetime(btc.ts, unit='ms').dt.strftime('%Y-%m-%d')).groupby('date').c.last()
nav = lambda name: pd.Series(json.loads((H / f'results/cohort/ens_heatf10_{name}_cohort_H84.json').read_text())['daily'])
mdd = lambda x: (x / np.maximum.accumulate(np.r_[1.0, x.to_numpy()])[1:] - 1).min() * 100
st, bt, starts = [], [], []
for s in (3, 2, 1, 0):
    n = nav(f'direct_4h_s{s}'); starts.append(sum(len(x) for x in st))
    st.append(n.pct_change().dropna()); bt.append(btc.reindex(n.index).pct_change().dropna())
S = (1 + pd.concat(st).reset_index(drop=True)).cumprod(); B = (1 + pd.concat(bt).reset_index(drop=True)).cumprod()
h = nav('hold_4h_s4'); hb = btc.reindex(h.index); hb = hb / hb.iloc[0]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 3.1), gridspec_kw={'width_ratios': [2.2, 1]})
lab = lambda name, x: f'{name} ×{x.iloc[-1]:.2f}, MDD {mdd(x):.1f}%'.replace('-', '−')
a1.plot(np.r_[0, S.index + 1], np.r_[1, S], color=RED, lw=1.3, label=lab('열지도 CNN 롱숏', S))
a1.plot(np.r_[0, B.index + 1], np.r_[1, B], color=GRAY, lw=1, label=lab('BTC 보유', B))
for k, x0 in zip('ABCD', starts):
    if x0: a1.axvline(x0, color='#bbbbbb', lw=.6, ls=':')
    a1.text(x0 + 4, 1.72, f'구간 {k}', fontsize=7.5, color='#555555')
a1.set_ylim(.4, 1.8); a1.set_xlim(0, len(S) + 2)
a1.set_xlabel('네 평가 구간을 이은 일수(구간 사이는 불연속)', fontsize=8); a1.set_ylabel('누적 자산(시작 = 1)', fontsize=8)
a1.set_title('네 평가 구간 A→D', fontsize=9, loc='left')
a2.plot(range(len(h)), h.to_numpy(), color=RED, lw=1.3, label=lab('열지도 CNN 롱숏', h))
a2.plot(range(len(hb)), hb.to_numpy(), color=GRAY, lw=1, label=lab('BTC 보유', hb))
a2.set_xticks([0, len(h) - 1], [h.index[0].replace('-', '.'), h.index[-1].replace('-', '.')], fontsize=7.5); a2.set_ylim(.9, 1.36)
a2.set_title('남겨 둔 기간 2026년 3~5월', fontsize=9, loc='left')
for ax in (a1, a2):
    ax.axhline(1, color='#555555', lw=.7); ax.grid(axis='y', color='#e4e4e4', lw=.5); ax.tick_params(labelsize=7.5)
    ax.legend(frameon=False, fontsize=7.5, loc='lower left' if ax is a1 else 'upper left', handlelength=1.4)
fig.tight_layout(w_pad=2); fig.savefig(A / 'heatmap_nav_v3.png'); plt.close(fig)
print('saved', A / 'heatmap_nav_v3.png', round(S.iloc[-1], 3), round(mdd(S), 1), round(B.iloc[-1], 3), round(h.iloc[-1], 3), round(hb.iloc[-1], 3))

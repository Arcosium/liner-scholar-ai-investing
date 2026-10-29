"""판단 시각마다 종목별 4시간봉 60개(판단 봉 마감까지) 차트를 그린다. 날짜·종목명·가격 수준 없음."""
import json, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('/home/arcosium/.local/share/fonts/msttcore-arial/Arial.TTF')

D = Path(__file__).resolve().parent
SRC = D.parent / 'llm_full_20261019'
BARS = Path('/home/arcosium/projects/HYFE_QTPA/work/bars_hold/4h')
OUT = D / 'images'
W = 60
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})


def draw(b, path):
    o, h, l, c, v = (b[k].to_numpy(float) for k in 'ohlcv')
    base = o[0]
    o, h, l, c = ((x / base - 1) * 100 for x in (o, h, l, c))  # 창 첫 봉 시가 대비 %
    x = np.arange(len(b))
    fig, (ax, av) = plt.subplots(2, 1, figsize=(5.12, 3.2), dpi=100, sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': .05})
    up = c >= o
    ax.vlines(x, l, h, color='#111111', lw=.7)
    ax.bar(x[up], (c - o)[up], .7, o[up], facecolor='white', edgecolor='#111111', lw=.7)
    ax.bar(x[~up], (c - o)[~up], .7, o[~up], color='#111111', lw=0)
    ax.axhline(0, color='#999999', lw=.5, ls=':')
    ax.grid(axis='y', color='#dddddd', lw=.4); ax.set_ylabel('% from window start')
    av.bar(x, v, .7, color='#999999'); av.set_yticks([]); av.set_ylabel('volume')
    av.set_xticks([0, 15, 30, 45, 59], ['-59', '-44', '-29', '-14', 'now']); av.set_xlabel('4-hour bars (last 60, ~10 days)')
    fig.subplots_adjust(left=.13, right=.98, top=.97, bottom=.15)
    fig.savefig(path); plt.close(fig)


def job(item):
    base, stamps = item
    b = pd.read_parquet(BARS / f'{base}.parquet').set_index('ts')
    for ts in stamps:
        path = OUT / str(ts) / f'{base}.png'
        if path.exists():
            continue
        i = b.index.get_loc(ts)
        w = b.iloc[i - W + 1:i + 1]
        assert len(w) == W and w.index[-1] == ts
        path.parent.mkdir(parents=True, exist_ok=True)
        draw(w.reset_index(), path)
    return base, len(stamps)


if __name__ == '__main__':
    tasks = [t for t in json.loads((SRC / 'tasks.json').read_text()) if t['arm'] == 'chart' and t['repeat'] == 0]
    by_base = {}
    for t in tasks:
        for b in t['bases']:
            by_base.setdefault(b, []).append(t['ts'])
    items = sorted(by_base.items())
    if len(sys.argv) > 1 and sys.argv[1] == 'sample':
        t = tasks[len(tasks) // 2]
        for b in t['bases'][:4]:
            job((b, [t['ts']]))
        print(t['ts'], t['bases'][:4]); sys.exit()
    with Pool(16) as pool:
        for base, n in pool.imap_unordered(job, items):
            pass
    print('done', sum(len(v) for v in by_base.values()), 'images')

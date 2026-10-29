"""열지도 Haiku 점수를 뒤집으면(롱·숏 다리 맞바꿈) 어떻게 되는가. 저장된 예측(npz)만 읽고 같은 cohort 코드로 다시 계산한다.
새 API 호출·학습 없음. 임시 npz 는 scratch 에만 쓴다. 비용 0(총수익)과 0.1%(기본)을 같이 본다.
"""
import os, sys, json, tempfile
import numpy as np, pandas as pd
R = '/home/arcosium/projects/HYFE_QTPA'
sys.path.insert(0, R); os.environ['HYFE_BARS'] = R + '/work/bars_hold'
from hyfe import cohort
from hyfe.perf import stats, nw_pvalue
cohort.B.OUT = R + '/work/bars_hold'
W = '/home/arcosium/projects/라이너스콜라_공모전/_workspace/'
TMP = '/tmp/claude-1000/-home-arcosium/09e0f88a-ddf7-4141-b9b1-476c7474e2ea/scratchpad/review2/rev'
os.makedirs(TMP, exist_ok=True)
SETS = {'열지도 Haiku': (W + 'haiku_heatmap_20261029/evaluation/', 'LLM_heat'),
        '차트 Haiku': (W + 'haiku_image_20261028/evaluation/', 'LLM_image'),
        '지표37 Haiku': (W + 'haiku_full_20261028/evaluation/', 'LLM_context')}
rows = []
for name, (folder, var) in SETS.items():
    for rep in range(3):
        for mode in ('원래', '뒤집음'):
            for cost in (0.001, 0.0):
                pieces = []
                for fold in (3, 2, 1, 0):
                    z = np.load(f'{folder}{var}_r{rep}_s{fold}_pred.npz', allow_pickle=True)
                    p = z['p'].copy()
                    if mode == '뒤집음':
                        p[:, 1] = -p[:, 1]
                    path = f'{TMP}/{var}_r{rep}_s{fold}_{mode}.npz'
                    np.savez_compressed(path, ts=z['ts'], base=z['base'], p=p)
                    r, eq = cohort.run(path, '4h', 84, q=.1, cost=cost)
                    pieces.append(r)
                pooled = pd.concat(pieces); eq = (1 + pooled).cumprod()
                st = stats(pooled, eq, 14)
                rows.append(dict(set=name, rep=rep, mode=mode, cost=cost, sharpe=float(pooled.mean() / pooled.std(ddof=1) * np.sqrt(365)),
                                 t=st['t'], ret_pct=float((eq.iloc[-1] - 1) * 100), vol_pct=float(pooled.std() * np.sqrt(365) * 100), days=len(pooled)))
                print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df.to_csv('/home/arcosium/projects/라이너스콜라_공모전/_workspace/paper_v3_review_20261029/reverse_check.csv', index=False)
print(df.round(2).to_string(index=False))

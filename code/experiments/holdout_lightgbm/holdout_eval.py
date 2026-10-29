"""별도 기간 2026-03~05 평가: 열지도 CNN 10시드 앙상블과 LightGBM v2 를 같은 코호트 코드로(14일 보유, 상·하위 10%, 왕복 0.2%, 펀딩 반영 따로).
python3 holdout_eval.py [cnn|all]"""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
R = Path('/home/arcosium/projects/HYFE_QTPA'); D = Path(__file__).resolve().parent
sys.path.insert(0, str(R)); os.environ['HYFE_BARS'] = str(R / 'work/bars_hold'); os.chdir(R)
from hyfe import cohort
from hyfe.perf import stats
cohort.B.OUT = str(R / 'work/bars_hold')
END = pd.Timestamp('2026-06-01').value // 10**6
SRC = {'CNN': R / 'work/results/ens_heatf10_hold_4h_s4_pred.npz', 'GBM': D / 'gbm/ens_v2hold_4h_s0_pred.npz'}
rows = []
for name in (['CNN'] if sys.argv[1:] == ['cnn'] else ['CNN', 'GBM']):
    z = np.load(SRC[name], allow_pickle=True); m = z['ts'] < END
    tmp = D / f'_{name}_0305_pred.npz'; np.savez_compressed(tmp, **{k: z[k][m] for k in z.files})
    for fund in ('', str(R / 'work/funding.parquet')):
        r, _ = cohort.run(str(tmp), '4h', 84, q=.1, cost=.001, funding=fund)
        r = r[(r.index >= '2026-03-01') & (r.index < '2026-06-01')]; eq = (1 + r).cumprod()   # 논문 CNN 2.49 와 같은 방식(신호 기간 92일 일별 수익)
        v = stats(r, eq, 14)
        rows.append({'model': name, 'funding': bool(fund), 'days': len(r), 'sharpe': round(float(r.mean() / r.std(ddof=1) * np.sqrt(365)), 2), 't': v['t'], 'p': v['p'],
                     'return_pct': round(float((eq.iloc[-1] - 1) * 100), 2), 'mdd_pct': round(float((eq / eq.cummax() - 1).min() * 100), 2),
                     'rows': int(m.sum()), 'first': str(pd.to_datetime(z['ts'][m].min(), unit='ms'))[:10], 'last': str(pd.to_datetime(z['ts'][m].max(), unit='ms'))[:10]})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(D / 'holdout_eval.csv', index=False)

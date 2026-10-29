"""무작위 대조(CNN 점수 셔플)의 합산 NW t와 구간별 NW t. 열지도 Haiku 의 합산 t(−3.82~−3.35)와 같은 잣대로 비교한다."""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/arcosium/projects/HYFE_QTPA')
from hyfe.perf import nw_pvalue
C='/home/arcosium/projects/HYFE_QTPA/results/cohort/'
for k in (1,2):
    parts=[pd.Series(json.load(open(f'{C}ens_heatf10_direct_4h_s{s}_cohort_H84_shuf{k}.json'))['daily']).pct_change().dropna() for s in (3,2,1,0)]
    r=pd.concat(parts)
    t,_=nw_pvalue(r,14)
    print('무작위',k,'days',len(r),'NW t(lag14)',round(t,2),'Sharpe',round(r.mean()/r.std(ddof=1)*np.sqrt(365),2))

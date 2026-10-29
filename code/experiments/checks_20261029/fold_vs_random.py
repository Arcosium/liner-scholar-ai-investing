"""구간별 열지도 Haiku(3회 평균) 대 무작위 대조 Sharpe, 합산 변동성, 비용 몫. 저장된 순자산만 읽는다."""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/arcosium/projects/HYFE_QTPA')
from hyfe.perf import nw_pvalue
W='/home/arcosium/projects/라이너스콜라_공모전/_workspace/'
C='/home/arcosium/projects/HYFE_QTPA/results/cohort/'
sh=lambda r: r.mean()/r.std(ddof=1)*np.sqrt(365)
rand={k:{s:pd.Series(json.load(open(f'{C}ens_heatf10_direct_4h_s{s}_cohort_H84_shuf{k}.json'))['daily']).pct_change().dropna() for s in (3,2,1,0)} for k in (1,2)}
names={3:'A',2:'B',1:'C',0:'D'}
print('구간별 Sharpe: 열지도 Haiku(r0,r1,r2,평균) | 무작위(1,2)')
for s in (3,2,1,0):
    hs=[sh(pd.read_csv(f'{W}haiku_heatmap_20261029/evaluation/LLM_heat_r{r}_s{s}_daily_nav.csv',index_col=0)['nav'].pct_change().dropna()) for r in range(3)]
    print(names[s], np.round(hs,2), round(np.mean(hs),2), '|', [round(sh(rand[k][s]),2) for k in (1,2)], 'days', len(rand[1][s]))
for k in (1,2):
    r=pd.concat(rand[k].values()); eq=(1+r).cumprod()
    print('무작위',k,'합산 Sharpe',round(sh(r),2),'누적',round((eq.iloc[-1]-1)*100,2),'연변동성',round(r.std()*np.sqrt(365)*100,2),'days',len(r))
# 37개·차트 연 변동성
for nm,f,v in (('차트',W+'haiku_image_20261028/evaluation/','LLM_image'),('지표37',W+'haiku_full_20261028/evaluation/','LLM_context'),('열지도',W+'haiku_heatmap_20261029/evaluation/','LLM_heat')):
    out=[]
    for r in range(3):
        x=pd.concat([pd.read_csv(f'{f}{v}_r{r}_s{s}_daily_nav.csv',index_col=0)['nav'].pct_change().dropna() for s in (3,2,1,0)])
        out.append(round(x.std()*np.sqrt(365)*100,2))
    print(nm,'연 변동성',out)

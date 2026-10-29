import json, sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/arcosium/projects/HYFE_QTPA')
from hyfe.perf import nw_pvalue
W='/home/arcosium/projects/라이너스콜라_공모전/_workspace/'
C='/home/arcosium/projects/HYFE_QTPA/results/cohort/'
def rand(sh):
    out={}
    for s in (3,2,1,0):
        nav=pd.Series(json.load(open(f'{C}ens_heatf10_direct_4h_s{s}_cohort_H84_shuf{sh}.json'))['daily']); out[s]=nav.pct_change().dropna()
    return out
res=[]
for name,folder,var in (('차트 이미지',W+'haiku_image_20261028/evaluation/','LLM_image'),('지표 37개',W+'haiku_full_20261028/evaluation/','LLM_context')):
    for rep in range(3):
        for sh in (1,2):
            R=rand(sh); diffs=[]; mine=[]
            for s in (3,2,1,0):
                nav=pd.read_csv(f'{folder}{var}_r{rep}_s{s}_daily_nav.csv',index_col=0)['nav']
                h=nav.pct_change().dropna()
                r=R[s]
                n=min(len(h),len(r)); 
                diffs.append(pd.Series(h.to_numpy()[:n]-r.to_numpy()[:n]))
                mine.append(h)
            d=pd.concat(diffs,ignore_index=True)
            t,p=nw_pvalue(d,14)
            res.append((name,rep,sh,len(d),d.sum()*100,t))
            h=pd.concat(mine)
df=pd.DataFrame(res,columns=['조건','반복','셔플','일수','일별차이합(%p, 단순합)','NW t'])
print(df.round(2).to_string(index=False))
print(df.groupby('조건')['NW t'].agg(['min','max']).round(2))

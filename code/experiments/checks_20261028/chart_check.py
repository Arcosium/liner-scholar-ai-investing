import json, sqlite3, numpy as np, pandas as pd
from pathlib import Path
W = Path('/home/arcosium/projects/라이너스콜라_공모전/_workspace')
SRC = W/'llm_full_20261019'; D = W/'haiku_image_20261028'
FEATURES = ['ret_W','ret_q','ret_5','rsi','z_W','pos_W','vol_W','amihud','v_last_rel']
tasks = json.loads((SRC/'tasks.json').read_text())
con = sqlite3.connect(f'file:{D/"results.sqlite"}?mode=ro', uri=True)
got = {i:(s,st) for i,s,st in con.execute('SELECT id, score, state FROM results')}
print('rows', len(got), 'states', pd.Series([v[1] for v in got.values()]).value_counts().to_dict())
chart = [t for t in tasks if t['arm']=='chart' and t['repeat']==0]
ctx = {(t['fold'],t['ts']):t for t in tasks if t['arm']=='context' and t['repeat']==0}
rel_raw, rel_id, tie = [], [], []
for t in chart:
    c = ctx[(t['fold'], t['ts'])]
    p = json.loads((SRC/'inputs'/c['input']).read_text())
    x = pd.DataFrame(p['rows'], columns=p['columns']).set_index('id').rename(index=dict(zip(c['ids'], c['bases'])))
    for rep in range(3):
        v = []
        for aid, base in zip(t['ids'], t['bases']):
            s, st = got[f"s{t['fold']}_{t['ts']}_{aid}_r{rep}"]
            if st=='ok': v.append((aid, base, s))
        n = len(v)
        order = sorted(v, key=lambda a:(-a[2], a[0]))
        rk = pd.Series({b:(n-k) for k,(a,b,s) in enumerate(order)})
        raw = pd.Series({b:s for a,b,s in v})
        rel_raw.append({'fold':t['fold'], **{f: raw.corr(x[f].reindex(raw.index), method='spearman') for f in FEATURES}})
        rel_id.append({'fold':t['fold'], **{f: rk.corr(x[f].reindex(rk.index), method='spearman') for f in FEATURES}})
        # leg composition: top/bottom q by pct rank of ID-broken rank
        pr = rk.rank(pct=True, method='average')
        lm = pr >= 0.9; sm = pr <= 0.1
        sc = raw.reindex(rk.index)
        out = {'fold':t['fold'], 'n':n, 'distinct':sc.nunique()}
        for name, m in (('long',lm),('short',sm)):
            leg = sc[m]; nleg = len(leg)
            edge = leg.min() if name=='long' else leg.max()
            grp = (sc==edge).sum(); inleg = (leg==edge).sum()
            split = grp > inleg
            out[name+'_n'] = nleg
            out[name+'_edge_inleg'] = inleg if split else 0   # ID로 가려낸 종목 수
            out[name+'_edge_group'] = grp
            out[name+'_split'] = split
            # 점수가 같은 종목이 leg 크기 이상인 묶음에 속한 종목 비율(leg 전체가 한 묶음 안)
        tie.append(out)
for nm, rel, fn in (('raw(동점 평균순위, reliance.csv 방식)', rel_raw,'chart_reliance_raw.csv'), ('ID순 순위(지표 실험과 같은 잣대)', rel_id,'chart_reliance_idrank.csv')):
    d = pd.DataFrame(rel).groupby('fold')[FEATURES].mean().round(3)
    d.to_csv(fn)
    print(nm); print(d.to_string())
T = pd.DataFrame(tie)
print('distinct median', T.distinct.median(), 'mean', T.distinct.mean(), 'n obs', len(T), 'n assets median', T.n.median())
for name in ('long','short'):
    print(name, 'leg n median', T[name+'_n'].median(), 'split share', T[name+'_split'].mean().round(3),
          'ID로 갈린 종목/leg 크기 평균', (T[name+'_edge_inleg']/T[name+'_n']).mean().round(3),
          'edge group size median', T[name+'_edge_group'].median())
print('분포 ID로 갈린 비율 분위수 long', (T.long_edge_inleg/T.long_n).quantile([.1,.5,.9]).round(3).to_dict())
print('분포 ID로 갈린 비율 분위수 short', (T.short_edge_inleg/T.short_n).quantile([.1,.5,.9]).round(3).to_dict())
T.to_csv('/tmp/claude-1000/-home-arcosium/09e0f88a-ddf7-4141-b9b1-476c7474e2ea/scratchpad/review_v3/chart_tie.csv', index=False)

"""열지도 Haiku 결과 보조 분석(evaluate.py 뒤에 실행, 결과 파일만 읽는다). python3 analysis.py > analysis_out.txt

1 무작위 대조(CNN 점수 셔플 1·2) 대비 일별 차이 NW t   2 연 변동성   3 동점·ID 순 비율(차트 조건 chart_check.py 와 같은 잣대)
4 지표 37개와의 스피어만 상관(점수 원값)   5 열지도 CNN 점수·14일 미래수익과의 시각별 스피어만 상관
"""
import json, sqlite3, sys
from pathlib import Path
import numpy as np, pandas as pd

D = Path(__file__).resolve().parent
SRC = D.parent / 'llm_full_20261019'
H = Path('/home/arcosium/projects/HYFE_QTPA')
sys.path.insert(0, str(H))
from hyfe.perf import nw_pvalue

E = D / 'evaluation'
FOLDS = (3, 2, 1, 0)

# 1·2
def daily(path):
    return pd.read_csv(path, index_col=0)['nav'].pct_change().dropna()

rand = {sh: {s: pd.Series(json.load(open(H / f'results/cohort/ens_heatf10_direct_4h_s{s}_cohort_H84_shuf{sh}.json'))['daily']).pct_change().dropna()
             for s in FOLDS} for sh in (1, 2)}
rows = []
for rep in range(3):
    mine = {s: daily(E / f'LLM_heat_r{rep}_s{s}_daily_nav.csv') for s in FOLDS}
    vol = pd.concat(mine.values()).std() * np.sqrt(365) * 100
    for sh in (1, 2):
        d = pd.concat([pd.Series(mine[s].to_numpy()[:n] - rand[sh][s].to_numpy()[:n]) for s in FOLDS
                       for n in [min(len(mine[s]), len(rand[sh][s]))]], ignore_index=True)
        t, p = nw_pvalue(d, 14)
        rows.append({'repeat': rep, 'shuffle': sh, 'days': len(d), 'diff_sum_pctp': d.sum() * 100, 'nw_t': t, 'vol_pct': vol})
rv = [pd.concat(rand[sh].values()).std() * np.sqrt(365) * 100 for sh in (1, 2)]
print('== 무작위 대조 대비 일별 차이 (NW 14)'); print(pd.DataFrame(rows).round(2).to_string(index=False)); print('무작위 연 변동성 %', np.round(rv, 2))

# 3·4·5
tasks = json.loads((SRC / 'tasks.json').read_text())
con = sqlite3.connect(f'file:{D / "results.sqlite"}?mode=ro', uri=True)
got = {i: (s, st) for i, s, st in con.execute('SELECT id, score, state FROM results')}
ctx = {(t['fold'], t['ts']): t for t in tasks if t['arm'] == 'context' and t['repeat'] == 0}
cnn = {}
for s in FOLDS:
    q = np.load(H / f'work/results/ens_heatf10_direct_4h_s{s}_pred.npz', allow_pickle=True)
    cnn[s] = pd.DataFrame({'ts': q['ts'], 'base': q['base'].astype(str), 'cnn': q['p'][:, 1] - q['p'][:, 2], 'fwd': q['fwd']}).set_index(['ts', 'base'])
rel, tie = [], []
for t in (t for t in tasks if t['arm'] == 'chart' and t['repeat'] == 0):
    c = ctx[(t['fold'], t['ts'])]
    p = json.loads((SRC / 'inputs' / c['input']).read_text())
    x = pd.DataFrame(p['rows'], columns=p['columns']).set_index('id').rename(index=dict(zip(c['ids'], c['bases'])))
    feats = [k for k in p['columns'] if k != 'id']
    z = cnn[t['fold']].xs(t['ts'], level='ts')
    for rep in range(3):
        v = [(aid, base, got[f"s{t['fold']}_{t['ts']}_{aid}_r{rep}"][0]) for aid, base in zip(t['ids'], t['bases'])
             if got[f"s{t['fold']}_{t['ts']}_{aid}_r{rep}"][1] == 'ok']
        raw = pd.Series({b: s for a, b, s in v})
        rel.append({'repeat': rep, 'fold': t['fold'], **{f: raw.corr(x[f].reindex(raw.index), method='spearman') for f in feats},
                    'cnn_score': raw.corr(z.cnn.reindex(raw.index), method='spearman'), 'fwd_14d': raw.corr(z.fwd.reindex(raw.index), method='spearman')})
        order = sorted(v, key=lambda a: (-a[2], a[0])); n = len(order)
        rk = pd.Series({b: n - k for k, (a, b, s) in enumerate(order)}); pr = rk.rank(pct=True); sc = raw.reindex(rk.index)
        out = {'repeat': rep, 'fold': t['fold'], 'n': n, 'distinct': sc.nunique()}
        for name, m in (('long', pr >= 0.9), ('short', pr <= 0.1)):
            leg = sc[m]; edge = leg.min() if name == 'long' else leg.max()
            grp, inleg = (sc == edge).sum(), (leg == edge).sum()
            out[name + '_id_share'] = (inleg if grp > inleg else 0) / len(leg)
        tie.append(out)
R = pd.DataFrame(rel); T = pd.DataFrame(tie)
R.to_csv(E / 'reliance_all37.csv', index=False)
print('\n== 동점: 시각당 서로 다른 점수 중앙값', T.distinct.median(), '/ 종목 중앙값', T.n.median())
print('ID 순으로 갈린 종목 비율(다리 평균) long', round(T.long_id_share.mean(), 3), 'short', round(T.short_id_share.mean(), 3))
m = R.drop(columns='repeat').groupby('fold').mean()
print('\n== 스피어만 상관 구간별 평균(세 반복), |평균| 큰 순 15개')
print(m[m.abs().mean().sort_values(ascending=False).index[:15]].round(3).to_string())
print('\n== 열지도 CNN 점수·14일 미래수익과의 상관, 반복별 전체 평균')
print(R.groupby('repeat')[['cnn_score', 'fwd_14d']].mean().round(4).to_string())
print(R.groupby('fold')[['cnn_score', 'fwd_14d']].mean().round(4).to_string())
for rep in range(3):   # 시각별 IC 의 NW t(시각 단위, 84봉 겹침 보정 위해 lag 84)
    ic = R[R.repeat == rep].fwd_14d.dropna().reset_index(drop=True)
    print('IC rep', rep, 'mean', round(ic.mean(), 4), 'NW t(lag84)', round(nw_pvalue(ic, 84)[0], 2), 'n', len(ic))

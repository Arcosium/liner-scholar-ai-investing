"""차트 이미지 조건 평가(PROTOCOL.txt). python3 evaluate.py [반복 번호...]  기본 0 1 2.

종목별 점수 → 시각마다 점수 내림차순·동점은 익명 ID 오름차순으로 순위 → 원 실험과 같은 cohort·Sharpe·부트스트랩.
9회 모두 실패한 종목은 그 시각 순위에서 빼고, 유효 종목이 20개 미만인 시각은 새 포지션을 열지 않는다.
"""
from pathlib import Path
import json, os, sqlite3, sys
import numpy as np, pandas as pd

D = Path(__file__).resolve().parent
SRC = D.parent / 'llm_full_20261019'
REF = D.parent / 'haiku_full_20261028/evaluation/daily_returns.csv'
R = Path('/home/arcosium/projects/HYFE_QTPA')
FEATURES = ['ret_W', 'ret_q', 'ret_5', 'rsi', 'z_W', 'pos_W', 'vol_W', 'amihud', 'v_last_rel']
MIN_ASSETS = 20


def main(reps):
    sys.path.insert(0, str(R)); os.environ['HYFE_BARS'] = str(R / 'work/bars_hold')
    from hyfe import cohort
    from hyfe.perf import stats
    cohort.B.OUT = str(R / 'work/bars_hold')
    out = D / 'evaluation'; out.mkdir(exist_ok=True)
    tasks = json.loads((SRC / 'tasks.json').read_text())
    con = sqlite3.connect(f'file:{D / "results.sqlite"}?mode=ro', uri=True)
    got = {i: (s, st) for i, s, st in con.execute('SELECT id, score, state FROM results')}

    rows, skipped, quality = [], [], []
    for rep in reps:
        for t in (t for t in tasks if t['arm'] == 'chart' and t['repeat'] == 0):
            valid = []
            for aid, base in zip(t['ids'], t['bases']):
                s, st = got.get(f"s{t['fold']}_{t['ts']}_{aid}_r{rep}", (None, None))
                if st == 'ok':
                    valid.append((aid, base, s))
                elif st != 'failed':
                    raise RuntimeError(f'repeat {rep} not finished: {t["ts"]} {aid} {st}')
            n_all = len(t['ids'])
            quality.append({'repeat': rep, 'fold': t['fold'], 'ts': t['ts'], 'assets': n_all, 'valid': len(valid),
                            'distinct_scores': len({s for *_, s in valid}), 'ties_beyond_first': len(valid) - len({s for *_, s in valid})})
            if len(valid) < MIN_ASSETS:
                skipped.append({'repeat': rep, 'ts': t['ts'], 'valid': len(valid)}); continue
            order = sorted(valid, key=lambda v: (-v[2], v[0]))
            rows += [{'repeat': rep, 'fold': t['fold'], 'ts': t['ts'], 'base': b, 'aid': a, 'raw': s, 'score': len(order) - k}
                     for k, (a, b, s) in enumerate(order)]
    g = pd.DataFrame(rows); q = pd.DataFrame(quality)
    q.to_csv(out / 'output_quality.csv', index=False)

    summaries, daily = [], {}

    def summarize(name, fold, r, eq):
        v = stats(r, eq, 14)
        v.update(variant=name, fold=fold, sharpe=float(r.mean() / r.std(ddof=1) * np.sqrt(365)),
                 return_pct=float((eq.iloc[-1] - 1) * 100), mdd=float((eq / eq.cummax() - 1).min() * 100))
        summaries.append(v)

    for rep in reps:
        name, pieces = f'LLM_image_r{rep}', []
        for fold in [3, 2, 1, 0]:
            z = g[(g.repeat == rep) & (g.fold == fold)].sort_values(['ts', 'base'])
            path = out / f'{name}_s{fold}_pred.npz'
            np.savez_compressed(path, ts=z.ts.to_numpy(), base=z.base.to_numpy(str), p=np.c_[np.zeros(len(z)), z.score.to_numpy(float), np.zeros(len(z))])
            r, eq = cohort.run(str(path), '4h', 84, q=.1, cost=.001)
            summarize(name, str(fold), r, eq); pieces.append(r)
            eq.to_csv(out / f'{name}_s{fold}_daily_nav.csv', header=['nav'])
        pooled = pd.concat(pieces); summarize(name, 'pooled_observation_days', pooled, (1 + pooled).cumprod())
        unique = pooled.groupby(level=0).mean(); daily[name] = unique
        summarize(name, 'unique_calendar_days', unique, (1 + unique).cumprod())
    pd.DataFrame(summaries).to_csv(out / 'performance.csv', index=False)

    ref = pd.read_csv(REF, index_col=0, parse_dates=True)
    d = pd.concat([pd.DataFrame(daily), ref], axis=1); d.to_csv(out / 'daily_returns.csv')
    paired, rng = [], np.random.default_rng(20261019)
    for name in daily:
        for other in ['GBM', 'CNN']:
            x = d[[name, other]].dropna().to_numpy(); n = len(x); values = []
            for _ in range(2000):
                starts = rng.integers(0, n, size=int(np.ceil(n / 14)))
                a = x[((starts[:, None] + np.arange(14)) % n).ravel()[:n]]
                sh = a.mean(0) / a.std(0, ddof=1) * np.sqrt(365); values.append(sh[0] - sh[1])
            paired.append({'variant': name, 'reference': other, 'days': n,
                           'sharpe_difference': float((x.mean(0) / x.std(0, ddof=1) * np.sqrt(365)) @ np.array([1, -1])),
                           'bootstrap_95_low': float(np.quantile(values, .025)), 'bootstrap_95_high': float(np.quantile(values, .975))})
    pd.DataFrame(paired).to_csv(out / 'paired_sharpe_intervals.csv', index=False)

    # 점수가 어떤 지표를 따랐나: 같은 시각 맥락 조건 입력(37개 지표)과 종목명으로 잇고 시각마다 스피어만 상관.
    rel = []
    for t in (t for t in tasks if t['arm'] == 'context' and t['repeat'] == 0):
        p = json.loads((SRC / 'inputs' / t['input']).read_text())
        x = pd.DataFrame(p['rows'], columns=p['columns']).set_index('id').rename(index=dict(zip(t['ids'], t['bases'])))
        for rep, z in g[g.ts == t['ts']].groupby('repeat'):
            s = z.set_index('base').raw
            rel.append({'repeat': rep, 'fold': t['fold'], **{f: s.corr(x[f].reindex(s.index), method='spearman') for f in FEATURES}})
    pd.DataFrame(rel).groupby('fold')[FEATURES].mean().round(2).to_csv(out / 'reliance.csv')

    calls = con.execute("SELECT count(*), sum(attempts), sum(state='failed') FROM results WHERE substr(id,-2) IN (%s)" % ','.join(f"'r{r}'" for r in reps)).fetchone()
    spent = con.execute('SELECT sum(cost) FROM batches').fetchone()[0]
    summary = {'repeats': reps, 'requests': calls[0], 'attempts': calls[1], 'failed_after_9': calls[2],
               'times_skipped_lt20': skipped, 'cost_usd_all_repeats': round(spent, 4),
               'mean_distinct_share': float((q.distinct_scores / q.valid).mean()), 'times_all_equal': int((q.distinct_scores == 1).sum()),
               'top_scores': g.raw.round(2).value_counts().head(8).to_dict()}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    p = pd.DataFrame(summaries)
    print(p[p.fold.isin(['3', '2', '1', '0', 'unique_calendar_days', 'pooled_observation_days'])][['variant', 'fold', 'sharpe', 'return_pct', 'mdd']].round(2).to_string(index=False))
    print(pd.DataFrame(paired).round(2).to_string(index=False))
    print(pd.read_csv(out / 'reliance.csv').to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main([int(a) for a in sys.argv[1:]] or [0, 1, 2])

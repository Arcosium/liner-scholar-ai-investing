"""Paired retrospective replay of archived ten-seed refresh policies.

No fitting or deployment. Writes only to the explicit output directory.
A three-month refresh is compared with keeping the initial model six months.
Both policies carry pre-refresh cohorts to maturity. Historical price/missing
price conventions and transaction costs match hyfe.cohort.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hyfe.cohort import run
from hyfe.paper_rules import ensemble
from hyfe.perf import nw_pvalue


def read_prediction(path):
    with np.load(path, allow_pickle=True) as z:
        d = pd.DataFrame({'ts': z['ts'], 'base': z['base'].astype(str),
                          'score': z['p'][:, 1] - z['p'][:, 2]})
    assert not d.duplicated(['ts', 'base']).any()
    assert np.isfinite(d.score).all()
    return d


def ten_seeds(root, split):
    merged = None
    files = []
    for seed in range(10):
        suffix = '' if seed == 0 else f'_sd{seed}'
        p = root / f'hold_i1_heatf{suffix}_4h_W60_H84_s{split}_pred.npz'
        meta_path = Path(str(p).replace('_pred.npz', '.json'))
        meta = json.loads(meta_path.read_text())
        args = meta['args']
        assert all(args[k] == v for k, v in {
            'seed': seed, 'model': 'i1', 'render': 'heatf',
            'res': '4h', 'W': 60, 'H': 84, 'label': 'relbin',
            'stride': 1, 'epochs': 8, 'cap': 5000000,
            'val': '2025-12' if split == 4 else '2026-03',
            'test': '2026-03' if split == 4 else '2026-06',
            'test_end': '2026-09'}.items()), 'Unexpected training configuration'
        d = read_prediction(p).rename(columns={'score': f's{seed}'})
        if merged is None:
            merged = d
        else:
            before = len(merged)
            merged = merged.merge(d, on=['ts', 'base'], validate='one_to_one')
            assert len(merged) == before == len(d), 'Seed coverage differs'
        files.append({'file': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
        files.append({'file': str(meta_path), 'sha256': hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                      'training_args': args})
    merged['score'] = ensemble(merged, [f's{s}' for s in range(10)])
    return merged[['ts', 'base', 'score']], files


def save_pred(d, p):
    s = d.score.to_numpy()
    np.savez_compressed(p, ts=d.ts.to_numpy(), base=d.base.to_numpy(dtype=str),
                        p=np.c_[np.zeros(len(d)), np.maximum(s, 0), np.maximum(-s, 0)])


def summary(r):
    r = pd.Series(r).astype(float)
    eq = np.r_[1., (1+r).cumprod().to_numpy()]
    t, p = nw_pvalue(r, 14)
    return {'days': len(r), 'return_pct': float((eq[-1]-1)*100),
            'sharpe': float(r.mean()/r.std(ddof=1)*np.sqrt(365)),
            'mdd_pct': float((eq/np.maximum.accumulate(eq)-1).min()*100),
            'nw_t': t, 'nw_p_positive': p}


def paired(a, b, block, reps=10000):
    """Paired circular moving-block CI. Seed and blocks fixed before analysis."""
    a, b = np.asarray(a), np.asarray(b)
    assert len(a) == len(b) and len(a) > block
    rng = np.random.default_rng(20261015 + block)
    starts = rng.integers(0, len(a), (reps, int(np.ceil(len(a)/block))))
    ix = ((starts[:, :, None] + np.arange(block)) % len(a)).reshape(reps, -1)[:, :len(a)]
    aa, bb = a[ix], b[ix]
    delta = (aa-bb).mean(axis=1)*10000
    ds = (aa.mean(1)/aa.std(1, ddof=1)-bb.mean(1)/bb.std(1, ddof=1))*np.sqrt(365)
    t, p = nw_pvalue(a-b, 14)
    return {'block_days': block, 'reps': reps, 'daily_delta_bp': float((a-b).mean()*10000),
            'daily_delta_bp_ci95': np.quantile(delta, [.025,.975]).tolist(),
            'sharpe_delta': summary(a)['sharpe']-summary(b)['sharpe'],
            'sharpe_delta_ci95': np.quantile(ds, [.025,.975]).tolist(),
            'paired_nw_t': t, 'paired_nw_p_two_sided': float(2*min(p,1-p))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out', required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    root=Path('work/results'); old, f4=ten_seeds(root,4); new,f5=ten_seeds(root,5)
    switch=int(pd.Timestamp('2026-06-01').value//1_000_000)
    assert old.ts.min() < switch == new.ts.min()
    assert old.ts.max() == new.ts.max()
    cutoff=pd.to_datetime(old.ts.max(),unit='ms').normalize()
    manifest={'source_files':f4+f5,'switch':'2026-06-01','last_signal_day':str(cutoff.date()),
              'seeds':list(range(10)),'H_bars':84,'bar_hours':4,'round_trip_cost':.002,
              'bootstrap_seed':20261015,'funding':False,
              'scope':'Retrospective archived-prediction replay, not new model fitting or forward live performance',
              'limitations':['One refresh date; no 1- or 2-month policy is measured.',
                'Refresh changes training/validation vintage and universe; not a pure model-age intervention.',
                'Common-row sensitivity equalizes tradable names, not historical feature/training universes.',
                'June-August 2026 was already examined in September; this is not an untouched holdout.',
                'Predictions stop August 17 due to 14-day label horizon; primary evaluation stops there.',
                'Historical same-bar close entry and missing-price convention; not an executable fill study.']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[];comparisons=[];series={}
    for universe in ['native','common_rows']:
        o,n=old.copy(),new.copy()
        if universe=='common_rows':
            keys=o[o.ts>=switch][['ts','base']].merge(n[['ts','base']],on=['ts','base'],validate='one_to_one')
            o=pd.concat([o[o.ts<switch],o.merge(keys,on=['ts','base'],validate='one_to_one')],ignore_index=True)
            n=n.merge(keys,on=['ts','base'],validate='one_to_one')
        policies={'keep6':o,'refresh3':pd.concat([o[o.ts<switch],n],ignore_index=True)}
        for policy,d in policies.items():
            assert not d.duplicated(['ts','base']).any()
            p=out/f'{universe}_{policy}_pred.npz';save_pred(d,p)
            r,eq=run(str(p),'4h',84)
            series[f'{universe}_{policy}']=r
            for period,start,end in [('full','2026-03-02',str(cutoff.date())),
                                     ('paired','2026-06-01',str(cutoff.date())),
                                     ('tail_included','2026-06-01','2026-08-31')]:
                rr=r.loc[start:end]
                rows.append({'universe':universe,'policy':policy,'period':period,
                             'start':str(rr.index.min().date()),'end':str(rr.index.max().date()),**summary(rr)})
            print(universe,policy,summary(r.loc['2026-06-01':cutoff]),flush=True)
        x=series[f'{universe}_refresh3']; y=series[f'{universe}_keep6']
        assert x.index.equals(y.index)
        assert np.allclose(x.loc[:'2026-05-31'],y.loc[:'2026-05-31'])
        for period,end in [('paired',str(cutoff.date())),('tail_included','2026-08-31')]:
            for block in [7,14,28]:
                comparisons.append({'universe':universe,'period':period,
                                    **paired(x.loc['2026-06-01':end],y.loc['2026-06-01':end],block)})
    frame=pd.DataFrame(series);frame.index.name='date';frame.to_csv(out/'daily_returns.csv')
    pd.DataFrame(rows).to_csv(out/'policy_summary.csv',index=False)
    (out/'paired_inference.json').write_text(json.dumps(comparisons,indent=2))
    monthly=[]
    for name,r in series.items():
        for m,g in r.groupby(r.index.to_period('M')):
            monthly.append({'policy':name,'month':str(m),**summary(g)})
    pd.DataFrame(monthly).to_csv(out/'monthly.csv',index=False)
    # Descriptive age profile from all five original three-month folds.
    ages=[]
    for split,start in [(3,'2024-06-01'),(2,'2024-12-01'),(1,'2025-09-01'),(0,'2025-12-01'),(4,'2026-03-01')]:
        p=root/(f'ens_heatf10_direct_4h_s{split}_cohort_H84.json' if split<4 else 'ens_heatf10_hold_4h_s4_cohort_H84.json')
        d=json.loads(p.read_text());eq=pd.Series(d['daily'],dtype=float);eq.index=pd.to_datetime(eq.index);r=eq.pct_change().dropna()
        for age in range(1,4):
            s=pd.Timestamp(start)+pd.DateOffset(months=age-1);e=s+pd.DateOffset(months=1)
            g=r[(r.index>=s)&(r.index<e)]
            ages.append({'fold':split,'deployment_month':age,'calendar_month':str(s.to_period('M')),**summary(g)})
    pd.DataFrame(ages).to_csv(out/'descriptive_age.csv',index=False)
    print('DONE',out,flush=True)


if __name__=='__main__':main()

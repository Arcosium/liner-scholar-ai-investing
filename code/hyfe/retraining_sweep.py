"""Run and evaluate the frozen 1..6-calendar-month retraining experiment.

Fits use the existing paper trainer unchanged. A manifest fixes all inputs and
hyperparameters before any new results. No operation changes live releases.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic(path, data):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)


def month_age(ts, origin):
    d=pd.DatetimeIndex(pd.to_datetime(ts,unit='ms'));o=pd.Timestamp(origin)
    return (d.year-o.year)*12+d.month-o.month


def schedule_month(age, interval):
    if interval not in range(1,7):raise ValueError('interval must be 1..6')
    a=np.asarray(age)
    if (a<0).any():raise ValueError('negative model age')
    return (a//interval)*interval


def job_command(job, run):
    return [sys.executable,'-m','hyfe.train_cnn','--res','4h','--W','60','--H','84',
            '--model','i1','--render','heatf','--label','relbin','--stride','1',
            '--bases','@'+str(run/'universe.txt'),'--seed',str(job['seed']),
            '--cap','5000000','--epochs','8','--bs','256','--start','2023-01',
            '--val',job['val'],'--test',job['test'],'--test_end',job['test_end'],
            '--out',str(run/'models'/f"{job['id']}.json"),'--save_pred']


def verify_inputs(plan):
    for p,h in plan['inputs'].items():
        if sha(p)!=h:raise ValueError(f'Input changed: {p}')
    for p,h in plan.get('code',{}).items():
        if sha(p)!=h:raise ValueError(f'Code changed: {p}')


def complete(job,run):
    p=run/'models'/f"{job['id']}.json";done=run/'models'/f"{job['id']}.complete.json"
    if not done.exists():return False
    record=json.loads(done.read_text())
    for ext,h in record['sha256'].items():
        path=Path(str(p).replace('.json',ext))
        if not path.exists() or sha(path)!=h:raise ValueError('Completed model changed')
    return True


def fit(args):
    run=Path(args.run);plan=json.loads((run/'plan.json').read_text());verify_inputs(plan)
    (run/'models').mkdir(exist_ok=True);(run/'logs').mkdir(exist_ok=True)
    jobs=[j for j in plan['jobs'] if j['seed'] in args.seeds]
    if args.job:jobs=[j for j in jobs if j['id']==args.job]
    env=dict(os.environ,HYFE_BARS='work/bars_hold',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for j in jobs:
        if complete(j,run):continue
        start=time.time();atomic(run/f'worker_{args.seeds[0]}.json',{'job':j['id'],'state':'training','started_at':start})
        print('START',j['id'],flush=True)
        with (run/'logs'/f"{j['id']}.log").open('a') as log:
            proc=subprocess.run(job_command(j,run),env=env,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:raise RuntimeError(f"Training failed: {j['id']}; see log")
        p=run/'models'/f"{j['id']}.json";meta=json.loads(p.read_text());a=meta['args']
        for k in ['seed','val','test','test_end']:
            if a[k]!=j[k]:raise ValueError('Wrong model metadata')
        if meta['n_train']<=0 or meta['n_val']<=0:raise ValueError('Empty fit')
        pred=Path(str(p).replace('.json','_pred.npz'))
        with np.load(pred,allow_pickle=True) as z:
            if not np.isfinite(z['p']).all():raise ValueError('Nonfinite predictions')
            if z['ts'].min()<pd.Timestamp(j['test']).value//10**6:raise ValueError('Prediction before deployment')
        atomic(run/'models'/f"{j['id']}.complete.json",{
            'wall_seconds':time.time()-start,'completed_at':time.time(),
            'plan_sha256':sha(run/'plan.json'),
            'sha256':{ext:sha(Path(str(p).replace('.json',ext))) for ext in ['.json','.pt','_pred.npz']}})
        print('DONE',j['id'],round(time.time()-start),flush=True)
    atomic(run/f'worker_{args.seeds[0]}.json',{'state':'complete','completed_at':time.time()})


def holm(p):
    p=np.asarray(p);order=np.argsort(p);adjusted=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]));out=np.empty(len(p));out[order]=adjusted;return out


def evaluate(args):
    from hyfe.paper_rules import ensemble
    from hyfe.cohort import run as replay
    from hyfe.retraining_study import read_prediction,save_pred,summary,paired
    run=Path(args.run);plan=json.loads((run/'plan.json').read_text());verify_inputs(plan)
    if not all(complete(j,run) for j in plan['jobs']):raise ValueError('All 120 fits must complete; no partial-seed analysis')
    out=run/'analysis';out.mkdir(exist_ok=True);months=pd.date_range(plan['evaluation_start'],periods=12,freq='MS')
    vintage={};cost=[]
    for i,m in enumerate(months):
        frame=None
        for seed in plan['seeds']:
            ident=m.strftime('%Y%m')+f'_s{seed}';p=run/'models'/f'{ident}_pred.npz'
            x=read_prediction(p).rename(columns={'score':f's{seed}'})
            if frame is None:frame=x
            else:
                n=len(frame);frame=frame.merge(x,on=['ts','base'],validate='one_to_one')
                if len(frame)!=n or len(x)!=n:raise ValueError('Seed coverage differs')
            meta=json.loads((run/'models'/f'{ident}.json').read_text());rec=json.loads((run/'models'/f'{ident}.complete.json').read_text())
            cost.append({'month':i,'seed':seed,'train_seconds':meta['sec'],'wall_seconds':rec['wall_seconds'],'epochs':len(meta['hist']),'n_train':meta['n_train']})
        frame['score']=ensemble(frame,[f's{s}' for s in plan['seeds']]);vintage[i]=frame
    daily={};rows=[];seed_rows=[];reference_keys=None
    end=pd.Timestamp(plan['evaluation_end_exclusive']);start=pd.Timestamp(plan['evaluation_start'])
    for k in plan['periods_months']:
        chosen=[]
        for age in range(12):
            v=int(schedule_month(age,k));d=vintage[v];mask=month_age(d.ts,plan['evaluation_start'])==age;chosen.append(d[mask])
        frame=pd.concat(chosen,ignore_index=True).sort_values(['ts','base']).reset_index(drop=True)
        keys=frame[['ts','base']]
        if reference_keys is None:reference_keys=keys
        elif not reference_keys.equals(keys):raise ValueError('Schedules have different tradable rows')
        p=out/f'interval{k}_pred.npz';save_pred(frame,p)
        r,eq=replay(str(p),'4h',84)
        # run() drops the first day's return; restore it from initial unit equity.
        r=pd.concat([pd.Series([eq.iloc[0]-1],index=eq.index[:1]),r]);r=r[(r.index>=start)&(r.index<end)]
        if len(r)!=365:raise ValueError('Expected 365 complete daily observations')
        daily[f'{k}m']=r
        jobs=pd.DataFrame(cost);selected=jobs[jobs.month.isin(range(0,12,k))]
        rows.append({'interval_months':k,'releases':len(range(0,12,k)),**summary(r),
                     'training_gpu_hours':float(selected.train_seconds.sum()/3600)})
        for seed in plan['seeds']:
            d=frame[['ts','base',f's{seed}']].rename(columns={f's{seed}':'score'})
            sp=out/f'interval{k}_seed{seed}_pred.npz';save_pred(d,sp);sr,se=replay(str(sp),'4h',84)
            sr=pd.concat([pd.Series([se.iloc[0]-1],index=se.index[:1]),sr]);sr=sr[(sr.index>=start)&(sr.index<end)]
            seed_rows.append({'interval_months':k,'seed':seed,**summary(sr)})
            sp.unlink()
        print('EVALUATED',k,rows[-1],flush=True)
    df=pd.DataFrame(daily);df.to_csv(out/'daily_returns.csv',index_label='date')
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False);pd.DataFrame(seed_rows).to_csv(out/'seed_summary.csv',index=False)
    comparisons=[]
    for block in [7,14,28]:
        group=[]
        for a,b in itertools.combinations(plan['periods_months'],2):
            group.append({'a_months':a,'b_months':b,**paired(df[f'{a}m'],df[f'{b}m'],block)})
        for g,p in zip(group,holm([g['paired_nw_p_two_sided'] for g in group])):g['holm_p_15_tests']=float(p)
        comparisons.extend(group)
    atomic(out/'paired_inference.json',comparisons);pd.DataFrame(cost).to_csv(out/'training_cost.csv',index=False)
    monthly=[]
    for col in df:
        for m,r in df[col].groupby(df.index.to_period('M')):monthly.append({'interval':col,'month':str(m),**summary(r)})
    pd.DataFrame(monthly).to_csv(out/'monthly.csv',index=False)
    atomic(out/'complete.json',{'completed_at':time.time(),'plan_sha256':sha(run/'plan.json'),'days':len(df),'models':len(plan['jobs'])})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['fit','evaluate']);ap.add_argument('--run',default='work/retraining_1to6_20261016');ap.add_argument('--seeds',type=int,nargs='+',default=list(range(10)));ap.add_argument('--job')
    a=ap.parse_args();(fit if a.mode=='fit' else evaluate)(a)


if __name__=='__main__':main()

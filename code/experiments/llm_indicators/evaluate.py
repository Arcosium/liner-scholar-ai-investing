from pathlib import Path
import hashlib,json,os,sqlite3,sys
import numpy as np
import pandas as pd
D=Path(__file__).resolve().parent
R=Path('/home/arcosium/projects/HYFE_QTPA')
def main():
    # Recheck inputs before scoring; changed historical data invalidates a comparison.
    for path,h in json.loads((D/'source_hashes.json').read_text()).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h:raise RuntimeError('source changed: '+path)
    sys.path.insert(0,str(R));os.environ['HYFE_BARS']=str(R/'work/bars_hold')
    from hyfe import cohort
    from hyfe.perf import stats
    cohort.B.OUT=str(R/'work/bars_hold')
    panel=pd.read_parquet(D/'panel.parquet');tasks=json.loads((D/'tasks.json').read_text())
    con=sqlite3.connect(D/'responses_v2.sqlite');results={i:json.loads(r) for i,r in con.execute("SELECT id,ranking FROM results WHERE state='ok'")}
    out=D/'evaluation';out.mkdir(exist_ok=True)
    quality=[]
    for ident,raw in con.execute("SELECT id,response FROM results WHERE state='ok'"):
        values=list(json.loads(json.loads(raw)['choices'][0]['message']['content'])['scores'].values())
        quality.append({'id':ident,'assets':len(values),'distinct_scores':len(set(values)),'all_equal':len(set(values))==1,'ties_beyond_first':len(values)-len(set(values))})
    pd.DataFrame(quality).to_csv(out/'output_quality.csv',index=False)
    variants={'GBM':panel[['fold','ts','base','GBM']].rename(columns={'GBM':'score'}),'CNN':panel[['fold','ts','base','CNN']].rename(columns={'CNN':'score'})}
    missing={}
    for arm in ['chart','context']:
        for rep in range(3):
            name=f'LLM_{arm}_r{rep}';subset=[t for t in tasks if t['arm']==arm and t['repeat']==rep]
            absent=[t['id'] for t in subset if t['id'] not in results]
            if absent:missing[name]=absent;continue
            rows=[]
            for t in subset:
                ranks={ident:len(t['ids'])-i for i,ident in enumerate(results[t['id']])}
                rows.extend({'fold':t['fold'],'ts':t['ts'],'base':base,'score':ranks[ident]} for ident,base in zip(t['ids'],t['bases']))
            variants[name]=pd.DataFrame(rows)
    summaries=[];daily={}
    def summarize(name,fold,r,eq):
        v=stats(r,eq,14);v.update(variant=name,fold=fold,sharpe=float(r.mean()/r.std(ddof=1)*np.sqrt(365)),return_pct=float((eq.iloc[-1]-1)*100),mdd=float((eq/eq.cummax()-1).min()*100));summaries.append(v)
    for name,g in variants.items():
        pieces=[]
        for fold in [3,2,1,0]:
            z=g[g.fold==fold].sort_values(['ts','base']);score=z.score.to_numpy(float)
            path=out/f'{name}_s{fold}_pred.npz'
            np.savez_compressed(path,ts=z.ts.to_numpy(),base=z.base.to_numpy(str),p=np.c_[np.zeros(len(z)),score,np.zeros(len(z))])
            r,eq=cohort.run(str(path),'4h',84,q=.1,cost=.001)
            summarize(name,str(fold),r,eq);pieces.append(r)
            eq.to_csv(out/f'{name}_s{fold}_daily_nav.csv',header=['nav'])
        pooled=pd.concat(pieces);summarize(name,'pooled_observation_days',pooled,(1+pooled).cumprod())
        # Calendar-date sensitivity: average concurrent fold returns, not duplicated days.
        unique=pooled.groupby(level=0).mean();daily[name]=unique
        summarize(name,'unique_calendar_days',unique,(1+unique).cumprod())
    pd.DataFrame(summaries).to_csv(out/'performance.csv',index=False)
    pd.DataFrame(daily).to_csv(out/'daily_returns.csv')
    paired=[]
    rng=np.random.default_rng(20261019)
    for name in daily:
        if not name.startswith('LLM'):continue
        for ref in ['GBM','CNN']:
            x=pd.concat([daily[name],daily[ref]],axis=1).dropna().to_numpy();n=len(x)
            values=[]
            for _ in range(2000):
                starts=rng.integers(0,n,size=int(np.ceil(n/14)))
                ix=((starts[:,None]+np.arange(14))%n).ravel()[:n];a=x[ix]
                sh=a.mean(axis=0)/a.std(axis=0,ddof=1)*np.sqrt(365)
                values.append(sh[0]-sh[1])
            paired.append({'variant':name,'reference':ref,'days':n,'sharpe_difference':float((x.mean(0)/x.std(0,ddof=1)*np.sqrt(365))@np.array([1,-1])),'bootstrap_95_low':float(np.quantile(values,.025)),'bootstrap_95_high':float(np.quantile(values,.975))})
    pd.DataFrame(paired).to_csv(out/'paired_sharpe_intervals.csv',index=False)
    (out/'missing.json').write_text(json.dumps(missing,indent=2))
    (out/'README.txt').write_text('동일 종목·시각·보유·비용 조건으로 계산. pooled_observation_days는 폴드 경계의 중복 달력을 포함한다. unique_calendar_days는 같은 날짜의 폴드 수익을 평균한다. Sharpe는 일 수익률 표본표준편차, 연율 sqrt(365), 무위험수익률 0. 가격 결측은 기존 cohort의 다리 내 재정규화 관례를 따른다. 판단봉 종가 체결 가정이며 실제 주문 지연·펀딩은 제외한다. 14일 원형 블록 부트스트랩 2,000회 신뢰구간은 탐색적이며 다중비교 보정 전이다. 실패가 있는 LLM 조건은 전체 성과표에서 제외하고 missing.json에 남긴다. 반복 호출은 독립 시장 표본이 아니다.\n')
if __name__=='__main__':main()

from pathlib import Path
import ast, hashlib, json, sys
import numpy as np
import pandas as pd
D=Path(__file__).resolve().parent
R=Path('/home/arcosium/projects/HYFE_QTPA')
src=(R/'hyfe/features.py').read_text()
tree=ast.parse(src)
parts=['import numpy as np\nimport pandas as pd\nfrom numpy.lib.stride_tricks import sliding_window_view\nEPS=1e-12']
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name in ('_rsi','make'):
        s=ast.get_source_segment(src,node)
        if node.name=='make':
            s=s.replace('if n < W + H + 2:', 'if n < W + 2:')
            s=s.replace('idx = _idx(ts, n, W, H, stride, align)', 'idx = np.arange(W, n + 1, stride)')
            s=s.replace('fwd = logc[e + H] - logc[e]', 'fwd = np.zeros(len(e))')
            s=s.replace('sigH = sig1[e] * np.sqrt(H)', 'sigH = sig1[e]')
            s=s.replace('label = label_of(fwd, sigH, k, mode, fixed)', 'label = np.zeros(len(e), dtype=int)')
            s=s.replace('return f\n','return f\n')
        parts.append(s)
    elif isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='FEATURES' for x in node.targets):
        parts.append(ast.get_source_segment(src,node))
(D/'causal_features.py').write_text('\n\n'.join(parts)+'\n')
from causal_features import make,FEATURES
sys.path.insert(0,str(R))
from hyfe.features import make as original
b=pd.read_parquet(R/'work/bars_hold/4h/BTC.parquet')
f=make(b,60,84,stride=1,bars_per_30d=180).set_index('ts')[FEATURES]
p=make(b.iloc[:1000],60,84,stride=1,bars_per_30d=180).set_index('ts')[FEATURES]
np.testing.assert_allclose(f.loc[p.index],p,rtol=1e-10,atol=1e-10)
o=original(b,60,84,stride=1,bars_per_30d=180).set_index('ts')[FEATURES]
np.testing.assert_allclose(f.loc[o.index],o,rtol=1e-10,atol=1e-10)
manifest={}; grids=[]
for s in [3,2,1,0]:
    pair=[]
    for name,stem in [('GBM','ens_v2_4h'),('CNN','ens_heatf10_direct_4h')]:
        path=R/f'work/results/{stem}_s{s}_pred.npz'
        manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        z=np.load(path,allow_pickle=False)
        pair.append(pd.DataFrame({'ts':z['ts'],'base':z['base'].astype(str),name:z['p'][:,1]-z['p'][:,2]}))
    g=pair[0].merge(pair[1],on=['ts','base'],validate='one_to_one');g['fold']=s;grids.append(g)
g=pd.concat(grids,ignore_index=True)
frames=[]
for base in sorted(g.base.unique()):
    path=R/f'work/bars_hold/4h/{base}.parquet'
    manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    b=pd.read_parquet(path)
    f=make(b,60,84,stride=1,bars_per_30d=180)[['ts']+FEATURES]; f['base']=base
    frames.append(f[f.ts.isin(g.loc[g.base==base,'ts'])])
f=pd.concat(frames,ignore_index=True)
g=g.merge(f,on=['ts','base'],how='left',validate='many_to_one')
assert g[FEATURES].notna().all().all(), 'missing causal features: do not shrink universe silently'
xs=['ret_W','ret_q','vol_W','vol_ratio','rng_mean','qv_rel30','amihud','v_last_rel','rsi','z_W']
for c in xs:g['xs_'+c]=g.groupby(['fold','ts'])[c].rank(pct=True)
g.to_parquet(D/'panel.parquet',index=False)
(D/'inputs').mkdir(exist_ok=True)
tasks=[]
for (fold,ts), frame in g.groupby(['fold','ts'],sort=True):
    frame=frame.sort_values('base').sample(frac=1,random_state=int(ts//14400000)%2**32)
    ids=[f'A{i:03}' for i in range(len(frame))]
    for arm,cols in [('chart',FEATURES),('context',FEATURES+['xs_'+c for c in xs])]:
        payload={'columns':['id']+cols,'rows':[[i]+[float(format(x,'.6g')) for x in row] for i,row in zip(ids,frame[cols].to_numpy())]}
        key=f's{fold}_{ts}_{arm}'
        (D/'inputs'/f'{key}.json').write_text(json.dumps(payload,separators=(',',':')))
        for repeat in range(3):tasks.append({'id':f'{key}_r{repeat}','input':key+'.json','fold':int(fold),'ts':int(ts),'arm':arm,'repeat':repeat,'ids':ids,'bases':frame.base.tolist()})
# Interleave all arms/repeats at each chronological decision; no performance feedback.
tasks.sort(key=lambda t:(-t['fold'],t['ts'],t['repeat'],t['arm']))
(D/'tasks.json').write_text(json.dumps(tasks,separators=(',',':')))
(D/'source_hashes.json').write_text(json.dumps(manifest,indent=2))
(D/'validation.json').write_text(json.dumps({'causal_prefix_invariance':True,'matches_original_27_features':True,'timestamps':int(g.groupby(['fold','ts']).ngroups),'tasks':len(tasks),'assets':g.base.nunique(),'rows':len(g),'min_cross_section':int(g.groupby(['fold','ts']).size().min()),'max_cross_section':int(g.groupby(['fold','ts']).size().max())},indent=2))
print((D/'validation.json').read_text(),flush=True)

"""열지도 CNN 이 본 48행×60봉 열지도를 판단 시각·종목마다 다시 만든다(HYFE train_cnn heatf 그대로, CPU).
표준화 통계는 원 학습과 같이 그 구간 학습 집합에서 구한다. python3 build_heat.py [구간...]  기본 0 1 2 3
verify: 구간 0 에서 원 모델 가중치로 다시 예측해 저장된 예측과 비교한다.
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

D = Path(__file__).resolve().parent
SRC = D.parent / 'llm_full_20261019'
R = Path('/home/arcosium/projects/HYFE_QTPA')
os.environ['HYFE_BARS'] = str(R / 'work/bars_hold')
os.chdir(R); sys.path.insert(0, str(R))
import torch
from hyfe import train_cnn as T, features as F
from hyfe.pilot_gbm import add_xs, XS_FEATS

FEATS = list(F.FEATURES) + list(XS_FEATS)
OUT = D / 'heat'


def fold(s, verify=False):
    a = json.load(open(R / f'work/results/full_i1_heatf_liq_4h_W60_H84_s{s}.json'))['args']
    assert a['render'] == 'heatf' and a['train_months'] == 0 and a['px'] == 3 and a['img_h'] == 96
    u = pd.read_csv('work/universe.csv'); u = u[u.exclude.fillna('') == '']; holdout = set(u[u.holdout].base)
    bases = [l.strip() for l in open(a['bases'][1:]) if l.strip()]
    wins = T.Windows(bases, a['res'], a['W'], a['H'], a['k'], 'cpu', a['start'], a['test_end'], a['label'], a['fixed'], need_feats=True, stride=a['stride'] or None)
    ix = add_xs(wins.idx); wins.idx = ix
    emb = (a['W'] + a['H']) * 240 * 60_000
    tr = ix[~ix.base.isin(holdout) & (ix.ts < T.ms(a['val']) - emb)]
    assert len(tr) <= a['cap']   # 표본 추출이 없었으므로 통계가 시드와 무관
    mu, sd = tr[FEATS].mean(), tr[FEATS].std() + 1e-9
    te = ix[(ix.ts >= T.ms(a['test'])) & (ix.ts < T.ms(a['test_end']))]
    pairs = pd.DataFrame([(t['ts'], b) for t in json.loads((SRC / 'tasks.json').read_text())
                          if t['arm'] == 'chart' and t['repeat'] == 0 and t['fold'] == s for b in t['bases']], columns=['ts', 'base'])
    sel = pairs.merge(te, on=['ts', 'base'], how='left', indicator=True)
    missing = int((sel._merge != 'both').sum()); sel = sel[sel._merge == 'both']
    print(f's{s}: tasks {len(pairs)} matched {len(sel)} missing {missing} train {len(tr)}', flush=True)
    f = lambda d: torch.tensor(((d[FEATS] - mu) / sd).to_numpy(np.float32))
    mats = []
    for i in range(0, len(sel), 4096):
        z = sel.iloc[i:i + 4096]
        x = wins.heatf(torch.tensor(z.g.to_numpy()), f(z))
        mats.append((x[:, 0, ::2, ::3].clamp(0, 1) * 255).round().to(torch.uint8).numpy())   # 96×180 → 48×60(행 2px·봉 3px 반복 되돌림)
    m = np.concatenate(mats)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f's{s}.npz', ts=sel.ts.to_numpy(), base=sel.base.to_numpy(str), m=m, missing=missing)
    if verify:   # 원 모델(시드 0) 가중치로 다시 예측해 저장된 예측과 비교
        model = T.SmallCNN(in_ch=1); model.load_state_dict(torch.load(R / f'work/models/full_i1_heatf_liq_4h_W60_H84_s{s}.pt', map_location='cpu')); model.eval()
        z = sel.sample(3000, random_state=0)
        with torch.no_grad():
            p = torch.softmax(model(wins.heatf(torch.tensor(z.g.to_numpy()), f(z)), None), 1).numpy()
        q = np.load(R / f'work/results/full_i1_heatf_liq_4h_W60_H84_s{s}_pred.npz', allow_pickle=True)
        ref = pd.DataFrame({'ts': q['ts'], 'base': q['base'].astype(str), 'ref': q['p'][:, 1] - q['p'][:, 2]})
        c = z[['ts', 'base']].assign(new=p[:, 1] - p[:, 2]).merge(ref, on=['ts', 'base'])
        r = float(np.corrcoef(c.new, c.ref)[0, 1]); err = float((c.new - c.ref).abs().max())
        print(f'verify s{s}: n {len(c)} corr {r:.6f} max_abs_diff {err:.5f}', flush=True)
        (D / 'verify.json').write_text(json.dumps({'fold': s, 'n': len(c), 'corr': r, 'max_abs_diff': err}, indent=1))
        assert r > 0.999, 'heatmap does not reproduce the CNN input'


if __name__ == '__main__':
    folds = [int(x) for x in sys.argv[1:]] or [0, 1, 2, 3]
    for s in folds:
        fold(s, verify=(s == 0))

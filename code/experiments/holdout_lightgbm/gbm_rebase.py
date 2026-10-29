"""GBM 모의 장부를 CNN 과 같은 시작 시각(2026-09-08 13:00 KST = 04:00 UTC)으로 맞춘다(10/29 사장 지시).
시작 시각에 열려 있던 포지션을 그 시각 4시간봉 마감가로 평가한 순자산 M0 를 시작자산으로 삼는다.
  평가 순자산 기준(현재 평가):  M_T / M0 - 1
  정산 기준(장부 기준):          (E_T - (M0 - 1)) / M0 - 1   (시작 전에 생긴 평가손익을 첫 보유분에서 뺀다)
평가식은 AutoCrypto web/app.py 와 같다: M = E + Σ size·부호·(가격/진입가 - 1). 진입가 검산도 한다.
python3 gbm_rebase.py  → gbm_rebase.json
"""
import json
from pathlib import Path
import pandas as pd

LIVE = Path('/home/arcosium/projects/HYFE_QTPA/work/live'); HIST = Path('/home/arcosium/projects/CryptoBars/data/history')
B = 1788840000000   # 2026-09-08 04:00 UTC
_cache = {}


def close_at(base, ts):
    """ts 에 끝나는 4시간봉의 마감가 = ts-1분에 시작한 1분봉 종가."""
    if base not in _cache:
        fs = [HIST / f'base={base}/part-{m}.parquet' for m in ('2026-09',)]
        _cache[base] = pd.concat(pd.read_parquet(f, columns=['ts', 'close']) for f in fs if f.exists()).drop_duplicates('ts').set_index('ts').close.sort_index()
    s = _cache[base]; s = s[s.index <= ts - 60_000]
    return float(s.iloc[-1]) if len(s) else None


out = {}
for key in ('v2', 'v1', 'v2d'):
    tr = [json.loads(l) for l in open(LIVE / f'dir_{key}_trades.jsonl')]
    st = json.loads((LIVE / 'state.json').read_text())['dir'][key]
    opened = [t for t in tr if t['entry_ts'] <= B < t['exit_ts']] + [dict(v, base=k) for k, v in (st.get('positions') or {}).items() if v.get('entry_ts', B + 1) <= B]
    closed_before = [t for t in tr if t['exit_ts'] <= B]
    E_B = 1.0 + sum(t['pnl'] for t in closed_before)
    u, chk = 0.0, []
    for t in opened:
        px = close_at(t['base'], B); sign = 1 if t['side'] == 'long' else -1
        u += float(t['size']) * sign * (px / t['entry'] - 1)
        e = close_at(t['base'], t['entry_ts'])
        if e: chk.append(abs(e / t['entry'] - 1))
    M0 = E_B + u
    out[key] = {'base_ts': B, 'E_B': E_B, 'M0': M0, 'open_at_base': len(opened), 'closed_before': len(closed_before),
                'entry_check_max_rel_err': max(chk) if chk else None, 'entry_check_n': len(chk)}
    print(key, out[key])
(Path(__file__).parent / 'gbm_rebase.json').write_text(json.dumps(out, indent=1))

"""heat/s*.npz(48행×60봉 열지도 수치) → 종목별 PNG. 행 이름 칸은 한 번만 그려 붙인다. 날짜·종목명·가격 수준 없음.
python3 render.py sample  # 견본 4장 → sample_sheet.png
"""
import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFont

D = Path(__file__).resolve().parent
OUT = D / 'images'
sys.path.insert(0, '/home/arcosium/projects/HYFE_QTPA')
from hyfe import features as F
from hyfe.pilot_gbm import XS_COLS

ROWS = (['open', 'high', 'low', 'close', 'volume / window mean', 'volume / own 30d avg', 'hour (UTC)', 'weekday',
         'xs rank: 24h return', 'xs rank: 24h volatility', 'xs rank: rel. volume'] + list(F.FEATURES) + [f'xs_{c}' for c in XS_COLS])
GAP_AFTER = {3, 5, 7, 10, 37}          # 행 묶음: 가격 4 · 거래량 2 · 시간 2 · 순위 3 · 지표 27 · 횡단면 10
CW, CH, GAP, LEFT, TOP, BOTTOM, RIGHT = 8, 10, 4, 132, 16, 28, 10
BG = (217, 230, 242)                   # 칸 사이·여백은 옅은 파랑(흰 칸과 구분)
FONT = ImageFont.truetype('/home/arcosium/.local/share/fonts/msttcore-arial/Arial.TTF', 9)
assert len(ROWS) == 48
Y = [TOP + i * CH + GAP * sum(g < i for g in GAP_AFTER) for i in range(48)]
SIZE = (LEFT + 60 * CW + RIGHT, Y[-1] + CH + BOTTOM)


def frame():
    im = Image.new('RGB', SIZE, BG); d = ImageDraw.Draw(im)
    d.text((LEFT, 2), 'dark = low, light = high', fill=(40, 40, 40), font=FONT)
    for y, name in zip(Y, ROWS):
        d.text((LEFT - 5, y + CH / 2), name, fill=(20, 20, 20), font=FONT, anchor='rm')
    for k, lab in ((0, '-59'), (15, '-44'), (30, '-29'), (45, '-14'), (59, 'now')):
        d.text((LEFT + k * CW + CW / 2, Y[-1] + CH + 3), lab, fill=(20, 20, 20), font=FONT, anchor='ma')
    d.text((LEFT + 30 * CW, Y[-1] + CH + 15), '4-hour bars (last 60, ~10 days)', fill=(20, 20, 20), font=FONT, anchor='ma')
    return im


BASE = frame()


def draw(m):
    im = BASE.copy(); a = np.asarray(im).copy()
    for i, y in enumerate(Y):
        a[y:y + CH, LEFT:LEFT + 60 * CW] = np.repeat(m[i], CW)[None, :, None]
    return Image.fromarray(a)


def job(item):
    s, k, n_parts = item
    z = np.load(D / f'heat/s{s}.npz'); n = 0
    for ts, base, m in list(zip(z['ts'], z['base'], z['m']))[k::n_parts]:
        p = OUT / str(ts) / f'{base}.png'
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix('.tmp')   # 다 쓴 뒤 이름을 바꿔 중단 시 빈 파일이 남지 않게(RECOVERY_20261029.txt)
        draw(m).save(tmp, format='PNG', optimize=True); tmp.replace(p); n += 1
    return s, n


if __name__ == '__main__':
    if sys.argv[1:] == ['sample']:
        z = np.load(D / 'heat/s0.npz'); ims = [draw(m) for m in z['m'][:4]]
        sheet = Image.new('RGB', (SIZE[0] * 2, SIZE[1] * 2), 'white')
        for k, im in enumerate(ims):
            sheet.paste(im, ((k % 2) * SIZE[0], (k // 2) * SIZE[1]))
        sheet.save(D / 'sample_sheet.png'); ims[0].save(D / 'sample_one.png'); print(SIZE); sys.exit()
    folds = [int(x) for x in sys.argv[1:]] or [0, 1, 2, 3]
    items = [(s, k, 4) for s in folds for k in range(4)]   # 구간마다 4조각, 16프로세스
    with Pool(16) as pool:
        for s, n in pool.imap_unordered(job, items):
            print('fold', s, n, 'images', flush=True)
    print('done', flush=True)

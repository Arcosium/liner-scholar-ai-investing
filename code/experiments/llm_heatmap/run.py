"""Claude Haiku 5.5 열지도 이미지 실험(PROTOCOL.txt). 종목마다 열지도 한 장 → 점수 하나.

python3 run.py smoke   # 실제 이미지 40건 동기 호출(결과에 넣지 않음)
python3 run.py run     # Batch API 전체 실행, 중단 뒤 다시 실행하면 이어서
"""
from pathlib import Path
import base64, json, math, re, sqlite3, sys, time
import anthropic

D = Path(__file__).resolve().parent
SRC = D.parent / 'llm_full_20261019'
IMG = D / 'images'
MODEL = 'claude-haiku-5-5'
REPEATS = 3
MAX_ATTEMPTS = 9
BATCH = 5000          # 이미지 약 26KB, base64 포함 약 200MB(한도 256MB)
IN_FLIGHT = 6
BUDGET_USD = 100.0
PRICE = {'input_tokens': 0.05, 'output_tokens': 0.25}  # Batch, 100만 토큰당
SYSTEM = '''You are shown a grayscale heatmap of one anonymous cryptocurrency perpetual future at a decision time. Columns are the last 60 four-hour bars (about 10 days), oldest at left, newest at right. Each row is one labeled variable; dark means low and light means high. Rows open, high, low, close: log change from the last close, mid-gray = equal to the last close, white = 10% or more above, black = 10% or more below. Row volume / window mean: black = zero, white = three times the window mean or more. Row volume / own 30d avg: log scale, brighter = higher. Rows hour (UTC) and weekday: calendar fields drawn on a very dark scale. Rows xs rank: at each bar, this asset's percentile rank among all assets at that bar for 24-hour return, 24-hour volatility and volume relative to its 30-day average (black = lowest, white = highest). The remaining 37 rows are constant across columns: indicator values at the decision time, standardized with training-period statistics so that mid-gray is a typical value and lighter is higher. Bars are 4 hours; W=60 bars, q=15 bars. ret fields are log returns; vol fields are standard deviations of log returns; rng fields describe high-low ranges; pos_W is closing price position within the window; z_W is closing price z-score; c_maq,c_maW,maq_maW are moving-average ratios minus one; rsi is 0-100; v_last_rel is volume divided by W-bar mean; v_recent_share is recent q-bar share of W-bar volume; qv_rel30 is W-bar quote volume relative to its trailing 30-day average; amihud is mean absolute return divided by quote volume plus one; top5_share,hhi,n_spike,zero_ratio describe volume concentration; updown_vol is up-bar/down-bar volume ratio; n_big counts large-return bars; hour,dow are UTC calendar fields. xs_ fields are contemporaneous cross-sectional percentile ranks of the corresponding feature. Many other assets are scored at the same decision time, each separately. Estimate this asset's expected simple return over the next 14 days relative to the average of the other assets at the same time. Use only the heatmap. Do not assume any strategy or input design will succeed or fail. Return only a JSON object {"score": x} where x is a number in [-1,1]. Higher scores mean higher expected relative 14-day return; use up to four decimal places. Scores are ordinal preference strengths, not probabilities or calibrated return forecasts. No explanations or other fields.'''


def client():
    for line in open(Path.home() / 'vault/projects-root/.env'):
        m = re.match(r'\s*(?:export\s+)?CLAUDE_API_KEY=(.*)', line)
        if m:
            return anthropic.Anthropic(api_key=m.group(1).strip().strip('"\''), max_retries=8)
    raise RuntimeError('CLAUDE_API_KEY not found')


def jobs():
    """id → (ts, base). id 는 원 실험의 익명 ID 를 쓴다(동점 순서도 같다)."""
    out = {}
    for t in json.loads((SRC / 'tasks.json').read_text()):
        if t['arm'] == 'chart' and t['repeat'] == 0:
            for aid, base in zip(t['ids'], t['bases']):
                for rep in range(REPEATS):
                    out[f"s{t['fold']}_{t['ts']}_{aid}_r{rep}"] = (t['ts'], base)
    return out


def params(ts, base):
    data = base64.b64encode((IMG / str(ts) / f'{base}.png').read_bytes()).decode()
    return dict(model=MODEL, max_tokens=64, thinking={'type': 'disabled'}, system=[{'type': 'text', 'text': SYSTEM}],
                messages=[{'role': 'user', 'content': [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': data}},
                                                       {'type': 'text', 'text': 'Score this heatmap.'}]}])


def judge(message):
    """(score, text, error). 응답 전체를 감싼 코드 울타리 한 겹만 벗긴다."""
    text = ''.join(b['text'] for b in message['content'] if b['type'] == 'text')
    body = text
    fence = re.fullmatch(r'\s*```(?:json)?[ \t]*\n(.*)\n[ \t]*```\s*', text, re.S)
    if fence:
        body = fence.group(1)
    try:
        if message['stop_reason'] != 'end_turn':
            raise ValueError('incomplete generation: ' + str(message['stop_reason']))
        obj = json.loads(body)
        if not isinstance(obj, dict) or set(obj) != {'score'}:
            raise ValueError('expected {"score": x} only')
        x = obj['score']
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError('score must be a finite number')
        return float(x), text, None
    except Exception as e:
        return None, text, f'{type(e).__name__}: {str(e)[:300]}'


def cost(usage):
    return sum((usage.get(k) or 0) * p for k, p in PRICE.items()) / 1e6


def smoke():
    c = client(); J = jobs()
    ids = [i for i in sorted(i for i in J if i.endswith('_r0')) if (IMG / str(J[i][0]) / f'{J[i][1]}.png').exists()][::997][:40]
    rows = []
    for i in ids:
        m = c.messages.create(**params(*J[i])).model_dump(mode='json')
        score, text, error = judge(m)
        rows.append({'id': i, 'ok': score is not None, 'score': score, 'text': text[:200], 'error': error, 'usage': m['usage']})
        print(rows[-1]['ok'], score, repr(text[:80]), m['usage']['input_tokens'], m['usage']['output_tokens'], flush=True)
    (D / 'smoke.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print('valid', sum(r['ok'] for r in rows), '/', len(rows))


def run():
    c = client(); J = jobs()
    con = sqlite3.connect(D / 'results.sqlite')
    con.execute('CREATE TABLE IF NOT EXISTS results (id TEXT PRIMARY KEY, state TEXT, attempts INTEGER, score REAL, text TEXT, usage TEXT, error TEXT, batch TEXT)')
    con.execute('CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, size INTEGER, submitted REAL, ended REAL, cost REAL, failures INTEGER, first_try INTEGER)')
    con.commit()

    def spent():
        return con.execute('SELECT coalesce(sum(cost),0) FROM batches').fetchone()[0]

    def status(stage, error=None):
        counts = dict(con.execute('SELECT state,count(*) FROM results GROUP BY state').fetchall())
        (D / 'status.json').write_text(json.dumps({'stage': stage, 'total': len(J), 'ok': counts.get('ok', 0), 'failed_final': counts.get('failed', 0),
            'retry': counts.get('retry', 0), 'inflight': counts.get('inflight', 0), 'cost_usd': round(spent(), 4), 'budget_usd': BUDGET_USD,
            'error': error, 'updated': time.time()}, ensure_ascii=False, indent=2))

    def stop(msg):
        status('stopped', msg); sys.exit(1)

    def finish(batch_id):
        total, failures, first_try, n = 0.0, 0, True, 0
        for r in c.messages.batches.results(batch_id):
            tid = r.custom_id.rsplit('-a', 1)[0]
            attempt = con.execute('SELECT attempts FROM results WHERE id=?', (tid,)).fetchone()[0]
            first_try = first_try and attempt == 1
            score, text, usage, error = None, None, None, None
            if r.result.type == 'succeeded':
                m = r.result.message.model_dump(mode='json')
                total += cost(m['usage']); usage = json.dumps(m['usage'])
                score, text, error = judge(m)
            else:
                error = f'{r.result.type}: ' + (json.dumps(r.result.error.model_dump(mode='json'))[:300] if r.result.type == 'errored' else '')
                if r.result.type == 'errored' and r.result.error.error.type == 'invalid_request_error' and 'image cannot be empty' not in error:   # 빈 이미지 = 렌더 중단으로 남은 0바이트 파일(RECOVERY_20261029.txt), 다시 그린 뒤 재시도
                    con.commit(); stop('invalid_request_error: ' + error)
            state = 'ok' if score is not None else ('failed' if attempt >= MAX_ATTEMPTS else 'retry')
            failures += score is None; n += 1
            con.execute('UPDATE results SET state=?, score=?, text=?, usage=?, error=?, batch=NULL WHERE id=?', (state, score, (text or '')[:400], usage, error, tid))
        con.execute('UPDATE batches SET ended=?, cost=?, failures=?, first_try=? WHERE id=?', (time.time(), total, failures, int(first_try), batch_id))
        con.commit(); status('running')
        if first_try and n and failures >= 0.2 * n:
            stop(f'{failures}/{n} failures in first-try batch {batch_id}')

    while True:
        for (bid,) in con.execute('SELECT id FROM batches WHERE ended IS NULL').fetchall():
            if c.messages.batches.retrieve(bid).processing_status == 'ended':
                finish(bid)
        done = {i for (i,) in con.execute("SELECT id FROM results WHERE state IN ('ok','failed','inflight')")}
        pending = [i for i in sorted(J, key=lambda i: (i[-2:], i)) if i not in done and (IMG / str(J[i][0]) / f'{J[i][1]}.png').exists()]
        inflight = con.execute('SELECT count(*) FROM batches WHERE ended IS NULL').fetchone()[0]
        if not pending and not inflight:
            if len(done) >= len(J):
                break
            status('waiting for images'); time.sleep(60); continue
        while pending and inflight < IN_FLIGHT:
            group, pending = pending[:BATCH], pending[BATCH:]
            if spent() + len(group) * 0.00007 > BUDGET_USD:
                stop(f'budget {BUDGET_USD} USD would be exceeded')
            reqs = []
            for i in group:
                prev = con.execute('SELECT attempts FROM results WHERE id=?', (i,)).fetchone()
                reqs.append({'custom_id': f'{i}-a{(prev[0] if prev else 0) + 1}', 'params': params(*J[i])})
            b = c.messages.batches.create(requests=reqs)
            con.execute('INSERT INTO batches VALUES (?,?,?,?,?,?,?)', (b.id, len(group), time.time(), None, None, None, None))
            for i in group:
                con.execute("INSERT INTO results (id, state, attempts, batch) VALUES (?, 'inflight', 1, ?) ON CONFLICT(id) DO UPDATE SET state='inflight', attempts=attempts+1, batch=excluded.batch", (i, b.id))
            con.commit(); inflight += 1
        status('running'); time.sleep(60)
    status('completed_with_missing' if con.execute("SELECT count(*) FROM results WHERE state='failed'").fetchone()[0] else 'completed')


if __name__ == '__main__':
    {'smoke': smoke, 'run': run}[sys.argv[1]]()

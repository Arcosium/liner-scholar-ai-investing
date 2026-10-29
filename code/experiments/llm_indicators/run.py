"""지표 표 실험(입력·작업 목록은 llm_full_20261019)을 Claude Haiku 5.5 로 실행한다.

동결된 입력·지시문·작업 목록을 그대로 쓰고 Batch API + 프롬프트 캐시로 보낸다.
출력 스키마는 보내지 않는다(197개 필수 키는 문법이 너무 커 거부됨). 완전성은 원 실험의 로컬 검사로 본다.
입력 묶음마다 1회차를 먼저 보내 캐시를 쓰고, 끝나면 2·3회차를 보내 캐시를 읽는다.
원 실험 폴더는 읽기만 한다. 결과 DB 형식은 원 실험과 같아 evaluate.py 를 그대로 쓴다.

python3 run.py smoke   # 실제 입력 6건 형식 관문 + 1건 Batch 시험
python3 run.py run     # 전체 실행(중단 후 다시 실행하면 이어서 한다)
python3 run.py eval    # 성과 계산 → evaluation/
"""
from pathlib import Path
import hashlib, json, re, sqlite3, sys, time
from itertools import groupby
import anthropic

D = Path(__file__).resolve().parent
SRC = D.parent / 'llm_full_20261019'
sys.dont_write_bytecode = True  # 원 실험 폴더에 아무것도 쓰지 않는다
sys.path.insert(0, str(SRC))
from runner import SYSTEM
from structured_output import validate, normalize_scores

MODEL = 'claude-haiku-5-5'
SAMPLING = {'thinking': {'type': 'disabled'}}  # Haiku 5.5 는 temperature·top_p·top_k 를 거부한다(PROTOCOL.txt)
BUDGET_USD = 200.0               # 누적 비용이 이 값을 넘으면 새 묶음을 보내지 않는다
WAVE = 500                       # 한 번에 다루는 입력 수(1회차 500건, 2·3회차 1,000건)
MAX_ATTEMPTS = 9  # 개정 3: 3회씩 세 순회(원 실험 RECOVERY_V2_1 의 후속 순회 재시도)
# Batch 가격, 100만 토큰당(Haiku 5.5, 요청 10만 토큰 이하). 캐시 쓰기는 1시간 값으로 보수적으로 센다.
PRICE = {'input_tokens': 0.05, 'cache_creation_input_tokens': 0.10, 'cache_read_input_tokens': 0.005, 'output_tokens': 0.25}


def client():
    for line in open(Path.home() / 'vault/projects-root/.env'):
        m = re.match(r'\s*(?:export\s+)?CLAUDE_API_KEY=(.*)', line)
        if m:
            return anthropic.Anthropic(api_key=m.group(1).strip().strip('"\''), max_retries=8)
    raise RuntimeError('CLAUDE_API_KEY not found')


def params(task):
    user = json.dumps(json.loads((SRC / 'inputs' / task['input']).read_text()), separators=(',', ':'))
    return dict(model=MODEL, max_tokens=8192, **SAMPLING,
                system=[{'type': 'text', 'text': SYSTEM}],
                messages=[{'role': 'user', 'content': [{'type': 'text', 'text': user, 'cache_control': {'type': 'ephemeral', 'ttl': '1h'}}]}])


def cost(usage):
    return sum((usage.get(k) or 0) * p for k, p in PRICE.items()) / 1e6


def judge(task, message):
    """(ranking, wrapped_response, error). 원 runner 의 검사와 같다.

    응답 전체를 감싼 마크다운 코드 울타리 한 겹만 벗긴다(내용은 바꾸지 않고, 원문은 claude 키에 남는다).
    """
    text = ''.join(b['text'] for b in message['content'] if b['type'] == 'text')
    fence = re.fullmatch(r'\s*```(?:json)?[ \t]*\n(.*)\n[ \t]*```\s*', text, re.S)
    if fence:
        text = fence.group(1)
    wrapped = {'choices': [{'message': {'content': text}, 'finish_reason': message['stop_reason']}],
               'usage': message['usage'], 'model': message['model'], 'fence_stripped': bool(fence), 'claude': message}
    try:
        if message['stop_reason'] != 'end_turn':
            raise ValueError('incomplete generation: ' + str(message['stop_reason']))
        ranking = validate(text, task['ids'])
        normalized, scale = normalize_scores(json.loads(text)['scores'])
        if scale > 1:
            with open(D / 'range_repairs.jsonl', 'a') as out:
                out.write(json.dumps({'id': task['id'], 'time': time.time(), 'scale': scale, 'action': 'common_positive_rescaling',
                                      'ranking_source': 'unaltered_raw_scores', 'normalized_scores': normalized}) + '\n')
        return ranking, wrapped, None
    except Exception as e:
        return None, wrapped, f'{type(e).__name__}: {str(e)[:500]}'


def check_frozen():
    frozen = json.loads((SRC / 'frozen.json').read_text())
    for path, h in frozen.items():
        if hashlib.sha256((SRC / path).read_bytes()).hexdigest() != h:
            raise RuntimeError('frozen file changed: ' + path)
    hashes = json.loads((SRC / 'input_hashes.json').read_text())
    for name, h in hashes.items():
        if hashlib.sha256((SRC / 'inputs' / name).read_bytes()).hexdigest() != h:
            raise RuntimeError('input changed: ' + name)


def smoke():
    c = client()
    tasks = json.loads((SRC / 'tasks.json').read_text())
    first = tasks[0]['input']
    gate = [t for t in tasks if t['input'] in (first, first.replace('_chart', '_context'))]
    report = {'model': c.models.retrieve(MODEL).model_dump(mode='json'), 'calls': []}
    for t in sorted(gate, key=lambda t: (t['input'], t['repeat'])):
        began = time.time()
        m = c.messages.create(**params(t)).model_dump(mode='json')
        ranking, wrapped, error = judge(t, m)
        report['calls'].append({'id': t['id'], 'ok': ranking is not None, 'error': error, 'seconds': round(time.time() - began, 1),
                                'usage': m['usage'], 'cost_usd_nonbatch': cost(m['usage']) * 2, 'blocks': [b['type'] for b in m['content']]})
        print(report['calls'][-1], flush=True)
    t = gate[0]
    b = c.messages.batches.create(requests=[{'custom_id': t['id'] + '-smoke', 'params': params(t)}])
    while (b := c.messages.batches.retrieve(b.id)).processing_status != 'ended':
        time.sleep(20)
    r = next(iter(c.messages.batches.results(b.id)))
    entry = {'batch_id': b.id, 'type': r.result.type}
    if r.result.type == 'succeeded':
        m = r.result.message.model_dump(mode='json')
        ranking, _, error = judge(t, m)
        entry.update(ok=ranking is not None, error=error, usage=m['usage'], cost_usd=cost(m['usage']))
    report['batch'] = entry
    print(entry, flush=True)
    (D / 'smoke.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))


def run():
    check_frozen()
    c = client()
    tasks = {t['id']: t for t in json.loads((SRC / 'tasks.json').read_text())}
    con = sqlite3.connect(D / 'responses_v2.sqlite')
    con.execute('CREATE TABLE IF NOT EXISTS results (id TEXT PRIMARY KEY, state TEXT, attempts INTEGER, response TEXT, ranking TEXT, elapsed REAL, error TEXT)')
    con.execute('CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, label TEXT, size INTEGER, submitted REAL, ended REAL, cost REAL, failures INTEGER)')
    con.commit()

    def spent():
        return con.execute('SELECT coalesce(sum(cost),0) FROM batches').fetchone()[0]

    def status(stage, error=None):
        counts = dict(con.execute('SELECT state,count(*) FROM results GROUP BY state').fetchall())
        (D / 'status.json').write_text(json.dumps({'stage': stage, 'total': len(tasks), 'completed': counts.get('ok', 0),
            'failed_final': counts.get('failed', 0), 'retrying': counts.get('retry', 0), 'cost_usd': round(spent(), 4),
            'budget_usd': BUDGET_USD, 'error': error, 'updated': time.time()}, ensure_ascii=False, indent=2))

    def finish(batch_id):
        submitted = con.execute('SELECT submitted FROM batches WHERE id=?', (batch_id,)).fetchone()[0]
        while (b := c.messages.batches.retrieve(batch_id)).processing_status != 'ended':
            time.sleep(30)
        total, failures, first_try = 0.0, 0, True
        for r in c.messages.batches.results(batch_id):
            tid = r.custom_id.rsplit('-a', 1)[0]
            task = tasks[tid]
            prev = con.execute('SELECT attempts FROM results WHERE id=?', (tid,)).fetchone()
            attempt = (prev[0] if prev else 0) + 1
            first_try = first_try and attempt == 1
            if r.result.type == 'succeeded':
                m = r.result.message.model_dump(mode='json')
                total += cost(m['usage'])
                ranking, wrapped, error = judge(task, m)
            else:
                ranking, wrapped = None, None
                error = f'{r.result.type}: ' + (json.dumps(r.result.error.model_dump(mode='json'))[:500] if r.result.type == 'errored' else '')
                if r.result.type == 'errored' and r.result.error.error.type == 'invalid_request_error':
                    con.execute('UPDATE batches SET ended=?,cost=? WHERE id=?', (time.time(), total, batch_id)); con.commit()
                    status('stopped', 'invalid_request_error: ' + error); sys.exit(1)
            if ranking is None:
                failures += 1
                with open(D / 'attempts.jsonl', 'a') as out:
                    out.write(json.dumps({'id': tid, 'attempt': attempt, 'time': time.time(), 'error': error, 'response': wrapped}, ensure_ascii=False) + '\n')
            state = 'ok' if ranking else ('failed' if attempt >= MAX_ATTEMPTS else 'retry')
            con.execute('INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?)',
                        (tid, state, attempt, json.dumps(wrapped), json.dumps(ranking), time.time() - submitted, error))
        con.execute('UPDATE batches SET ended=?,cost=?,failures=? WHERE id=?', (time.time(), total, failures, batch_id)); con.commit()
        status('running')
        size = con.execute('SELECT size FROM batches WHERE id=?', (batch_id,)).fetchone()[0]
        if first_try and failures >= max(12, 0.1 * size):  # 개정 2·3: 첫 시도 묶음의 체계적 형식 문제만 멈춘다
            status('stopped', f'{failures} failures in batch {batch_id}; inspect attempts.jsonl'); sys.exit(1)

    def submit(label, group):
        if not group:
            return
        if spent() + len(group) * 0.006 > BUDGET_USD:
            status('stopped', f'budget {BUDGET_USD} USD would be exceeded'); sys.exit(1)
        attempts = {t['id']: (r[0] if (r := con.execute('SELECT attempts FROM results WHERE id=?', (t['id'],)).fetchone()) else 0) for t in group}
        b = c.messages.batches.create(requests=[{'custom_id': f"{t['id']}-a{attempts[t['id']] + 1}", 'params': params(t)} for t in group])
        con.execute('INSERT INTO batches VALUES (?,?,?,?,?,?,?)', (b.id, label, len(group), time.time(), None, None, None)); con.commit()
        status('waiting ' + label)
        finish(b.id)

    for (batch_id,) in con.execute('SELECT id FROM batches WHERE ended IS NULL').fetchall():
        finish(batch_id)  # 중단 전에 보낸 묶음은 다시 보내지 않고 결과만 받는다
    status('running')
    while True:
        done = {i for (i,) in con.execute("SELECT id FROM results WHERE state IN ('ok','failed')")}
        todo = sorted((t for t in tasks.values() if t['id'] not in done), key=lambda t: (t['input'], t['repeat']))
        if not todo:
            break
        inputs = [list(g) for _, g in groupby(todo, key=lambda t: t['input'])]
        for w in range(0, len(inputs), WAVE):
            wave = inputs[w:w + WAVE]
            submit(f'wave{w // WAVE}-write', [g[0] for g in wave])
            submit(f'wave{w // WAVE}-read', [t for g in wave for t in g[1:]])
    failed = con.execute("SELECT count(*) FROM results WHERE state='failed'").fetchone()[0]
    status('completed_with_missing' if failed else 'completed')


def rejudge():
    # 규칙 개정 뒤 이미 받은 응답만 다시 판정한다. 다시 추론해 좋은 응답을 고르지 않는다.
    tasks = {t['id']: t for t in json.loads((SRC / 'tasks.json').read_text())}
    con = sqlite3.connect(D / 'responses_v2.sqlite')
    rows = con.execute("SELECT id,response,error FROM results WHERE state='retry' AND response!='null'").fetchall()
    fixed = 0
    for tid, raw, previous in rows:
        ranking, wrapped, error = judge(tasks[tid], json.loads(raw)['claude'])
        if ranking:
            con.execute("UPDATE results SET state='ok',response=?,ranking=?,error=NULL WHERE id=?", (json.dumps(wrapped), json.dumps(ranking), tid))
            with open(D / 'rejudged.jsonl', 'a') as out:
                out.write(json.dumps({'id': tid, 'time': time.time(), 'rule': 'strip_outer_markdown_fence', 'previous_error': previous}) + '\n')
            fixed += 1
    con.commit()
    print(f'{fixed}/{len(rows)} rejudged ok')


def evaluate():
    # 큰 입력은 복사하지 않고 원본을 가리킨다.
    for name in ('panel.parquet', 'tasks.json', 'source_hashes.json'):
        if not (D / name).exists():
            (D / name).symlink_to(SRC / name)
    import evaluate as e
    e.D = D
    e.main()


if __name__ == '__main__':
    {'smoke': smoke, 'run': run, 'rejudge': rejudge, 'eval': evaluate}[sys.argv[1]]()

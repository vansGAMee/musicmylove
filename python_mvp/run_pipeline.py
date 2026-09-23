"""One entry point. All work is isolated by run directory; failed gates stop immediately."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['all', 'data', 'graph', 'train-graph', 'train-taste', 'train-ranker', 'evaluate', 'release', 'recommend', 'smoke'])
    p.add_argument('tracks', nargs='?', default=str(ROOT/'example_tracks.txt'))
    p.add_argument('--run', type=Path, default=ROOT/'data'/'main')
    p.add_argument('--days', type=int, default=7)
    p.add_argument('--user-fraction', type=float, default=.1)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    args = p.parse_args()
    if args.action in ('all', 'release') and len(set(args.seeds)) < 3:
        p.error('Release pipeline requires at least three distinct seeds')
    run = args.run.resolve()
    os.environ['MUSICMVP_RUN'] = str(run)
    reports = run/'reports'; reports.mkdir(parents=True, exist_ok=True)
    if (reports/'final_consumed.lock').exists() and args.action not in ('recommend','evaluate'):
        p.error('Final split already consumed: frozen run cannot be trained/rebuilt again')

    def execute(script, options=(), resume=False, expected=()):
        command = [sys.executable, str(ROOT/script), *map(str, options)]
        signature = hashlib.sha256(json.dumps(command).encode()).hexdigest()[:16]
        marker = reports/(signature + '.step.json')
        # Resume only identical commands AND unchanged input/code/output hashes.
        def digest_files(paths):
            result = {}
            for path in paths:
                if not path.exists():
                    continue
                h=hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda:stream.read(1024*1024), b''): h.update(block)
                result[str(path)] = h.hexdigest()
            return result
        inputs = list(ROOT.glob('*.py')) + [run/'data'/'dataset.json', run/'data'/'split_manifest.json']
        if script != 'build_graph.py': inputs += [run/'data'/'graph.json'] + list((run/'data').glob('*.npz'))
        if script == 'train_taste.py': inputs += [run/'models'/f'graph_{options[1]}.pt']
        if script == 'train_ranker.py': inputs += [run/'models'/f'graph_{options[1]}.pt', run/'models'/f'taste_{options[1]}.pt']
        before = digest_files(inputs)
        if resume and marker.exists():
            saved=json.loads(marker.read_text())
            if saved.get('inputs')==before and saved.get('outputs')==digest_files(expected) and all(x.exists() for x in expected):
                print('Already completed:', script, *options, flush=True); return
            raise RuntimeError('Existing completed stage is incompatible; create a new --run directory')
        log = reports/(signature + '.log')
        print('Running:', ' '.join(command), '\nLog:', log, flush=True)
        started=time.monotonic()
        with log.open('w') as output:
            child=subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=output, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired: print(f'{script}: running {time.monotonic()-started:.0f}s; {log}', flush=True)
            except KeyboardInterrupt:
                child.terminate(); child.wait(timeout=15); raise
        record={'command':command,'config':vars(args) | {'run':str(run)},'reason':'user-requested pipeline',
                'timestamp':datetime.now(timezone.utc).isoformat(), 'elapsed_seconds':time.monotonic()-started,
                'exit_code':child.returncode,'inputs':before,'outputs':digest_files(expected),
                'result':'PASS' if child.returncode==0 else 'FAIL',
                'decision':'continue' if child.returncode==0 else 'stop; inspect log; do not relax quality gates',
                'metrics_report':str(log)}
        with (reports/'experiments.jsonl').open('a') as output: output.write(json.dumps(record)+'\n')
        if child.returncode:
            print(log.read_text()[-6000:], file=sys.stderr)
            raise RuntimeError(f'STAGE_FAILED: {script}; log={log}')
        marker.write_text(json.dumps(record,indent=2)+'\n')

    def data():
        execute('collect_data.py',['--days',args.days,'--user-fraction',args.user_fraction])
        sources=json.loads((reports/'collection.json').read_text())
        execute('prepare_data.py',[r['output'] for r in sources])

    def graph():
        execute('build_graph.py', resume=True, expected=[run/'data'/'graph.json'] + [run/'data'/(key+'.npz') for key in ('global','local','users','sessions','confidence','user_incidence','session_incidence')])

    def train(stage):
        for seed in args.seeds:
            execute(f'train_{stage}.py',['--seed',seed,'--epochs',args.epochs],resume=True,
                    expected=[run/'models'/f'{stage}_{seed}.pt'])

    try:
        if args.action == 'all':
            data(); graph()
            for stage in ('graph','taste','ranker'): train(stage)
            execute('release_check.py',['--split','shadow','--seeds',*args.seeds])
            execute('release_check.py',['--split','final','--seeds',*args.seeds])
        elif args.action == 'data': data()
        elif args.action == 'graph': graph()
        elif args.action.startswith('train-'): train(args.action.removeprefix('train-'))
        elif args.action == 'evaluate':
            for seed in args.seeds: execute('evaluate.py',['--split','dev','--seed',seed])
        elif args.action == 'release':
            execute('release_check.py',['--split','shadow','--seeds',*args.seeds])
            execute('release_check.py',['--split','final','--seeds',*args.seeds])
        elif args.action == 'recommend':
            execute('recommend.py',[str(Path(args.tracks).resolve()),'--seed',args.seeds[0]])
            result=json.loads((reports/'last_recommendation.json').read_text())
            for row in result['top50']: print(f"{row['rank']:2} | {row['artist']} | {row['title']} | {row['neural_score']:.5f}")
        elif args.action == 'smoke':
            data(); graph(); execute('smoke_real.py')
    except (RuntimeError, OSError, ValueError) as error:
        print(error, file=sys.stderr); return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())

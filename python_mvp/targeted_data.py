"""Bounded public recent-listen collection for missing taste directions.

No website co-occurrence guesses. Only existing representation TRAIN users are
queried. Official API: https://listenbrainz.readthedocs.io/en/latest/users/api/core.html
Use --collect to perform network requests and prepare a NEW dataset.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from .config import ROOT, fingerprint, write_json
from .prepare_data import normalize, identify
from .expand_data import prepare_bounded

UA = 'MusicMyLoveResearch/0.4 (bounded public ListenBrainz history; local music recommendation experiment)'


def select_users(engine, coverage, limit):
    missing = [r for r in coverage['tracks'] if r['status'] in ('in_raw_not_catalog', 'absent_from_raw')]
    exact = {tid for r in missing for tid in r.get('raw_ids', [])}
    artists = {normalize(r['input'].split(' - ',1)[0]) for r in missing if ' - ' in r['input']}
    wanted = {}
    for i, track in enumerate(engine.data['tracks']):
        if track['id'] in exact:
            wanted[i] = 5
        elif normalize(track['artist']) in artists or any(normalize(a) in artists for a, _ in track.get('aliases', [])):
            wanted[i] = 1
    allowed = set(engine.meta['train_users'])
    scores = []
    for user in engine.data['users']:
        if user['id'] not in allowed:
            continue
        score = sum(wanted.get(t,0) for t in set(user['tracks'])) / math.sqrt(max(1,len(user['tracks'])))
        if score:
            scores.append((score,user['id']))
    return [uid for _,uid in sorted(scores,key=lambda x:(-x[0],x[1]))[:limit]]


def api_request(url):
    if not url.startswith('https://api.listenbrainz.org/1/user/'):
        raise ValueError('Only official public ListenBrainz user endpoints allowed')
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':UA}), timeout=30) as response:
                value = json.load(response)
                remaining = response.headers.get('X-RateLimit-Remaining')
                reset = response.headers.get('X-RateLimit-Reset-In','1')
                delay = min(60,max(1,float(reset))) if remaining == '0' else 1
            time.sleep(delay)
            return value
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return {'payload':{'listens':[]}, 'unavailable':True}
            if error.code not in (429,500,502,503,504) or attempt == 3:
                raise
            retry = error.headers.get('Retry-After','1')
            time.sleep(min(60,max(2**attempt,int(retry) if retry.isdigit() else 1)))
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError('API request failed')


def collect_user(handle, cache, upper, lower, pages=3, request=api_request):
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    rows, boundary = [], upper
    for _ in range(pages):
        url = 'https://api.listenbrainz.org/1/user/'+urllib.parse.quote(handle,safe='')+'/listens?'+urllib.parse.urlencode({'count':100,'max_ts':boundary})
        path = cache/(hashlib.sha256(url.encode()).hexdigest()+'.json')
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['url'] != url:
                raise RuntimeError('API cache URL mismatch')
            payload = saved['response']
        else:
            payload = request(url)
            write_json(path, {'url':url,'response':payload})
        listens = payload.get('payload',{}).get('listens')
        if not isinstance(listens,list):
            raise ValueError('API response has no listens list')
        if not listens:
            break
        timestamps = []
        for listen in listens:
            timestamp = listen.get('listened_at')
            if not isinstance(timestamp,int) or timestamp <= 0 or timestamp >= boundary:
                raise ValueError('API response violates requested time boundary')
            timestamps.append(timestamp)
            if timestamp >= lower:
                row = {'user_name':handle,'listened_at':timestamp,'track_metadata':listen.get('track_metadata',{})}
                identify(row)  # reject malformed recording identities; never infer them
                rows.append(row)
        oldest = min(timestamps)
        if oldest < lower or len(listens) < 100:
            break
        boundary = oldest
    return rows


def find_handles(paths, user_ids):
    wanted = set(user_ids); result = {}
    for path in paths:
        print(f'Resolving source user names in {Path(path).name}', flush=True)
        with Path(path).open() as source:
            for line in source:
                try:
                    row = json.loads(line); name = row.get('user_name')
                    if not isinstance(name,str):
                        continue
                    uid = hashlib.sha256(name.encode()).hexdigest()
                    if uid in wanted:
                        result[uid] = name
                        if len(result) == len(wanted):
                            return result
                except (ValueError,AttributeError):
                    continue
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library', type=Path)
    parser.add_argument('--source', type=Path, default=ROOT/'data/expanded-v1')
    parser.add_argument('--model', type=Path, default=ROOT/'data/expanded-v1/model')
    parser.add_argument('--run', type=Path, default=ROOT/'data/targeted-v1')
    parser.add_argument('--users', type=int, default=100)
    parser.add_argument('--pages', type=int, default=3)
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--collect', action='store_true', help='Run bounded downloads and prepare new data; no training')
    args = parser.parse_args()
    if not 1 <= args.users <= 500 or not 1 <= args.pages <= 3 or not 1 <= args.days <= 30:
        parser.error('users 1..500, pages 1..3, days 1..30 required')
    source, run = args.source.resolve(), args.run.resolve()
    if run in (source,args.model.resolve()):
        parser.error('Output must be a new data directory')
    reports = json.loads((source/'reports/collection.json').read_text())
    original = json.loads((source/'data/split_manifest.json').read_text())
    config = {'source':str(source), 'model':str(args.model.resolve()), 'users':args.users,'pages':args.pages,'days':args.days,
        'library_sha256':fingerprint(args.library.expanduser()), 'source_manifest_sha256':fingerprint(source/'data/split_manifest.json'),
        'model_sha256':fingerprint(args.model/'ranker.pt'),
        'implementation':{n:fingerprint(ROOT/n) for n in ('targeted_data.py','coverage.py','expand_data.py','prepare_data.py')}}
    manifest = run/'targeting.json'
    if run.exists():
        if not manifest.exists() or json.loads(manifest.read_text())['config'] != config:
            raise RuntimeError('Targeted run belongs to another configuration; choose a new --run')
        plan = json.loads(manifest.read_text())
    else:
        from .cli import open_engine, read_path
        from .coverage import diagnose
        print('Loading the verified model and diagnosing missing profile tracks...',flush=True)
        engine, _ = open_engine(args.model, source)
        if engine.meta['fingerprint'] != original['fingerprint']:
            raise RuntimeError('Model/source dataset mismatch')
        coverage = diagnose(engine, read_path(str(args.library)))
        selected = select_users(engine,coverage,args.users)
        if not selected:
            raise RuntimeError('No existing representation users connect to missing directions; no blind download')
        if any(original['users'].get(uid) != 'train' for uid in selected):
            raise RuntimeError('Targeted collection would include held-out users')
        plan = {'config':config,'users':selected,'upper':int(time.time()),'coverage':coverage,
                'scope':'bounded recent listens, not full user history; timestamp boundary ties may be truncated'}
        del engine; gc.collect()
        run.mkdir(parents=True); write_json(manifest,plan)
    print(f"Selected {len(plan['users'])} representation TRAIN users; at most {len(plan['users'])*args.pages} API pages.",flush=True)
    if not args.collect:
        print(f'Plan only: {manifest}. Add --collect to run bounded collection.'); return
    complete = run/'complete.json'
    if complete.exists():
        expected = json.loads(complete.read_text())
        if any(not (run/n).exists() or fingerprint(run/n) != sha for n,sha in expected.items()):
            raise RuntimeError('Completed targeted dataset changed')
    else:
        paths = [Path(r['output']) for r in reports]
        for path, report in zip(paths,reports):
            if fingerprint(path) != report['sha256']:
                raise RuntimeError('Source extraction checksum mismatch')
        cache = run/'data/cache/listenbrainz'; cache.mkdir(parents=True,exist_ok=True)
        handles_path = cache/'handles.json'
        if handles_path.exists():
            handles = json.loads(handles_path.read_text())
        else:
            handles = find_handles(paths,plan['users']); write_json(handles_path,handles)
        if set(handles) != set(plan['users']) or any(hashlib.sha256(name.encode()).hexdigest() != uid for uid,name in handles.items()):
            raise RuntimeError('Source handles could not be verified for every selected user')
        extra = cache/'extra.jsonl'; temporary=extra.with_suffix('.tmp')
        count=0
        with temporary.open('w') as destination:
            for i,uid in enumerate(plan['users']):
                rows = collect_user(handles[uid],cache,plan['upper'],plan['upper']-args.days*86400,args.pages)
                for row in rows:
                    destination.write(json.dumps(row,ensure_ascii=False)+'\n')
                count += len(rows)
                print(f'Collected {i+1}/{len(handles)} users, {count} bounded listens',flush=True)
        temporary.replace(extra)
        if not count:
            raise RuntimeError('No additional listens collected; no dataset published')
        prepare_bounded([*paths,extra],run/'data',run/'reports',expected_original=original['users'])
        write_json(run/'reports/targeted_collection.json',{'users':len(handles),'listens':count,'sha256':fingerprint(extra),
                   'only_representation_users':True,'frozen_final_used':False,'scope':plan['scope']})
    # Preserve and verify FINAL consumption just as with offline expansion.
    lineage = {name: path for name,path in {'source':source/'reports/final_consumed.lock',
               'model':args.model/'reports/final_consumed.lock'}.items() if path.exists()}
    copied = run/'reports/final_consumed.lock'
    if lineage:
        frozen = next(iter(lineage.values()))
        if copied.exists() and fingerprint(copied) != fingerprint(frozen):
            raise RuntimeError('Source FINAL marker changed')
        if not copied.exists():
            shutil.copyfile(frozen,copied)
    write_json(run/'reports/final_lineage.json',{name:fingerprint(path) for name,path in lineage.items()})
    files=['data/dataset.json','data/split_manifest.json','reports/data_audit.json','reports/targeted_collection.json',
           'reports/final_lineage.json']
    if copied.exists():
        files.append('reports/final_consumed.lock')
    write_json(complete,{n:fingerprint(run/n) for n in files})
    import shlex
    print('Data ready, no training performed. Next: python -m python_mvp.honest_training --source '
          +shlex.quote(str(run))+' --run '+shlex.quote(str(run/'model')))


if __name__ == '__main__':
    main()

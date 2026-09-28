"""Resumable catalog preview lookup and optional CLAP encoding. No full songs."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
from .recommend import match_key
from .sound import SoundCache,DEFAULT_CACHE
from .sound_encoder import ClapEncoder,MODEL_DIR
from .sound_import import build

API='https://api.deezer.com/search'
UA='MusicMyLove/1.0 (bounded music preview research)'


def choose_preview(track,results):
    # No fuzzy matches, genre tags, provider popularity or recommendations.
    identity=(match_key(track['artist']),match_key(track['title']))
    matches=[r for r in results if (match_key(r.get('artist',{}).get('name','')),match_key(r.get('title','')))==identity
             and str(r.get('preview','')).startswith('https://') and r.get('isrc')]
    if not matches:return None
    recordings={r.get('isrc') or f'id:{r["id"]}' for r in matches}
    if len(recordings)!=1:return None
    row=min(matches,key=lambda r:int(r['id']))
    return {k:row[k] for k in ('id','title','artist','preview','isrc') if k in row}


class PreviewSearch:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.last=0.

    def search(self,track):
        query=track['artist']+' '+track['title']
        key=hashlib.sha256(query.encode()).hexdigest();path=self.root/(key+'.json')
        if path.exists():
            saved=json.loads(path.read_text())
            # Preview URLs expire; keep metadata for one day, vectors indefinitely.
            if time.time()-saved['saved_at']<86400:return saved['data']
        url=API+'?'+urllib.parse.urlencode(dict(q=query,limit=25))
        for attempt in range(3):
            time.sleep(max(0,.55-(time.monotonic()-self.last)));self.last=time.monotonic()
            try:
                req=urllib.request.Request(url,headers={'User-Agent':UA})
                with urllib.request.urlopen(req,timeout=15) as response:
                    raw=response.read(2*1024*1024+1)
                if len(raw)>2*1024*1024:raise ValueError('API response too large')
                data=json.loads(raw)
                if 'error' in data:raise ValueError(f'Preview API error: {data["error"]}')
                result=choose_preview(track,data.get('data',[]))
                payload=dict(saved_at=time.time(),data=result)
                temp=path.with_suffix('.tmp');temp.write_text(json.dumps(payload));temp.replace(path)
                return result
            except (urllib.error.URLError,TimeoutError,ValueError):
                if attempt==2:raise
                time.sleep(2**attempt)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('python_mvp/data/targeted-v1/model'))
    p.add_argument('--cache',type=Path,default=DEFAULT_CACHE)
    p.add_argument('--model-dir',type=Path,default=MODEL_DIR)
    p.add_argument('--limit',type=int,default=100,help='Total prefix of deterministic catalog sample; increase to expand coverage; reruns resume this target')
    p.add_argument('--library',type=Path,help='Process matched favorite tracks before the rest of the catalog')
    p.add_argument('--encode',action='store_true',help='Download previews temporarily and compute vectors')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=p.parse_args()
    if args.limit<1:p.error('--limit must be positive')
    tracks=json.loads((args.run/'data/graph.json').read_text())['tracks']
    priority=set()
    if args.library:
        from .discovery_engine import read_library
        from .profile_input import resolve_profile
        profile=resolve_profile(read_library(args.library.expanduser()),tracks)
        priority={tracks[i]['id'] for i in profile['seeds']}
        print(f'Priority profile: {len(priority)} matched tracks',flush=True)
    catalog_size=len(tracks)
    tracks=sorted(tracks,key=lambda t:(t['id'] not in priority,hashlib.sha256(t['id'].encode()).hexdigest()))[:args.limit]
    cache=SoundCache(args.cache);search=PreviewSearch(args.cache/'preview-search')
    encoder=None
    def get_encoder():
        nonlocal encoder
        if encoder is None:encoder=ClapEncoder(args.model_dir,args.device)
        return encoder
    # Validate model once before doing a large network job.
    if args.encode:get_encoder()
    report=dict(inspected=0,matched=0,cached=0,encoded=0,missing=0,errors=[],
                identity='exact artist/title, unique provider ISRC; NOT fingerprint verification',
                catalog_size=catalog_size,priority_profile_tracks=len(priority))
    start=time.monotonic();consecutive_errors=0
    try:
        for track in tracks:
            report['inspected']+=1
            if cache.get(track['id']) is not None:
                report['cached']+=1;continue
            try:
                result=search.search(track)
                if result:
                    report['matched']+=1
                    if args.encode:
                        r=build([dict(track_id=track['id'],url=result['preview'],provider='deezer',provider_id=result['id'],
                                      isrc=result.get('isrc'),artist=result['artist']['name'],title=result['title'],
                                      identity='exact artist/title and unique provider ISRC; not fingerprint verified')],cache,get_encoder,limit=1)
                        report['encoded']+=r['encoded'];report['errors'].extend(r['failed'])
                        if r['failed']:
                            consecutive_errors+=1
                            if consecutive_errors>=5:raise ImportError('Five preview failures; stop and retry later')
                        else:consecutive_errors=0
                    else:consecutive_errors=0
                else:
                    report['missing']+=1;consecutive_errors=0
            except (OSError,ValueError,RuntimeError) as exc:
                report['errors'].append(dict(track_id=track['id'],error=str(exc)))
                consecutive_errors+=1
                if consecutive_errors>=5:raise RuntimeError('Five consecutive failures; stop instead of hammering provider') from exc
            if report['inspected']%10==0:
                elapsed=time.monotonic()-start
                print(f'{report["inspected"]}/{len(tracks)} matched={report["matched"]} encoded={report["encoded"]} cached={report["cached"]} elapsed={elapsed:.0f}s',flush=True)
    finally:
        report['elapsed_seconds']=round(time.monotonic()-start,2)
        work=report['inspected']-report['cached']
        report['estimated_remaining_hours_at_measured_rate']=round(report['elapsed_seconds']/max(1,work)*max(0,catalog_size-report['inspected'])/3600,2) if work else None
        report['available_fraction']=(report['matched']+report['cached'])/max(1,report['inspected'])
        args.cache.mkdir(parents=True,exist_ok=True)
        temp=args.cache/'collection-report.tmp';temp.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        temp.replace(args.cache/'collection-report.json')
        print(json.dumps(report,ensure_ascii=False,indent=2))
    if report['errors']:raise SystemExit(1)


if __name__=='__main__':main()

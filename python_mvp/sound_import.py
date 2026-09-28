"""Prepare real CLAP vectors. Explicit download/build; never invoked by CLI inference."""
import argparse
import json
from pathlib import Path
import tempfile
import urllib.request
import urllib.parse
from .sound import SoundCache,DEFAULT_CACHE,sha256
from .sound_encoder import ClapEncoder,MODEL_DIR,download_model,duration
from .recommend import match_key

EXTENSIONS={'.mp3','.wav','.flac','.ogg','.m4a','.opus'}


def resolve_files(directory,tracks,mapping):
    catalog={str(t['id']):t for t in tracks}; names={}
    for tid,t in catalog.items():names.setdefault((match_key(t['artist']),match_key(t['title'])),[]).append(tid)
    rows=[];skipped=[]
    if mapping:
        mapping=Path(mapping);entries=json.loads(mapping.read_text())
        if not isinstance(entries,list):raise ValueError('Mapping must be a list of {track_id,path} or {track_id,url}')
        for entry in entries:
            tid=str(entry['track_id'])
            if tid not in catalog:raise ValueError(f'Unknown catalog track_id: {tid}')
            if ('path' in entry)==('url' in entry):raise ValueError('Specify exactly one of path or url')
            if 'path' in entry:
                path=Path(entry['path']).expanduser()
                if not path.is_absolute():path=mapping.resolve().parent/path
                if not path.is_file():raise ValueError(f'Missing audio file: {path}')
                rows.append(dict(track_id=tid,path=str(path.resolve())))
            else:
                url=entry['url'];parsed=urllib.parse.urlsplit(url)
                if parsed.scheme not in {'http','https'} or not parsed.hostname:
                    raise ValueError('Preview URL must be HTTP(S)')
                rows.append(dict(track_id=tid,url=url))
    if directory:
        for path in sorted(Path(directory).expanduser().rglob('*')):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:continue
            ids=[path.stem] if path.stem in catalog else []
            if not ids and ' - ' in path.stem:
                a,t=path.stem.split(' - ',1);ids=names.get((match_key(a),match_key(t)),[])
            if len(ids)!=1:
                skipped.append(dict(path=str(path),reason='No unique exact ID or Artist - Title filename match'));continue
            rows.append(dict(track_id=ids[0],path=str(path.resolve())))
    by_id={}
    for row in rows:
        if row['track_id'] in by_id and by_id[row['track_id']]!=row:
            raise ValueError(f'Multiple files for {row["track_id"]}; choose one explicitly')
        by_id[row['track_id']]=row
    return [by_id[k] for k in sorted(by_id)],skipped


def fetch_preview(url,destination):
    """Only explicitly mapped URLs; <=12 MiB and <=60 seconds, no search guesses."""
    import time
    limit=12*1024*1024;start=time.monotonic()
    request=urllib.request.Request(url,headers={'User-Agent':'MusicMyLove/1.0 (user-selected audio preview)'})
    with urllib.request.urlopen(request,timeout=20) as response,Path(destination).open('wb') as out:
        if urllib.parse.urlsplit(response.url).scheme not in {'http','https'}:raise ValueError('Invalid redirect')
        if int(response.headers.get('Content-Length',0))>limit:raise ValueError('Preview exceeds 12 MiB')
        size=0
        while True:
            chunk=response.read(65536)
            if not chunk:break
            size+=len(chunk)
            if size>limit or time.monotonic()-start>60:raise ValueError('Preview download limit exceeded')
            out.write(chunk)
    if duration(destination)>60:raise ValueError('Remote audio must be a preview <=60 seconds')


def build(rows,cache,encoder_factory,limit=200,*,prepared_path=None,quiet=False):
    if limit<1:raise ValueError('limit must be positive')
    if prepared_path is not None and len(rows)!=1:raise ValueError('Prepared preview requires exactly one track')
    encoder=None;report=dict(encoded=0,cached=0,failed=[],remaining=max(0,len(rows)-limit))
    for number,row in enumerate(rows[:limit],1):
        tid=row['track_id']
        try:
            old=cache.read(tid)
            # URLs are immutable identity claims in this manifest; use --refresh for changed remote files.
            if 'url' in row and old and old[1]['source'].get('url')==row['url']:
                report['cached']+=1;continue
            with tempfile.TemporaryDirectory(prefix='music-preview-') as temp:
                path=Path(prepared_path) if prepared_path is not None else (Path(row['path']) if 'path' in row else Path(temp)/'preview.audio')
                if 'url' in row and prepared_path is None:fetch_preview(row['url'],path)
                digest=sha256(path)
                if cache.current(tid,digest):report['cached']+=1;continue
                if encoder is None:
                    try:encoder=encoder_factory()
                    except Exception as exc:raise ImportError(f'Cannot initialize audio model: {exc}') from exc
                vector=encoder.encode(path)
                if sha256(path)!=digest:raise ValueError('Audio file changed during encoding')
                source=dict(row)
                source.update(file_sha256=digest,identity=row.get('identity','explicit ID or exact filename; not acoustic identification'))
                cache.put(tid,vector,source)
                report['encoded']+=1
        except ImportError:
            raise
        except Exception as exc:
            report['failed'].append(dict(track_id=tid,error=str(exc)))
            print(f'Failed {tid}: {exc}',flush=True)

        if not quiet:print(f'{number}/{min(limit,len(rows))}: encoded={report["encoded"]}, cached={report["cached"]}',flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('download');d.add_argument('--model-dir',type=Path,default=MODEL_DIR)
    b=sub.add_parser('build');b.add_argument('--run',type=Path,default=Path('python_mvp/data/targeted-v1/model'))
    b.add_argument('--audio-dir',type=Path);b.add_argument('--map',type=Path)
    b.add_argument('--model-dir',type=Path,default=MODEL_DIR);b.add_argument('--cache',type=Path,default=DEFAULT_CACHE)
    b.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');b.add_argument('--limit',type=int,default=200)
    b.add_argument('--playlist',type=Path,help='Restrict to IDs in exported playlist plus resolved input seeds')
    b.add_argument('--refresh',action='store_true',help='Recheck remote previews; keep cached vectors until successful replacement')
    args=p.parse_args()
    if args.command=='download':download_model(args.model_dir);print('CLAP checkpoint downloaded and verified');return
    if not args.audio_dir and not args.map:p.error('build requires --audio-dir or --map')
    tracks=json.loads((args.run/'data/graph.json').read_text())['tracks']
    rows,skipped=resolve_files(args.audio_dir,tracks,args.map)
    if args.playlist:
        from .profile_input import resolve_profile
        result=json.loads(args.playlist.read_text());ids={r['id'] for r in result['top50']}
        profile=resolve_profile(result.get('input',[]),tracks)
        ids.update(tracks[i]['id'] for i in profile['seeds'])
        rows=[r for r in rows if r['track_id'] in ids]
    cache=SoundCache(args.cache)
    if args.refresh:
        # Skip only the remote URL shortcut, still use content hashes to avoid redundant inference.
        class RefreshCache(SoundCache):
            def read(self,tid):
                record=super().read(tid)
                if record:record[1]['source'].pop('url',None)
                return record
        cache=RefreshCache(args.cache)
    report=build(rows,cache,lambda:ClapEncoder(args.model_dir,args.device),args.limit)
    report.update(matched=len(rows),unmatched=skipped)
    args.cache.mkdir(parents=True,exist_ok=True)
    (args.cache/'last-import.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not rows or report['failed']:raise SystemExit(1)


if __name__=='__main__':main()

from pathlib import Path
import numpy as np
import pytest
from python_mvp.tests.test_discovery_policy import fixture


def test_sound_cache_validates_and_resumes(tmp_path):
    from python_mvp.sound import SoundCache
    cache=SoundCache(tmp_path)
    v=np.ones(512,dtype=np.float32)
    cache.put('track',v,{'file_sha256':'a'})
    assert np.isclose(np.linalg.norm(cache.get('track')),1)
    assert cache.current('track','a') and not cache.current('track','b')
    with pytest.raises(ValueError):cache.put('bad',np.zeros(512),{})
    with pytest.raises(ValueError):cache.put('bad',np.full(512,np.nan),{})
    assert SoundCache(tmp_path).get('missing') is None


def test_sound_filter_absence_identity_and_direction_specific(tmp_path):
    from python_mvp.sound import SoundCache,filter_pool
    from python_mvp.discovery_policy import prepare_discovery
    e,s=fixture();pool=prepare_discovery(e,[0,1],[])
    before={k:set(v) for k,v in pool.groups.items()}
    cache=SoundCache(tmp_path)
    filter_pool(pool,[0,1],cache)
    assert before=={k:set(v) for k,v in pool.groups.items()}
    def vec(x):
        v=np.zeros(512);v[0]=x;v[1]=np.sqrt(1-x*x);return v
    cache.put(e.tracks[0]['id'],vec(1),{})
    for i in range(2,22):cache.put(e.tracks[i]['id'],vec((i-2)/20),{})
    report=filter_pool(pool,[0,1],cache)
    assert report['rejected_edges']>0
    assert 2 not in pool.groups['seed a'] and 21 in pool.groups['seed a']
    assert before['seed b']==set(pool.groups['seed b'])
    assert 25 in pool.groups['seed a'] # missing audio stays


def test_manifest_refuses_ambiguous_or_missing_identity(tmp_path):
    from python_mvp.sound_import import resolve_files
    tracks=[{'id':'one','artist':'A','title':'T'},{'id':'two','artist':'A','title':'T'}]
    (tmp_path/'A - T.wav').write_bytes(b'not used')
    rows,skips=resolve_files(tmp_path,tracks,None)
    assert not rows and skips


def test_bounded_decode_real_wav(tmp_path):
    import wave
    from python_mvp.sound_encoder import decode_segment
    p=tmp_path/'a.wav'
    with wave.open(str(p),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(48000)
        f.writeframes((np.sin(np.arange(48000)*.1)*10000).astype('<i2').tobytes())
    a=decode_segment(p,0)
    assert a.shape==(480000,) and np.isfinite(a).all()
    bad=tmp_path/'bad.wav';bad.write_bytes(b'broken')
    with pytest.raises(Exception):decode_segment(bad,0)


def test_preview_matching_rejects_wrong_artist_and_ambiguous_versions():
    from python_mvp.sound_collect import choose_preview
    track=dict(id='a',artist='Hum',title='Stars')
    row=dict(id=1,artist={'name':'Hum'},title='Stars',preview='https://cdn.example/a',isrc='A')
    assert choose_preview(track,[row])['id']==1
    assert choose_preview(track,[dict(row,artist={'name':'Other'})]) is None
    assert choose_preview(track,[row,dict(row,id=2,isrc='B')]) is None
    assert choose_preview(track,[dict(row,title='Stars (Live)')]) is None


def test_import_resumes_without_loading_model(tmp_path):
    from python_mvp.sound import SoundCache,sha256
    from python_mvp.sound_import import build
    p=tmp_path/'audio';p.write_bytes(b'source')
    cache=SoundCache(tmp_path/'cache');cache.put('a',np.ones(512),{'file_sha256':sha256(p)})
    def fail():raise AssertionError('must not load encoder for cached tracks')
    report=build([dict(track_id='a',path=str(p))],cache,fail)
    assert report['cached']==1 and report['encoded']==0


def test_missing_isrc_not_claimed_as_verified():
    from python_mvp.sound_collect import choose_preview
    assert choose_preview(dict(artist='A',title='B'),[dict(id=1,artist={'name':'A'},title='B',preview='https://x')]) is None


def test_cli_empty_sound_cache_preserves_exact_playlist(tmp_path):
    from python_mvp.cli import generate
    from python_mvp.sound import SoundCache
    e,s=fixture();e.features=lambda seeds:(np.ones((len(e.tracks),22)),set(range(len(e.tracks))),{})
    e.score=lambda model,features:s
    args=(e,None,['Seed A - song0','Seed B - song1'],[],'2')
    plain=generate(*args,feedback_dir=tmp_path,personal=False)
    sound=generate(*args,feedback_dir=tmp_path,personal=False,sound_cache=SoundCache(tmp_path/'sound'))
    assert [t['id'] for t in plain['top50']]==[t['id'] for t in sound['top50']]
    assert sound['sound_report']['compared_edges']==0


def test_bad_first_file_does_not_block_later_files(tmp_path):
    from python_mvp.sound_import import build
    from python_mvp.sound import SoundCache
    p=tmp_path/'valid';p.write_bytes(b'fixture')
    class Encoder:
        def encode(self,path):return np.ones(512)
    report=build([dict(track_id='bad',path=str(tmp_path/'missing')),dict(track_id='ok',path=str(p))],
                 SoundCache(tmp_path/'cache'),Encoder)
    assert report['encoded']==1 and len(report['failed'])==1


def test_cache_rejects_wrong_space(tmp_path):
    from python_mvp.sound import SoundCache
    cache=SoundCache(tmp_path);cache.put('a',np.ones(512),{})
    import json
    path=cache.path('a')
    with np.load(path) as f:meta=json.loads(str(f['metadata'].item()));v=f['vector']
    meta['contract']['model']='different'
    np.savez(path,vector=v,metadata=json.dumps(meta))
    with pytest.raises(ValueError):cache.get('a')


def test_collector_stops_after_repeated_preview_failures(tmp_path,monkeypatch):
    import json,sys
    from python_mvp import sound_collect as collect
    run=tmp_path/'run';(run/'data').mkdir(parents=True)
    (run/'data/graph.json').write_text(json.dumps({'tracks':[dict(id=str(i),artist='A',title='T') for i in range(8)]}))
    class Search:
        def __init__(self,*args):pass
        def search(self,t):return dict(id=1,artist={'name':'A'},title='T',isrc='x',preview='https://example/a')
        def invalidate(self,t):invalidated.append(t['id'])
    calls=[];invalidated=[]
    def failing_build(*args,**kwargs):
        calls.append(1);return dict(encoded=0,failed=[dict(error='unavailable')])
    monkeypatch.setattr(collect,'PreviewSearch',Search)
    monkeypatch.setattr(collect,'ClapEncoder',lambda *args:object())
    monkeypatch.setattr(collect,'build',failing_build)
    monkeypatch.setattr(sys,'argv',['sound_collect','--run',str(run),'--cache',str(tmp_path/'cache'),'--encode'])
    with pytest.raises(ImportError,match='Five preview failures'):collect.main()
    assert len(calls)==5
    assert len(invalidated)==5


def test_search_cache_cannot_reuse_another_artist_title_pair(tmp_path,monkeypatch):
    import json,time
    from python_mvp.sound_collect import PreviewSearch
    import urllib.request
    answers=iter([dict(data=[dict(id=1,artist={'name':'A B'},title='C',isrc='one',preview='https://cdn/one')]),
                  dict(data=[dict(id=2,artist={'name':'A'},title='B C',isrc='two',preview='https://cdn/two')])])
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,*args):return json.dumps(next(answers)).encode()
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:Response())
    monkeypatch.setattr(time,'sleep',lambda *a:None)
    search=PreviewSearch(tmp_path)
    assert search.search(dict(artist='A B',title='C'))['id']==1
    assert search.search(dict(artist='A',title='B C'))['id']==2


def test_search_refreshes_expiring_signed_preview(tmp_path,monkeypatch):
    import json,time,urllib.request
    from python_mvp.sound_collect import PreviewSearch
    clock=[1000.];calls=[]
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,*args):
            calls.append(1)
            return json.dumps(dict(data=[dict(id=len(calls),artist={'name':'A'},title='B',
                isrc='same',preview=f'https://cdn/audio?hdnea=exp={int(clock[0]+120)}~acl=x')])).encode()
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:Response())
    monkeypatch.setattr(time,'time',lambda:clock[0]);monkeypatch.setattr(time,'sleep',lambda *a:None)
    search=PreviewSearch(tmp_path);track=dict(artist='A',title='B')
    assert search.search(track)['id']==1
    clock[0]+=60
    assert search.search(track)['id']==1
    clock[0]+=70
    assert search.search(track)['id']==2
    search.invalidate(track)
    assert search.search(track)['id']==3

import threading
from pathlib import Path


def test_ordered_prefetch_overlaps_but_bounds_work():
    from python_mvp.sound_prefetch import ordered_prefetch
    gate=threading.Barrier(3,timeout=3)
    active=[];lock=threading.Lock()
    def prepare(i):
        with lock:active.append(i)
        if i<3:gate.wait()
        return i*2
    with ordered_prefetch(range(8),prepare,3) as results:
        assert next(results)==0
        assert len(active)==3 # no whole-catalog submission
        assert [0]+list(results)==list(range(0,16,2))


def test_prefetch_cleans_up_on_consumer_error(tmp_path):
    from python_mvp.sound_prefetch import preview_jobs
    from python_mvp.sound import SoundCache
    class Search:
        def search(self,t):return dict(preview='https://example/clip')
    paths=[]
    def fetch(url,path):path.write_bytes(b'clip');paths.append(path)
    try:
        with preview_jobs([dict(id=str(i)) for i in range(10)],SoundCache(tmp_path),Search(),3,fetch) as jobs:
            first=next(jobs)
            assert first.path.read_bytes()==b'clip'
            raise RuntimeError('stop')
    except RuntimeError:pass
    assert paths and all(not p.exists() for p in paths)

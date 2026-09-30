"""Versioned real-audio vectors and an optional discovery gate. No downloads."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import numpy as np
from .recommend import match_key

MODEL = 'laion/larger_clap_music'
REVISION = 'a0b4534a14f58e20944452dff00a22a06ce629d1'
WEIGHTS_SHA256 = '5c289311f4a030d768af7ffbfdecd01b008aa64824211899a4e59f4f9d154fd1'
CONTRACT = dict(model=MODEL, revision=REVISION, weights_sha256=WEIGHTS_SHA256,
                preprocessing='hf-4.57.6-clap-48k-mono-1x10s-center-repeatpad-v1',dimensions=512)
DEFAULT_CACHE = Path(__file__).resolve().parent / 'data/cache/sound-clap-v1'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class SoundCache:
    """One atomic float32 record per track; no model or waveform loaded at serving."""
    def __init__(self,root=DEFAULT_CACHE):
        self.root=Path(root)

    def path(self,track_id):
        return self.root/(hashlib.sha256(str(track_id).encode()).hexdigest()+'.npz')

    def read(self,track_id):
        path=self.path(track_id)
        if not path.exists():return None
        try:
            with np.load(path,allow_pickle=False) as data:
                meta=json.loads(str(data['metadata'].item()))
                vector=data['vector'].astype(np.float32)
            if meta['contract']!=CONTRACT or meta['track_id']!=str(track_id):
                raise ValueError('Incompatible audio cache contract')
            if vector.shape!=(512,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector),1,atol=1e-4):
                raise ValueError('Invalid audio vector')
            return vector,meta
        except (OSError,KeyError,ValueError) as exc:
            raise ValueError(f'Invalid audio cache {path}: {exc}') from exc

    def get(self,track_id):
        record=self.read(track_id)
        return None if record is None else record[0]

    def current(self,track_id,file_sha256):
        record=self.read(track_id)
        return record is not None and record[1]['source'].get('file_sha256')==file_sha256

    def put(self,track_id,vector,source):
        v=np.asarray(vector,dtype=np.float32)
        if v.shape!=(512,) or not np.isfinite(v).all() or np.linalg.norm(v)<1e-8:
            raise ValueError('Expected a finite nonzero 512D audio vector')
        v=v/np.linalg.norm(v)
        meta=dict(contract=CONTRACT,track_id=str(track_id),source=source)
        self.root.mkdir(parents=True,exist_ok=True)
        fd,name=tempfile.mkstemp(dir=self.root,suffix='.tmp')
        try:
            with os.fdopen(fd,'wb') as f:
                np.savez(f,vector=v,metadata=json.dumps(meta,sort_keys=True))
                f.flush();os.fsync(f.fileno())
            os.replace(name,self.path(track_id))
        finally:
            if os.path.exists(name):os.unlink(name)


def filter_pool(pool,seeds,cache,drop_fraction=.20):
    """Remove the weakest audio fifth per direction if >=10 comparisons exist.

    A relative product policy, NOT a calibrated probability or genre verdict.
    Missing audio never rejects a track. All same-direction seed vectors compete.
    """
    if not 0<=drop_fraction<=.5:raise ValueError('drop_fraction must be between 0 and .5')
    seed_vectors={}; loaded={}
    for i in sorted(set(seeds),key=lambda i:pool.tracks[i]['id']):
        t=pool.tracks[i];v=cache.get(t['id'])
        if v is not None:seed_vectors.setdefault(match_key(t['artist']),[]).append((t['id'],v))
    report=dict(model=MODEL,policy='relative-bottom-fifth-v1',drop_fraction=drop_fraction,
                seeds_with_audio=sum(map(len,seed_vectors.values())),seed_count=len(set(seeds)),
                compared_edges=0,rejected_edges=0,missing_audio_edges=0,directions={})
    for direction,group in sorted(pool.groups.items()):
        measured={};sv=seed_vectors.get(direction,[])
        for i,item in group.items():
            item['audio_compared']=False
            if not sv:
                report['missing_audio_edges']+=1;continue
            tid=pool.tracks[i]['id']
            if tid not in loaded:loaded[tid]=cache.get(tid)
            v=loaded[tid]
            if v is None:
                report['missing_audio_edges']+=1;continue
            similarity,seed_id=max((float(v@u),sid) for sid,u in sv)
            measured[i]=similarity
            item.update(audio_compared=True,audio_similarity=similarity,audio_seed_id=seed_id)
        cutoff=float(np.quantile(list(measured.values()),drop_fraction)) if len(measured)>=10 else None
        rejected=[i for i,s in measured.items() if cutoff is not None and s<cutoff]
        for i in rejected:del group[i]
        report['compared_edges']+=len(measured);report['rejected_edges']+=len(rejected)
        report['directions'][direction]=dict(compared=len(measured),rejected=len(rejected),cutoff=cutoff)
    pool.candidates={i for group in pool.groups.values() for i in group}
    pool.unsupported=sorted(a for a,g in pool.groups.items() if not g)
    report['candidate_count_after']=len(pool.candidates)
    return report

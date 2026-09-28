"""Export ONLY compact inference artifacts; no CLAP/audio/history payloads."""
import json
from pathlib import Path
import numpy as np
from .sound import sha256
from .joint import DIM,K


def export_browser(root,pack,z,weights,status,report):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    np.asarray(z,dtype='<f4').tofile(root/'vectors.f32')
    np.asarray(pack['neighbors'],dtype='<i4').tofile(root/'neighbors.i32')
    np.asarray(pack['x'][:,[609,608]],dtype='<f4').tofile(root/'info.f32')
    catalog=dict(tracks=[{k:t[k] for k in ('id','artist','title','aliases') if k in t} for t in pack['tracks']],
                 artists=pack['artists'].tolist(),families=pack['families'].tolist())
    (root/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,separators=(',',':')))
    (root/'ranker.json').write_text(json.dumps(weights,separators=(',',':')))
    files=['vectors.f32','neighbors.i32','info.f32','catalog.json','ranker.json']
    sizes={f:(root/f).stat().st_size for f in files};total=sum(sizes.values())
    if total>32*1024*1024:raise ValueError(f'Browser budget exceeded: {total} bytes. Do not publish this export.')
    manifest=dict(schema='joint-browser-v1',tracks=len(z),dimensions=DIM,neighbors=K,status=status,
                  bytes=total,sizes=sizes,sha256={f:sha256(root/f) for f in files},evaluation=report)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest

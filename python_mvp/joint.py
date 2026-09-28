"""Compact learned graph+audio ranking, shared mathematical contract with browser."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DIM=32
K=24
MAX_SEEDS=64


class JointRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.track=nn.Sequential(nn.Linear(610,64),nn.ReLU(),nn.Linear(64,DIM))
        self.rank=nn.Sequential(nn.Linear(6,16),nn.ReLU(),nn.Linear(16,1))

    def encode(self,x):return F.normalize(self.track(x),dim=-1)

    def score(self,seeds,candidates,extra,mask=None):
        sim=candidates@seeds.transpose(-1,-2)
        if mask is None:mask=torch.ones(seeds.shape[:-1],device=seeds.device,dtype=torch.bool)
        maximum=sim.masked_fill(~mask[:,None,:],-2).max(-1).values
        mean=(sim*mask[:,None,:]).sum(-1)/mask.sum(-1)[:,None].clamp_min(1)
        return self.rank(torch.cat((maximum[...,None],mean[...,None],extra),-1)).squeeze(-1)


def select_seeds(seeds,artists):
    groups={}
    key=lambda i:((int(i)+1)*2654435761)&0xffffffff
    for i in sorted(set(map(int,seeds)),key=key):groups.setdefault(int(artists[i]),[]).append(i)
    queues=sorted(groups.values(),key=lambda q:key(q[0]))
    result=[]
    for pos in range(max(map(len,queues),default=0)):
        for q in queues:
            if pos<len(q):result.append(q[pos])
            if len(result)==MAX_SEEDS:return result
    return result


def retrieve(seeds,neighbors,families,artists):
    blocked={int(families[i]) for i in seeds}
    chosen=select_seeds(seeds,artists)
    return sorted({int(i) for i in neighbors[chosen].reshape(-1) if i>=0 and int(families[i]) not in blocked})


def extras(seeds,candidates,artists,pop,audio):
    familiar=np.isin(artists[candidates],artists[seeds]).astype(np.float32)
    return np.column_stack((pop[candidates],familiar,audio[candidates],
                            np.full(len(candidates),audio[seeds].mean(),dtype=np.float32))).astype(np.float32)


def numpy_scores(z,seeds,candidates,extra,weights):
    if not len(candidates):return np.empty(0,np.float32)
    sim=z[candidates]@z[seeds].T
    x=np.column_stack((sim.max(1),sim.mean(1),extra))
    h=np.maximum(x@np.asarray(weights['w1']).T+weights['b1'],0)
    return (h@np.asarray(weights['w2']).T+weights['b2']).ravel()


def select(ids,scores,tracks,families,artists,size=50):
    # Only identity and repetition safeguards. No genre, audio percentile or popularity quota.
    counts={};used=set();result=[];previous=None
    remaining=sorted(ids,key=lambda i:(-float(scores[i]),tracks[i]['id']))
    while remaining and len(result)<size:
        position=next((j for j,i in enumerate(remaining) if families[i] not in used and
                       counts.get(int(artists[i]),0)<2 and int(artists[i])!=previous),None)
        if position is None:break
        i=remaining.pop(position);a=int(artists[i]);result.append(i)
        used.add(families[i]);counts[a]=counts.get(a,0)+1;previous=a
    return result


class JointEngine:
    def __init__(self,root):
        from .sound import sha256
        root=Path(root);self.manifest=json.loads((root/'manifest.json').read_text())
        if self.manifest['schema']!='joint-browser-v1':raise ValueError('Unknown joint model schema')
        for name,digest in self.manifest['sha256'].items():
            if sha256(root/name)!=digest:raise ValueError(f'Corrupt joint artifact: {name}')
        catalog=json.loads((root/'catalog.json').read_text())
        self.tracks=catalog['tracks'];n=len(self.tracks)
        self.z=np.fromfile(root/'vectors.f32',dtype='<f4').reshape(n,DIM)
        self.neighbors=np.fromfile(root/'neighbors.i32',dtype='<i4').reshape(n,K)
        self.artists=np.asarray(catalog['artists'],dtype=np.int32)
        self.families=np.asarray(catalog['families'],dtype=np.int32)
        info=np.fromfile(root/'info.f32',dtype='<f4').reshape(n,2)
        self.pop,self.audio=info[:,0],info[:,1]
        self.weights=json.loads((root/'ranker.json').read_text())
        self.embeddings=torch.from_numpy(self.z.copy())
        self.refinement_status=self.manifest['status']

    def features(self,seeds,user=None):
        candidates=retrieve(seeds,self.neighbors,self.families,self.artists)
        selected=select_seeds(seeds,self.artists)
        x=extras(selected,candidates,self.artists,self.pop,self.audio)
        scores=numpy_scores(self.z,selected,candidates,x,self.weights)
        features=np.full((len(self.tracks),1),-1e9,dtype=np.float32)
        features[candidates,0]=scores
        return features,set(candidates),{}

    def score(self,model,features):return features[:,0].copy()

    def select_playlist(self,scores,candidates,lines,**kwargs):
        from .recommend import known_track_indices
        from .discovery_engine import input_keys,song_key
        keys=input_keys(lines)
        known=known_track_indices(lines,self.tracks)
        known.update(i for i,t in enumerate(self.tracks) if
                     ({song_key(t['artist'],t['title'])}|{song_key(*a) for a in t.get('aliases',[])})&keys)
        blocked=set(self.families[sorted(known)])
        return select([i for i in candidates if self.families[i] not in blocked],scores,self.tracks,self.families,self.artists)

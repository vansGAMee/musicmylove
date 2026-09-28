"""Train compact graph+audio neural ranking on disjoint users; export browser assets."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .joint import JointRanker,K,DIM,MAX_SEEDS,select_seeds,retrieve,extras,select
from .sound import SoundCache,DEFAULT_CACHE,CONTRACT,sha256
from .honest_training import mapped_users,raw_protection,episode
from .evaluate import metrics,paired_interval


def pairs(targets,known,candidates,families,rng):
    protected=set(families[sorted(known)])
    tf=set(families[sorted(targets)])
    positives=sorted({int(families[i]):i for i in sorted(candidates,reverse=True) if families[i] in tf}.values())
    negatives=[i for i in candidates if families[i] not in protected]
    if not positives or not negatives:return [],[]
    p=rng.choice(positives,min(8,len(positives)),replace=False)
    return np.repeat(p,4).tolist(),rng.choice(negatives,len(p)*4).tolist()


def atomic_torch(path,value):
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def prepare(source,cache,run):
    from .cli import open_engine
    from .ranker_data import load_ranker_data
    engine,_=open_engine(source,None,inference_only=True)
    engine.data=load_ranker_data(source/'data/dataset.json',engine.meta)
    n=len(engine.tracks);x=np.zeros((n,610),np.float32)
    x[:,:96]=engine.embeddings.numpy()
    sound=SoundCache(cache);audio_count=0
    for i,t in enumerate(engine.tracks):
        v=sound.get(t['id'])
        if v is not None:x[i,96:608]=v;x[i,608]=1;audio_count+=1
    if audio_count<100:raise ValueError('Need at least 100 real audio vectors; continue existing collection first')
    x[:,609]=np.log1p(engine.pop)/max(1,float(np.log1p(engine.pop).max()))
    neighbors=np.full((n,K),-1,np.int32)
    affinity=engine.graph['global'].multiply(engine.graph['confidence']).tocsr()
    for i in range(n):
        row=affinity.getrow(i)
        weights={int(j):float(v) for j,v in zip(row.indices,row.data) if j!=i and v>0}
        ids=sorted(weights,key=lambda j:(-weights[j],engine.tracks[j]['id']))
        novel=[j for j in ids if engine.artist_ids[j]!=engine.artist_ids[i]][:16]
        own=[j for j in ids if engine.artist_ids[j]==engine.artist_ids[i]][:8]
        chosen=novel+own;neighbors[i,:len(chosen)]=chosen
    protected=raw_protection(engine.data,engine.meta);rng=np.random.default_rng(42)
    train=[];dev=[];stats=dict(audio_tracks=audio_count,tracks=n,train_users=0,dev_users=0,skipped_train_episodes=0)
    train_users=mapped_users(engine.data,engine.meta,'ranker_train')
    dev_users=mapped_users(engine.data,engine.meta,'dev')
    if set(u for u,_ in train_users)&(set(engine.user_index)|set(u for u,_ in dev_users)):
        raise ValueError('User split overlap')
    if set(u for u,_ in dev_users)&set(engine.user_index):raise ValueError('DEV entered representation graph')
    for split,users in [('train',train_users),('dev',dev_users)]:
        for number,(uid,known) in enumerate(users):
            for discovery in (False,True):
                seeds,targets=episode(known,engine.tracks,engine.families,rng,discovery)
                if not seeds or not targets:continue
                chosen=select_seeds(seeds,engine.artist_ids)
                candidates=retrieve(seeds,neighbors,engine.families,engine.artist_ids)
                if split=='train':
                    p,neg=pairs(targets,known|protected.get(uid,set()),candidates,engine.families,rng)
                    if not p:stats['skipped_train_episodes']+=1
                    for a,b in zip(p,neg):train.append((chosen,a,b))
                else:dev.append(dict(user=uid,seeds=chosen,candidates=candidates,targets=sorted(targets),discovery=discovery))
            if (number+1)%500==0:print(f'Prepare {split}: {number+1}/{len(users)} users',flush=True)
        stats[split+'_users']=len(users)
    if not train or len(dev)<20:raise ValueError('Insufficient independent training/DEV examples')
    seeds=np.zeros((len(train),MAX_SEEDS),np.int64);mask=np.zeros_like(seeds,dtype=bool)
    pos=[];neg=[]
    for i,(s,p,q) in enumerate(train):seeds[i,:len(s)]=s;mask[i,:len(s)]=True;pos.append(p);neg.append(q)
    pack=dict(x=x,neighbors=neighbors,artists=engine.artist_ids.astype(np.int32),families=engine.families,
              tracks=engine.tracks,seeds=seeds,mask=mask,pos=np.asarray(pos),neg=np.asarray(neg),dev=dev,stats=stats)
    atomic_torch(run/'prepared.pt',pack)
    return pack


def evaluate(model,x,pack,device):
    with torch.no_grad():z=model.encode(x)
    results={}
    pop=x[:,609].cpu().numpy();audio=x[:,608].cpu().numpy()
    for q in pack['dev']:
        ids=q['candidates'];seeds=q['seeds'];value=0.
        if ids:
            extra=extras(seeds,ids,pack['artists'],pop,audio)
            with torch.no_grad():scores=model.score(z[seeds][None],z[ids][None],torch.tensor(extra[None],device=device))[0].cpu().numpy()
            all_scores=np.full(len(x),-1e9);all_scores[ids]=scores
            ordered=select(ids,all_scores,pack['tracks'],pack['families'],pack['artists'])
            target={int(pack['families'][i]) for i in q['targets']}
            value=metrics([int(pack['families'][i]) for i in ordered],target)['NDCG@50']
        results.setdefault(q['user'],[]).append(value)
    return {u:float(np.mean(v)) for u,v in results.items()}


def fit(pack,run,device,epochs,audio=True):
    name='joint' if audio else 'graph-only';path=run/(name+'.pt')
    if path.exists():return torch.load(path,map_location='cpu',weights_only=False)
    torch.manual_seed(42)
    x=torch.tensor(pack['x'],device=device)
    if not audio:x[:,96:609]=0
    model=JointRanker().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    artists=torch.tensor(pack['artists'],device=device)
    seed=torch.tensor(pack['seeds'],device=device);mask=torch.tensor(pack['mask'],device=device)
    pos=torch.tensor(pack['pos'],device=device);neg=torch.tensor(pack['neg'],device=device)
    best=-1;state=None;history=[];stale=0
    for epoch in range(epochs):
        model.train();losses=[]
        for batch in torch.randperm(len(pos),device=device).split(512):
            si=seed[batch];ma=mask[batch];ids=torch.stack((pos[batch],neg[batch]),1)
            zs=model.encode(x[si]);zc=model.encode(x[ids])
            familiar=((artists[ids][:,:,None]==artists[si][:,None,:])&ma[:,None,:]).any(-1).float()
            coverage=(x[si,608]*ma).sum(1)/ma.sum(1)
            extra=torch.stack((x[ids,609],familiar,x[ids,608],coverage[:,None].expand(-1,2)),dim=-1)
            score=model.score(zs,zc,extra,ma)
            loss=F.softplus(score[:,1]-score[:,0]).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            losses.append(float(loss.detach()))
        model.eval();report=evaluate(model,x,pack,device);quality=float(np.mean(list(report.values())))
        row=dict(model=name,epoch=epoch+1,loss=float(np.mean(losses)),dev_ndcg50=quality)
        history.append(row);print(json.dumps(row),flush=True)
        if quality>best+1e-6:
            best=quality;stale=0;state=dict(weights=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()}),dev=report,epoch=epoch+1)
        else:stale+=1
        if stale>=3:break
    state['history']=history;atomic_torch(path,state)
    return state


def export(pack,state,run,status,report):
    from .joint_export import export_browser
    model=JointRanker();model.load_state_dict(state['weights']);model.eval()
    with torch.no_grad():z=model.encode(torch.tensor(pack['x'])).numpy()
    weights=dict(w1=model.rank[0].weight.detach().tolist(),b1=model.rank[0].bias.detach().tolist(),
                 w2=model.rank[2].weight.detach().tolist(),b2=model.rank[2].bias.detach().tolist())
    export_browser(run/'browser',pack,z,weights,status,report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('python_mvp/data/targeted-v1/model'))
    p.add_argument('--run',type=Path,default=Path('python_mvp/data/joint-v1'))
    p.add_argument('--sound-cache',type=Path,default=DEFAULT_CACHE)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--epochs',type=int,default=12)
    args=p.parse_args()
    if args.epochs<1:p.error('epochs must be positive')
    args.run.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (args.run/'training.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4)
        if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        contract=dict(source=str(args.source.resolve()),representation=sha256(args.source/'representation.pt'),
                      graph=sha256(args.source/'data/graph.json'),dataset=sha256(args.source/'data/dataset.json'),
                      sound=CONTRACT,code={f:sha256(Path(__file__).parent/f) for f in ['joint.py','joint_training.py','joint_export.py']})
        manifest=args.run/'training-contract.json'
        if manifest.exists() and json.loads(manifest.read_text())!=contract:raise ValueError('Changed source/code: choose a new --run')
        manifest.write_text(json.dumps(contract,indent=2))
        if (args.run/'prepared.pt').exists():pack=torch.load(args.run/'prepared.pt',weights_only=False)
        else:pack=prepare(args.source,args.sound_cache,args.run)
        print(json.dumps(pack['stats']),flush=True)
        joint=fit(pack,args.run,args.device,args.epochs,True)
        baseline=fit(pack,args.run,args.device,args.epochs,False)
        users=sorted(joint['dev']);a=[joint['dev'][u] for u in users];b=[baseline['dev'][u] for u in users]
        interval=paired_interval(a,b)
        report=dict(joint_ndcg50=float(np.mean(a)),graph_only_ndcg50=float(np.mean(b)),
                    paired_interval=interval,dev_users=len(users),final_used=False,audio_snapshot_tracks=pack['stats']['audio_tracks'])
        status='JOINT_DEV_GAIN' if interval['ci95'][0]>0 else 'JOINT_NO_PROVEN_DEV_GAIN'
        export(pack,joint,args.run,status,report)
        (args.run/'evaluation.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2));print(f'Browser export: {args.run / "browser"}')


if __name__=='__main__':main()

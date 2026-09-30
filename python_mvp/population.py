"""Train on verified non-targeted archives; reserve users before joint training.

No downloads and no personal-library argument. Reuses the neutral graph and
existing measured audio, never the personalized joint weights or old ranker.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .config import ROOT, fingerprint, write_json
from .sound import DEFAULT_CACHE, CONTRACT

POLICY='population-ranker-holdout-v1'


def split_ranker_users(users):
    train=[];audit=[]
    for uid,known in sorted(users):
        bucket=int(hashlib.sha256((POLICY+':'+uid).encode()).hexdigest()[:16],16)%5
        (audit if bucket==0 else train).append((uid,known))
    return train,audit


def verify_source(data_source,model):
    """Fail closed on targeted enrichment, unknown lineage or changed artifacts."""
    data_source,model=Path(data_source).resolve(),Path(model).resolve()
    if (data_source/'targeting.json').exists() or (data_source/'reports/targeted_collection.json').exists():
        raise ValueError('Rejecting personal-library-targeted source')
    expansion=json.loads((data_source/'expansion.json').read_text())
    collection=json.loads((data_source/'reports/collection.json').read_text())
    complete=json.loads((data_source/'complete.json').read_text())
    audit=json.loads((data_source/'reports/data_audit.json').read_text())
    if expansion['user_fraction']!=1 or not collection or any(r['user_fraction']!=1 for r in collection):
        raise ValueError('Require all users from non-targeted archive extraction')
    if {r['archive']:r['archive_sha256'] for r in collection}!=expansion['archives']:
        raise ValueError('Archive provenance mismatch')
    if {str(Path(r['output']).resolve()):r['sha256'] for r in collection}!={s['path']:s['sha256_decoded'] for s in audit['sources']}:
        raise ValueError('Additional or changed listens entered neutral dataset')
    for name in ('data/dataset.json','data/split_manifest.json','reports/data_audit.json'):
        if fingerprint(data_source/name)!=complete.get(name):raise ValueError('Neutral dataset integrity mismatch: '+name)
    partition=json.loads((model/'partition.json').read_text())
    if partition['source_dataset_sha256']!=complete['data/dataset.json']:
        raise ValueError('Graph was trained on a different, possibly targeted dataset')
    from .resume_training import validate_checkpoint
    validate_checkpoint(model)  # Verifies frozen graph, partition, representation and code.
    return dict(policy=POLICY,source=str(model),neutral_source=str(data_source),
                neutral_dataset=complete['data/dataset.json'],representation=fingerprint(model/'representation.pt'),
                graph=fingerprint(model/'data/graph.json'),dataset=fingerprint(model/'data/dataset.json'),
                sound=CONTRACT,implementation={n:fingerprint(ROOT/n) for n in
                    ('population.py','joint.py','joint_training.py','joint_export.py','ranker_data.py','sound.py')})


def source_engine(source):
    """Only frozen representation inputs; no old ranker inference or raw sessions."""
    from .build_graph import load_graph
    from .honest_training import family_index
    from .recommend import match_key
    source=Path(source);meta,graph=load_graph(source/'data')
    state=torch.load(source/'representation.pt',map_location='cpu',weights_only=True)
    tracks=meta['tracks']
    if state['vocabulary']!=[t['id'] for t in tracks]:raise ValueError('Representation vocabulary mismatch')
    artists=sorted({match_key(t['artist']) for t in tracks});index={a:i for i,a in enumerate(artists)}
    return SimpleNamespace(meta=meta,graph=graph,tracks=tracks,embeddings=state['embeddings'],
                           families=family_index(tracks),artist_ids=np.array([index[match_key(t['artist'])] for t in tracks]),
                           pop=np.asarray(graph['user_incidence'].sum(0)).ravel(),user_index={u:i for i,u in enumerate(meta['train_users'])})


def query_metrics(candidates,ranked,targets,seeds,families,artists):
    from .evaluate import metrics
    target=set(families[list(targets)]);familiar=set(artists[seeds])
    return dict(ndcg50=metrics([int(families[i]) for i in ranked],target)['NDCG@50'],
                recall50=len(set(families[ranked])&target)/max(1,len(target)),
                candidate_recall=len(set(families[candidates])&target)/max(1,len(target)),
                new_artist_share=sum(artists[i] not in familiar for i in ranked)/max(1,len(ranked)),length=len(ranked))


def independent_report(pack,states):
    """Evaluate the two already selected checkpoints; do not select/retrain here."""
    from .joint import JointRanker, extras, numpy_scores, select
    from .evaluate import paired_interval
    encoded={}
    for name,state in states.items():
        model=JointRanker().eval();model.load_state_dict(state['weights'])
        x=torch.tensor(pack['x'])
        if name=='graph_only':x[:,96:609]=0
        with torch.no_grad():z=model.encode(x).numpy()
        weights=dict(w1=model.rank[0].weight.detach().numpy(),b1=model.rank[0].bias.detach().numpy(),
                     w2=model.rank[2].weight.detach().numpy(),b2=model.rank[2].bias.detach().numpy())
        encoded[name]=(z,weights,x[:,608].numpy())
    # Strata thresholds are computed only from representation-derived catalog popularity.
    pop=pack['x'][:,609];cutoffs=np.quantile(pop,[1/3,2/3]);groups={}
    for q in pack['audit']:
        ids=q['candidates'];seed=q['seeds'];full=q['full_seeds'];row={}
        for name,(z,weights,audio) in encoded.items():
            scores=np.full(len(z),-1e9);scores[ids]=numpy_scores(z,seed,ids,extras(seed,ids,pack['artists'],pop,audio),weights)
            ranked=select(ids,scores,pack['tracks'],pack['families'],pack['artists'])
            row[name]=query_metrics(ids,ranked,q['targets'],full,pack['families'],pack['artists'])
        task='new_artist' if q['discovery'] else 'regular'
        bucket=('lower_popularity','middle_popularity','higher_popularity')[int(np.searchsorted(cutoffs,float(np.mean(pop[full]))))]
        for group in (task,task+'/'+bucket):groups.setdefault(group,[]).append(row)
    report={}
    for group,rows in sorted(groups.items()):
        report[group]={key:dict(joint=float(np.mean([r['joint'][key] for r in rows])),
                                graph_only=float(np.mean([r['graph_only'][key] for r in rows])),
                                delta=paired_interval([r['joint'][key] for r in rows],[r['graph_only'][key] for r in rows]))
                       for key in rows[0]['joint']}
        report[group]['users']=len(rows)
    return dict(policy=POLICY,split='reserved_ranker_users',users=len({q['user'] for q in pack['audit']}),
                preparation=pack['stats'],
                final_used=False,used_for_checkpoint_selection=False,personal_library_used=False,
                population_scope='ListenBrainz archive users, not all music listeners',
                prior_project_exposure_possible=True,release_certified=False,metrics=report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-source',type=Path,default=ROOT/'data/expanded-v1')
    p.add_argument('--source',type=Path,default=ROOT/'data/expanded-v1/model')
    p.add_argument('--run',type=Path,default=ROOT/'data/population-v1')
    p.add_argument('--sound-cache',type=Path,default=DEFAULT_CACHE)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if args.epochs<1:p.error('epochs must be positive')
    if args.run.resolve() in (args.source.resolve(),args.data_source.resolve()):p.error('Use a separate run')
    print('Verifying neutral archives and graph provenance…',flush=True)
    contract=verify_source(args.data_source,args.source)
    contract.update(epochs=args.epochs,sound_cache=str(args.sound_cache.resolve()))
    partition=json.loads((args.source/'partition.json').read_text())
    train,audit=split_ranker_users([(u,set()) for u in partition['ranker_users']])
    print(json.dumps(dict(ranker_train_users=len(train),reserved_audit_users=len(audit),personal_library_used=False)),flush=True)
    if args.check_only:return
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU training')
    from .joint_training import prepare,fit,export
    import fcntl
    args.run.mkdir(parents=True,exist_ok=True)
    with (args.run/'training.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        marker=args.run/'population-contract.json'
        if marker.exists():
            if json.loads(marker.read_text())!=contract:raise ValueError('Population inputs/code changed: choose a new --run')
        else:
            if any(x.name!='training.lock' for x in args.run.iterdir()):raise ValueError('Run already contains another experiment')
            write_json(marker,contract)
        torch.set_num_threads(4)
        snapshot=args.run/'prepared.pt';seal=args.run/'snapshot.json'
        if snapshot.exists():
            digest=fingerprint(snapshot)
            if seal.exists():
                if json.loads(seal.read_text())!={'sha256':digest}:raise ValueError('Prepared training/audit snapshot changed')
            elif any((args.run/f).exists() for f in ('joint.pt','graph-only.pt')):
                raise ValueError('Checkpoint without a sealed training/audit snapshot')
            pack=torch.load(snapshot,weights_only=False)
        else:
            if seal.exists() or any((args.run/f).exists() for f in ('joint.pt','graph-only.pt')):
                raise ValueError('Training snapshot is missing; preserve this run')
            pack=prepare(args.source,args.sound_cache,args.run,population=True)
            digest=fingerprint(snapshot)
        write_json(seal,{'sha256':digest})
        print(json.dumps(pack['stats']),flush=True)
        joint=fit(pack,args.run,args.device,args.epochs,True)
        graph=fit(pack,args.run,args.device,args.epochs,False)
        report_path=args.run/'population-evaluation.json'
        hashes={name:fingerprint(args.run/file) for name,file in [('joint','joint.pt'),('graph_only','graph-only.pt')]}
        if report_path.exists():
            report=json.loads(report_path.read_text())
            if report['checkpoint_sha256']!=hashes:raise ValueError('Evaluated checkpoints changed')
        else:
            report=independent_report(pack,dict(joint=joint,graph_only=graph))
            report['checkpoint_sha256']=hashes
            write_json(report_path,report)
        export(pack,joint,args.run,'POPULATION_EXPERIMENT_NOT_RELEASE',report)
        print(json.dumps({task:report['metrics'][task] for task in ('regular','new_artist') if task in report['metrics']},indent=2),flush=True)
        print(f'Finished. Independent report: {report_path}\nBrowser model: {args.run / "browser"}',flush=True)
        print('No guaranteed gain and no automatic production deployment. Inspect regular/new-artist and population-stratum metrics.')


if __name__=='__main__':main()

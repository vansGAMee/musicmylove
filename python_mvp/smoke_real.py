"""Real-data engineering smoke, isolated from accepted checkpoints and frozen splits.
A successful run proves execution/backprop/serialization, NEVER recommendation quality.
"""
import json
import numpy as np
import torch
try:
    from .config import load_data, MODELS, REPORTS, deterministic, write_json
    from .build_graph import load_graph, adjacency
    from .networks import GraphEncoder, MultiInterest, NeuralRanker
    from .evaluate import queries, evaluate
    from .sampling import sample_negatives, protect
    from .retrieval import candidate_range, score_rows
    from .artifacts import contract
except ImportError:
    from config import load_data, MODELS, REPORTS, deterministic, write_json
    from build_graph import load_graph, adjacency
    from networks import GraphEncoder, MultiInterest, NeuralRanker
    from evaluate import queries, evaluate
    from sampling import sample_negatives, protect
    from retrieval import candidate_range, score_rows
    from artifacts import contract


def smoke():
    deterministic(42)
    data=load_data(); meta,graph=load_graph()
    ui,si=graph['user_incidence'],graph['session_incidence']
    ug,sg=adjacency(ui),adjacency(si)
    encoder=GraphEncoder(ui.shape[1],ui.shape[0],si.shape[0])
    taste,ranker=MultiInterest(),NeuralRanker()
    rng=np.random.default_rng(42)
    popularity=np.asarray(ui.sum(0)).ravel()
    graph_updates=0
    optimizer=torch.optim.AdamW(encoder.parameters(),lr=.003)
    for u in range(min(ui.shape[0],128)):
        known=set(ui.getrow(u).indices.tolist())
        if not known: continue
        positive=min(known)
        emb,users=encoder(ug,sg)
        neg=sample_negatives(rng,len(emb),known,protect([positive],graph),popularity,positive,(emb@users[u]).detach().numpy())
        if not len(neg): continue
        logits=(emb[[positive,*neg]]@users[u])/.1
        loss=-torch.log_softmax(logits,0)[0]
        optimizer.zero_grad();loss.backward();optimizer.step();graph_updates+=1
        if graph_updates==8: break
    with torch.no_grad(): emb,_=encoder(ug,sg)
    optimizer=torch.optim.AdamW(taste.parameters(),lr=.003)
    examples=list(queries(data,meta,'train'))[:32]
    taste_updates=0
    for q in examples:
        heads,masses,assignment=taste(emb,q['seeds'])
        positive=min(q['targets'])
        neg=sample_negatives(rng,len(emb),q['known'],protect([positive],graph),popularity,positive,(emb@heads.T).max(-1).values.detach().numpy())
        if not len(neg): continue
        logits=torch.logsumexp(emb[[positive,*neg]]@heads.T/.1 + masses.clamp_min(1e-8).log(),-1)
        loss=-torch.log_softmax(logits,0)[0]+taste.regularization(heads,masses,assignment)
        optimizer.zero_grad();loss.backward();optimizer.step();taste_updates+=1
    optimizer=torch.optim.AdamW(ranker.parameters(),lr=.003)
    ranker_updates=0
    for q in examples:
        with torch.no_grad():
            heads,masses,assignment=taste(emb,q['seeds'])
            rows=candidate_range(q['seeds'],heads,masses,assignment,emb,graph)
        protected=protect(sorted(q['targets']),graph)
        rows=[r for r in rows if r['track'] in q['targets'] or (r['track'] not in q['known'] and r['track'] not in protected)]
        ids=sorted({r['track'] for r in rows})
        positives=torch.tensor([t in q['targets'] for t in ids],dtype=torch.bool)
        if not positives.any() or positives.all(): continue
        per_head=score_rows(rows,ranker,emb,heads)
        scores=torch.stack([per_head[[i for i,r in enumerate(rows) if r['track']==t]].max() for t in ids])
        loss=torch.logsumexp(scores,0)-torch.logsumexp(scores[positives],0)
        optimizer.zero_grad();loss.backward();optimizer.step();ranker_updates+=1
    before=taste(emb,examples[0]['seeds'])[0] if examples else None
    location=MODELS/'diagnostic'/'smoke.pt';location.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'status':'DIAGNOSTIC_NOT_RELEASE','contract':contract(),'graph':encoder.state_dict(),
                'taste':taste.state_dict(),'ranker':ranker.state_dict(),'embeddings':emb},location)
    restored=torch.load(location,weights_only=True); taste.load_state_dict(restored['taste'])
    identical=before is not None and torch.equal(before,taste(restored['embeddings'],examples[0]['seeds'])[0])
    report={'status':'ENGINEERING_PASS' if min(graph_updates,taste_updates,ranker_updates)>0 and identical else 'ENGINEERING_INCOMPLETE',
            'quality_proven':False,'real_users':len(data['users']),'raw_tracks':len(data['tracks']),
            'seen_tracks':len(emb),'optimizer_updates':{'graph':graph_updates,'taste':taste_updates,'ranker':ranker_updates},
            'checkpoint_roundtrip_equal':bool(identical), 'fingerprint':data['fingerprint']}
    report['dev']=evaluate(data,meta,graph,embeddings=emb,taste=taste,ranker=ranker)
    write_json(REPORTS/'smoke_real.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='dev'},indent=2))
    if report['status']!='ENGINEERING_PASS': raise SystemExit(2)


if __name__=='__main__': smoke()

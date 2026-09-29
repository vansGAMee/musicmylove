import numpy as np
import pytest


def test_population_holdout_depends_only_on_user_identity_and_is_disjoint():
    from python_mvp.population import split_ranker_users
    users=[(str(i),{i,i+1}) for i in range(200)]
    train,audit=split_ranker_users(users)
    assert 20<len(audit)<70
    assert not {u for u,_ in train}&{u for u,_ in audit}
    assert len(train)+len(audit)==len(users)
    assert {u for u,_ in audit}=={u for u,_ in split_ranker_users([(u,{999}) for u,_ in reversed(users)])[1]}


def test_targeted_source_is_rejected_before_training(tmp_path):
    from python_mvp.population import verify_source
    (tmp_path/'targeting.json').write_text('{}')
    with pytest.raises(ValueError,match='personal'):
        verify_source(tmp_path,tmp_path/'model')


def test_audit_is_not_dev_and_is_not_used_by_fit(tmp_path):
    import torch
    from python_mvp.joint_training import fit
    rng=np.random.default_rng(5);n=40;x=rng.normal(size=(n,610)).astype(np.float32)
    x[:,608]=1;x[:,609]=.2
    pack=dict(x=x,artists=np.arange(n),families=np.arange(n),tracks=[dict(id=str(i)) for i in range(n)],
              seeds=np.tile([0,1],(8,1)),mask=np.ones((8,2),bool),pos=np.full(8,2),neg=np.arange(8)+3,
              dev=[dict(user='dev',seeds=[0,1],candidates=list(range(2,n)),targets=[2])],
              audit=[dict(user='audit',seeds=[0],candidates=[99999],targets=[99999])])
    torch.set_num_threads(2)
    state=fit(pack,tmp_path,'cpu',1)
    assert list(state['dev'])==['dev']  # Invalid audit IDs would crash if used in selection.


def test_population_evaluation_retains_unretrievable_targets():
    from python_mvp.population import query_metrics
    families=np.arange(10);artists=np.arange(10)
    v=query_metrics([2,3],[],{8},[0,1],families,artists)
    assert v['candidate_recall']==v['ndcg50']==v['recall50']==0
    assert v['length']==0


def test_preparation_keeps_reserved_users_out_of_training_and_dev(tmp_path,monkeypatch):
    import torch
    from scipy import sparse
    from types import SimpleNamespace
    from python_mvp import population,joint_training,ranker_data
    n=150;tracks=[dict(id=f't{i:03}',artist=f'a{i}',title=f's{i}') for i in range(n)]
    users=[dict(id='train',split='ranker_train',tracks=list(range(15)))]
    users += [dict(id=f'dev{i}',split='dev',tracks=list(range(15))) for i in range(10)]
    users += [dict(id=f'audit{i}',split='ranker_train',tracks=list(range(100,115))) for i in range(100)]
    meta=dict(seen=list(range(n)),tracks=tracks)
    engine=SimpleNamespace(tracks=tracks,meta=meta,embeddings=torch.zeros(n,96),pop=np.ones(n),
                           artist_ids=np.arange(n),families=np.arange(n),user_index={'representation':0},
                           graph={'global':sparse.csr_matrix(np.ones((n,n))), 'confidence':sparse.csr_matrix(np.ones((n,n)))})
    monkeypatch.setattr(population,'source_engine',lambda _:engine)
    monkeypatch.setattr(population,'split_ranker_users',lambda rows:([r for r in rows if r[0]=='train'],[r for r in rows if r[0]!='train']))
    monkeypatch.setattr(ranker_data,'load_ranker_data',lambda *args:dict(tracks=tracks,users=users))
    monkeypatch.setattr(joint_training,'SoundCache',lambda _:SimpleNamespace(get=lambda _:np.ones(512,dtype=np.float32)/np.sqrt(512)))
    pack=joint_training.prepare(tmp_path,tmp_path,tmp_path,population=True)
    assert set(pack['seeds'][pack['mask']]) <= set(range(15))
    assert set(pack['pos']) <= set(range(15))
    assert {q['user'] for q in pack['dev']}=={f'dev{i}' for i in range(10)}
    assert {q['user'] for q in pack['audit']}=={f'audit{i}' for i in range(100)}
    assert all(set(q['targets'])<=set(range(100,115)) for q in pack['audit'])


def test_population_report_uses_reserved_queries_and_keeps_tasks_separate():
    from python_mvp.population import independent_report
    from python_mvp.joint import JointRanker
    n=12;x=np.zeros((n,610),np.float32);x[:,609]=np.arange(n)/n
    query=dict(user='reserved',seeds=[0,1],full_seeds=[0,1],candidates=[2,3],targets=[2])
    pack=dict(x=x,artists=np.arange(n),families=np.arange(n),tracks=[dict(id=str(i)) for i in range(n)],stats={},
              audit=[dict(query,discovery=False),dict(query,discovery=True)],dev=[{'invalid':'must not read'}])
    state=dict(weights=JointRanker().state_dict())
    report=independent_report(pack,dict(joint=state,graph_only=state))
    assert report['users']==1 and not report['release_certified']
    assert report['metrics']['regular']['users']==report['metrics']['new_artist']['users']==1
    assert report['metrics']['regular']['ndcg50']['delta']['mean']==0


def test_population_command_exports_resumes_and_detects_snapshot_changes(tmp_path,monkeypatch):
    import json,sys
    from python_mvp import population,joint_training
    source=tmp_path/'source';source.mkdir()
    (source/'partition.json').write_text(json.dumps({'ranker_users':[str(i) for i in range(200)]}))
    run=tmp_path/'run';n=40;rng=np.random.default_rng(5)
    x=rng.normal(size=(n,610)).astype(np.float32);x[:,608]=1;x[:,609]=np.arange(n)/n
    query=dict(user='audit',seeds=[0,1],full_seeds=[0,1],candidates=list(range(2,n)),targets=[2])
    pack=dict(x=x,neighbors=np.tile(np.arange(2,26,dtype=np.int32),(n,1)),artists=np.arange(n),families=np.arange(n),
              tracks=[dict(id=str(i),artist=str(i),title='song') for i in range(n)],
              seeds=np.tile([0,1],(8,1)),mask=np.ones((8,2),bool),pos=np.full(8,2),neg=np.arange(8)+3,
              dev=[dict(query,user='dev')],audit=[dict(query,discovery=False),dict(query,discovery=True)],stats={})
    calls=[]
    def prepare(source,cache,run,*,population):
        assert population;calls.append(1);joint_training.atomic_torch(run/'prepared.pt',pack);return pack
    monkeypatch.setattr(joint_training,'prepare',prepare)
    monkeypatch.setattr(population,'verify_source',lambda *args:dict(policy='test'))
    monkeypatch.setattr(sys,'argv',['population','--source',str(source),'--run',str(run),'--device','cpu','--epochs','1'])
    population.main()
    report=(run/'population-evaluation.json').read_bytes()
    assert (run/'browser/manifest.json').exists()
    population.main()
    assert len(calls)==1 and (run/'population-evaluation.json').read_bytes()==report
    with (run/'prepared.pt').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='snapshot changed'):population.main()

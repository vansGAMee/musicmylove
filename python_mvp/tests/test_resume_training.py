import json
import numpy as np
import pytest
import torch
from scipy import sparse


def partial_run(path):
    from python_mvp.config import deterministic, write_json
    from python_mvp.honest_training import SCHEMA, implementation
    from python_mvp.networks import MultiInterest
    deterministic(7)
    tracks = [dict(id=str(i), artist=f'artist{i}', title=str(i)) for i in range(12)]
    rows = [list(range(i, i+6)) for i in range(6)]
    ui = sparse.csr_matrix(([1.] * 36, ([i for i in range(6) for _ in range(6)], sum(rows, []))), shape=(6,12))
    co = (ui.T @ ui).tocsr(); co.setdiag(0); co.eliminate_zeros()
    confidence = co.copy(); confidence.data[:] = .5
    graph = dict(user_incidence=ui, session_incidence=ui, users=co, sessions=co,
                 confidence=confidence, **{'global':co,'local':co})
    meta = dict(fingerprint='fixture', seen=list(range(12)), tracks=tracks,
                train_users=[f'rep{i}' for i in range(6)])
    data = dict(fingerprint='fixture', tracks=tracks, users=[
        dict(id=f'rep{i}',split='train',tracks=row,sessions=[row]) for i,row in enumerate(rows)])
    data['users'] += [dict(id=split+str(i),split=split,tracks=list(range(i%4,i%4+7)))
                     for split in ('ranker_train','dev') for i in range(10)]
    write_json(path/'data/dataset.json',data)
    write_json(path/'data/graph.json',meta)
    write_json(path/'partition.json',dict(schema=SCHEMA, representation_users=meta['train_users'],
               ranker_users=[f'ranker_train{i}' for i in range(10)], config=dict(epochs=1,train_users=10,seed=7)))
    for key,matrix in graph.items():
        sparse.save_npz(path/'data'/f'{key}.npz',matrix)
    torch.save(dict(embeddings=torch.nn.functional.normalize(torch.randn(12,96),dim=1),
                    taste=MultiInterest().state_dict(), vocabulary=[t['id'] for t in tracks],
                    implementation=implementation()),path/'representation.pt')
    write_json(path/'training.json',[{'epoch':3,'loss':.5}])
    (path/'reports').mkdir(); (path/'reports/final_consumed.lock').write_text('preserve')


def test_recovery_keeps_representation_and_old_log_and_serves_result(tmp_path, monkeypatch):
    from python_mvp import resume_training as recovery
    from python_mvp import honest_training
    from python_mvp.config import fingerprint
    from python_mvp.cli import open_engine
    partial_run(tmp_path)
    original = {n:fingerprint(tmp_path/n) for n in ('representation.pt','data/dataset.json','reports/final_consumed.lock')}
    monkeypatch.setattr('sys.argv',['resume_training','--run',str(tmp_path),'--check-only'])
    recovery.main()
    assert not (tmp_path/'recovery.json').exists()
    monkeypatch.setattr('sys.argv',['resume_training','--run',str(tmp_path)])
    real_fit = honest_training.train_ranker
    def interrupted(*args,**kwargs):
        raise RuntimeError('synthetic interruption')
    monkeypatch.setattr(honest_training,'train_ranker',interrupted)
    with pytest.raises(RuntimeError,match='synthetic interruption'):
        recovery.main()
    assert (tmp_path/'recovery.json').exists()
    assert not (tmp_path/'ranker.pt').exists()
    monkeypatch.setattr(honest_training,'train_ranker',real_fit)
    recovery.main()
    assert (tmp_path/'ranker.pt').exists()
    assert json.loads((tmp_path/'recovery-original/training.json').read_text()) == [{'epoch':3,'loss':.5}]
    assert original == {n:fingerprint(tmp_path/n) for n in original}
    assert json.loads((tmp_path/'evaluation.json').read_text())['frozen_final_used'] is False
    engine, model = open_engine(tmp_path,None,inference_only=True)
    assert engine.data == {}  # inference keeps no raw catalog or listener histories
    x, _, _ = engine.features([0,2])
    assert np.isfinite(engine.score(model,x)).all()
    full_engine, full_model = open_engine(tmp_path,None)
    np.testing.assert_allclose(engine.score(model,x),full_engine.score(full_model,x))
    # Completed reruns do not train again, or even load the raw dataset.
    monkeypatch.setattr(honest_training,'train_ranker',interrupted)
    recovery.main()
    # A saved ranker can finish evaluation after an interruption without retraining.
    (tmp_path/'evaluation.json').unlink()
    recovery.main()
    assert (tmp_path/'evaluation.json').exists()
    (tmp_path/'representation.pt').write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='changed'):
        recovery.main()

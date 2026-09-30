"""Regression tests for actual discovery and honest artifact acceptance."""
import copy
import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix


def test_typographic_matching_and_ambiguous_known_exclusion():
    from python_mvp.recommend import resolve, known_track_indices
    tracks = [{'id': 'a', 'artist': 'X', 'title': "Don’t Stop"},
              {'id': 'b', 'artist': 'X', 'title': "Don’t Stop"}]
    assert resolve([" X — Don't Stop "], tracks) == ([], ["X — Don't Stop"])
    assert known_track_indices(["X – Don't Stop"], tracks) == {0, 1}
    assert resolve(["X — Don't Stop"], tracks[:1])[0] == [0]


def test_contract_detects_implementation_change():
    from python_mvp.artifacts import validate_contract
    old = {'schema': 2, 'model': {}, 'artifacts': {}, 'implementation': {'retrieval.py': 'old'}}
    new = copy.deepcopy(old)
    new['implementation']['retrieval.py'] = 'new'
    with pytest.raises(RuntimeError, match='implementation'):
        validate_contract(old, new)


def test_stability_exception_is_failure(monkeypatch):
    from python_mvp import verify_release as v
    def broken(*args, **kwargs):
        raise RuntimeError('broken checkpoint')
    monkeypatch.setattr(v, 'recommend', broken)
    ok, message = v.check_gate_5_stability({'tracks': [{'id': str(i)} for i in range(100)]})
    assert not ok
    assert 'broken checkpoint' in message


def test_candidates_respect_radius_and_include_every_active_interest():
    from python_mvp.retrieval import candidate_range
    matrix = csr_matrix([[0., 1., 1.], [1., 0., 0.], [1., 0., 0.]])
    graph = {k: matrix * (4 if k == 'users' else 1) for k in ('global', 'local', 'users', 'sessions', 'confidence')}
    emb = torch.tensor([[1., 0.], [.8, .6], [-1., 0.]])
    heads = torch.tensor([[1., 0.], [.8, .6]])
    rows = candidate_range([0, 2], heads, torch.tensor([.5, .5]), torch.eye(2), emb, graph, radius=.5)
    assert {(r['track'], r['head']) for r in rows} == {(1, 0), (1, 1)}
    assert candidate_range([0], heads[:1], torch.ones(1), torch.ones(1, 1), emb, graph, radius=.9) == []


def test_neural_fusion_can_learn_negative_evidence_effect():
    from python_mvp.networks import NeuralRanker
    torch.manual_seed(2)
    model = NeuralRanker(dim=2)
    candidates = torch.ones(2, 2)
    heads = candidates.clone()
    evidence = torch.zeros(2, 6)
    evidence[1, 3] = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    for _ in range(40):
        scores = model(candidates, heads, evidence)
        loss = torch.nn.functional.softplus(scores[1] - scores[0])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert model(candidates, heads, evidence)[0] > model(candidates, heads, evidence)[1] + 1


def test_discovery_episode_holds_out_whole_artist_without_negative_leakage():
    from python_mvp.sampling import profile_episode
    tracks = [{'artist': a} for a in ['A', 'A', 'B', 'B', 'C', 'C']]
    known = set(range(6))
    seeds, targets = profile_episode(known, tracks, np.random.default_rng(42), discovery=True)
    assert seeds and targets
    assert set(seeds) | targets == known
    assert not ({tracks[i]['artist'] for i in seeds} & {tracks[i]['artist'] for i in targets})


def test_refresh_import_preserves_vectors_and_rejects_changed_data(tmp_path):
    from python_mvp.refresh import import_graph
    from python_mvp.artifacts import contract
    from python_mvp.config import DIM
    source = tmp_path / 'graph.pt'
    current = contract(tmp_path)
    vectors = torch.nn.functional.normalize(torch.ones(2, DIM), dim=-1)
    state = {'contract': current, 'status': 'DEV_PASS', 'fingerprint': 'fp',
             'vocabulary': ['a', 'b'], 'embeddings': vectors, 'seed': 42, 'weights': {}}
    torch.save(state, source)
    meta = {'fingerprint': 'fp', 'tracks': [{'id': 'a'}, {'id': 'b'}]}
    imported = import_graph(source, meta, current, 42)
    assert torch.equal(imported['embeddings'], vectors)
    assert imported['import_provenance']['source_sha256']
    changed = copy.deepcopy(current)
    changed['artifacts']['global.npz'] = 'different'
    with pytest.raises(RuntimeError, match='artifacts'):
        import_graph(source, meta, changed, 42)


def test_explicit_run_never_falls_back_to_other_experiment(tmp_path, monkeypatch):
    from python_mvp import config, training, build_graph
    monkeypatch.setenv('MUSICMVP_RUN', str(tmp_path / 'missing'))
    with pytest.raises(RuntimeError, match='DATA_BLOCKED'):
        config.load_data()
    assert training.checkpoint('graph', 42) == tmp_path / 'missing/models/graph_42.pt'
    with pytest.raises(FileNotFoundError):
        build_graph.load_graph()


def test_recommendation_excludes_ambiguous_known_and_preserves_exact_order(monkeypatch):
    from python_mvp import recommend as module
    from python_mvp.networks import MultiInterest, NeuralRanker
    n = 60
    tracks = [{'id': str(i), 'artist': 'Artist', 'title': f'Song {i}'} for i in range(n)]
    tracks[2]['title'] = tracks[1]['title']
    matrix = csr_matrix(np.ones((n, n)) - np.eye(n))
    graph = {k: matrix * (4 if k == 'users' else 1) for k in ('global', 'local', 'users', 'sessions', 'confidence')}
    embeddings = torch.nn.functional.normalize(torch.ones(n, 96), dim=-1)
    components = ({'fingerprint': 'fixture'}, {'tracks': tracks, 'raw_tracks': n}, graph,
                  embeddings, MultiInterest(), NeuralRanker())
    monkeypatch.setattr(module, 'load_components', lambda *args, **kwargs: components)
    first = module.recommend(['0', 'Artist — Song 1'])
    other = module.recommend(['Artist — Song 1', '0', '0'])
    assert first['top50'] == other['top50']
    assert len(first['top50']) == 50
    assert not ({'0', '1', '2'} & {row['id'] for row in first['top50']})


def test_refresh_resume_propagates_newly_consumed_final_split(tmp_path):
    import json
    from python_mvp.refresh import prepare
    from python_mvp.artifacts import contract
    from python_mvp.config import DIM
    source, destination = tmp_path / 'source', tmp_path / 'new'
    for sub in ('data', 'models', 'reports'):
        (source / sub).mkdir(parents=True)
    meta = {'fingerprint': 'fp', 'tracks': [{'id': 'a'}, {'id': 'b'}]}
    (source / 'data/graph.json').write_text(json.dumps(meta))
    state = {'contract': contract(source / 'data'), 'status': 'DEV_PASS', 'fingerprint': 'fp',
             'vocabulary': ['a', 'b'], 'embeddings': torch.ones(2, DIM), 'seed': 42}
    torch.save(state, source / 'models/graph_42.pt')
    prepare(source, destination, [42])
    (source / 'reports/final_consumed.lock').write_text('consumed')
    prepare(source, destination, [42])
    assert (destination / 'reports/final_consumed.lock').read_text() == 'consumed'

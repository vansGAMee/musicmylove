"""Small artificial tensors/metadata test contracts, never recommendation quality."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_modules_exist():
    assert (ROOT / 'prepare_data.py').exists(), 'Missing identity-safe ingestion'
    assert (ROOT / 'networks.py').exists(), 'Missing trainable neural components'


def test_identity_preserves_recordings_and_ignores_msid():
    from python_mvp.prepare_data import identify
    meta = {'artist_name': 'Artist', 'track_name': 'Title', 'additional_info': {}}
    a = identify({'track_metadata': meta, 'recording_msid': 'not-a-recording'})
    assert a['id'].startswith('fallback:')
    meta['additional_info']['recording_mbid'] = '00000000-0000-4000-8000-000000000001'
    b = identify({'track_metadata': meta})
    meta['additional_info']['recording_mbid'] = '00000000-0000-4000-8000-000000000002'
    c = identify({'track_metadata': meta})
    assert len({a['id'], b['id'], c['id']}) == 3


def test_recording_aliases_and_fallback_distinct():
    from python_mvp.prepare_data import identify
    row1 = {'track_metadata': {'artist_name': 'Artist', 'track_name': 'Title',
                                'additional_info': {'artist_mbids': ['00000000-0000-4000-8000-000000000001']}}}
    row2 = {'track_metadata': {'artist_name': 'Artist feat. Other', 'track_name': 'Title',
                                'additional_info': {'artist_mbids': ['00000000-0000-4000-8000-000000000001']}}}
    assert identify(row1)['id'] != identify(row2)['id']
    mbid = '00000000-0000-4000-8000-000000000099'
    row1['track_metadata']['additional_info']['recording_mbid'] = mbid
    row2['track_metadata']['additional_info']['recording_mbid'] = mbid
    assert identify(row1)['id'] == identify(row2)['id'] == f'mbid:{mbid}'


def test_set_heads_ignore_order_and_duplicates_and_backpropagate():
    from python_mvp.networks import MultiInterest, NeuralRanker
    torch.manual_seed(5)
    embeddings = torch.randn(30, 96)
    model = MultiInterest()
    heads, masses, assignments = model(embeddings, [1, 2, 3, 4, 5])
    other = model(embeddings, [5, 2, 1, 1, 3, 4, 2])
    assert torch.equal(heads, other[0])
    assert torch.equal(masses, other[1])
    ranker = NeuralRanker()
    score = ranker(embeddings[:4], heads[0].expand(4, -1), torch.ones(4, 6))
    score.sum().backward()
    assert model.queries.grad is not None and model.queries.grad.abs().sum() > 0
    assert ranker.net[0].weight.grad.abs().sum() > 0


def test_negative_pool_excludes_entire_known_library():
    from python_mvp.training import sample_negatives
    rng = np.random.default_rng(1)
    for _ in range(10):
        got = sample_negatives(rng, 12, {0, 1, 2, 3, 4}, {5, 6}, np.arange(1, 13), 2, np.ones(12))
        assert set(got) <= {7, 8, 9, 10, 11}


def test_resolver_rejects_ambiguous_recording_titles():
    from python_mvp.recommend import resolve
    tracks = [{'id': 'a', 'artist': 'X', 'title': 'Song'}, {'id': 'b', 'artist': 'X', 'title': 'Song'}]
    resolved, unresolved = resolve(['X - Song', 'a'], tracks)
    assert resolved == [0]
    assert unresolved == ['X - Song']


def test_degree_shuffle_preserves_both_degrees():
    from python_mvp.evaluate import shuffle_edges
    edges = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 0), (3, 3)]
    shuffled = shuffle_edges(edges, 7)
    assert len(set(shuffled)) == len(edges)
    from collections import Counter
    assert Counter(u for u, t in edges) == Counter(u for u, t in shuffled)
    assert Counter(t for u, t in edges) == Counter(t for u, t in shuffled)
    assert set(shuffled) != set(edges)


def test_gate_rejects_unsupported_candidate_even_if_embedding_matches():
    from python_mvp.retrieval import candidate_range
    from scipy.sparse import csr_matrix
    graph = {'global': csr_matrix(([.8, .8], ([0, 1], [1, 0])), shape=(3, 3)),
             'local': csr_matrix((3, 3)),
             'users': csr_matrix(([8., 8.], ([0, 1], [1, 0])), shape=(3, 3)),
             'sessions': csr_matrix((3, 3)),
             'confidence': csr_matrix(([.8, .8], ([0, 1], [1, 0])), shape=(3, 3))}
    emb = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
    rows = candidate_range([0], emb[:1], torch.tensor([1.]), torch.tensor([[1.]]), emb, graph)
    assert {r['track'] for r in rows} == {1}


def test_graph_encoder_updates_and_checkpoint_reproduces_output(tmp_path):
    from python_mvp.networks import GraphEncoder
    from python_mvp.build_graph import adjacency
    from scipy.sparse import csr_matrix
    # Tensor-level unit fixture, not listening data or release evidence.
    matrix = csr_matrix(np.array([[1., 1., 0., 0.], [0., 1., 1., 1.]]))
    graph = adjacency(matrix)
    torch.manual_seed(4)
    model = GraphEncoder(4, 2, 2)
    before = model.track.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    embeddings, users = model(graph, graph)
    loss = torch.nn.functional.cross_entropy(users @ embeddings.T, torch.tensor([0, 3]))
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.track.weight)
    target = tmp_path / 'checkpoint.pt'
    torch.save(model.state_dict(), target)
    restored = GraphEncoder(4, 2, 2)
    restored.load_state_dict(torch.load(target, weights_only=True))
    assert torch.equal(model(graph, graph)[0], restored(graph, graph)[0])


def test_exact_ranking_metrics_do_not_hide_relevant_item_below_100():
    from python_mvp.evaluate import metrics
    result = metrics(list(range(500)), {300})
    assert result['Recall@100'] == 0
    assert result['MRR'] == pytest.approx(1 / 301)

import json
import numpy as np
import torch
from scipy import sparse


def test_export_reads_only_user_supplied_artist_and_title(tmp_path):
    from python_mvp.discovery_engine import read_library
    path = tmp_path / 'liked.json'
    path.write_text(json.dumps([{'Track Name': 'Song', 'Artist Name(s)': 'A;B',
                                 'Popularity': 100, 'Track URI': 'ignored', 'tempo': 190}]))
    assert read_library(path) == ['A - Song']


def test_listener_retrieval_reaches_track_without_projected_graph_edge():
    from python_mvp.discovery_engine import listener_scores
    ui = sparse.csr_matrix([[1., 1., 0.], [1., 1., 1.], [0., 0., 1.]])
    scores, support = listener_scores(ui, [0])
    assert scores[2] > 0 and support[2] == 1
    without, support_without = listener_scores(ui, [0], exclude_user=1)
    assert without[2] == 0 and support_without[2] == 0


def test_playlist_constraints_never_reintroduce_known_recordings_or_monoculture():
    from python_mvp.discovery_engine import select_playlist
    tracks = [{'id': str(i), 'artist': 'Known' if i < 20 else f'New{i//2}',
               'title': f'Song{i}'} for i in range(100)]
    tracks[1]['title'] = 'Song0 (2011 Remaster)'
    result = select_playlist(tracks, np.arange(100, 0, -1), set(range(100)), ['Known - Song0'])
    assert len(result) == 50
    assert not {0, 1}.intersection(result)
    from collections import Counter
    assert max(Counter(tracks[i]['artist'] for i in result).values()) <= 2
    assert sum(tracks[i]['artist'] != 'Known' for i in result) >= 35
    assert result == select_playlist(tracks, np.arange(100, 0, -1), set(reversed(range(100))), ['Known - Song0'] * 2)


def test_training_pairs_never_use_any_known_positive_as_negative():
    from python_mvp.discovery_engine import training_pairs
    features = np.arange(200, dtype=np.float32).reshape(20, 10)
    positive, negative = training_pairs(features, {2, 3}, set(range(10)), np.ones(20), np.random.default_rng(1))
    assert set(positive[:, 0]) <= {20., 30.}
    assert set(negative[:, 0]) <= set(np.arange(100, 200, 10, dtype=float))


def test_neural_ranker_learns_and_serializes_preference(tmp_path):
    from python_mvp.discovery_engine import DiscoveryRanker
    torch.manual_seed(4)
    model = DiscoveryRanker(3)
    positive = torch.tensor([[1., 0., 1.], [1., 1., 0.]])
    negative = torch.tensor([[0., 1., 0.], [0., 0., 1.]])
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    original = model.linear.weight.detach().clone()
    for _ in range(30):
        loss = torch.nn.functional.softplus(model(negative) - model(positive)).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert torch.all(model(positive) > model(negative) + 1)
    assert not torch.equal(original, model.linear.weight)
    path = tmp_path / 'weights.pt'
    torch.save(model.state_dict(), path)
    restored = DiscoveryRanker(3)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(restored(positive), model(positive))


def test_equal_listener_evidence_is_not_tied_to_catalog_position():
    from python_mvp.discovery_engine import rank_feature
    values = np.array([0., 2., 2., 0., 1.])
    scores = rank_feature(values)
    assert scores[1] == scores[2]
    assert scores[0] == scores[3]
    permutation = np.array([4, 2, 0, 1, 3])
    assert np.array_equal(rank_feature(values[permutation]), scores[permutation])


def test_short_playlist_still_has_at_least_seventy_percent_unfamiliar_artists():
    from python_mvp.discovery_engine import select_playlist
    tracks = [{'id': str(i), 'artist': f'Known{i//2}', 'title': f'Title{i}'} for i in range(10)]
    tracks.extend([{'id': str(i), 'artist': f'New{i}', 'title': f'Title{i}'} for i in range(10, 15)])
    lines = [f'Known{i} - Different song' for i in range(5)]
    output = select_playlist(tracks, np.arange(15, 0, -1), set(range(15)), lines)
    familiar = sum(tracks[i]['artist'].startswith('Known') for i in output)
    assert familiar <= len(output) * .30

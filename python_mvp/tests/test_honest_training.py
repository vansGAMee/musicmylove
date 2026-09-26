import numpy as np
from python_mvp.build_graph import supported_core
from python_mvp.honest_training import partition_users, family_index, episode, pair_indices


def test_support_is_counted_after_projected_singletons_are_removed():
    users = [{'id': str(i), 'tracks': [0, *range(10 + i*4, 14 + i*4)]} for i in range(3)]
    users += [{'id': 'core'+str(i), 'tracks': [1, 2, 3]} for i in range(3)]
    retained, seen = supported_core(users)
    assert seen == [1, 2, 3]
    assert {u['id'] for u in retained} == {'core0', 'core1', 'core2'}


def test_partition_never_reassigns_dev_shadow_final():
    users = [{'id': str(i), 'split': 'train'} for i in range(100)]
    users += [{'id': s, 'split': s} for s in ('dev', 'shadow', 'final')]
    a, b = partition_users(users)
    assert a and b and not a & b
    assert a | b == {str(i) for i in range(100)}
    assert partition_users(list(reversed(users))) == (a, b)


def test_song_variants_never_cross_episode_or_become_unknown_negatives():
    tracks = [dict(id=str(i), artist='a' if i < 3 else str(i), title=t)
              for i, t in enumerate(['Song', 'Song', 'Song - Remastered', 'Other', 'Third', 'Unknown'])]
    families = family_index(tracks)
    assert families[0] == families[1] == families[2]
    seeds, targets = episode({0, 1, 2, 3, 4}, tracks, families, np.random.default_rng(1))
    assert {families[i] for i in seeds}.isdisjoint(families[i] for i in targets)
    p, n = pair_indices({3, 4}, {0, 3, 4}, {1, 2, 3, 5}, families, np.arange(6), np.random.default_rng(2))
    assert set(p) == {3}  # target 4 is missing from serving candidates
    assert set(n) == {5}  # aliases 1/2 of known 0 remain protected
    assert set(p) | set(n) <= {1, 2, 3, 5}


def test_same_pool_and_policy_for_every_method():
    from python_mvp.honest_training import evaluate_one, HonestEngine
    tracks = [dict(id=str(i), artist=str(i), title=str(i)) for i in range(8)]
    engine = HonestEngine.__new__(HonestEngine)
    engine.tracks, engine.families = tracks, np.arange(8)
    engine.score = lambda model, x: x
    scores = np.arange(8, dtype=float)
    q = {'user': 'dev', 'seeds': [0, 1], 'targets': {3, 7}}
    row = evaluate_one(engine, None, q, scores, {2, 3},
                       dict(listeners=scores, ppr=scores, graph=scores), np.arange(8))
    assert row['candidate_recall'] == .5
    assert row['neural'] == row['listeners'] == row['ppr'] == row['graph']
    assert row['neural_recall'] == .5


def test_training_and_reload_on_tiny_synthetic_users(tmp_path):
    import json
    import torch
    from scipy import sparse
    from python_mvp.config import deterministic, write_json
    from python_mvp.honest_training import (train_representation, engine_from_parts,
        train_ranker, load_run)
    deterministic(7)
    n = 12
    tracks = [dict(id=str(i), artist='artist'+str(i), title=str(i)) for i in range(n)]
    rows = [list(range(i, i+6)) for i in range(6)]
    ui = sparse.csr_matrix(([1.] * 36, ([i for i in range(6) for _ in range(6)], sum(rows, []))), shape=(6,n))
    co = (ui.T @ ui).tocsr(); co.setdiag(0); co.eliminate_zeros()
    confidence = co.copy(); confidence.data[:] = .5
    graph = dict(user_incidence=ui, session_incidence=ui, users=co, sessions=co,
                 confidence=confidence, **{'global': co, 'local': co})
    meta = dict(seen=list(range(n)), tracks=tracks, train_users=['rep'+str(i) for i in range(6)])
    data = {'users': [{'id': 'rep'+str(i), 'split': 'train', 'tracks': row} for i, row in enumerate(rows)]}
    data['users'] += [{'id': split+str(i), 'split': split, 'tracks': list(range(i%4, i%4+7))}
                     for split in ('ranker_train', 'dev') for i in range(10)]
    write_json(tmp_path/'data/dataset.json', data)
    write_json(tmp_path/'data/graph.json', meta)
    write_json(tmp_path/'partition.json', {'synthetic': True})
    for key, matrix in graph.items():
        sparse.save_npz(tmp_path/'data'/f'{key}.npz', matrix)
    embeddings, taste, curves = train_representation(data, meta, graph, 1, 7)
    assert all(r['updates'] > 0 for r in curves)
    torch.save(dict(embeddings=embeddings, taste=taste.state_dict(), vocabulary=[t['id'] for t in tracks]), tmp_path/'representation.pt')
    engine = engine_from_parts(tmp_path, data, meta, graph, embeddings, taste)
    model = train_ranker(engine, tmp_path, epochs=1, train_limit=10, seed=7)
    loaded, restored = load_run(tmp_path)
    x, candidates, _ = engine.features([0, 2])
    xx, other, _ = loaded.features([2, 0, 2])
    assert candidates == other
    np.testing.assert_allclose(x, xx)
    np.testing.assert_allclose(engine.score(model, x), loaded.score(restored, xx))
    report = json.loads((tmp_path/'evaluation.json').read_text())
    assert report['frozen_final_used'] is False
    assert report['selection_data_reused'] is True


def test_positive_outside_catalog_protects_its_catalog_version():
    from python_mvp.honest_training import raw_protection
    meta = {'tracks': [dict(id='mbid', artist='A', title='Song - Remastered')], 'seen': [0]}
    data = {'tracks': [meta['tracks'][0], dict(id='fallback', artist='A', title='Song')],
            'users': [dict(id='ranker', split='ranker_train', tracks=[1]),
                      dict(id='final', split='final', tracks=[1])]}
    assert raw_protection(data, meta) == {'ranker': {0}}


def test_playlist_families_do_not_compress_metric_positions():
    from python_mvp.honest_training import HonestEngine
    engine = HonestEngine.__new__(HonestEngine)
    engine.tracks = [dict(id=str(i), artist='a'+str(i), title=str(i)) for i in range(5)]
    engine.tracks[0]['aliases'] = [('shared', 'song')]
    engine.tracks[1]['aliases'] = [('shared', 'song')]
    engine.families = family_index(engine.tracks)
    selected = engine.select_playlist(np.arange(5., 0., -1), set(range(5)), [], familiar_share=1.)
    assert selected == [0, 2, 3, 4]
    selected = engine.select_playlist(np.arange(5., 0., -1), set(range(5)), ['a1 - 1'], familiar_share=1.)
    assert selected == [2, 3, 4]

import pytest


def exposure():
    return {'checkpoint_sha256': 'base', 'input': ['A - Seed'], 'top50': [
        {'rank': i+1, 'id': str(i), 'artist': 'A', 'title': str(i), 'personal_features': [float(i), 1.], 'neural_score': .2}
        for i in range(4)]}


def test_preferences_and_familiarity_are_independent_and_undoable(tmp_path):
    from python_mvp.feedback import record, current, undo, exclusions
    record(tmp_path, exposure(), 'like', '1-2')
    record(tmp_path, exposure(), 'known', '1')
    rows = current(tmp_path)
    assert rows['0']['preference'] == 'like'
    assert rows['0']['familiarity'] == 'known'
    assert rows['1']['familiarity'] is None
    assert '2' not in rows  # unreviewed songs do not become dislikes
    record(tmp_path, exposure(), 'dislike', '2')
    assert current(tmp_path)['1']['preference'] == 'dislike'
    undo(tmp_path)
    assert current(tmp_path)['1']['preference'] == 'like'
    assert exclusions(tmp_path, discovery=True) == ['A - 0', 'A - 1']


def test_invalid_batch_is_atomic(tmp_path):
    from python_mvp.feedback import record, current
    with pytest.raises(ValueError):
        record(tmp_path, exposure(), 'like', '1 50')
    assert current(tmp_path) == {}


def test_latest_label_is_not_duplicated_and_wrong_base_ignored(tmp_path):
    from python_mvp.feedback import record, training_rows
    record(tmp_path, exposure(), 'like', '1')
    record(tmp_path, exposure(), 'dislike', '1')
    rows = training_rows(tmp_path, 'base')
    assert len(rows) == 1 and rows[0]['label'] == 0
    assert training_rows(tmp_path, 'different') == []


def test_cli_keeps_features_and_honors_persistent_ratings(tmp_path):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from python_mvp.cli import generate
    from python_mvp.feedback import record
    tracks = [dict(id=str(i), artist='artist'+str(i), title='song') for i in range(4)]
    engine = SimpleNamespace(tracks=tracks, embeddings=torch.eye(4),
        features=lambda seeds:(np.zeros((4,22),np.float32), {1,2,3}, {}),
        score=lambda model, features:np.array([0.,3.,2.,1.]))
    first = generate(engine, None, ['artist0 - song'], [], '2', feedback_dir=tmp_path, checkpoint='base')
    assert len(first['top50'][0]['personal_features']) == 26
    assert first['exposure_id'] and first['checkpoint_sha256'] == 'base'
    record(tmp_path, first, 'like', '1')
    record(tmp_path, first, 'dislike', '2')
    second = generate(engine, None, ['artist0 - song'], [], '2', feedback_dir=tmp_path, checkpoint='base')
    assert [r['id'] for r in second['top50']] == ['3']


def test_full_known_library_persists_without_profile_limit_and_can_be_undone(tmp_path):
    from python_mvp.feedback import import_known, known_lines, undo, current
    path = tmp_path/'known.txt'
    path.write_text('\n'.join(f'Artist - Song{i}' for i in range(2500)))
    directory = tmp_path/'feedback'
    assert len(import_known(directory, path)) == 2500
    assert len(known_lines(directory)) == 2500
    assert current(directory) == {}
    undo(directory)
    assert known_lines(directory) == []

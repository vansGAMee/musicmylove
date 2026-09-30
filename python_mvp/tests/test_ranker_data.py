import json
from pathlib import Path

import pytest


def example():
    tracks = [
        {'id': 'noise', 'artist': 'Other', 'title': 'Unused'},
        {'id': 'a', 'artist': 'A', 'title': 'Song', 'aliases': [['Alias A', 'Alternate']]},
        {'id': 'version', 'artist': 'A', 'title': 'Song - Remastered'},
        {'id': 'b', 'artist': 'B', 'title': 'Трек "quoted" \\ title'},
        {'id': 'alias', 'artist': 'Unrelated', 'title': 'Canonical',
         'aliases': [['Alias A', 'Alternate']]},
        {'id': 'discard', 'artist': 'C', 'title': 'Other'},
    ]
    users = [
        {'id': 'rep', 'split': 'train', 'tracks': [1], 'sessions': [[1]]},
        {'id': 'rank', 'split': 'ranker_train', 'tracks': [0, 2, 4], 'sessions': [[0, 2, 4]]},
        {'id': 'dev', 'split': 'dev', 'tracks': [1, 2, 3, 5], 'sessions': [[1, 3]]},
        {'id': 'frozen', 'split': 'final', 'tracks': [1, 3], 'sessions': [[1, 3]]},
    ]
    return ({'fingerprint': 'fixture', 'tracks': tracks, 'users': users},
            {'fingerprint': 'fixture', 'seen': [1, 3], 'tracks': [tracks[1], tracks[3]]})


@pytest.mark.parametrize('indent', [None, 2])
@pytest.mark.parametrize('chunk_size', [1, 7, 64])
def test_compact_data_preserves_raw_indices_and_known_positive_aliases(tmp_path, indent, chunk_size):
    from python_mvp.honest_training import mapped_users, raw_protection
    from python_mvp.ranker_data import load_ranker_data

    original, meta = example()
    path = tmp_path / 'dataset.json'
    path.write_text(json.dumps(original, indent=indent, ensure_ascii=False))
    data = load_ranker_data(path, meta, chunk_size=chunk_size)

    assert data['fingerprint'] == 'fixture'
    assert set(data['tracks']) == {1, 2, 3, 4}
    assert data['users'] == [
        {'id': 'rank', 'split': 'ranker_train', 'tracks': [2, 4]},
        {'id': 'dev', 'split': 'dev', 'tracks': [1, 2, 3]},
    ]
    for split in ('ranker_train', 'dev'):
        assert mapped_users(data, meta, split) == mapped_users(original, meta, split)
    assert raw_protection(data, meta) == {'rank': {0}}
    assert raw_protection(data, meta)['rank'] == raw_protection(original, meta)['rank']


@pytest.mark.parametrize('cut', [1, 2, 11, 80])
def test_truncated_stream_is_rejected(tmp_path, cut):
    from python_mvp.ranker_data import load_ranker_data

    original, meta = example()
    path = tmp_path / 'dataset.json'
    path.write_text(json.dumps(original)[:-cut])
    with pytest.raises(ValueError):
        load_ranker_data(path, meta, chunk_size=7)


@pytest.mark.parametrize('change', ['fingerprint', 'id', 'missing_seen'])
def test_incompatible_catalog_is_rejected(tmp_path, change):
    from python_mvp.ranker_data import load_ranker_data

    original, meta = example()
    if change == 'fingerprint':
        original['fingerprint'] = 'other'
    elif change == 'id':
        original['tracks'][1] = dict(original['tracks'][1], id='changed')
    else:
        meta['seen'] = [1, 90]
    path = tmp_path / 'dataset.json'
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError):
        load_ranker_data(path, meta, chunk_size=7)


def test_rejects_users_before_tracks_and_trailing_input(tmp_path):
    from python_mvp.ranker_data import load_ranker_data

    original, meta = example()
    path = tmp_path / 'dataset.json'
    reordered = {'users': original['users'], 'tracks': original['tracks'], 'fingerprint': 'fixture'}
    path.write_text(json.dumps(reordered))
    with pytest.raises(ValueError, match='tracks before users'):
        load_ranker_data(path, meta)
    path.write_text(json.dumps(original) + '{}')
    with pytest.raises(ValueError):
        load_ranker_data(path, meta)


def test_dataset_is_read_in_bounded_chunks(tmp_path, monkeypatch):
    from python_mvp.ranker_data import load_ranker_data

    original, meta = example()
    path = tmp_path / 'dataset.json'
    path.write_text(json.dumps(original))
    open_file = Path.open

    class BoundedFile:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size=-1):
            assert 0 < size <= 17, 'The dataset must never be read into memory at once'
            return self.stream.read(size)

    monkeypatch.setattr(Path, 'open', lambda self, **kwargs: BoundedFile(open_file(self, **kwargs)))
    assert load_ranker_data(path, meta, chunk_size=17)['users'][0]['tracks'] == [2, 4]


@pytest.mark.parametrize('indent', [None, 2])
def test_projected_coverage_preserves_reports_and_representation_support(tmp_path, indent):
    from types import SimpleNamespace
    from python_mvp.coverage import diagnose
    from python_mvp.ranker_data import load_coverage_data

    raw = [
        dict(id='unused', artist='Unused', title='Ignored'),
        dict(id='a', artist='A', title='One'),
        dict(id='rare', artist='C', title='Rare'),
        dict(id='version', artist='C', title='Rare - Remastered'),
        dict(id='alias', artist='Other', title='Name', aliases=[['E', 'Alias']]),
        dict(id='heldout', artist='H', title='Solo'),
        dict(id='b1', artist='B', title='Ambiguous'),
        dict(id='b2', artist='B', title='Ambiguous'),
    ]
    original = {'fingerprint': 'fixture', 'tracks': raw, 'users': [
        dict(id='rep1', split='train', tracks=[0, 1, 2, 2, 4], sessions=[[2, 4]]),
        dict(id='rep2', split='train', tracks=[3, 4], sessions=[[3, 4]]),
        dict(id='dev', split='dev', tracks=[2, 5], sessions=[[2, 5]]),
        dict(id='final', split='final', tracks=[2, 5], sessions=[[2, 5]]),
    ]}
    meta = {'fingerprint': 'fixture', 'seen': [1, 6, 7], 'tracks': [raw[1], raw[6], raw[7]],
            'train_users': ['rep1', 'rep2']}
    lines = ['A - One', 'B - Ambiguous', 'C - Rare', 'E - Alias', 'H - Solo', 'D - Missing']
    path = tmp_path / 'dataset.json'
    path.write_text(json.dumps(original, indent=indent))
    compact = load_coverage_data(path, meta, lines, chunk_size=5)
    full_report = diagnose(SimpleNamespace(tracks=meta['tracks'], meta=meta, data=original), lines)
    report = diagnose(SimpleNamespace(tracks=meta['tracks'], meta=meta, data=compact), lines)

    assert report == full_report
    assert [t['id'] for t in compact['tracks']] == ['a', 'rare', 'version', 'alias', 'heldout', 'b1', 'b2']
    assert {u['id'] for u in compact['users']} == {'rep1', 'rep2'}
    assert all('sessions' not in u for u in compact['users'])
    by_input = {row['input']: row for row in report['tracks']}
    assert by_input['C - Rare']['representation_support'] == 1
    assert by_input['E - Alias']['representation_support'] == 2
    assert by_input['H - Solo']['representation_support'] == 0
    assert by_input['D - Missing']['status'] == 'absent_from_raw'

import json
import pytest
from python_mvp.prepare_data import prepare


def test_bounded_preparation_matches_original_sessions_and_splits(tmp_path):
    from python_mvp.expand_data import prepare_bounded
    path = tmp_path/'listens.jsonl'
    rows = []
    for i in range(600):
        for j in range(5):
            row = {'user_name': f'u{i}', 'listened_at': 1700000000+j*1000,
                   'track_metadata': {'artist_name': f'a{i%20}', 'track_name': f't{j}'}}
            rows.extend([row, row])
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows) + 'invalid json\n')
    prepare([path], tmp_path/'reference', tmp_path/'reference_reports')
    prepare_bounded([path], tmp_path/'bounded', tmp_path/'bounded_reports', shards=8)
    original = json.loads((tmp_path/'reference/dataset.json').read_text())
    bounded = json.loads((tmp_path/'bounded/dataset.json').read_text())
    assert original['tracks'] == bounded['tracks']
    assert sorted(original['users'], key=lambda u:u['id']) == sorted(bounded['users'], key=lambda u:u['id'])
    assert json.loads((tmp_path/'reference/split_manifest.json').read_text()) == json.loads((tmp_path/'bounded/split_manifest.json').read_text())
    assert original['fingerprint'] == bounded['fingerprint']
    assert json.loads((tmp_path/'bounded_reports/data_audit.json').read_text())['invalid_rows'] == 1


def test_archive_validation_never_downloads_and_detects_changes(tmp_path):
    from python_mvp.expand_data import cached_archives
    from python_mvp.config import fingerprint
    archive = tmp_path/'archive.tar.zst'; archive.write_bytes(b'local archive')
    (tmp_path/'reports').mkdir()
    report = [{'archive': str(archive), 'archive_sha256': fingerprint(archive), 'user_fraction': .25}]
    (tmp_path/'reports/collection.json').write_text(json.dumps(report))
    assert cached_archives(tmp_path) == [(archive, report[0]['archive_sha256'])]
    archive.write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='checksum'):
        cached_archives(tmp_path)


def test_split_preservation_rejects_old_user_reassignment():
    from python_mvp.expand_data import verify_splits
    verify_splits({'a':'final'}, {'a':'final', 'b':'train'})
    with pytest.raises(RuntimeError, match='split'):
        verify_splits({'a':'final'}, {'a':'train'})


@pytest.mark.parametrize('initially_frozen', [True, False])
def test_offline_expansion_resumes_and_keeps_frozen_split(tmp_path, monkeypatch, initially_frozen):
    import hashlib
    import io
    import tarfile
    from python_mvp import expand_data as expand
    from python_mvp.config import fingerprint, write_json
    from python_mvp.prepare_data import split_user
    source, run = tmp_path/'source', tmp_path/'expanded'
    source.mkdir()
    rows, original = [], {}
    for i in range(600):
        uid = hashlib.sha256(f'u{i}'.encode()).hexdigest()
        if int(uid[:16], 16)/2**64 < .25:
            original[uid] = split_user(uid)
        for j in range(5):
            rows.append({'user_name': f'u{i}', 'listened_at': 1700000000+j*1000,
                         'track_metadata': {'artist_name': f'a{i%20}', 'track_name': f't{j}'}})
    raw = ''.join(json.dumps(r)+'\n' for r in rows).encode()
    archive = source/'archive.tar'
    with tarfile.open(archive, 'w') as stream:
        entry = tarfile.TarInfo('nested/events.listens'); entry.size = len(raw)
        stream.addfile(entry, io.BytesIO(raw))
    write_json(source/'reports/collection.json', [{'archive': str(archive), 'archive_sha256': fingerprint(archive), 'user_fraction': .25}])
    write_json(source/'data/split_manifest.json', {'users': original})
    if initially_frozen:
        (source/'reports/final_consumed.lock').write_text('original consumption marker')
    monkeypatch.setattr('sys.argv', ['expand_data', '--source', str(source), '--run', str(run)])
    real_prepare = expand.prepare_bounded
    def interrupted(*a, **k):
        raise RuntimeError('simulated interruption')
    monkeypatch.setattr(expand, 'prepare_bounded', interrupted)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        expand.main()
    monkeypatch.setattr(expand, 'prepare_bounded', real_prepare)
    def forbidden(*a, **k):
        raise AssertionError('Verified completed extraction must be reused')
    monkeypatch.setattr(expand, 'extract_listens', forbidden)
    expand.main()
    audit = json.loads((run/'reports/data_audit.json').read_text())
    assert audit['users'] == 600
    if initially_frozen:
        assert (run/'reports/final_consumed.lock').read_text() == 'original consumption marker'
    else:
        (source/'reports/final_consumed.lock').write_text('original consumption marker')
    assert len(original) < audit['users']
    monkeypatch.setattr(expand, 'prepare_bounded', forbidden)
    expand.main()  # completed rerun does not re-extract or re-prepare
    assert (run/'reports/final_consumed.lock').read_text() == 'original consumption marker'
    (run/'reports/final_consumed.lock').unlink()
    with pytest.raises(RuntimeError, match='changed'):
        expand.main()


def test_rejected_split_never_publishes_trainable_data(tmp_path):
    from python_mvp.expand_data import prepare_bounded
    path = tmp_path/'input.jsonl'
    rows = [{'user_name': f'u{i}', 'listened_at': 1700000000,
             'track_metadata': {'artist_name': 'a', 'track_name': str(i%100)}} for i in range(600)]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(RuntimeError, match='split'):
        prepare_bounded([path], tmp_path/'data', tmp_path/'reports', expected_original={'missing':'final'})
    assert not (tmp_path/'data/dataset.json').exists()
    assert not (tmp_path/'reports/data_audit.json').exists()

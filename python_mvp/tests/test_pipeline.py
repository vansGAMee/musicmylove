import json
import pytest


def test_range_config_rejects_unsafe_thresholds():
    from python_mvp.artifacts import RangeConfig
    with pytest.raises(ValueError):
        RangeConfig(confidence=0)
    with pytest.raises(ValueError):
        RangeConfig(min_users=1)


def test_contract_detects_graph_or_vocabulary_change(tmp_path):
    from python_mvp.artifacts import contract, validate_contract
    from python_mvp.config import write_json
    for name, content in [('dataset.json', {'fingerprint':'a'}), ('split_manifest.json', {'users':{'x':'train'}}), ('graph.json', {'tracks':[{'id':'a'}]})]:
        write_json(tmp_path/name, content)
    (tmp_path/'global.npz').write_bytes(b'graph1')
    original=contract(tmp_path)
    validate_contract(original, original)
    (tmp_path/'global.npz').write_bytes(b'graph2')
    with pytest.raises(RuntimeError, match='ARTIFACT_MISMATCH'):
        validate_contract(original, contract(tmp_path))


@pytest.mark.parametrize('compressed', [False, True])
def test_archive_extract_preserves_users_and_time(tmp_path, compressed):
    import tarfile, io
    from python_mvp.collect_data import extract_listens
    row={'user_name':'real-format-fixture', 'timestamp':123, 'track_metadata':{'artist_name':'A','track_name':'T'}}
    archive=tmp_path/'sample.tar'
    with tarfile.open(archive,'w') as tar:
        body=(json.dumps(row)+'\n').encode()
        info=tarfile.TarInfo('dump/listens/2026/9.listens'); info.size=len(body)
        tar.addfile(info, io.BytesIO(body))
    if compressed:
        import subprocess, shutil
        if not shutil.which('zstd'): pytest.skip('zstd is not installed')
        subprocess.run(['zstd', '-q', str(archive)], check=True)
        archive=archive.with_suffix('.tar.zst')
    output=tmp_path/'out.jsonl'
    report=extract_listens(archive, output, fraction=1.)
    assert json.loads(output.read_text())==row
    assert report['listens']==1
    assert report['users']==1


def test_checkpoint_without_contract_is_rejected():
    from python_mvp.artifacts import validate_contract
    with pytest.raises(RuntimeError, match='ARTIFACT_MISMATCH'):
        validate_contract({}, {'schema':2})


def test_block_graph_support_counts_independent_users():
    from scipy import sparse
    import numpy as np
    from python_mvp.build_graph import association
    # Three independent contexts support 0<->1, only one supports 1<->2.
    matrix=sparse.csr_matrix([[1.,1.,0.],[1.,1.,0.],[1.,1.,1.]])
    affinity,support,degree=association(matrix)
    assert support[0,1]==3
    assert support[1,2]==0
    assert affinity[0,1]==pytest.approx(1.)
    assert np.array_equal(degree,[3,3,1])


def test_original_recording_msid_is_never_recording_mbid():
    from python_mvp.prepare_data import identify
    row={'recording_msid':'00000000-0000-4000-8000-000000000001',
         'track_metadata':{'artist_name':'A','track_name':'T','additional_info':{}}}
    assert identify(row)['recording_mbid'] is None
    assert identify(row)['id'].startswith('fallback:')


def test_projection_preserves_learned_geometry_at_random_initialization():
    import torch
    from python_mvp.networks import MultiInterest
    torch.manual_seed(7)
    embedding=torch.nn.functional.normalize(torch.randn(1,96),dim=-1)
    heads,_,_=MultiInterest()(embedding,[0])
    assert torch.all((heads@embedding.T)>.9)


def test_download_resets_partial_when_server_ignores_range(tmp_path, monkeypatch):
    import io, hashlib
    from python_mvp import collect_data as c
    payload=b'complete real-format HTTP body'
    (tmp_path/'file.tar.zst.part').write_bytes(b'old partial bytes')
    class Response(io.BytesIO):
        status=200
        headers={}
    monkeypatch.setattr(c,'text',lambda url:hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(c,'request',lambda url,headers=None:Response(payload))
    result=c.download('https://data.metabrainz.org/file.tar.zst',tmp_path)
    assert result.read_bytes()==payload
    assert not (tmp_path/'file.tar.zst.part').exists()


def test_download_does_not_publish_wrong_checksum(tmp_path, monkeypatch):
    import io
    from python_mvp import collect_data as c
    class Response(io.BytesIO):
        status=200
        headers={}
    monkeypatch.setattr(c,'text',lambda url:'a'*64)
    monkeypatch.setattr(c,'request',lambda url,headers=None:Response(b'corrupt'))
    with pytest.raises(RuntimeError,match='CHECKSUM_FAIL'):
        c.download('https://data.metabrainz.org/file.tar.zst',tmp_path)
    assert not (tmp_path/'file.tar.zst').exists()

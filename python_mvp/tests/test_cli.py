import pytest


def test_pasted_tracks_are_deduplicated_and_empty_input_rejected():
    from python_mvp.cli import validate_lines
    assert validate_lines([' A - Song ', '', 'A - Song']) == ['A - Song']
    with pytest.raises(ValueError):
        validate_lines([])
    with pytest.raises(ValueError):
        validate_lines([str(i) for i in range(2001)])


def test_missing_weights_never_start_training(tmp_path):
    from python_mvp.cli import load_ranker
    with pytest.raises(FileNotFoundError, match='Обучение'):
        load_ranker(None, tmp_path)


def test_export_keeps_each_playlist_and_its_input(tmp_path):
    from python_mvp.cli import export_playlist
    result = {'top50': [{'rank': 1, 'artist': 'A', 'title': 'Song'}], 'input': ['B - Other']}
    first = export_playlist(result, tmp_path)
    second = export_playlist(result, tmp_path)
    assert first != second
    assert (first / 'playlist.txt').read_text().strip() == '1. A — Song'
    import json
    assert json.loads((first / 'result.json').read_text())['input'] == ['B - Other']


def test_focus_changes_retrieval_but_keeps_original_library_excluded(tmp_path):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from python_mvp.cli import generate, focus_from_result
    tracks = [dict(id=str(i), artist=f'artist{i}', title='song') for i in range(7)]
    # Two recordings have the same display name: focus must preserve the chosen ID.
    tracks[6].update(artist='artist1', title='song')
    def features(seeds):
        values = np.zeros((7, 22), np.float32)
        values[:, 0] = [10, 9, 8, 7, 6, 5, 4] if seeds == [0] else [10, 9, 8, 1, 6, 7, 4]
        assert seeds in ([0], [1, 2])
        return values, set(range(7)), {}
    engine = SimpleNamespace(tracks=tracks, embeddings=torch.eye(7), features=features,
                             score=lambda model, values:values[:, 0].copy())
    def run(focus=None):
        return generate(engine, None, ['artist0 - song - Remastered'], ['artist4 - song'], '1',
                        feedback_dir=tmp_path, checkpoint='base', focus=focus)
    original = run()
    focus = focus_from_result(original, '2 1 1')
    assert focus == ['1', '2']
    focused = run(focus)
    assert [t['id'] for t in focused['top50']] == ['5', '3']
    assert focused['library_input'] == ['artist0 - song - Remastered']
    assert set(original['known_exclusions']) <= set(focused['known_exclusions'])
    assert focused['input'] == ['1', '2']
    assert run(list(reversed(focus)))['top50'] == focused['top50']
    assert run()['top50'] == original['top50']
    assert not (tmp_path/'feedback.json').exists()
    with pytest.raises(ValueError):
        focus_from_result(original, '1 99')
    with pytest.raises(ValueError):
        focus_from_result(None, '1')


def test_interactive_focus_invalid_selection_and_reset(tmp_path, monkeypatch, capsys):
    import json
    import numpy as np
    import torch
    from types import SimpleNamespace
    from python_mvp import cli
    run = tmp_path/'model'; run.mkdir(); (run/'ranker.pt').write_bytes(b'fixture')
    library = tmp_path/'library.txt'; library.write_text('artist0 - song')
    tracks = [dict(id=str(i), artist=f'artist{i}', title='song') for i in range(5)]
    def features(seeds):
        values = np.zeros((5,22),np.float32)
        values[:,0] = [10,9,8,7,6] if seeds == [0] else [10,9,8,6,7]
        return values, set(range(5)), {}
    engine = SimpleNamespace(tracks=tracks, embeddings=torch.eye(5), features=features,
                             score=lambda model, values:values[:,0].copy())
    monkeypatch.setattr(cli,'open_engine',lambda *a,**kw:(engine,None))
    commands = iter(['ещё как 1 99', 'еще как 1 2', '3', 'весь профиль', '0'])
    monkeypatch.setattr('builtins.input',lambda *a:next(commands))
    monkeypatch.setattr('sys.argv',['cli',str(library),'--run',str(run),'--output',str(tmp_path/'out'),
                                   '--feedback',str(tmp_path/'ratings')])
    assert cli.main() == 0
    results = [json.loads(p.read_text()) for p in (tmp_path/'out').glob('*/result.json')]
    assert len(results) == 4  # invalid selection does not replace the last playlist
    focused = [r for r in results if r.get('focus')]
    assert len(focused) == 2
    assert all([t['id'] for t in r['top50']] == ['4','3'] for r in focused)
    full = [r for r in results if not r.get('focus')]
    assert len(full) == 2
    assert all('0' not in [t['id'] for t in r['top50']] for r in results)
    assert not (tmp_path/'ratings/feedback.json').exists()
    assert 'Не получилось:' in capsys.readouterr().out

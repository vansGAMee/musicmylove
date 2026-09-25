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

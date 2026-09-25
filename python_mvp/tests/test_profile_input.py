from python_mvp.profile_input import resolve_profile


def test_metadata_record_is_explicit_and_does_not_merge_recordings():
    tracks = [{'id': 'fallback:a', 'artist': 'A', 'title': 'Song'},
              {'id': 'mbid:b', 'artist': 'A', 'title': 'Song'}]
    result = resolve_profile(['A - Song'], tracks)
    assert result['seeds'] == [0]
    assert result['matches'][0]['kind'] == 'metadata_name'
    assert result['matches'][0]['alternatives'] == 2
    assert resolve_profile(['A - Song'], tracks, strict=True)['seeds'] == []


def test_different_authoritative_recordings_remain_ambiguous():
    tracks = [{'id': 'mbid:a', 'artist': 'A', 'title': 'Song'},
              {'id': 'mbid:b', 'artist': 'A', 'title': 'Song'}]
    assert resolve_profile(['A - Song'], tracks)['unresolved'] == ['A - Song']


def test_only_remaster_suffix_can_be_used_as_name_hint():
    tracks = [{'id': 'fallback:a', 'artist': 'A', 'title': 'Song'}]
    result = resolve_profile(['A - Song - 2011 Remaster'], tracks)
    assert result['seeds'] == [0]
    assert result['matches'][0]['kind'] == 'remaster_name'
    assert resolve_profile(['A - Song (Live)'], tracks)['seeds'] == []


def test_input_order_and_duplicates_do_not_change_profile():
    tracks = [{'id': 'fallback:a', 'artist': 'A', 'title': 'Song'}]
    assert resolve_profile(['A - Song'] * 2, tracks) == resolve_profile(['A - Song'], tracks)

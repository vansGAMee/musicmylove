from ml.examples import build_examples, is_suitable, negatives_from_candidates


def recordings(count=30):
    return [{"recording_mbid": f"m{i}", "track_name": f"T{i}", "artist_name": f"A{i % 7}", "listen_count": count - i + 2} for i in range(count)]


def test_suitability_requires_twenty_valid_and_twelve_repeated():
    assert is_suitable(recordings(30))
    assert not is_suitable(recordings(19))
    assert not is_suitable([{**row, "listen_count": 1} for row in recordings(30)])


def test_builds_three_deterministic_seed_hidden_examples():
    first = build_examples("alice", recordings(), 41)
    second = build_examples("alice", list(reversed(recordings())), 41)
    assert first == second
    assert len(first) == 3
    assert all(len(example["seeds"]) == 5 and len(example["hidden"]) == 3 for example in first)
    assert all(not (set(example["seeds"]) & set(example["hidden"])) for example in first)


def test_known_positives_are_never_implicit_negatives():
    assert negatives_from_candidates({"p1", "p2"}, ["p2", "n1", "p1", "n2"]) == ["n1", "n2"]

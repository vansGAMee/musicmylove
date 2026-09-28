import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

import ml.tastelift_data as tastelift_data
from ml.tastelift_data import build_dataset, build_popularity, fnv1a_utf8, hashed_subword_ids


def _key(username: str) -> str:
    return hashlib.sha256(username.encode("utf-8")).hexdigest()


def _recording(mbid: str, count: int, artist: str | None = None, title: str | None = None) -> dict:
    return {
        "recording_mbid": mbid,
        "artist_name": artist or f"Artist {mbid}",
        "track_name": title or f"Title {mbid}",
        "listen_count": count,
    }


def _write_user(root: Path, username: str, recordings: list[dict], rows: list[dict] | None = None) -> None:
    key = _key(username)
    (root / "data/cache/real-profiles").mkdir(parents=True, exist_ok=True)
    (root / "data/cache/real-retrieval").mkdir(parents=True, exist_ok=True)
    (root / f"data/cache/real-profiles/{key}.json").write_text(json.dumps({"payload": {"recordings": recordings}}))
    (root / f"data/cache/real-retrieval/{key}.json").write_text(json.dumps({"rows": rows or []}))


def _write_splits(root: Path, train: list[str], validation: list[str] | None = None, test: list[str] | None = None) -> None:
    target = root / "data/manifests/real-splits-final.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"train": train, "validation": validation or [], "test": test or []}))


def test_episodes_have_canonical_unique_5_to_30_strong_seeds_and_a_held_out_positive(tmp_path: Path):
    username = "train-user"
    tracks = [_recording(f"track-{index:02d}", 40 - index) for index in range(40)]
    _write_splits(tmp_path, [username])
    _write_user(tmp_path, username, tracks, [{"reference_mbid": "track-00", "recording_mbid": "negative", "recording_name": "Negative", "artist_credit_name": "Other"}])

    dataset = build_dataset(tmp_path, masks_per_user=12, seed=17)

    assert len(dataset["episodes"]) == 12
    known = {track["recording_mbid"] for track in tracks}
    for episode in dataset["episodes"]:
        assert 5 <= len(episode["seed_ids"]) <= 30
        assert episode["seed_ids"] == sorted(set(episode["seed_ids"]))
        assert episode["positive_id"] in known - set(episode["seed_ids"])


def test_held_out_positive_is_always_an_unseeded_repeated_listen(tmp_path: Path):
    username = "repeated-listen-user"
    strong = [_recording(f"strong-{index}", 3) for index in range(6)]
    weak = [_recording(f"weak-{index}", 1) for index in range(8)]
    _write_splits(tmp_path, [username])
    _write_user(tmp_path, username, strong + weak, [{"reference_mbid": "strong-0", "recording_mbid": "negative", "recording_name": "Negative", "artist_credit_name": "Other"}])

    dataset = build_dataset(tmp_path, masks_per_user=12, seed=7)

    strong_ids = {track["recording_mbid"] for track in strong}
    assert dataset["episodes"]
    assert all(episode["positive_id"] in strong_ids - set(episode["seed_ids"]) for episode in dataset["episodes"])


def test_repeated_masks_are_deterministic_and_not_single_fixed_mask(tmp_path: Path):
    username = "repeat-user"
    _write_splits(tmp_path, [username])
    _write_user(tmp_path, username, [_recording(f"track-{index:02d}", 80 - index) for index in range(50)], [{"reference_mbid": "track-00", "recording_mbid": "negative", "recording_name": "Negative", "artist_credit_name": "Other"}])

    first = build_dataset(tmp_path, masks_per_user=12, seed=41)
    second = build_dataset(tmp_path, masks_per_user=12, seed=41)

    assert first == second
    assert len({tuple(episode["seed_ids"]) for episode in first["episodes"]}) == 12


def test_episodes_keep_users_and_partitions_isolated_without_raw_usernames(tmp_path: Path):
    train, validation, test = "train-user", "validation-user", "test-user"
    _write_splits(tmp_path, [train], [validation], [test])
    for prefix, user in zip(("a", "b", "c"), (train, validation, test), strict=True):
        _write_user(tmp_path, user, [_recording(f"{prefix}-track-{index:02d}", 40 - index) for index in range(15)])

    dataset = build_dataset(tmp_path, masks_per_user=2, seed=3)

    assert {episode["partition"] for episode in dataset["episodes"]} == {"train", "validation", "test"}
    assert {episode["user_key"] for episode in dataset["episodes"]} == {_key(train), _key(validation), _key(test)}
    assert train not in json.dumps(dataset)
    assert validation not in json.dumps(dataset)
    assert test not in json.dumps(dataset)


def test_negative_is_not_any_known_positive_and_matches_empirical_user_frequency_band(tmp_path: Path):
    target, peer_a, peer_b = "target", "peer-a", "peer-b"
    _write_splits(tmp_path, [target, peer_a, peer_b])
    positives = [_recording(f"p-{index:02d}", 30 - index) for index in range(12)]
    matching_negative = _recording("popular-negative", 1, "Other", "Matched")
    rows = [{"reference_mbid": "p-00", "recording_mbid": "popular-negative", "recording_name": "Matched", "artist_credit_name": "Other", "score": 99}]
    _write_user(tmp_path, target, positives, rows)
    _write_user(tmp_path, peer_a, positives + [matching_negative])
    _write_user(tmp_path, peer_b, positives + [matching_negative])

    dataset = build_dataset(tmp_path, masks_per_user=1, seed=11)
    target_episode = next(episode for episode in dataset["episodes"] if episode["user_key"] == _key(target))
    assert target_episode["negative_id"] == "popular-negative"
    assert target_episode["negative_id"] not in {track["recording_mbid"] for track in positives}
    assert target_episode["negative_band"] == target_episode["positive_band"]


def test_exact_band_corpus_negative_beats_nearest_band_retrieval_candidate():
    popularity = {
        "retrieval-near": {"user_frequency": 3, "percentile": 0.4, "band": 4},
        "corpus-exact": {"user_frequency": 4, "percentile": 0.5, "band": 5},
    }

    selected = tastelift_data._negative_id({"positive"}, {"retrieval-near"}, {"retrieval-near", "corpus-exact"}, 5, popularity, random.Random(3))

    assert selected == ("corpus-exact", "corpus", 5)


def test_subword_hash_is_nfkc_lowercase_utf8_fnv1a_and_deduplicated():
    assert fnv1a_utf8("é") == 0x1E9DE8C1
    assert hashed_subword_ids("ＣＡＦÉ", buckets=101, min_n=3, max_n=3) == hashed_subword_ids("café", buckets=101, min_n=3, max_n=3)
    assert hashed_subword_ids("aaaa", buckets=101, min_n=3, max_n=3) == sorted(set(hashed_subword_ids("aaaa", buckets=101, min_n=3, max_n=3)))


def test_popularity_bands_use_distinct_user_frequency_not_listen_count():
    popularity = build_popularity({"a": {"x", "y"}, "b": {"x"}, "c": {"x"}})
    assert popularity["x"]["user_frequency"] == 3
    assert popularity["y"]["user_frequency"] == 1
    assert popularity["x"]["band"] > popularity["y"]["band"]


def test_builder_script_runs_directly_from_repository_root():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "scripts/build_tastelift_data.py", "--help"], cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_atomic_checkpoint_resumes_unchanged_inputs_and_rejects_changed_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    first, second = "checkpoint-a", "checkpoint-b"
    first_tracks = [_recording(f"a-{index}", 4) for index in range(8)]
    second_tracks = [_recording(f"b-{index}", 4) for index in range(8)]
    _write_splits(tmp_path, [first, second])
    _write_user(tmp_path, first, first_tracks, [{"reference_mbid": "a-0", "recording_mbid": "candidate-a", "recording_name": "Candidate A", "artist_credit_name": "Other"}])
    _write_user(tmp_path, second, second_tracks, [{"reference_mbid": "b-0", "recording_mbid": "candidate-b", "recording_name": "Candidate B", "artist_credit_name": "Other"}])
    checkpoint = tmp_path / "data/cache/tastelift/checkpoint.json"
    original = tastelift_data._episodes_for_user
    calls = 0

    def interrupt_after_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(tastelift_data, "_episodes_for_user", interrupt_after_first)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        build_dataset(tmp_path, masks_per_user=2, seed=5, checkpoint_path=checkpoint, checkpoint_every=1)
    assert checkpoint.exists()
    monkeypatch.setattr(tastelift_data, "_episodes_for_user", original)

    resumed = build_dataset(tmp_path, masks_per_user=2, seed=5, checkpoint_path=checkpoint, checkpoint_every=1)
    clean = build_dataset(tmp_path, masks_per_user=2, seed=5)
    assert resumed == clean

    _write_user(tmp_path, first, first_tracks + [_recording("changed-track", 1)], [{"reference_mbid": "a-0", "recording_mbid": "candidate-a", "recording_name": "Candidate A", "artist_credit_name": "Other"}])
    with pytest.raises(ValueError, match="checkpoint"):
        build_dataset(tmp_path, masks_per_user=2, seed=5, checkpoint_path=checkpoint, checkpoint_every=1)

    checkpoint.unlink()
    rebuilt = build_dataset(tmp_path, masks_per_user=2, seed=5, checkpoint_path=checkpoint, checkpoint_every=1)
    assert "changed-track" in {track["id"] for track in rebuilt["tracks"]}

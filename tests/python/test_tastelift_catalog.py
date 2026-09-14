import hashlib
import json
from pathlib import Path

from ml.tastelift_catalog import collect_train_histories, collect_train_tracks, quantize_vectors


def _profile(root: Path, username: str, recordings: list[dict]) -> None:
    key = hashlib.sha256(username.encode("utf-8")).hexdigest()
    target = root / "data/cache/real-profiles" / f"{key}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"payload": {"recordings": recordings}}))


def _recording(mbid: str, artist: str, title: str, release: str = "") -> dict:
    return {
        "recording_mbid": mbid,
        "artist_name": artist,
        "track_name": title,
        "release_name": release,
        "listen_count": 3,
    }


def test_catalog_contains_only_deduplicated_train_user_history_tracks(tmp_path: Path):
    manifest = tmp_path / "data/manifests/real-splits-final.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"train": ["train-a", "train-b"], "validation": ["validation"], "test": ["test"]}))
    shared = _recording("train-shared", "Björk", "Jóga", "Homogenic")
    _profile(tmp_path, "train-a", [shared, _recording("train-a", "Artist A", "Song A")])
    _profile(tmp_path, "train-b", [shared, _recording("train-b", "Artist B", "Song B")])
    _profile(tmp_path, "validation", [_recording("validation-only", "Leak", "Validation")])
    _profile(tmp_path, "test", [_recording("test-only", "Leak", "Test")])

    tracks = collect_train_tracks(tmp_path, manifest, {"train-shared": 0.75, "train-a": 0.25})

    assert [track["mbid"] for track in tracks] == ["train-a", "train-b", "train-shared"]
    assert tracks[-1] == {
        "mbid": "train-shared",
        "artist": "Björk",
        "title": "Jóga",
        "release": "Homogenic",
        "popularityPercentile": 0.75,
    }
    assert "validation-only" not in json.dumps(tracks)
    assert "test-only" not in json.dumps(tracks)
    assert "train-a" not in json.dumps(tracks).replace('"mbid": "train-a"', "")
    histories = collect_train_histories(tmp_path, manifest, {track["mbid"]: index for index, track in enumerate(tracks)})
    assert histories == [[0, 2], [1, 2]]
    assert "validation" not in json.dumps(histories)


def test_catalog_vectors_use_deterministic_symmetric_int8_quantization():
    encoded, maximum_error = quantize_vectors([[1.0, -1.0, 0.0, 0.5], [0.25, -0.25, 0.1, -0.1]])

    assert encoded == bytes([127, 129, 0, 64, 32, 224, 13, 243])
    assert maximum_error <= 1 / 254

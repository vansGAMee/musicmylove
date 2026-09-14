"""Deterministic, offline TasteLift episodes built only from local caches.

Hash constants are part of the Python/TypeScript model contract: text is Unicode
NFKC-normalized, lowercased, whitespace-collapsed, encoded as UTF-8, then hashed
with 32-bit FNV-1a into 65,536 buckets. Character subwords use padded 3--5-grams.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_VERSION = 1
DEFAULT_MASKS_PER_USER = 12
MIN_SEEDS = 5
MAX_SEEDS = 30
STRONG_LISTEN_COUNT = 2
POPULARITY_BANDS = 10
FNV1A_OFFSET_BASIS = 0x811C9DC5
FNV1A_PRIME = 0x01000193
HASH_BUCKETS = 65_536
CHAR_NGRAM_MIN = 3
CHAR_NGRAM_MAX = 5


def normalize_text(value: str) -> str:
    """Stable NFKC-lower text representation shared by future TS inference."""
    return " ".join(unicodedata.normalize("NFKC", value).lower().split())


def fnv1a_utf8(value: str) -> int:
    """Return the unsigned 32-bit FNV-1a hash of UTF-8 text."""
    state = FNV1A_OFFSET_BASIS
    for byte in value.encode("utf-8"):
        state ^= byte
        state = (state * FNV1A_PRIME) & 0xFFFFFFFF
    return state


def hashed_subword_ids(value: str, buckets: int = HASH_BUCKETS, min_n: int = CHAR_NGRAM_MIN, max_n: int = CHAR_NGRAM_MAX) -> list[int]:
    """Return sorted unique FNV-1a bucket IDs for padded normalized char n-grams."""
    if buckets <= 0 or min_n <= 0 or max_n < min_n:
        raise ValueError("invalid subword hash configuration")
    text = f"^{normalize_text(value)}$"
    values = {
        fnv1a_utf8(text[start : start + width]) % buckets
        for width in range(min_n, max_n + 1)
        for start in range(max(0, len(text) - width + 1))
    }
    return sorted(values)


def _cache_key(username: str) -> str:
    return hashlib.sha256(username.encode("utf-8")).hexdigest()


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _load_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid local cache JSON: {path}") from error


def _profile_tracks(path: Path) -> dict[str, dict[str, Any]]:
    source = _load_json(path, {})
    recordings = source.get("payload", {}).get("recordings", []) if isinstance(source, dict) else []
    tracks: dict[str, dict[str, Any]] = {}
    if not isinstance(recordings, list):
        return tracks
    for row in recordings:
        if not isinstance(row, dict):
            continue
        mbid, count = row.get("recording_mbid"), row.get("listen_count")
        if not isinstance(mbid, str) or not mbid or isinstance(count, bool) or not isinstance(count, int) or count < 1:
            continue
        candidate = {
            "id": mbid,
            "artist": row.get("artist_name") if isinstance(row.get("artist_name"), str) else "Unknown artist",
            "title": row.get("track_name") if isinstance(row.get("track_name"), str) else mbid,
            "listen_count": count,
        }
        previous = tracks.get(mbid)
        if previous is None or (candidate["listen_count"], candidate["artist"], candidate["title"]) > (previous["listen_count"], previous["artist"], previous["title"]):
            tracks[mbid] = candidate
    return tracks


def _retrieval_tracks(path: Path) -> dict[str, dict[str, Any]]:
    source = _load_json(path, {})
    rows = source.get("rows", []) if isinstance(source, dict) else []
    tracks: dict[str, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return tracks
    for row in rows:
        if not isinstance(row, dict):
            continue
        mbid = row.get("recording_mbid")
        if not isinstance(mbid, str) or not mbid:
            continue
        artist = row.get("artist_credit_name") if isinstance(row.get("artist_credit_name"), str) else "Unknown artist"
        title = row.get("recording_name") if isinstance(row.get("recording_name"), str) else mbid
        tracks.setdefault(mbid, {"id": mbid, "artist": artist, "title": title, "listen_count": 0})
    return tracks


def build_popularity(user_track_ids: dict[str, set[str]], bands: int = POPULARITY_BANDS) -> dict[str, dict[str, float | int]]:
    """Compute train-user frequency percentiles and decile bands for every track."""
    if bands < 1:
        raise ValueError("popularity bands must be positive")
    frequency: Counter[str] = Counter(track for ids in user_track_ids.values() for track in set(ids))
    total = len(frequency)
    if total == 0:
        return {}
    lower_counts: dict[int, int] = {}
    running = 0
    for value in sorted(set(frequency.values())):
        lower_counts[value] = running
        running += sum(count == value for count in frequency.values())
    output: dict[str, dict[str, float | int]] = {}
    for track_id, value in frequency.items():
        percentile = 0.0 if total == 1 else lower_counts[value] / (total - 1)
        output[track_id] = {"user_frequency": value, "percentile": percentile, "band": min(bands - 1, int(percentile * bands))}
    return output


def _popularity_for(track_id: str, popularity: dict[str, dict[str, float | int]]) -> dict[str, float | int]:
    return popularity.get(track_id, {"user_frequency": 0, "percentile": 0.0, "band": 0})


def _episode_rng(seed: int, user_key: str, mask_index: int) -> random.Random:
    return random.Random(f"tastelift-v{DATASET_VERSION}:{seed}:{user_key}:{mask_index}")


def _negative_id(
    known_positive_ids: set[str],
    retrieval_ids: set[str],
    all_track_ids: set[str],
    positive_band: int,
    popularity: dict[str, dict[str, float | int]],
    rng: random.Random,
) -> tuple[str, str, int] | None:
    def eligible(ids: set[str]) -> list[str]:
        return sorted(track_id for track_id in ids if track_id not in known_positive_ids)

    retrieved = eligible(retrieval_ids)
    corpus = eligible(all_track_ids)
    for source, choices in (("retrieval", retrieved), ("corpus", corpus)):
        exact = [track_id for track_id in choices if int(_popularity_for(track_id, popularity)["band"]) == positive_band]
        if exact:
            chosen = exact[rng.randrange(len(exact))]
            return chosen, source, positive_band
        if source == "retrieval" and choices:
            distance = min(abs(int(_popularity_for(track_id, popularity)["band"]) - positive_band) for track_id in choices)
            nearest = [track_id for track_id in choices if abs(int(_popularity_for(track_id, popularity)["band"]) - positive_band) == distance]
            chosen = nearest[rng.randrange(len(nearest))]
            return chosen, source, int(_popularity_for(chosen, popularity)["band"])
    return None


def _episodes_for_user(
    partition: str,
    user_key: str,
    profile: dict[str, dict[str, Any]],
    retrieval: dict[str, dict[str, Any]],
    all_track_ids: set[str],
    popularity: dict[str, dict[str, float | int]],
    masks_per_user: int,
    seed: int,
) -> list[dict[str, Any]]:
    strong_ids = sorted(track_id for track_id, track in profile.items() if track["listen_count"] >= STRONG_LISTEN_COUNT)
    if len(strong_ids) < MIN_SEEDS + 1:
        return []
    output = []
    known_ids = set(profile)
    for mask_index in range(masks_per_user):
        rng = _episode_rng(seed, user_key, mask_index)
        seed_count = rng.randint(MIN_SEEDS, min(MAX_SEEDS, len(strong_ids) - 1))
        shuffled = list(strong_ids)
        rng.shuffle(shuffled)
        seed_ids = sorted(shuffled[:seed_count])
        held_out = sorted(known_ids - set(seed_ids))
        if not held_out:
            continue
        positive_id = held_out[rng.randrange(len(held_out))]
        positive = _popularity_for(positive_id, popularity)
        negative = _negative_id(known_ids, set(retrieval), all_track_ids, int(positive["band"]), popularity, rng)
        if negative is None:
            continue
        negative_id, negative_source, negative_band = negative
        output.append({
            "partition": partition,
            "user_key": user_key,
            "mask_index": mask_index,
            "seed_ids": seed_ids,
            "positive_id": positive_id,
            "positive_band": int(positive["band"]),
            "negative_id": negative_id,
            "negative_band": negative_band,
            "negative_source": negative_source,
        })
    return output


def _track_metadata(tracks: dict[str, dict[str, Any]], popularity: dict[str, dict[str, float | int]]) -> tuple[dict[str, dict[str, int]], list[dict[str, Any]]]:
    artists = sorted({normalize_text(track["artist"]) for track in tracks.values()})
    titles = sorted({normalize_text(track["title"]) for track in tracks.values()})
    vocabulary = {
        "artist": {value: index + 1 for index, value in enumerate(artists)},
        "title": {value: index + 1 for index, value in enumerate(titles)},
    }
    metadata = []
    for track_id in sorted(tracks):
        track = tracks[track_id]
        artist, title = normalize_text(track["artist"]), normalize_text(track["title"])
        metadata.append({
            "id": track_id,
            "artist": artist,
            "title": title,
            "artist_vocab_id": vocabulary["artist"][artist],
            "title_vocab_id": vocabulary["title"][title],
            "artist_subword_ids": hashed_subword_ids(artist),
            "title_subword_ids": hashed_subword_ids(title),
            "popularity": _popularity_for(track_id, popularity),
        })
    return vocabulary, metadata


def build_dataset(
    root: Path,
    masks_per_user: int = DEFAULT_MASKS_PER_USER,
    seed: int = 41,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 25,
) -> dict[str, Any]:
    """Build an in-memory dataset from cached profiles/retrieval with resumable episodes."""
    if masks_per_user < 1 or checkpoint_every < 1:
        raise ValueError("masks_per_user and checkpoint_every must be positive")
    manifest_path = root / "data/manifests/real-splits-final.json"
    splits = _load_json(manifest_path, {})
    if not isinstance(splits, dict):
        raise ValueError("final split manifest must be an object")
    users: list[tuple[str, str]] = []
    seen_users: set[str] = set()
    for partition in ("train", "validation", "test"):
        names = splits.get(partition, [])
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"final split manifest has invalid {partition} users")
        for username in names:
            if username in seen_users:
                raise ValueError("final split manifest is not user-disjoint")
            seen_users.add(username)
            users.append((partition, _cache_key(username)))
    users.sort(key=lambda item: (item[0], item[1]))
    signature = hashlib.sha256(json.dumps({"version": DATASET_VERSION, "masks_per_user": masks_per_user, "seed": seed, "users": users}, separators=(",", ":")).encode("utf-8")).hexdigest()

    profiles: dict[str, dict[str, dict[str, Any]]] = {}
    retrievals: dict[str, dict[str, dict[str, Any]]] = {}
    tracks: dict[str, dict[str, Any]] = {}
    train_track_ids: dict[str, set[str]] = {}
    for partition, user_key in users:
        profile = _profile_tracks(root / f"data/cache/real-profiles/{user_key}.json")
        retrieval = _retrieval_tracks(root / f"data/cache/real-retrieval/{user_key}.json")
        profiles[user_key], retrievals[user_key] = profile, retrieval
        for track_id, track in sorted((profile | retrieval).items()):
            tracks.setdefault(track_id, track)
        if partition == "train" and profile:
            train_track_ids[user_key] = set(profile)
    popularity = build_popularity(train_track_ids)
    vocabulary, metadata = _track_metadata(tracks, popularity)

    completed: dict[str, list[dict[str, Any]]] = {}
    if checkpoint_path and checkpoint_path.exists():
        checkpoint = _load_json(checkpoint_path, {})
        if checkpoint.get("signature") != signature or not isinstance(checkpoint.get("episodes_by_user"), dict):
            raise ValueError("checkpoint does not match this deterministic dataset build")
        completed = {key: value for key, value in checkpoint["episodes_by_user"].items() if isinstance(key, str) and isinstance(value, list)}

    processed_since_checkpoint = 0
    all_track_ids = set(tracks)
    for partition, user_key in users:
        if user_key in completed:
            continue
        completed[user_key] = _episodes_for_user(partition, user_key, profiles[user_key], retrievals[user_key], all_track_ids, popularity, masks_per_user, seed)
        processed_since_checkpoint += 1
        if checkpoint_path and processed_since_checkpoint >= checkpoint_every:
            _atomic_json_write(checkpoint_path, {"version": DATASET_VERSION, "signature": signature, "episodes_by_user": completed})
            processed_since_checkpoint = 0
    if checkpoint_path:
        _atomic_json_write(checkpoint_path, {"version": DATASET_VERSION, "signature": signature, "episodes_by_user": completed})

    episodes = [episode for _, user_key in users for episode in completed.get(user_key, [])]
    aggregate = {
        "users_in_final_manifest": len(users),
        "users_with_profile_cache": sum(bool(profiles[user_key]) for _, user_key in users),
        "episodes": len(episodes),
        "episodes_by_partition": {partition: sum(episode["partition"] == partition for episode in episodes) for partition in ("train", "validation", "test")},
        "negative_sources": {source: sum(episode["negative_source"] == source for episode in episodes) for source in ("retrieval", "corpus")},
    }
    return {
        "version": DATASET_VERSION,
        "split_manifest": "data/manifests/real-splits-final.json",
        "hashing": {"normalization": "NFKC-lower-whitespace", "encoding": "UTF-8", "algorithm": "FNV-1a-32", "offset_basis": FNV1A_OFFSET_BASIS, "prime": FNV1A_PRIME, "buckets": HASH_BUCKETS, "char_ngram_min": CHAR_NGRAM_MIN, "char_ngram_max": CHAR_NGRAM_MAX},
        "vocabulary": vocabulary,
        "tracks": metadata,
        "episodes": episodes,
        "aggregate": aggregate,
    }


def write_dataset(path: Path, dataset: dict[str, Any]) -> None:
    """Atomically publish a completed dataset only after all episodes were built."""
    _atomic_json_write(path, dataset)

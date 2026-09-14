"""Build a compact neural retrieval catalog from cached training-user histories."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch

from ml.tastelift_data import _atomic_json_write
from ml.tastelift_model import TasteLift


def _cache_key(username: str) -> str:
    return hashlib.sha256(username.encode("utf-8")).hexdigest()


def collect_train_tracks(root: Path, manifest_path: Path, popularity: dict[str, float]) -> list[dict]:
    """Read only profiles assigned to train; raw usernames never enter the artifact."""
    manifest = json.loads(manifest_path.read_text())
    tracks: dict[str, dict] = {}
    for username in manifest["train"]:
        path = root / "data/cache/real-profiles" / f"{_cache_key(username)}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text()).get("payload", {})
        for row in payload.get("recordings", []):
            mbid = str(row.get("recording_mbid") or "").strip()
            artist = str(row.get("artist_name") or "").strip()
            title = str(row.get("track_name") or "").strip()
            if not mbid or not artist or not title:
                continue
            candidate = {
                "mbid": mbid,
                "artist": artist,
                "title": title,
                **({"release": str(row["release_name"]).strip()} if row.get("release_name") else {}),
                "popularityPercentile": float(popularity.get(mbid, 0.0)),
            }
            previous = tracks.get(mbid)
            if previous is None or json.dumps(candidate, ensure_ascii=False, sort_keys=True) < json.dumps(previous, ensure_ascii=False, sort_keys=True):
                tracks[mbid] = candidate
    return [tracks[mbid] for mbid in sorted(tracks)]


def quantize_vectors(vectors: Iterable[Iterable[float]]) -> tuple[bytes, float]:
    encoded = bytearray()
    maximum_error = 0.0
    for vector in vectors:
        for value in vector:
            clipped = max(-1.0, min(1.0, float(value)))
            quantized = max(-127, min(127, round(clipped * 127)))
            encoded.append(quantized & 0xFF)
            maximum_error = max(maximum_error, abs(clipped - quantized / 127))
    return bytes(encoded), maximum_error


def build_catalog(root: Path, dataset_path: Path, model_path: Path, manifest_path: Path, output_path: Path, batch_size: int = 2048) -> dict:
    dataset = json.loads(dataset_path.read_text())
    exported = json.loads(model_path.read_text())
    tracks_by_id = {track["id"]: track for track in dataset["tracks"]}
    model = TasteLift.from_export(model_path, dataset["tracks"])
    catalog_tracks = collect_train_tracks(root, manifest_path, exported.get("popularity", {}))
    vectors: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(catalog_tracks), batch_size):
            rows = []
            for track in catalog_tracks[offset:offset + batch_size]:
                source = tracks_by_id.get(track["mbid"])
                rows.append(source if source is not None else {"artist": track["artist"], "title": track["title"]})
            vectors.extend(model.encode_metadata(rows).cpu().tolist())
    packed, maximum_error = quantize_vectors(vectors)
    payload = {
        "format": "tastelift-catalog-v1",
        "dimensions": model.dim,
        "source": "listenbrainz-train-histories",
        "quantization": {"type": "symmetric-int8", "scale": 127, "encoding": "base64-row-major", "maximumError": maximum_error},
        "modelSha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "manifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "tracks": catalog_tracks,
        "vectors": base64.b64encode(packed).decode("ascii"),
    }
    _atomic_json_write(output_path, payload)
    return payload

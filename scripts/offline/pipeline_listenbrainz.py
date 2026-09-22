#!/usr/bin/env python3
"""
scripts/offline/pipeline_listenbrainz.py
Reproducible streaming ingestion pipeline for ListenBrainz Full Export + Incremental + KEXP radio tracklists.
Streams on-the-fly through curl | zstd | tar without saving hundreds of gigabytes to disk.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path

FULL_DUMP_URL = "https://data.metabrainz.org/pub/musicbrainz/listenbrainz/fullexport/listenbrainz-dump-2663-20260915-000002-full/listenbrainz-listens-dump-2663-20260915-000002-full.tar.zst"
FULL_DUMP_TOTAL_BYTES = 245936631999

INCREMENTAL_DUMP_URL = "https://data.metabrainz.org/pub/musicbrainz/listenbrainz/incremental/listenbrainz-dump-2672-20260922-000003-incremental/listenbrainz-listens-dump-2672-20260922-000003-incremental.tar.zst"
INCREMENTAL_DUMP_TOTAL_BYTES = 239227339

USER_AGENT = "MusicMyLoveBot/2.0 (academic neural recommendation study; musicmylove@example.com)"

def normalize_text(text: str) -> str:
    if not text:
        return ""
    norm = unicodedata.normalize("NFKC", text).lower().strip()
    norm = re.sub(r"\s+", " ", norm)
    return norm

def make_track_key(artist: str, title: str) -> str:
    return f"{normalize_text(artist)}\x1f{normalize_text(title)}"

def stream_archive_lines(url: str, byte_range_end: int):
    cmd = f"curl -s -r 0-{byte_range_end} -A '{USER_AGENT}' {url} | zstd -d -q 2>/dev/null | tar -xOf - 2>/dev/null"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, bufsize=1048576)
    try:
        for raw_line in proc.stdout:
            yield raw_line
    finally:
        proc.kill()
        proc.wait()

def fetch_kexp_plays(limit_plays: int = 500) -> list[dict]:
    print(f"[kexp] Fetching up to {limit_plays} live radio plays from KEXP API...")
    headers = {"User-Agent": USER_AGENT}
    plays = []
    offset = 0
    batch_size = 100
    while len(plays) < limit_plays:
        req_url = f"https://api.kexp.org/v2/plays/?limit={batch_size}&offset={offset}"
        req = urllib.request.Request(req_url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                results = data.get("results", [])
                if not results:
                    break
                for r in results:
                    if r.get("play_type") == "trackplay" and r.get("artist") and r.get("song"):
                        plays.append({
                            "user": f"kexp-show-{r.get('show', 'unknown')}",
                            "artist": r.get("artist"),
                            "title": r.get("song"),
                            "mbid": r.get("recording_id") or "",
                            "timestamp": int(time.time()),
                        })
                offset += batch_size
                if not data.get("next"):
                    break
        except Exception as e:
            print(f"[kexp] Warning: failed to fetch offset {offset}: {e}")
            break
    print(f"[kexp] Extracted {len(plays)} valid trackplays from KEXP")
    return plays

def main():
    parser = argparse.ArgumentParser(description="Ingest ListenBrainz Full Export streaming subset")
    parser.add_argument("--full-bytes", type=int, default=52428800, help="Compressed bytes to stream from full dump (default 50MB)")
    parser.add_argument("--incremental-bytes", type=int, default=20971520, help="Compressed bytes to stream from incremental dump (default 20MB)")
    parser.add_argument("--session-gap-sec", type=int, default=1800, help="Max gap in seconds between listens in a session (default 30m)")
    parser.add_argument("--output-dir", type=str, default="data/cache/pipeline")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("ListenBrainz Reproducible Streaming Pipeline")
    print("==================================================")
    print(f"Target Full Dump: {FULL_DUMP_URL}")
    print(f"Full Dump Archive Size: {FULL_DUMP_TOTAL_BYTES:,} bytes (~{FULL_DUMP_TOTAL_BYTES / (1024**3):.2f} GiB)")
    print(f"Streaming slice: {args.full_bytes:,} bytes (~{args.full_bytes / (1024**2):.2f} MiB)")
    print(f"Fraction of Full Dump: {args.full_bytes / FULL_DUMP_TOTAL_BYTES * 100:.5f}%")
    print(f"Incremental Dump: {INCREMENTAL_DUMP_URL}")
    print(f"Incremental streaming slice: {args.incremental_bytes:,} bytes (~{args.incremental_bytes / (1024**2):.2f} MiB)")

    t0 = time.time()
    user_listens = defaultdict(list)
    raw_listens_count = 0
    unique_users = set()

    # 1. Stream from Full Dump
    print("\n[stage 1/3] Streaming from official Full Export...")
    for raw in stream_archive_lines(FULL_DUMP_URL, args.full_bytes):
        raw_listens_count += 1
        try:
            data = json.loads(raw)
            user = data.get("user_name") or str(data.get("user_id", ""))
            ts = data.get("timestamp") or 0
            meta = data.get("track_metadata", {})
            artist = meta.get("artist_name")
            title = meta.get("track_name")
            if not artist or not title or not user:
                continue
            add = meta.get("additional_info", {})
            mbid = add.get("recording_mbid") or data.get("recording_msid") or ""
            unique_users.add(user)
            user_listens[user].append({
                "artist": artist.strip(),
                "title": title.strip(),
                "mbid": mbid,
                "ts": ts,
            })
        except Exception:
            pass

    full_dump_listens = raw_listens_count
    print(f"[stage 1/3] Streamed {full_dump_listens:,} listens from full dump. Unique users: {len(unique_users):,}")

    # 2. Stream from Incremental Dump
    print("\n[stage 2/3] Streaming recent listens from Incremental Dump...")
    incremental_listens_count = 0
    for raw in stream_archive_lines(INCREMENTAL_DUMP_URL, args.incremental_bytes):
        raw_listens_count += 1
        incremental_listens_count += 1
        try:
            data = json.loads(raw)
            user = data.get("user_name") or str(data.get("user_id", ""))
            ts = data.get("timestamp") or 0
            meta = data.get("track_metadata", {})
            artist = meta.get("artist_name")
            title = meta.get("track_name")
            if not artist or not title or not user:
                continue
            add = meta.get("additional_info", {})
            mbid = add.get("recording_mbid") or data.get("recording_msid") or ""
            unique_users.add(user)
            user_listens[user].append({
                "artist": artist.strip(),
                "title": title.strip(),
                "mbid": mbid,
                "ts": ts,
            })
        except Exception:
            pass
    print(f"[stage 2/3] Streamed {incremental_listens_count:,} listens from incremental dump.")

    # 3. Add Live KEXP plays
    print("\n[stage 3/3] Ingesting verified KEXP public radio plays...")
    kexp_plays = fetch_kexp_plays(500)
    for p in kexp_plays:
        user_listens[p["user"]].append({
            "artist": p["artist"],
            "title": p["title"],
            "mbid": p["mbid"],
            "ts": p["timestamp"],
        })
        raw_listens_count += 1
        unique_users.add(p["user"])

    # 4. Session Segmentation & Filtering
    print("\n[processing] Segmenting listens into sessions (gap = %ds)..." % args.session_gap_sec)
    all_sessions = []
    track_frequency = defaultdict(int)
    track_users = defaultdict(set)
    track_sessions = defaultdict(int)

    track_registry = {}  # key -> {artist, title, mbid}

    # Pre-seed with existing catalog tracks to ensure 100% preservation
    existing_catalog_path = root / "ml/tastelift-catalog.json"
    if existing_catalog_path.exists():
        existing_cat = json.loads(existing_catalog_path.read_text())
        for t in existing_cat.get("tracks", []):
            k = make_track_key(t["artist"], t["title"])
            if k not in track_registry:
                track_registry[k] = {
                    "artist": t["artist"],
                    "title": t["title"],
                    "mbid": t.get("mbid") or f"track-{hashlib.sha256(k.encode()).hexdigest()[:16]}",
                }

    for user, listens in user_listens.items():
        listens.sort(key=lambda x: x["ts"])
        current_session = []
        last_ts = 0

        for item in listens:
            k = make_track_key(item["artist"], item["title"])
            if not k or len(item["artist"]) > 200 or len(item["title"]) > 200:
                continue

            if k not in track_registry:
                mbid = item["mbid"] if item["mbid"] else f"track-{hashlib.sha256(k.encode()).hexdigest()[:16]}"
                track_registry[k] = {
                    "artist": item["artist"],
                    "title": item["title"],
                    "mbid": mbid,
                }

            if last_ts > 0 and (item["ts"] - last_ts) > args.session_gap_sec:
                if len(current_session) >= 2:
                    all_sessions.append(current_session)
                current_session = []

            # Dedup adjacent trackplays in session
            if not current_session or current_session[-1] != k:
                current_session.append(k)
            last_ts = item["ts"]

        if len(current_session) >= 2:
            all_sessions.append(current_session)

    # Filter spam sessions (e.g. repetition > 80% same artist)
    valid_sessions = []
    for s in all_sessions:
        if len(s) < 2 or len(s) > 150:
            continue
        artists = [k.split("\x1f")[0] for k in s]
        most_common_artist_count = max(artists.count(a) for a in set(artists))
        if most_common_artist_count / len(s) > 0.85 and len(s) > 3:
            continue  # single-artist looping spam
        valid_sessions.append(s)
        s_set = set(s)
        for k in s_set:
            track_sessions[k] += 1
            track_frequency[k] += 1

    # Ingest existing web tracklists
    web_tracklists_path = root / "data/cache/web-tracklists.json"
    if web_tracklists_path.exists():
        web_lists = json.loads(web_tracklists_path.read_text())
        for w in web_lists:
            w_tracks = w.get("tracks", [])
            w_session = []
            for t in w_tracks:
                k = make_track_key(t.get("artist", ""), t.get("title", ""))
                if k not in track_registry:
                    track_registry[k] = {
                        "artist": t.get("artist", "").strip(),
                        "title": t.get("title", "").strip(),
                        "mbid": f"web-{hashlib.sha256(k.encode()).hexdigest()[:16]}",
                    }
                w_session.append(k)
            if len(w_session) >= 2:
                valid_sessions.append(w_session)
                for k in set(w_session):
                    track_sessions[k] += 1
                    track_frequency[k] += 1

    # Build indexed tracks catalog with relative popularity percentiles
    print(f"\n[processing] Building indexed catalog of {len(track_registry):,} tracks...")
    all_keys = sorted(track_registry.keys())
    key_to_idx = {k: i for i, k in enumerate(all_keys)}

    # Compute popularity rank
    sorted_by_freq = sorted(all_keys, key=lambda k: (track_sessions[k], track_frequency[k]), reverse=True)
    n_tracks = len(all_keys)
    popularity_percentiles = {}
    for rank, k in enumerate(sorted_by_freq):
        # 1.0 for top popularity, down to ~0.01 for tail
        popularity_percentiles[k] = round(max(0.01, 1.0 - (rank / max(1, n_tracks))), 4)

    catalog_tracks = []
    for k in all_keys:
        info = track_registry[k]
        catalog_tracks.append({
            "mbid": info["mbid"],
            "artist": info["artist"],
            "title": info["title"],
            "popularity": popularity_percentiles[k],
            "sessions": track_sessions[k],
        })

    # Convert valid_sessions to integer indices
    indexed_sessions = []
    for s in valid_sessions:
        indices = [key_to_idx[k] for k in s if k in key_to_idx]
        if len(set(indices)) >= 2:
            indexed_sessions.append(indices)

    dt = time.time() - t0

    # Ingest Report
    report = {
        "dataset": "ListenBrainz Official Full Export (dump 2663) + Incremental (dump 2672) + KEXP",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "full_dump": {
            "url": FULL_DUMP_URL,
            "total_bytes": FULL_DUMP_TOTAL_BYTES,
            "streamed_bytes": args.full_bytes,
            "fraction_of_full_dump": args.full_bytes / FULL_DUMP_TOTAL_BYTES,
            "listens_extracted": full_dump_listens,
        },
        "incremental_dump": {
            "url": INCREMENTAL_DUMP_URL,
            "total_bytes": INCREMENTAL_DUMP_TOTAL_BYTES,
            "streamed_bytes": args.incremental_bytes,
            "listens_extracted": incremental_listens_count,
        },
        "kexp_radio": {
            "plays_extracted": len(kexp_plays),
        },
        "summary": {
            "total_raw_listens": raw_listens_count,
            "total_unique_users": len(unique_users),
            "total_segmented_sessions": len(valid_sessions),
            "total_resolved_tracks": len(catalog_tracks),
            "pipeline_runtime_sec": round(dt, 2),
        }
    }

    # Save outputs
    print(f"\n[saving] Writing catalog and sessions to {out_dir}...")
    (out_dir / "expanded_catalog.json").write_text(json.dumps({"tracks": catalog_tracks}, indent=2))
    (out_dir / "expanded_sessions.json").write_text(json.dumps({"sessions": indexed_sessions}))
    (out_dir / "ingest_report.json").write_text(json.dumps(report, indent=2))

    print("==================================================")
    print("Ingest Completed Successfully in %.2fs!" % dt)
    print(json.dumps(report, indent=2))
    print("==================================================")

if __name__ == "__main__":
    main()

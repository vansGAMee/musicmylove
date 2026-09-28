import hashlib
import json
import random
import re
import subprocess
import urllib.request
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

BASE_URL = "https://ftp.musicbrainz.org/pub/musicbrainz/listenbrainz/incremental/"
MAX_COMPRESSED_BYTES = 500 * 1024 * 1024
USER_AGENT = "MusicMyLove/0.2 (real-cohort recommendation research; contact: dev@example.com)"


def extract_usernames(lines: Iterable[bytes]) -> list[str]:
    names: set[str] = set()
    for line in lines:
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        name = value.get("user_name") if isinstance(value, dict) else None
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return sorted(names)


def choose_users(names: list[str], count: int, seed: int) -> list[str]:
    ordered = sorted(set(names))
    random.Random(seed).shuffle(ordered)
    return ordered[:count]


def count_user_listens(lines: Iterable[bytes]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for line in lines:
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        name = value.get("user_name") if isinstance(value, dict) else None
        if isinstance(name, str) and name.strip():
            counts[name.strip()] += 1
    return dict(sorted(counts.items()))


def choose_active_users(counts: dict[str, int], count: int, minimum_listens: int, seed: int) -> list[str]:
    return choose_users([name for name, listens in counts.items() if listens >= minimum_listens], count, seed)


def choose_active_users_excluding(counts: dict[str, int], count: int, minimum_listens: int, seed: int, excluded: set[str]) -> list[str]:
    return choose_users([name for name, listens in counts.items() if listens >= minimum_listens and name not in excluded], count, seed)


def _read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def discover_latest_archive() -> tuple[str, str]:
    listing = _read_url(BASE_URL).decode()
    directories = sorted(set(re.findall(r'href="(listenbrainz-dump-[^"]+-incremental)/"', listing)))
    if not directories:
        raise RuntimeError("No incremental ListenBrainz dumps listed")
    directory = directories[-1]
    detail_url = f"{BASE_URL}{directory}/"
    detail = _read_url(detail_url).decode()
    files = re.findall(r'href="([^"]+\.tar\.zst)"', detail)
    if not files:
        raise RuntimeError("Latest incremental dump has no listens archive")
    return f"{detail_url}{files[0]}", directory


def download_verified(url: str, target: Path) -> tuple[int, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as handle:
        declared = int(response.headers.get("Content-Length", "0"))
        if declared and declared > MAX_COMPRESSED_BYTES:
            raise RuntimeError(f"Archive exceeds 500 MB limit: {declared}")
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_COMPRESSED_BYTES:
                raise RuntimeError(f"Archive exceeds 500 MB limit: {total}")
            digest.update(chunk)
            handle.write(chunk)
    actual = digest.hexdigest()
    expected = _read_url(f"{url}.sha256").decode().strip().split()[0]
    if actual != expected:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch: expected {expected}, got {actual}")
    return total, actual


def usernames_from_archive(path: Path) -> list[str]:
    process = subprocess.Popen(
        ["tar", "--use-compress-program=unzstd", "-xOf", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    names = extract_usernames(process.stdout)
    stderr = process.stderr.read().decode() if process.stderr else ""
    if process.wait() != 0:
        raise RuntimeError(f"Unable to extract dump: {stderr[-1000:]}")
    return names


def activity_from_archive(path: Path) -> dict[str, int]:
    process = subprocess.Popen(
        ["tar", "--use-compress-program=unzstd", "-xOf", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    counts = count_user_listens(process.stdout)
    stderr = process.stderr.read().decode() if process.stderr else ""
    if process.wait() != 0:
        raise RuntimeError(f"Unable to extract dump: {stderr[-1000:]}")
    return counts

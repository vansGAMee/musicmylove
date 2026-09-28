"""Official incremental dump downloader. Subsets are NOT the full ListenBrainz corpus.
Resume files, bounded retries, authoritative SHA-256, stable user sampling.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
try:
    from .config import DATA, REPORTS, write_json, fingerprint
except ImportError:
    from config import DATA, REPORTS, write_json, fingerprint

BASE = 'https://data.metabrainz.org/pub/musicbrainz/listenbrainz/incremental/'
UA = os.environ.get('LISTENBRAINZ_USER_AGENT', 'MusicMyLoveOfflineResearch/0.2 (local neural recommender; Python urllib; noncommercial research)')


def request(url, headers=None):
    if not url.startswith('https://data.metabrainz.org/'):
        raise ValueError('Only official HTTPS dump origin is supported')
    error = None
    for attempt in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA, **(headers or {})}), timeout=60)
        except (urllib.error.URLError, TimeoutError) as exc:
            error = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                raise
            retry = exc.headers.get('Retry-After', '0') if isinstance(exc, urllib.error.HTTPError) else '0'
            time.sleep(min(30, max(2 ** attempt, int(retry) if retry.isdigit() else 0)))
    raise RuntimeError(f'DOWNLOAD_FAILED {url}: {error}')


def text(url):
    with request(url) as response:
        return response.read().decode()


def discover(days):
    dirs = re.findall(r'href="(listenbrainz-dump-(\d+)-\d+-\d+-incremental/)"', text(BASE))
    urls, dates = [], set()
    for name, _ in sorted(dirs, key=lambda x: int(x[1]), reverse=True):
        page = text(BASE + name)
        files = re.findall(r'href="(listenbrainz-listens-dump-[^"/]+\.tar\.zst)"', page)
        if not files:
            continue
        url = BASE + name + files[0]
        with request(url) as response:
            if int(response.headers.get('Content-Length', '0')) < 65536:
                continue  # Skip empty/header-only publications, not a daily listen corpus.
        date = name.split('-')[3]
        if date in dates:
            continue
        dates.add(date)
        urls.append(BASE + name + files[0])
        if len(urls) == days:
            break
    if len(urls) < days:
        raise RuntimeError(f'Only {len(urls)} official daily listens archives available; requested {days}')
    return sorted(urls)


def download(url, directory):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / url.rsplit('/', 1)[1]
    expected = text(url + '.sha256').split()[0].lower()
    if not re.fullmatch('[a-f0-9]{64}', expected):
        raise RuntimeError('Invalid official checksum')
    if target.exists():
        if fingerprint(target) != expected:
            raise RuntimeError(f'CHECKSUM_FAIL: remove corrupt cache file {target}')
        return target
    partial = target.with_suffix(target.suffix + '.part')
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with request(url, {'Range': f'bytes={offset}-'} if offset else {}) as response:
                resumed = response.status == 206
                if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise RuntimeError('Invalid resumed HTTP byte range')
                if not resumed:
                    offset = 0
                with partial.open('ab' if resumed else 'wb') as output:
                    shutil.copyfileobj(response, output, 1024 * 1024)
            if fingerprint(partial) != expected:
                raise RuntimeError(f'CHECKSUM_FAIL: {partial}; incomplete or corrupt download')
            partial.replace(target)
            return target
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            if attempt == 3:
                raise RuntimeError(f'DOWNLOAD_FAILED: partial kept at {partial}') from error
            time.sleep(2 ** attempt)
    raise RuntimeError('Download did not complete')


@contextmanager
def archive_stream(path):
    if path.suffix != '.zst':
        with tarfile.open(path, 'r|*') as archive:
            yield archive
        return
    # No shell interpolation; stream decompression; never extract tar paths to disk.
    if not shutil.which('zstd'):
        raise RuntimeError('Install zstd (Ubuntu: sudo apt install zstd; Arch: sudo pacman -S zstd)')
    process = subprocess.Popen(['zstd', '-d', '-q', '-c', str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            yield archive
        process.stdout.close()
        _, error = process.communicate(timeout=30)
        if process.returncode:
            raise RuntimeError('DECOMPRESS_FAILED: ' + error.decode())
    finally:
        if process.poll() is None:
            process.kill(); process.wait()


def extract_listens(path, target, fraction=.1):
    if not 0 < fraction <= 1:
        raise ValueError('user fraction must be in (0,1]')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix('.tmp')
    users, listens, scanned, decoded, malformed = set(), 0, 0, 0, 0
    digest = hashlib.sha256()
    with archive_stream(Path(path)) as archive, partial.open('wb') as output:
        for member in archive:
            if not member.isfile() or not member.name.endswith('.listens'):
                continue
            stream = archive.extractfile(member)
            for raw in stream:
                scanned += 1; decoded += len(raw)
                try:
                    row = json.loads(raw)
                    user = row.get('user_name')
                    if not isinstance(user, str) or not user:
                        raise ValueError('Missing user_name')
                    user_hash = hashlib.sha256(user.encode()).hexdigest()
                    if int(user_hash[:16], 16) / 2**64 >= fraction:
                        continue
                    output.write(raw); digest.update(raw)
                    users.add(user_hash); listens += 1
                except (ValueError, TypeError, AttributeError):
                    malformed += 1
    if not listens:
        raise RuntimeError('DATA_FAIL: no selected original listens; output not published')
    partial.replace(target)
    return {'archive': str(path), 'archive_bytes': Path(path).stat().st_size, 'archive_sha256': fingerprint(path), 'output': str(target), 'scanned_listens': scanned,
            'decoded_bytes_processed': decoded, 'listens': listens, 'users': len(users),
            'invalid_rows': malformed, 'user_fraction': fraction, 'sha256': digest.hexdigest(),
            'scope': 'stable user subset of daily incremental submissions, not full historical libraries'}


def collect(days=7, fraction=.1):
    cache = DATA / 'cache' / 'listenbrainz'
    manifest_path = cache / 'download_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['days'] != days or manifest['user_fraction'] != fraction:
            raise RuntimeError('Frozen download selection differs; use a new --run directory')
    else:
        manifest = {'days': days, 'user_fraction': fraction, 'urls': discover(days)}
        write_json(manifest_path, manifest)
    reports = []
    for url in manifest['urls']:
        print('Downloading/verifying ' + url, flush=True)
        archive = download(url, cache)
        target = archive.with_suffix('').with_suffix('.jsonl')
        report_path = target.with_suffix('.report.json')
        if target.exists() and report_path.exists():
            report = json.loads(report_path.read_text())
            if report['user_fraction'] != fraction or fingerprint(target) != report['sha256']:
                raise RuntimeError('Cached extracted subset mismatch')
        else:
            report = extract_listens(archive, target, fraction)
            write_json(report_path, report)
        reports.append(report)
        print(json.dumps(report), flush=True)
    write_json(REPORTS / 'collection.json', reports)
    return [r['output'] for r in reports]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--days', type=int, default=7)
    p.add_argument('--user-fraction', type=float, default=.1)
    args = p.parse_args()
    if not 1 <= args.days <= 28 or not 0 < args.user_fraction <= 1:
        p.error('days 1..28, fraction (0,1] required')
    collect(args.days, args.user_fraction)

"""Expand the original 25% user sample from already cached official archives.

No network, no training, no change to existing experiments. Events are sharded
on disk by user; dataset JSON is streamed to avoid duplicating it in RAM.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from .config import ROOT, fingerprint, write_json
from .collect_data import extract_listens
from .prepare_data import identify, normalize, split_user


def cached_archives(source):
    report = json.loads((source / 'reports/collection.json').read_text())
    archives = {}
    for entry in report:
        path = Path(entry['archive']).resolve()
        expected = entry['archive_sha256']
        if not path.is_file():
            raise RuntimeError(f'Cached archive missing: {path}; no automatic download')
        if fingerprint(path) != expected:
            raise RuntimeError(f'Archive checksum mismatch: {path}')
        archives[path] = expected
    if not archives:
        raise RuntimeError('No cached original archives')
    return sorted(archives.items())


def verify_splits(original, expanded):
    changed = [u for u, split in original.items() if expanded.get(u) != split]
    if changed:
        raise RuntimeError(f'Original user split changed or disappeared: {len(changed)} users')


def prepare_bounded(paths, output, reports, shards=32, expected_original=None):
    """Same identities/session rules as prepare_data; one user shard in memory.

    Canonical track metadata and its index still require catalog-sized memory;
    this bounds listen-event memory, not total memory independently of catalog.
    """
    output, reports = Path(output), Path(reports)
    output.mkdir(parents=True, exist_ok=True)
    registry, aliases, users = {}, defaultdict(set), {}
    sources, invalid, parsed, mbid_listens, msid_listens = [], 0, 0, 0, 0
    low, high = None, None
    with tempfile.TemporaryDirectory(prefix='prepare-', dir=output) as scratch:
        shard_paths = [Path(scratch)/f'{i}.jsonl' for i in range(shards)]
        with ExitStack() as stack:
            streams = [stack.enter_context(p.open('w')) for p in shard_paths]
            for path in sorted({Path(p).resolve() for p in paths}):
                digest, byte_count = hashlib.sha256(), 0
                opener = gzip.open if path.suffix == '.gz' else open
                with opener(path, 'rb') as stream:
                    for raw in stream:
                        digest.update(raw); byte_count += len(raw)
                        try:
                            row = json.loads(raw)
                            msid_listens += bool(row.get('recording_msid') or (row.get('track_metadata', {}).get('additional_info') or {}).get('recording_msid'))
                            track = identify(row)
                            user = row.get('user_name')
                            ts = row.get('listened_at', row.get('timestamp'))
                            if not isinstance(user, str) or not user.strip() or not isinstance(ts, int) or ts <= 0:
                                raise ValueError('Original user/timestamp required')
                            datetime.fromtimestamp(ts, timezone.utc)
                            uid = hashlib.sha256(user.encode()).hexdigest()
                            tid = track['id']
                            aliases[tid].add((normalize(track['artist']), normalize(track['title'])))
                            if tid not in registry or json.dumps(track, sort_keys=True) < json.dumps(registry[tid], sort_keys=True):
                                registry[tid] = track
                            users[uid] = split_user(uid)
                            parsed += 1; mbid_listens += bool(track['recording_mbid'])
                            low = ts if low is None else min(low, ts)
                            high = ts if high is None else max(high, ts)
                        except (ValueError, TypeError, AttributeError, OverflowError, OSError):
                            invalid += 1
                            continue
                        # Disk failures must abort, never masquerade as malformed input.
                        streams[int(uid[:8], 16) % shards].write(json.dumps([uid, ts, tid]) + '\n')
                sources.append({'path': str(path), 'file_bytes': path.stat().st_size,
                                'decoded_bytes_processed': byte_count, 'sha256_decoded': digest.hexdigest()})
                print(f'Indexed {parsed} listens; {len(registry)} identities; {len(users)} users', flush=True)
        digest = hashlib.sha256(json.dumps([(s['sha256_decoded'], s['decoded_bytes_processed']) for s in sources]).encode()).hexdigest()
        conflicts = {tid: sorted(names) for tid, names in aliases.items() if len(names) > 1 and not tid.startswith('mbid:')}
        counts = {s: sum(v == s for v in users.values()) for s in ('train', 'dev', 'shadow', 'final')}
        if conflicts or min(counts.values()) < 10 or len(registry) < 100:
            raise RuntimeError('Expanded data failed identity/size/split gates')
        manifest = {'fingerprint': digest, 'rule': 'sha256(python-mvp-v1:hashed-user), 75/10/10/5',
                    'users': dict(sorted(users.items()))}
        if expected_original is not None:
            verify_splits(expected_original, manifest['users'])
        target = output/'dataset.json'
        temporary = output/'dataset.tmp'
        ids = sorted(registry)
        raw_index = {tid: i for i, tid in enumerate(ids)}
        recording_mbids = sum(bool(t['recording_mbid']) for t in registry.values())
        alias_groups = sum(len(names) > 1 for tid, names in aliases.items() if tid.startswith('mbid:'))
        unique_listens, session_count = 0, 0
        with temporary.open('w') as destination:
            destination.write('{"fingerprint":' + json.dumps(digest) + ',"tracks":[')
            for i, tid in enumerate(ids):
                track = registry.pop(tid)
                track['aliases'] = [list(name) for name in sorted(aliases.pop(tid))]
                if i:
                    destination.write(',')
                json.dump(track, destination, ensure_ascii=False)
            del registry, aliases, ids
            destination.write('],"users":[')
            first = True
            for number, path in enumerate(shard_paths):
                events = defaultdict(set)
                with path.open() as stream:
                    for line in stream:
                        uid, ts, tid = json.loads(line)
                        events[uid].add((ts, raw_index[tid]))
                for uid in sorted(events):
                    ordered = sorted(events[uid])
                    unique_listens += len(ordered)
                    sessions, current, previous, start = [], set(), None, None
                    for ts, track in ordered:
                        if previous is not None and (ts - previous > 1800 or ts - start > 14400):
                            sessions.append(sorted(current)); current, start = set(), None
                        if start is None:
                            start = ts
                        current.add(track); previous = ts
                    if current:
                        sessions.append(sorted(current))
                    session_count += len(sessions)
                    user = {'id': uid, 'split': users[uid], 'tracks': sorted({t for _, t in ordered}), 'sessions': sessions}
                    if not first:
                        destination.write(',')
                    json.dump(user, destination, ensure_ascii=False)
                    first = False
                del events
                print(f'Prepared user shard {number+1}/{shards}', flush=True)
            destination.write(']}\n')
        audit = {'sources': sources, 'fingerprint': digest, 'parsed_listens': parsed, 'unique_listens': unique_listens,
                 'invalid_rows': invalid, 'users': len(users), 'tracks': len(raw_index), 'sessions': session_count,
                 'recording_mbids': recording_mbids, 'listens_with_recording_mbid': mbid_listens,
                 'rows_with_msid': msid_listens, 'split_users': counts, 'identity_conflicts': conflicts,
                 'recording_alias_groups': alias_groups, 'duplicate_ids': 0, 'status': 'AUDITED',
                 'period_utc': [datetime.fromtimestamp(t, timezone.utc).isoformat() for t in (low, high)],
                 'preparation': 'streamed-json-with-user-event-shards-v1'}
        write_json(reports/'data_audit.json', audit)
        write_json(output/'split_manifest.json', manifest)
        temporary.replace(target)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'data/main')
    parser.add_argument('--run', type=Path, default=ROOT/'data/expanded-v1')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    source, run = args.source.resolve(), args.run.resolve()
    if source == run:
        raise RuntimeError('Expansion must have a separate output directory')
    archives = cached_archives(source)
    original_path = source/'data/split_manifest.json'
    original = json.loads(original_path.read_text())['users']
    contract = {'archives': {str(p): sha for p, sha in archives}, 'user_fraction': 1.,
                'original_manifest_sha256': fingerprint(original_path),
                'implementation': {name: fingerprint(ROOT/name) for name in
                    ('expand_data.py', 'prepare_data.py', 'collect_data.py', 'config.py')}}
    marker = run/'expansion.json'
    if run.exists() and (not marker.exists() or json.loads(marker.read_text()) != contract):
        raise RuntimeError('Output belongs to another experiment; choose a new --run')
    print(f'{len(archives)} cached archives verified; all users will be included. No download or training.', flush=True)
    if args.check_only:
        return
    run.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        write_json(marker, contract)
    complete = run/'complete.json'
    files = ('data/dataset.json', 'data/split_manifest.json', 'reports/data_audit.json')
    if complete.exists():
        recorded = json.loads(complete.read_text())
        if (not all(name in recorded for name in files)
                or any(not (run/name).is_file() or fingerprint(run/name) != sha for name, sha in recorded.items())):
            raise RuntimeError('Completed expanded data changed; refusing silent reuse')
        print('Expanded data already complete. Nothing repeated.', flush=True)
    else:
        paths, reports = [], []
        for archive, expected in archives:
            cache = run/'data/cache/listenbrainz'; cache.mkdir(parents=True, exist_ok=True)
            target = cache/(archive.name + '.jsonl')
            report_path = target.with_suffix('.report.json')
            if target.exists() and report_path.exists():
                report = json.loads(report_path.read_text())
                if report['sha256'] != fingerprint(target) or report['archive_sha256'] != expected or report['user_fraction'] != 1.:
                    raise RuntimeError('Expanded extraction cache mismatch')
                print(f'Reusing verified extraction: {archive.name}', flush=True)
            else:
                print(f'Extracting all users: {archive.name}', flush=True)
                report = extract_listens(archive, target, fraction=1.)
                write_json(report_path, report)
            paths.append(target); reports.append(report)
        write_json(run/'reports/collection.json', reports)
        prepare_bounded(paths, run/'data', run/'reports', expected_original=original)
    # Include consumption provenance in integrity checks, including a source
    # marker created after this expansion first completed. Never remove a lock.
    frozen, copied = source/'reports/final_consumed.lock', run/'reports/final_consumed.lock'
    if frozen.exists():
        if copied.exists() and fingerprint(copied) != fingerprint(frozen):
            raise RuntimeError('Source FINAL consumption marker changed')
        if not copied.exists():
            shutil.copyfile(frozen, copied)
    if copied.exists():
        files = (*files, 'reports/final_consumed.lock')
    write_json(complete, {name: fingerprint(run/name) for name in files})
    import shlex
    print('Next, train in a NEW experiment:')
    print('python -m python_mvp.honest_training --source ' + shlex.quote(str(run))
          + ' --run ' + shlex.quote(str(run/'model')))
    print('No neural training was performed. More data does not guarantee better recommendations.')


if __name__ == '__main__':
    main()

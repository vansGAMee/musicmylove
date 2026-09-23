"""Read original ListenBrainz JSONL (optionally gzip), never legacy anonymous sessions.
No API traffic, generated listens, or inferred recording MBIDs.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import unicodedata
from pathlib import Path
import uuid
try:
    from .config import DATA, REPORTS, write_json
except ImportError:
    from config import DATA, REPORTS, write_json


def normalize(s):
    return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())


def identify(row):
    meta = row.get('track_metadata', {})
    artist, title = meta.get('artist_name'), meta.get('track_name')
    if not isinstance(artist, str) or not isinstance(title, str) or not artist.strip() or not title.strip():
        raise ValueError('Missing artist/title')
    info = meta.get('additional_info') or {}
    mbid = info.get('recording_mbid') or (meta.get('mbid_mapping') or {}).get('recording_mbid')
    if mbid:
        mbid = str(uuid.UUID(mbid))  # Invalid explicit IDs are rejected, never guessed.
    artist_ids = info.get('artist_mbids') or []
    identity = '|'.join(sorted(str(uuid.UUID(x)) for x in artist_ids)) if artist_ids else normalize(artist)
    key = identity + '\x1f' + normalize(title)
    return {'id': 'mbid:' + mbid if mbid else 'fallback:' + hashlib.sha256(key.encode()).hexdigest(),
            'artist': artist.strip(), 'title': title.strip(), 'recording_mbid': mbid,
            'fallback_key': key}


def split_user(user):
    bucket = int(hashlib.sha256(('python-mvp-v1:' + user).encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 75 else 'dev' if bucket < 85 else 'shadow' if bucket < 95 else 'final'


def prepare(paths, output=DATA, reports=REPORTS):
    output, reports = Path(output), Path(reports)
    registry, events = {}, defaultdict(set)
    msid_listens = 0
    sources, invalid, parsed, mbid_listens, aliases = [], 0, 0, 0, defaultdict(set)
    for path in sorted({Path(p).resolve() for p in paths}):
        digest, byte_count = hashlib.sha256(), 0
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rb') as stream:
            for raw in stream:
                digest.update(raw)
                byte_count += len(raw)
                try:
                    row = json.loads(raw)
                    msid_listens += bool(row.get('recording_msid') or (row.get('track_metadata', {}).get('additional_info') or {}).get('recording_msid'))
                    track = identify(row)
                    user = row.get('user_name')
                    timestamp = row.get('listened_at', row.get('timestamp'))
                    if not isinstance(user, str) or not user.strip() or not isinstance(timestamp, int) or timestamp <= 0:
                        raise ValueError('Original user_name and timestamp required')
                    datetime.fromtimestamp(timestamp, timezone.utc)
                    uid = hashlib.sha256(user.encode()).hexdigest()
                    aliases[track['id']].add((normalize(track['artist']), normalize(track['title'])))
                    # Canonical metadata selection is independent of source/line order.
                    if track['id'] not in registry or json.dumps(track, sort_keys=True) < json.dumps(registry[track['id']], sort_keys=True):
                        registry[track['id']] = track
                    events[uid].add((timestamp, track['id']))
                    parsed += 1
                    mbid_listens += bool(track['recording_mbid'])
                except (ValueError, TypeError, AttributeError, OverflowError, OSError):
                    invalid += 1
        sources.append({'path': str(path), 'file_bytes': path.stat().st_size,
                        'decoded_bytes_processed': byte_count, 'sha256_decoded': digest.hexdigest()})
    fingerprint = hashlib.sha256(json.dumps([(s['sha256_decoded'], s['decoded_bytes_processed']) for s in sources]).encode()).hexdigest()
    # Identity conflicts are explicit; no silent joining of different metadata.
    # Same authoritative Recording MBID may legitimately have spelling/credit aliases.
    conflicts = {k: sorted(v) for k, v in aliases.items() if len(v) > 1 and not k.startswith('mbid:')}
    for tid, names in aliases.items():
        registry[tid]['aliases'] = [list(name) for name in sorted(names)]
    counts = {s: sum(split_user(u) == s for u in events) for s in ('train', 'dev', 'shadow', 'final')}
    manifest = {'fingerprint': fingerprint, 'rule': 'sha256(python-mvp-v1:hashed-user), 75/10/10/5',
                'users': {u: split_user(u) for u in sorted(events)}}
    manifest_path = output / 'split_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError('Immutable split differs: use a separate experiment directory, never overwrite frozen test')
    ids = sorted(registry)
    index = {t: i for i, t in enumerate(ids)}
    users, all_times = [], []
    for uid, listens in sorted(events.items()):
        sessions, current, previous, start = [], [], None, None
        ordered = sorted(listens)
        for ts, tid in ordered:
            all_times.append(ts)
            # Cap continuous sessions at 4 hours; no enormous day-long context.
            if previous is not None and (ts - previous > 1800 or ts - start > 14400):
                sessions.append(sorted(set(current)))
                current, start = [], None
            if start is None:
                start = ts
            current.append(index[tid])
            previous = ts
        if current:
            sessions.append(sorted(set(current)))
        users.append({'id': uid, 'split': split_user(uid), 'tracks': sorted({index[t] for _, t in ordered}),
                      'sessions': sessions})
    audit = {'sources': sources, 'fingerprint': fingerprint, 'parsed_listens': parsed,
             'unique_listens': sum(map(len, events.values())), 'invalid_rows': invalid,
             'users': len(users), 'tracks': len(ids), 'sessions': sum(len(u['sessions']) for u in users),
             'recording_mbids': sum(bool(t['recording_mbid']) for t in registry.values()),
             'listens_with_recording_mbid': mbid_listens, 'rows_with_msid': msid_listens, 'split_users': counts,
             'period_utc': [datetime.fromtimestamp(t, timezone.utc).isoformat() for t in (min(all_times), max(all_times))] if all_times else [],
             'identity_conflicts': conflicts, 'recording_alias_groups': sum(len(v) > 1 for k, v in aliases.items() if k.startswith('mbid:')), 'duplicate_ids': 0,
             'status': 'IDENTITY_FAIL' if conflicts else 'DATA_BLOCKED' if min(counts.values()) < 10 or len(ids) < 100 else 'AUDITED'}
    write_json(reports / 'data_audit.json', audit)
    if audit['status'] != 'AUDITED':
        raise RuntimeError(audit['status'] + ': inspect reports/data_audit.json; no training data published')
    write_json(manifest_path, manifest)
    write_json(output / 'dataset.json', {'fingerprint': fingerprint, 'tracks': [registry[t] for t in ids], 'users': users})
    print(json.dumps({k: v for k, v in audit.items() if k not in ('sources', 'identity_conflicts')}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', nargs='+', help='Original listens JSONL or .jsonl.gz')
    a = p.parse_args()
    prepare(a.input)


if __name__ == '__main__':
    main()

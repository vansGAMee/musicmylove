"""Inspect cached artifacts themselves; do not trust historical ingest reports."""
from pathlib import Path
import hashlib
import json
try:
    from .config import ROOT, REPORTS, write_json
except ImportError:
    from config import ROOT, REPORTS, write_json


def audit():
    source = ROOT.parent / 'data/cache/pipeline'
    names = ['expanded_catalog.json', 'expanded_sessions.json', 'split_sessions.json']
    files, values = [], {}
    for name in names:
        path = source / name
        if not path.exists():
            continue
        raw = path.read_bytes()
        values[name] = json.loads(raw)
        files.append({'path': str(path), 'bytes_processed': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    catalog = values.get('expanded_catalog.json', {}).get('tracks', [])
    sessions = values.get('expanded_sessions.json', {}).get('sessions', [])
    indexed = {i for s in sessions for i in s}
    invalid = sum(not isinstance(i, int) or i < 0 or i >= len(catalog) for s in sessions for i in s)
    report = {'status': 'DATA_BLOCKED', 'files': files, 'bytes_processed': sum(f['bytes_processed'] for f in files),
              'catalog_rows': len(catalog), 'anonymous_sessions': len(sessions), 'session_item_occurrences': sum(map(len, sessions)),
              'unique_session_track_indices': len(indexed), 'out_of_bounds_indices': invalid,
              'stored_split_sessions': {k: len(v) for k, v in values.get('split_sessions.json', {}).items()},
              'verified_listens': None, 'verified_users': None, 'verified_recording_mbids': None, 'verified_period': None,
              'reasons': ['Sessions contain only catalog indices, no user identifiers or timestamps.',
                          'Old importer coalesces recording_mbid and recording_msid and merges by artist/title.',
                          'Catalog contains preseeded rows: row count is not trained or recommendable track count.',
                          'Raw ListenBrainz JSONL not found in repository data directory.'],
              'required_input': 'Original ListenBrainz JSONL with user_name, timestamp/listened_at, track_metadata and original recording_mbid when present.',
              'resume_command': 'python prepare_data.py data/cache/listens.jsonl'}
    write_json(REPORTS / 'existing_data_audit.json', report)
    # Standard audit path until original data successfully replaces this blocked audit.
    if not (REPORTS / 'data_audit.json').exists():
        write_json(REPORTS / 'data_audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    audit()

"""Stream ListenBrainz listen JSONL into a train-only human-history catalog.
Input: extracted listens JSONL, or '-' for stdin. No API calls or metadata vectors.
Use --max-users/--history-size to bound memory for an MVP sample.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--output', default='data/cache/offline/dump-histories.json')
    parser.add_argument('--max-users', type=int, default=10000)
    parser.add_argument('--history-size', type=int, default=200)
    args = parser.parse_args()
    if args.max_users < 1 or args.history_size < 2:
        parser.error('Positive user limit and history size >= 2 required')
    users, tracks = {}, {}
    digest = hashlib.sha256()
    stream = sys.stdin.buffer if args.input == '-' else open(args.input, 'rb')
    try:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            user = row.get('user_name')
            if not user:
                continue
            uid = hashlib.sha256(user.encode()).hexdigest()
            # Stable USER split; validation/test users never enter graph fitting.
            if int(uid[:8], 16) % 100 >= 80:
                continue
            if uid not in users and len(users) >= args.max_users:
                continue
            meta = row.get('track_metadata', {})
            info = meta.get('additional_info', {})
            mbid = info.get('recording_mbid') or meta.get('mbid_mapping', {}).get('recording_mbid')
            artist, title = meta.get('artist_name'), meta.get('track_name')
            if not all(isinstance(x, str) and x for x in (mbid, artist, title)):
                continue
            history = users.setdefault(uid, set())
            if len(history) >= args.history_size and mbid not in history:
                continue
            history.add(mbid)
            track = {'mbid': mbid, 'artist': artist, 'title': title}
            if mbid not in tracks or json.dumps(track, sort_keys=True) < json.dumps(tracks[mbid], sort_keys=True):
                tracks[mbid] = track
    finally:
        if args.input != '-':
            stream.close()
    ids = sorted(tracks)
    index = {mbid: i for i, mbid in enumerate(ids)}
    if not ids:
        raise ValueError('No mapped listens found; provide extracted ListenBrainz listens JSONL')
    result = {'source': 'listenbrainz-dump-train-only;input-sha256=' + digest.hexdigest(),
              'tracks': [tracks[i] for i in ids],
              'histories': [sorted(index[t] for t in users[u]) for u in sorted(users)],
              'split': 'sha256(user_name)[0:8] mod 100: train <80; validation 80..89; frozen test >=90'}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False))
    print(json.dumps({'tracks': len(ids), 'train_users': len(users), 'output': str(target)}))

if __name__ == '__main__':
    main()

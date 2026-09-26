"""Explain missing profile tracks without guessing recordings or touching weights."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from .config import ROOT, write_json
from .profile_input import resolve_profile, remaster_key
from .recommend import match_inputs


def diagnose(engine, lines):
    lines = sorted(set(lines))
    resolved = resolve_profile(lines, engine.tracks)
    matches = {m['input']: m for m in resolved['matches']}
    catalog = dict(match_inputs(resolved['unresolved'], engine.tracks))
    catalog_names = defaultdict(set)
    for i, track in enumerate(engine.tracks):
        for artist, title in [(track['artist'], track['title']), *track.get('aliases', [])]:
            catalog_names[remaster_key(artist+' - '+title)].add(i)
    for line in resolved['unresolved']:
        if not catalog.get(line):
            catalog[line] = catalog_names.get(remaster_key(line), set())
    wanted = defaultdict(set)
    for line in resolved['unresolved']:
        wanted[remaster_key(line)].add(line)
    raw_hits = defaultdict(set)
    # Only retain matches to requested names; no huge duplicate catalog index.
    for raw, track in enumerate(engine.data['tracks']):
        for artist, title in [(track['artist'], track['title']), *track.get('aliases', [])]:
            for line in wanted.get(remaster_key(artist+' - '+title), ()):
                raw_hits[line].add(raw)
    targets = set().union(*raw_hits.values()) if raw_hits else set()
    representation = set(engine.meta['train_users'])
    support = Counter()
    for user in engine.data['users']:
        if user['id'] in representation:
            support.update(set(user['tracks']) & targets)
    tracks, expansion = [], Counter()
    for line in lines:
        if line in matches:
            row = {'input': line, 'status': 'resolved', 'match': matches[line]}
        elif len(catalog.get(line, ())) > 1:
            row = {'input': line, 'status': 'ambiguous',
                   'candidates': [engine.tracks[i]['id'] for i in sorted(catalog[line])]}
        elif raw_hits[line]:
            ids = sorted(raw_hits[line])
            row = {'input': line, 'status': 'in_raw_not_catalog',
                   'raw_ids': [engine.data['tracks'][i]['id'] for i in ids],
                   'representation_support': max(support[i] for i in ids),
                   'note': 'Maximum per-recording support among actual graph users; aliases are not merged.'}
        else:
            row = {'input': line, 'status': 'absent_from_raw'}
        if row['status'] in ('in_raw_not_catalog', 'absent_from_raw') and ' - ' in line:
            expansion[line.split(' - ', 1)[0]] += 1
        tracks.append(row)
    return {'input_count': len(lines), 'resolved_recordings': len(resolved['seeds']),
            'counts': dict(Counter(r['status'] for r in tracks)), 'tracks': tracks,
            'expansion_targets': [{'artist': a, 'tracks': count} for a, count in sorted(expansion.items(), key=lambda x:(-x[1],x[0]))],
            'metadata_matches_are_not_audio_identity': True, 'heldout_users_used_for_support': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'data/coverage/latest.json')
    args = parser.parse_args()
    from .cli import default_run, open_engine, read_path
    run = args.run or default_run()
    engine, _ = open_engine(run, ROOT/'data/main')
    report = diagnose(engine, read_path(str(args.library)))
    write_json(args.output, report)
    print(report['counts'])
    print(f'Подробности и приоритеты расширения: {args.output}')


if __name__ == '__main__':
    main()

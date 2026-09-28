"""python recommend.py tracks.txt — neural output only, no manual score bonuses."""
import argparse
from collections import Counter, defaultdict
import json
import sys
import torch
try:
    from .config import REPORTS, MODELS, deterministic, write_json, fingerprint
    from .prepare_data import normalize
    from .training import load_components, checkpoint
    from .retrieval import candidate_range, score_rows
except ImportError:
    from config import REPORTS, MODELS, deterministic, write_json, fingerprint
    from prepare_data import normalize
    from training import load_components, checkpoint
    from retrieval import candidate_range, score_rows


def match_key(value):
    import re
    value = normalize(value).translate(str.maketrans({'’': "'", '‘': "'", '‐': '-', '‑': '-', '–': '-', '—': '-'}))
    return re.sub(r'\s+-\s+', ' - ', value)


def match_inputs(lines, tracks):
    by_id, by_name = {}, defaultdict(set)
    for i, track in enumerate(tracks):
        by_id[track['id']] = i
        if track.get('recording_mbid'):
            by_id[track['recording_mbid']] = i
        names = {(track['artist'], track['title'])}
        names.update(tuple(a) for a in track.get('aliases', []))
        for artist, title in names:
            by_name[match_key(artist + ' - ' + title)].add(i)
    for line in sorted({line.strip() for line in lines if line.strip()}):
        yield line, ({by_id[line]} if line in by_id else by_name.get(match_key(line), set()))


def resolve(lines, tracks):
    resolved, unresolved = set(), []
    for line, matches in match_inputs(lines, tracks):
        if len(matches) == 1:
            resolved.update(matches)
        else:
            unresolved.append(line)
    return sorted(resolved), unresolved


def known_track_indices(lines, tracks):
    # Ambiguous names cannot form a taste vector, but all matching recordings
    # must still be excluded from discovery. Never pretend they are unknown.
    return {i for _, matches in match_inputs(lines, tracks) for i in matches}


def recommend(lines, seed=42, artist_cap=None):
    deterministic(seed)
    data, meta, graph, embeddings, taste, ranker = load_components(seed, require_ranker=True)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        embeddings = embeddings.to(device)
        taste = taste.to(device)
        ranker = ranker.to(device)
    lines = list(lines)
    known = known_track_indices(lines, meta['tracks'])
    seeds, unresolved = resolve(lines, meta['tracks'])
    if not seeds:
        raise RuntimeError('UNRESOLVED: no unambiguous input recordings in trained catalog')
    with torch.no_grad():
        heads, masses, assignments = taste(embeddings, seeds)
        rows = candidate_range(seeds, heads, masses, assignments, embeddings, graph)
        scores = score_rows(rows, ranker, embeddings, heads)
    ranked, used, artists = [], set(), Counter()
    # Max neural head score for each candidate. Tie break is stable recording ID.
    for row, score in sorted(zip(rows, scores.tolist()), key=lambda x: (-x[1], meta['tracks'][x[0]['track']]['id'], x[0]['head'])):
        tid = row['track']
        track = meta['tracks'][tid]
        artist = normalize(track['artist'])
        if tid in used or tid in known or (artist_cap is not None and artists[artist] >= artist_cap):
            continue
        used.add(tid); artists[artist] += 1
        ranked.append({'rank': len(ranked) + 1, 'id': track['id'], 'artist': track['artist'],
                       'title': track['title'], 'neural_score': score, 'taste_head': row['head'],
                       'graph_support': {'users': row['support_users'], 'sessions': row['support_sessions'],
                                         'confidence': row['features'][4]},
                       'modality': row.get('modality', 'graph')})
        if len(ranked) == 50:
            break
    return {'fingerprint': data['fingerprint'], 'seed': seed, 'status': 'EXPERIMENTAL',
            'known_excluded_count': len(known),
            'resolved_ids': [meta['tracks'][i]['id'] for i in seeds], 'unresolved_input': unresolved,
            'taste_head_masses': masses.tolist(), 'active_tastes': len(set(assignments.argmax(-1).tolist())),
            'candidate_counts_per_head': dict(Counter(r['head'] for r in rows)),
            'candidate_range': len({r['track'] for r in rows}), 'top50': ranked,
            'shortfall': max(0, 50 - len(ranked)), 'artist_cap': artist_cap,
            'raw_tracks': meta['raw_tracks'], 'train_seen_tracks': len(embeddings),
            'trained_embeddings': len(embeddings), 'ann_indexed_tracks': 0,
            'recommendable_tracks': int((graph['confidence'].max(1).toarray() >= .35).sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('tracks')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--experimental', action='store_true', help='Dev-passed weights only; does not claim release quality')
    p.add_argument('--artist-cap', type=int)
    a = p.parse_args()
    try:
        if a.artist_cap is not None and a.artist_cap < 1:
            p.error('--artist-cap must be positive')
        lines = open(a.tracks, encoding='utf-8').read().splitlines()
        if sum(bool(line.strip()) for line in lines) > 2000:
            p.error('At most 2000 nonempty input lines')
        if not a.experimental:
            release = REPORTS / 'release_final.json'
            if not release.exists():
                raise RuntimeError('BLOCKED: no release evidence; collect data and train/evaluate first. --experimental accepts only dev-passed weights')
            evidence = json.loads(release.read_text())
            if evidence['status'] != 'PASS' or str(a.seed) not in evidence['checkpoints']:
                raise RuntimeError('BLOCKED: release gates failed or seed was not evaluated')
            expected = evidence['checkpoints'][str(a.seed)]
            if any(fingerprint(checkpoint(stage, a.seed)) != expected[stage] for stage in ('graph', 'taste', 'ranker')):
                raise RuntimeError('BLOCKED: checkpoints changed after release evaluation')
        result = recommend(lines, a.seed, a.artist_cap)
        result['status'] = 'EXPERIMENTAL' if a.experimental else 'RELEASE'
        write_json(REPORTS / 'last_recommendation.json', result)
        print(f"resolved: {len(result['resolved_ids'])}\nunresolved: {len(result['unresolved_input'])}\nactive tastes: {result['active_tastes']}\ncandidate range: {result['candidate_range']}")
        print('rank | artist | title | neural_score | taste_head | graph_support')
        for r in result['top50']:
            print(f"{r['rank']} | {r['artist']} | {r['title']} | {r['neural_score']:.6f} | {r['taste_head']} | {r['graph_support']['users']} users")
        if result['shortfall']:
            print(f"INSUFFICIENT_SUPPORTED_CANDIDATES: {result['shortfall']} missing; no padding", file=sys.stderr)
    except (RuntimeError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())

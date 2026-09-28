"""DEV-only fair comparison on identical candidates and playlist constraints. Never trains."""
import argparse
from pathlib import Path
import json
import numpy as np
from .config import ROOT, write_json, fingerprint
from .cli import load_ranker, MODES
from .discovery_engine import DiscoveryEngine, select_playlist
from .evaluate import queries, metrics, paired_interval
from .recommend import match_key


def audit_query(engine, model, query, features, candidates, signals, mode):
    candidates = set(candidates) - set(query['seeds'])
    scores = {'neural': engine.score(model, features), **{k: signals[k] for k in ('listeners', 'ppr', 'graph')}}
    lines = [engine.tracks[i]['artist'] + ' - ' + engine.tracks[i]['title'] for i in query['seeds']]
    familiar = {match_key(engine.tracks[i]['artist']) for i in query['seeds']}
    novel_targets = {i for i in query['targets'] if match_key(engine.tracks[i]['artist']) not in familiar}
    result = {'user': query['user'], 'candidate_recall': len(candidates & query['targets']) / len(query['targets'])}
    _, cap, share = MODES[mode]
    for name, values in scores.items():
        ranked = sorted(candidates, key=lambda i: (-float(values[i]), engine.tracks[i]['id']))
        playlist = select_playlist(engine.tracks, values, candidates, lines, artist_cap=cap, familiar_share=share)
        entry = {'rank_ndcg': metrics(ranked, query['targets'])['NDCG@50'],
                 'playlist_ndcg': metrics(playlist, query['targets'])['NDCG@50'],
                 'playlist_recall': metrics(playlist, query['targets'])['Recall@50'], 'size': len(playlist)}
        if novel_targets:
            entry['playlist_cross_ndcg'] = metrics(playlist, novel_targets)['NDCG@50']
        result[name] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/main')
    parser.add_argument('--run', type=Path, default=ROOT / 'data/discovery-v3')
    parser.add_argument('--mode', choices=MODES, default='2')
    args = parser.parse_args()
    engine = DiscoveryEngine(args.source)
    model = load_ranker(engine, args.run)
    records = []
    for q in queries(engine.data, engine.meta, 'dev'):
        x, ids, signals = engine.features(q['seeds'])
        records.append(audit_query(engine, model, q, x, ids, signals, args.mode))
    if not records:
        raise RuntimeError('No eligible DEV users')
    methods = ('neural', 'listeners', 'ppr', 'graph')
    mean = {name: {key: float(np.mean([r[name][key] for r in records if key in r[name]]))
                   for key in set().union(*(r[name].keys() for r in records))} for name in methods}
    comparisons = {}
    for baseline in methods[1:]:
        comparisons[baseline] = {key: paired_interval(
            [r['neural'][key] for r in records if key in r['neural']],
            [r[baseline][key] for r in records if key in r['neural']])
            for key in ('rank_ndcg', 'playlist_ndcg', 'playlist_cross_ndcg')}
    output = {'status': 'DEV_DIAGNOSTIC_NOT_RELEASE', 'split': 'dev', 'users': len(records),
              'mode': MODES[args.mode][0], 'identical_candidates_and_policy': True,
              'selection_overlap': 'DEV includes model-selection users; not an independent test',
              'checkpoint_sha256': fingerprint(args.run / 'ranker.pt'),
              'audit_implementation_sha256': fingerprint(__file__),
              'candidate_recall': float(np.mean([r['candidate_recall'] for r in records])),
              'metrics': mean, 'neural_minus_baseline': comparisons, 'per_user': records}
    write_json(args.run / f'fair_audit_mode{args.mode}.json', output)
    print(json.dumps({k: v for k, v in output.items() if k != 'per_user'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

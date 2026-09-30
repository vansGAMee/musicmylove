"""One fixed discovery-policy probe on SHADOW users, never frozen FINAL.

This is a diagnostic, not automatic model selection or a publication certificate.
SHADOW was available to earlier project evaluations, so it is not a pristine
project-wide release holdout. Do not tune policy constants on this report.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .joint import JointEngine, select, select_seeds
from .honest_training import episode
from .ranker_data import _dataset_rows
from .evaluate import metrics, paired_interval
from .sound import sha256


def less_familiar(ids, scores, tracks, families, artists, seeds):
    """Predeclared probe: at most one track per familiar artist, two per new one."""
    familiar = set(artists[seeds]); counts = {}; used = set(); out = []; previous = None
    remaining = sorted(ids, key=lambda i: (-float(scores[i]), tracks[i]['id']))
    while remaining and len(out) < 50:
        position = next((p for p, i in enumerate(remaining)
                         if families[i] not in used and artists[i] != previous
                         and counts.get(artists[i], 0) < (1 if artists[i] in familiar else 2)), None)
        if position is None:
            break
        i = remaining.pop(position); artist = artists[i]
        out.append(i); used.add(families[i]); counts[artist] = counts.get(artist, 0) + 1
        previous = artist
    return out


def values(engine, ranked, targets, seeds):
    families = engine.families
    familiar = set(engine.artists[seeds])
    target = set(families[list(targets)])
    novel_target = {families[i] for i in targets if engine.artists[i] not in familiar}
    return {
        'ndcg50': metrics([int(families[i]) for i in ranked], target)['NDCG@50'],
        'recall50': len({families[i] for i in ranked} & target) / max(1, len(target)),
        'new_artist_share': sum(engine.artists[i] not in familiar for i in ranked) / max(1, len(ranked)),
        'new_artist_ndcg50': metrics([int(families[i]) for i in ranked], novel_target)['NDCG@50'] if novel_target else 0.,
        'length': len(ranked),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('python_mvp/data/joint-v1'))
    parser.add_argument('--output', type=Path, default=Path('python_mvp/data/release-audit.json'))
    args = parser.parse_args()
    contract = json.loads((args.run / 'training-contract.json').read_text())
    source = Path(contract['source'])
    dataset = source / 'data/dataset.json'
    if sha256(dataset) != contract['dataset']:
        raise ValueError('Dataset changed since training')
    engine = JointEngine(args.run / 'browser')
    meta = json.loads((source / 'data/graph.json').read_text())
    if [t['id'] for t in meta['tracks']] != [t['id'] for t in engine.tracks]:
        raise ValueError('Catalog differs from source')
    index = dict(zip(meta['seen'], range(len(engine.tracks))))
    representation = set(meta['train_users'])
    users = []; forbidden = set(representation)
    for field, _, row in _dataset_rows(dataset, 65536):
        if field != 'users':
            continue
        if row['split'] in ('train', 'ranker_train', 'dev'):
            forbidden.add(row['id'])
        elif row['split'] == 'shadow':
            users.append((row['id'], {index[t] for t in row['tracks'] if t in index}))
    if forbidden & {u for u, _ in users}:
        raise ValueError('Shadow overlaps training or DEV')
    results = {'regular': [], 'held_out_artist': []}
    retrieval = []; seen_users = set()
    for uid, known in sorted(users):
        rng = np.random.default_rng(int(hashlib.sha256(('release-probe-v1:' + uid).encode()).hexdigest()[:16], 16))
        for discovery, name in [(False, 'regular'), (True, 'held_out_artist')]:
            seeds, targets = episode(known, engine.tracks, engine.families, rng, discovery)
            if not seeds or not targets:
                continue
            features, candidates, _ = engine.features(seeds)
            scores = features[:, 0]
            baseline = select(candidates, scores, engine.tracks, engine.families, engine.artists)
            proposed = less_familiar(candidates, scores, engine.tracks, engine.families, engine.artists, seeds)
            results[name].append((values(engine, baseline, targets, seeds), values(engine, proposed, targets, seeds)))
            tf = set(engine.families[list(targets)])
            retrieval.append(len(set(engine.families[list(candidates)]) & tf) / len(tf))
            seen_users.add(uid)
    summary = {}
    for name, rows in results.items():
        summary[name] = {k: {'baseline': float(np.mean([a[k] for a, _ in rows])),
                             'candidate': float(np.mean([b[k] for _, b in rows])),
                             'delta': paired_interval([b[k] for _, b in rows], [a[k] for a, _ in rows])}
                         for k in ['ndcg50', 'recall50', 'new_artist_share', 'new_artist_ndcg50', 'length']} if rows else {}
    # Strict zero-loss evidence, not "p > .05 means equal". Require all endpoints.
    passes = all(summary.get(s) and all(summary[s][k]['delta']['ci95'][0] >= 0
                 for k in ('ndcg50', 'recall50', 'new_artist_ndcg50', 'length'))
                 and summary[s]['new_artist_share']['delta']['ci95'][0] > 0 for s in results)
    report = dict(policy='familiar-cap-one-v1', users=len(seen_users), split='shadow',
                  model_manifest_sha256=sha256(args.run / 'browser/manifest.json'),
                  code_sha256=sha256(Path(__file__)), final_used=False,
                  shadow_previously_available_to_project=True,
                  candidate_passes_probe=bool(passes), production_policy_changed=False,
                  release_certified=False, candidate_recall=float(np.mean(retrieval)) if retrieval else 0., metrics=summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

"""One offline command: train a small neural ranker if needed, then export discovery TOP50."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .config import ROOT, deterministic, fingerprint, write_json
from .discovery_engine import (DiscoveryEngine, DiscoveryRanker, read_library, training_pairs,
                               select_playlist, input_keys, song_key)
from .evaluate import queries, metrics, paired_interval
from .recommend import resolve, match_key
from .sampling import profile_episode


def signature(engine):
    files = ('discovery.py', 'discovery_engine.py', 'networks.py', 'recommend.py', 'evaluate.py', 'sampling.py')
    return {'representation': engine.provenance,
            'implementation': {f: fingerprint(ROOT / f) for f in files}}


def evaluate_query(engine, query, features, candidates, signals, model):
    neural = engine.score(model, features)
    known_artists = {match_key(engine.tracks[i]['artist']) for i in query['seeds']}
    excluded = {i for i, t in enumerate(engine.tracks) if match_key(t['artist']) in known_artists}
    cross_targets = query['targets'] - excluded
    record = {'user': query['user'], 'candidate_recall': len(candidates & query['targets']) / len(query['targets'])}
    for name, scores in {'neural': neural, 'ppr': signals['ppr'], 'listeners': signals['listeners'], 'graph': signals['graph']}.items():
        allowed = candidates if name == 'neural' else set(range(len(scores))) - set(query['seeds'])
        ranked = sorted(allowed, key=lambda i: (-float(scores[i]), engine.tracks[i]['id']))
        record[name] = metrics(ranked, query['targets'])['NDCG@50']
        record[name + '_recall'] = metrics(ranked, query['targets'])['Recall@50']
        if cross_targets:
            record[name + '_cross'] = metrics([i for i in ranked if i not in excluded], cross_targets)['NDCG@50']
    lines = [engine.tracks[i]['artist'] + ' - ' + engine.tracks[i]['title'] for i in query['seeds']]
    playlist = select_playlist(engine.tracks, neural, candidates, lines)
    record['playlist_ndcg'] = metrics(playlist, query['targets'])['NDCG@50']
    record['playlist_recall'] = metrics(playlist, query['targets'])['Recall@50']
    record['playlist_size'] = len(playlist)
    if cross_targets:
        record['playlist_cross'] = metrics(playlist, cross_targets)['NDCG@50']
    return record


def summary(records):
    keys = sorted(set().union(*(r.keys() for r in records)) - {'user'})
    return {key: float(np.mean([r[key] for r in records if key in r])) for key in keys}


def fit(engine, destination, epochs=30, train_limit=2400, dev_limit=120):
    rng = np.random.default_rng(42)
    destination.mkdir(parents=True, exist_ok=True)
    train = list(queries(engine.data, engine.meta, 'train'))
    chosen = rng.permutation(len(train))[:train_limit]
    positives, negatives = [], []
    start = time.monotonic()
    print(f'Preparing TRAIN examples for {len(chosen)} users; graph and taste stay frozen.', flush=True)
    for counter, index in enumerate(chosen):
        query = train[index]
        for discovery in (False, True):
            seeds, targets = profile_episode(query['known'], engine.tracks, rng, discovery=discovery)
            features, _, signals = engine.features(seeds, user=query['user'])
            p, n = training_pairs(features, targets, query['known'], signals['listeners'], rng)
            if len(p):
                positives.append(p); negatives.append(n)
        if (counter + 1) % 300 == 0:
            print(f'TRAIN features: {counter+1}/{len(chosen)}, {time.monotonic()-start:.0f}s', flush=True)
    if not positives:
        raise RuntimeError('No valid training examples')
    positive = torch.from_numpy(np.concatenate(positives)); negative = torch.from_numpy(np.concatenate(negatives))
    del positives, negatives
    model = DiscoveryRanker(positive.shape[1])
    combined = torch.cat((positive, negative))
    model.mean.copy_(combined.mean(0)); model.scale.copy_(combined.std(0).clamp_min(.05))
    del combined
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.002)
    dev_queries = list(queries(engine.data, engine.meta, 'dev'))[:dev_limit]
    if len(dev_queries) < 10:
        raise RuntimeError('At least 10 DEV users required')
    dev = [(q, *engine.features(q['seeds'])) for q in dev_queries]
    print(f'Training {len(positive)} implicit preference pairs; selecting on {len(dev)} DEV users.', flush=True)
    best, best_state, stale, curves = -float('inf'), None, 0, []
    for epoch in range(epochs):
        model.train(); loss_sum = 0.; updates = 0
        order = torch.randperm(len(positive))
        for ids in order.split(4096):
            # Unknowns are unlabeled, not explicit dislikes; all known positives
            # have already been removed from the comparison pool.
            loss = F.softplus(model(negative[ids]) - model(positive[ids])).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()); updates += 1
        model.eval()
        report = summary([evaluate_query(engine, q, x, ids, signals, model) for q, x, ids, signals in dev])
        objective = report['neural'] + report.get('neural_cross', 0.)
        curve = {'epoch': epoch + 1, 'loss': loss_sum / updates, 'updates': updates, 'dev': report}
        curves.append(curve)
        print(json.dumps(curve), flush=True)
        if objective > best:
            best, stale = objective, 0
            best_state = copy.deepcopy(model.state_dict())
            state = {'weights': best_state, 'epoch': epoch + 1, 'dimensions': positive.shape[1],
                     'signature': signature(engine), 'training_users': len(chosen), 'pairs': len(positive),
                     'status': 'DEV_SELECTED_NOT_RELEASE', 'selection_users': len(dev)}
            temp = destination / 'ranker.tmp'
            torch.save(state, temp); temp.replace(destination / 'ranker.pt')
        else:
            stale += 1
        write_json(destination / 'training.json', curves)
        if stale >= 5:
            break
    model.load_state_dict(best_state); model.eval()
    full_evaluation(engine, destination, model, len(dev))
    return model


def full_evaluation(engine, destination, model, selection_users):
    records = []
    for q in queries(engine.data, engine.meta, 'dev'):
        x, ids, signals = engine.features(q['seeds'])
        records.append(evaluate_query(engine, q, x, ids, signals, model))
    report = {'status': 'DEV_EVALUATED_NOT_RELEASE', 'split': 'dev', 'users': len(records),
              'metrics': summary(records), 'per_user': records,
              'frozen_final_used': False, 'selection_users': selection_users,
              'checkpoint_sha256': fingerprint(destination / 'ranker.pt'),
              'comparisons': {b: paired_interval([r['neural'] for r in records], [r[b] for r in records])
                              for b in ('ppr', 'listeners', 'graph')}}
    outside = records[selection_users:]
    if outside:
        report['outside_selection_subset'] = {'users': len(outside), 'metrics': summary(outside)}
    write_json(destination / 'evaluation.json', report)
    print('FULL DEV: ' + json.dumps(report['metrics']), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library', type=Path)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/main')
    parser.add_argument('--run', type=Path, default=ROOT / 'data/discovery-v2')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--train-users', type=int, default=2400)
    args = parser.parse_args()
    if args.epochs < 1 or args.train_users < 10:
        parser.error('Positive epochs and at least 10 training users required')
    deterministic(42)
    lines = read_library(args.library)
    engine = DiscoveryEngine(args.source)
    checkpoint = args.run / 'ranker.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state.get('signature') != signature(engine):
            raise RuntimeError('Model/code/data changed: choose a new --run, never overwrite an old experiment')
        model = DiscoveryRanker(state['dimensions']); model.load_state_dict(state['weights']); model.eval()
        print('Using trained discovery ranker.', flush=True)
        report_path = args.run / 'evaluation.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if report.get('checkpoint_sha256') != fingerprint(checkpoint):
            print('Completing full DEV evaluation for these exact weights.', flush=True)
            full_evaluation(engine, args.run, model, state['selection_users'])
    else:
        model = fit(engine, args.run, args.epochs, args.train_users)
    seeds, unresolved = resolve(lines, engine.tracks)
    features, candidates, signals = engine.features(seeds)
    scores = engine.score(model, features)
    selected = select_playlist(engine.tracks, scores, candidates, lines)
    known_artists = {a for a, _ in input_keys(lines)}
    rows = [{'rank': rank + 1, **{k: engine.tracks[i][k] for k in ('id', 'artist', 'title')},
             'neural_score': float(scores[i]), 'supporting_listeners': int(signals['support'][i]),
             'artist_not_in_library': match_key(engine.tracks[i]['artist']) not in known_artists,
             'taste_head': int(signals['head'][i])} for rank, i in enumerate(selected)]
    result = {'status': 'EXPERIMENTAL_DEV_EVALUATED', 'library_tracks': len(lines), 'resolved': len(seeds),
              'unresolved': unresolved, 'candidates': len(candidates), 'top50': rows,
              'shortfall': 50 - len(rows), 'new_artist_tracks': sum(r['artist_not_in_library'] for r in rows),
              'distinct_artists': len({match_key(r['artist']) for r in rows}),
              'policy': {'max_per_artist': 2, 'max_familiar_artist_share': .30,
                         'known_tracks_and_versions_excluded': True},
              'source': engine.provenance, 'frozen_final_used': False}
    write_json(args.run / 'playlist.json', result)
    text = '\n'.join(f"{r['rank']:2}. {r['artist']} — {r['title']}" for r in rows) + '\n'
    (args.run / 'playlist.txt').write_text(text, encoding='utf-8')
    print(text)
    print(f"Resolved {len(seeds)}/{len(lines)}; artists absent from library: {result['new_artist_tracks']}/{len(rows)}; distinct artists: {result['distinct_artists']}")
    print(f'Playlist: {(args.run / "playlist.txt").resolve()}')
    if len(rows) < 50:
        print(f'Only {len(rows)} supported candidates satisfy discovery constraints; no padding.')


if __name__ == '__main__':
    main()

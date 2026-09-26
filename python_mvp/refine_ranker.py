"""Refine an existing honest run, reusing its graph and learned representation.

No downloads, graph rebuilding, or representation training. Repeated invocation
loads a completed result, or resumes cached examples after interruption.
"""
import argparse
import copy
import json
from pathlib import Path
import shlex
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .config import ROOT, deterministic, fingerprint, write_json
from .honest_training import (load_run, implementation, mapped_users, raw_protection,
                              episode, evaluation_queries, evaluate_one, aggregate, playlist_order)
from .evaluate import paired_interval, metrics
from .recommend import match_key

SCHEMA = 'listener-residual-v1'


class ResidualRanker(nn.Module):
    """Listener log-score plus a learned, bounded correction. No random initial order."""
    def __init__(self, dimensions=22):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dimensions))
        self.register_buffer('scale', torch.ones(dimensions))
        self.net = nn.Sequential(nn.Linear(dimensions, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def correction(self, x):
        return 2 * torch.tanh(self.net(((x - self.mean) / self.scale).clamp(-12, 12)).squeeze(-1))

    def forward(self, x):
        # Column 2 is log1p(listener_score): monotone, including ties and zero.
        return x[:, 2] + self.correction(x)


def sample_episode(features, targets, known, candidates, families, rng):
    protected = set(families[sorted(known)])
    target_families = set(families[sorted(targets)])
    positive = {}
    for i in sorted(candidates):
        if families[i] in target_families:
            positive.setdefault(int(families[i]), i)
    positives = np.array(list(positive.values()), dtype=int)
    if len(positives) > 12:
        positives = rng.choice(positives, 12, replace=False)
    unknown = [i for i in sorted(candidates) if families[i] not in protected]
    p, n = [], []
    for i in positives:
        # Compare within familiar/new-artist strata. The network must distinguish
        # plausible songs, not merely predict whether an artist occurred in seeds.
        allowed = np.array([j for j in unknown if features[j, 13] == features[i, 13]], dtype=int)
        if not len(allowed):
            continue
        hard = allowed[np.argsort(-features[allowed, 2], kind='stable')[:200]]
        negative = [*rng.choice(hard, 3).tolist(), int(rng.choice(allowed))]
        p.extend([int(i)] * 4); n.extend(negative)
    return np.asarray(p, dtype=int), np.asarray(n, dtype=int)


def acceptable_epoch(report, baseline, best):
    return (report['neural'] >= baseline['neural']
            and report.get('neural_cross', 0.) >= baseline.get('neural_cross', 0.)
            and report['neural'] + report.get('neural_cross', 0.) > best + 1e-8)


def save_torch(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary); temporary.replace(path)


def signature(source):
    return {'schema': SCHEMA, 'implementation': {**implementation(),
              'refine_ranker.py': fingerprint(Path(__file__))},
            'parent_ranker_sha256': fingerprint(source / 'ranker.pt'),
            'parent_representation_sha256': fingerprint(source / 'representation.pt')}


def load_refined(run):
    run = Path(run).resolve()
    manifest = json.loads((run / 'refinement.json').read_text())
    source = Path(manifest['source'])
    if manifest['signature'] != signature(source):
        raise RuntimeError('Refinement code/source changed; select a new run directory')
    engine, _ = load_run(source)  # independently validates parent graph/data/code
    state = torch.load(run / 'ranker.pt', map_location='cpu', weights_only=True)
    if state.get('signature') != manifest['signature']:
        raise RuntimeError('Ranker and refinement provenance differ')
    model = ResidualRanker(); model.load_state_dict(state['weights']); model.eval()
    engine.refinement_status = state['status']
    return engine, model


def prepare_examples(engine, run, repetitions=2):
    cache = run / 'examples'; cache.mkdir(exist_ok=True)
    protected = raw_protection(engine.data, engine.meta)
    users = mapped_users(engine.data, engine.meta, 'ranker_train')
    if {uid for uid, _ in users} & set(engine.user_index):
        raise RuntimeError('Ranker users overlap graph users')
    positives, negatives, weights, stats = [], [], [], []
    for counter, (uid, known) in enumerate(users):
        # Raw dataset IDs are hashes; avoid trusting them as paths in general.
        import hashlib
        key = hashlib.sha256(uid.encode()).hexdigest()
        path = cache / (key + '.pt')
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=True)
        else:
            rng = np.random.default_rng(int(key[:8], 16))
            pos, neg = [], []
            attempted, skipped = 0, 0
            for _ in range(repetitions):
                for discovery in (False, True):
                    seeds, targets = episode(known, engine.tracks, engine.families, rng, discovery)
                    attempted += 1
                    if not targets:
                        skipped += 1; continue
                    x, candidates, _ = engine.features(seeds, user=uid)
                    p, n = sample_episode(x, targets, known | protected[uid], candidates, engine.families, rng)
                    if not len(p):
                        skipped += 1; continue
                    pos.append(x[p]); neg.append(x[n])
            saved = {'user': uid, 'attempted': attempted, 'skipped': skipped,
                     'positive': torch.from_numpy(np.concatenate(pos)) if pos else torch.empty((0, 22)),
                     'negative': torch.from_numpy(np.concatenate(neg)) if neg else torch.empty((0, 22))}
            save_torch(path, saved)
        if saved['user'] != uid:
            raise RuntimeError('Example cache user mismatch')
        count = len(saved['positive'])
        stats.append({'user': uid, 'pairs': count, 'attempted': saved['attempted'], 'skipped': saved['skipped']})
        if count:
            positives.append(saved['positive']); negatives.append(saved['negative'])
            weights.append(torch.full((count,), 1 / count))  # equal total weight per user
        if (counter+1) % 100 == 0:
            print(f'Examples: {counter+1}/{len(users)} users (cached, resumable)', flush=True)
    write_json(run / 'sampling.json', {'per_user': stats, 'users_with_pairs': len(positives),
               'pairs': sum(len(p) for p in positives), 'unknowns_are_not_dislikes': True})
    if not positives:
        raise RuntimeError('No admissible comparisons')
    weight = torch.cat(weights); weight /= weight.mean()
    return torch.cat(positives), torch.cat(negatives), weight


def dev_cache(engine, run, limit=None):
    folder = run / 'dev_examples'; folder.mkdir(exist_ok=True)
    import hashlib
    queries = list(evaluation_queries(engine, engine.families))
    if limit is not None:
        queries = sorted(queries, key=lambda q: hashlib.sha256(('select:'+q['user']).encode()).hexdigest())[:limit]
    for q in queries:
        path = folder / (hashlib.sha256(q['user'].encode()).hexdigest() + '.pt')
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=True)
            x, candidates = saved['features'].numpy(), set(saved['candidates'])
            signals = {k: v.numpy() for k, v in saved['signals'].items()}
        else:
            x, candidates, signals = engine.features(q['seeds'])
            save_torch(path, {'features': torch.from_numpy(x), 'candidates': sorted(candidates),
                             'signals': {k: torch.from_numpy(v) for k, v in signals.items()}})
        yield q, x, candidates, signals


def evaluate_cached(engine, model, dev):
    # Baselines never change across epochs; do not rebuild their playlists each time.
    cache = getattr(engine, '_refinement_baselines', {})
    engine._refinement_baselines = cache
    rows = []
    for q, x, ids, signals in dev:
        key = (q['user'], tuple(q['seeds']), tuple(sorted(q['targets'])))
        if key not in cache:
            cache[key] = evaluate_one(engine, model, q, x, ids, signals, engine.families)
        row = dict(cache[key])
        scores = engine.score(model, x)
        playlist = playlist_order(engine, scores, ids, q['seeds'], engine.families)
        ranked = [int(engine.families[i]) for i in playlist]
        targets = set(engine.families[sorted(q['targets'])])
        values = metrics(ranked, targets)
        row['neural'], row['neural_recall'] = values['NDCG@50'], values['Recall@50']
        seed_artists = {match_key(engine.tracks[i]['artist']) for i in q['seeds']}
        cross = {int(engine.families[i]) for i in q['targets'] if match_key(engine.tracks[i]['artist']) not in seed_artists}
        if cross:
            row['neural_cross'] = metrics(ranked, cross)['NDCG@50']
        rows.append(row)
    return rows


def finish_report(engine, model, run, dev):
    rows = evaluate_cached(engine, model, dev)
    comparisons = {b: paired_interval([r['neural'] for r in rows], [r[b] for r in rows])
                   for b in ('listeners', 'ppr', 'graph')}
    cross = [r for r in rows if 'neural_cross' in r]
    cross_comparisons = {b: paired_interval([r['neural_cross'] for r in cross], [r[b+'_cross'] for r in cross])
                         for b in ('listeners', 'ppr', 'graph')}
    state = torch.load(run / 'ranker.pt', map_location='cpu', weights_only=True)
    report = {'status': state['status'], 'epoch': state['epoch'], 'split': 'dev',
              'frozen_final_used': False, 'selection_data_reused': True,
              'checkpoint_sha256': fingerprint(run / 'ranker.pt'), 'metrics': aggregate(rows),
              'comparisons': comparisons, 'cross_comparisons': cross_comparisons, 'per_user': rows}
    write_json(run / 'evaluation.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'per_user'}), flush=True)


def fit(engine, run, epochs=20):
    positive, negative, weight = prepare_examples(engine, run)
    model = ResidualRanker()
    combined = torch.cat((positive, negative))
    model.mean.copy_(combined.mean(0)); model.scale.copy_(combined.std(0).clamp_min(.05))
    del combined
    selection = list(dev_cache(engine, run, limit=120))
    if len(selection) < 10:
        raise RuntimeError('At least 10 eligible DEV users required')
    # Hash order, not lexicographic prefix; selection remains explicitly DEV.
    baseline = aggregate(evaluate_cached(engine, model, selection))
    best = baseline['neural'] + baseline.get('neural_cross', 0.)
    best_state, best_epoch, start, stale, curves = copy.deepcopy(model.state_dict()), 0, 0, 0, []
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    progress = run / 'progress.pt'
    if progress.exists():
        saved = torch.load(progress, map_location='cpu', weights_only=True)
        model.load_state_dict(saved['weights']); optimizer.load_state_dict(saved['optimizer'])
        best_state, best_epoch, best = saved['best_weights'], saved['best_epoch'], saved['best']
        start, stale, curves = saved['epoch'], saved['stale'], saved['curves']
        torch.set_rng_state(saved['rng'])
    for epoch in range(start, epochs):
        if stale >= 5:
            break
        model.train(); losses = []
        for ids in torch.randperm(len(positive)).split(512):
            p, n = positive[ids], negative[ids]
            preference = F.softplus(model(n) - model(p))
            regularizer = .02 * (model.correction(p).square() + model.correction(n).square())
            loss = ((preference + regularizer) * weight[ids]).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        report = aggregate(evaluate_cached(engine, model, selection))
        curves.append({'epoch': epoch+1, 'loss': float(np.mean(losses)), 'dev': report})
        print(json.dumps(curves[-1]), flush=True)
        if acceptable_epoch(report, baseline, best):
            best = report['neural'] + report.get('neural_cross', 0.)
            best_state, best_epoch, stale = copy.deepcopy(model.state_dict()), epoch+1, 0
        else:
            stale += 1
        save_torch(progress, {'weights': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'best_weights': best_state, 'best_epoch': best_epoch, 'best': best,
                   'epoch': epoch+1, 'stale': stale, 'curves': curves, 'rng': torch.get_rng_state()})
        write_json(run / 'training.json', {'baseline': baseline, 'curves': curves})
    manifest = json.loads((run / 'refinement.json').read_text())
    # Keep the actual learned candidate even when the DEV gate rejects it.
    save_torch(run / 'candidate.pt', {'weights': model.state_dict(), 'status': 'UNPROMOTED_TRAINED_CANDIDATE'})
    status = 'EXPERIMENTAL_RESIDUAL_DEV_SELECTED' if best_epoch else 'LISTENER_BASELINE_NO_ACCEPTED_NEURAL_GAIN'
    model.load_state_dict(best_state); model.eval()
    save_torch(run / 'ranker.pt', {'weights': best_state, 'epoch': best_epoch,
                                 'status': status, 'signature': manifest['signature']})
    del selection
    finish_report(engine, model, run, dev_cache(engine, run))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/honest-v1')
    parser.add_argument('--run', type=Path, default=ROOT / 'data/refined-v1')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    source, run = args.source.resolve(), args.run.resolve()
    deterministic(42)
    engine, _ = load_run(source)
    manifest = {'source': str(source), 'signature': signature(source), 'epochs': args.epochs, 'seed': 42}
    path = run / 'refinement.json'
    if run.exists():
        if not path.exists() or json.loads(path.read_text()) != manifest:
            raise RuntimeError('Run belongs to another configuration; select a new --run (nothing overwritten)')
    print(f'Existing graph verified: {len(engine.tracks)} tracks. No graph training or downloads.', flush=True)
    if args.check_only:
        print('Preflight passed; no training started.'); return
    run.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        write_json(path, manifest)
    if (run / 'ranker.pt').exists():
        engine, model = load_refined(run)
        report_path = run / 'evaluation.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if report.get('checkpoint_sha256') != fingerprint(run / 'ranker.pt'):
            finish_report(engine, model, run, dev_cache(engine, run))
        print('Completed model already exists; no retraining.', flush=True)
    else:
        fit(engine, run, args.epochs)
    print('CLI: python -m python_mvp.cli --run ' + shlex.quote(str(run)) + ' --mode 2', flush=True)


if __name__ == '__main__':
    main()

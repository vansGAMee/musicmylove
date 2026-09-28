"""Offline training with disjoint representation/ranker users; never consumes FINAL.

python -m python_mvp.honest_training --check-only
python -m python_mvp.honest_training
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F
from .config import ROOT, deterministic, fingerprint, write_json, set_run
from .build_graph import build, load_graph, adjacency, supported_core
from .discovery_engine import DiscoveryEngine, DiscoveryRanker, song_key
from .networks import GraphEncoder, MultiInterest
from .recommend import match_key
from .evaluate import metrics, paired_interval

SCHEMA = 'disjoint-users-v1'
FILES = ('honest_training.py', 'build_graph.py', 'config.py', 'networks.py',
         'discovery_engine.py', 'retrieval.py', 'recommend.py', 'evaluate.py')


def partition_users(users):
    ids = [u['id'] for u in users]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate users in dataset')
    representation, ranker = set(), set()
    for u in users:
        if u['split'] == 'train':
            value = int(hashlib.sha256(('representation-v1:' + u['id']).encode()).hexdigest()[:8], 16)
            (representation if value % 10 < 7 else ranker).add(u['id'])
    return representation, ranker


def family_index(tracks):
    """Conservative exclusion groups, not claims of recording equivalence.

    A shared alias links identities transitively. Never merge catalog embeddings.
    """
    parent = list(range(len(tracks)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    keys = {}
    for i, t in enumerate(tracks):
        aliases = [*t.get('aliases', []), (t['artist'], t['title'])]
        for a, title in aliases:
            key = song_key(a, title)
            if key in keys:
                x, y = root(i), root(keys[key])
                parent[max(x, y)] = min(x, y)
            else:
                keys[key] = i
    return np.array([root(i) for i in range(len(tracks))])


def episode(known, tracks, families, rng, discovery=False):
    groups = {}
    for i in sorted(known):
        groups.setdefault(int(families[i]), []).append(i)
    if len(groups) < 3:
        return [], set()
    ordered = sorted(groups)
    if discovery:
        # Hold out complete artists; reject episodes whose aliases span both sides.
        artists = sorted({match_key(tracks[i]['artist']) for i in known})
        choices = []
        for artist in artists:
            held = {f for f, ids in groups.items() if any(match_key(tracks[i]['artist']) == artist for i in ids)}
            if len(groups) - len(held) >= 2:
                choices.append(held)
        if not choices:
            return [], set()
        held = choices[int(rng.integers(len(choices)))]
    else:
        shuffled = rng.permutation(ordered).tolist()
        held = set(shuffled[:max(1, len(shuffled) // 5)])
    seeds = sorted(i for f, ids in groups.items() if f not in held for i in ids)[:2000]
    return seeds, {i for f in held for i in groups[f]}


def pair_indices(targets, known, candidates, families, retrieval, rng):
    protected = set(families[sorted(known)])
    target_families = set(families[sorted(targets)])
    positive = np.array(sorted(i for i in candidates if families[i] in target_families), dtype=int)
    allowed = np.array(sorted(i for i in candidates if families[i] not in protected), dtype=int)
    if not len(positive) or not len(allowed):
        return np.array([], dtype=int), np.array([], dtype=int)
    # One positive per exclusion family prevents repeated releases dominating loss.
    representatives = {}
    for i in positive:
        representatives.setdefault(int(families[i]), int(i))
    positive = np.array(list(representatives.values()))
    positive = rng.choice(positive, min(12, len(positive)), replace=False)
    hard = allowed[np.argsort(-retrieval[allowed], kind='stable')[:500]]
    negative = np.concatenate((rng.choice(hard, 3 * len(positive)), rng.choice(allowed, len(positive))))
    return np.tile(positive, 4), negative


def implementation():
    return {name: fingerprint(ROOT / name) for name in FILES}


def artifact_hashes(run):
    paths = [run / 'data/dataset.json', run / 'data/graph.json', run / 'partition.json',
             *sorted((run / 'data').glob('*.npz'))]
    return {str(p.relative_to(run)): fingerprint(p) for p in paths}


class HonestEngine(DiscoveryEngine):
    def select_playlist(self, scores, candidates, lines, artist_cap=2, familiar_share=.30):
        from .discovery_engine import select_playlist
        from .recommend import known_track_indices
        families = self.families
        known = known_track_indices(lines, self.tracks)
        blocked = set(families[sorted(known)])
        representatives = {}
        for i in sorted(candidates, key=lambda i: (-float(scores[i]), self.tracks[i]['id'])):
            if families[i] not in blocked:
                representatives.setdefault(int(families[i]), i)
        return select_playlist(self.tracks, scores, set(representatives.values()), lines,
                               artist_cap=artist_cap, familiar_share=familiar_share)


def engine_from_parts(run, data, meta, graph, embeddings, taste):
    # Shared exact serving features, without importing unrelated historical weights.
    engine = HonestEngine.__new__(HonestEngine)
    engine.source, engine.data, engine.meta, engine.graph = run, data, meta, graph
    engine.embeddings, engine.taste, engine.tracks = embeddings.detach(), taste.eval(), meta['tracks']
    engine.families = family_index(engine.tracks)
    for p in taste.parameters():
        p.requires_grad_(False)
    engine.ui = graph['user_incidence']
    engine.user_index = {u: i for i, u in enumerate(meta['train_users'])}
    engine.pop = np.asarray(engine.ui.sum(0)).ravel()
    engine.artist_names = sorted({match_key(t['artist']) for t in engine.tracks})
    artist_index = {a: i for i, a in enumerate(engine.artist_names)}
    engine.artist_ids = np.array([artist_index[match_key(t['artist'])] for t in engine.tracks])
    mapping = sparse.csr_matrix((np.ones(len(engine.tracks)), (np.arange(len(engine.tracks)), engine.artist_ids)),
                                shape=(len(engine.tracks), len(artist_index)))
    engine.artist_ui = (engine.ui @ mapping).tocsr(); engine.artist_ui.data[:] = 1
    engine.artist_pop = np.asarray(engine.artist_ui.sum(0)).ravel()
    engine.provenance = {'schema': SCHEMA, 'artifacts': artifact_hashes(run)}
    return engine


def load_run(run):
    run = Path(run).resolve()
    state = torch.load(run / 'ranker.pt', map_location='cpu', weights_only=True)
    if (state.get('schema') != SCHEMA or state.get('implementation') != implementation()
            or state.get('artifacts') != artifact_hashes(run)
            or state.get('representation_sha256') != fingerprint(run / 'representation.pt')):
        raise RuntimeError('Training artifacts/code changed; use a new run, never relabel old weights')
    saved = torch.load(run / 'representation.pt', map_location='cpu', weights_only=True)
    data = json.loads((run / 'data/dataset.json').read_text())
    meta, graph = load_graph(run / 'data')
    if saved['vocabulary'] != [t['id'] for t in meta['tracks']]:
        raise RuntimeError('Representation vocabulary mismatch')
    taste = MultiInterest(); taste.load_state_dict(saved['taste'])
    engine = engine_from_parts(run, data, meta, graph, saved['embeddings'], taste)
    model = DiscoveryRanker(state['dimensions']); model.load_state_dict(state['weights']); model.eval()
    return engine, model


def mapped_users(data, meta, split):
    index = {raw: i for i, raw in enumerate(meta['seen'])}
    return [(u['id'], {index[t] for t in u['tracks'] if t in index})
            for u in sorted(data['users'], key=lambda u: u['id']) if u['split'] == split]


def raw_protection(data, meta):
    """Protect catalog aliases of positives even when their raw ID is out of catalog."""
    keys = {}
    for i, t in enumerate(meta['tracks']):
        for artist, title in [*t.get('aliases', []), (t['artist'], t['title'])]:
            keys.setdefault(song_key(artist, title), set()).add(i)
    raw_tracks = data.get('tracks', meta['tracks'])
    result = {}
    for u in data['users']:
        if u['split'] not in ('train', 'ranker_train'):
            continue
        known = set()
        for raw in u['tracks']:
            t = raw_tracks[raw]
            for artist, title in [*t.get('aliases', []), (t['artist'], t['title'])]:
                known.update(keys.get(song_key(artist, title), ()))
        result[u['id']] = known
    return result


def train_representation(data, meta, graph, epochs, seed):
    """Fixed training budget on representation users only. No validation pass claims."""
    ui, si = graph['user_incidence'], graph['session_incidence']
    ug, sg = adjacency(ui), adjacency(si)
    model = GraphEncoder(ui.shape[1], ui.shape[0], si.shape[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    rng = np.random.default_rng(seed)
    families = family_index(meta['tracks'])
    known = [set(ui.getrow(i).indices) for i in range(ui.shape[0])]
    protected = raw_protection(data, meta)
    pool = [np.flatnonzero(~np.isin(families, families[sorted(k | protected[meta['train_users'][i]])]))
            for i, k in enumerate(known)]
    curves = []
    for epoch in range(epochs):
        losses = []
        for batch in np.array_split(rng.permutation(len(known)), max(1, (len(known)+127)//128)):
            users, positives, negatives = [], [], []
            for u in batch:
                if len(pool[u]):
                    users.extend([int(u)] * 4)
                    positives.extend(rng.choice(sorted(known[u]), 4).tolist())
                    negatives.extend(rng.choice(pool[u], 4).tolist())
            if not users:
                continue
            embeddings, user_vectors = model(ug, sg)
            pos = (embeddings[positives] * user_vectors[users]).sum(-1)
            neg = (embeddings[negatives] * user_vectors[users]).sum(-1)
            loss = F.softplus((neg - pos) / .1).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        if not losses:
            raise RuntimeError('No valid graph training updates')
        row = {'stage': 'graph', 'epoch': epoch+1, 'loss': float(np.mean(losses)), 'updates': len(losses)}
        curves.append(row); print(json.dumps(row), flush=True)
    with torch.no_grad():
        embeddings, _ = model(ug, sg)
    embeddings = embeddings.detach()
    taste = MultiInterest()
    optimizer = torch.optim.AdamW(taste.parameters(), lr=.0005, weight_decay=.001)
    for epoch in range(epochs):
        losses = []
        for u in rng.permutation(len(known))[:1200]:
            seeds, targets = episode(known[u], meta['tracks'], families, rng, discovery=bool(u % 2))
            if not targets or not len(pool[u]):
                continue
            heads, masses, assignments = taste(embeddings, seeds)
            positive = int(rng.choice(sorted(targets)))
            negative = rng.choice(pool[u], 16).tolist()
            logits = torch.logsumexp(embeddings[[positive, *negative]] @ heads.T / .1
                                     + masses.clamp_min(1e-8).log(), -1)
            loss = F.softplus(logits[1:] - logits[0]).mean() + taste.regularization(heads, masses, assignments)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        if not losses:
            raise RuntimeError('No valid taste training updates')
        row = {'stage': 'taste', 'epoch': epoch+1, 'loss': float(np.mean(losses)), 'updates': len(losses)}
        curves.append(row); print(json.dumps(row), flush=True)
    return embeddings, taste.eval(), curves


def evaluation_queries(engine, families):
    for uid, known in mapped_users(engine.data, engine.meta, 'dev'):
        rng = np.random.default_rng(int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16))
        seeds, targets = episode(known, engine.tracks, families, rng)
        if targets:
            yield {'user': uid, 'seeds': seeds, 'targets': targets}


def playlist_order(engine, scores, candidates, seeds, families):
    """Same defaults as CLI discovery: 50 tracks, <=2/artist, <=30% familiar."""
    lines = [engine.tracks[i]['artist'] + ' - ' + engine.tracks[i]['title'] for i in seeds]
    return engine.select_playlist(scores, candidates, lines)


def evaluate_one(engine, model, query, features, candidates, signals, families):
    targets = set(families[sorted(query['targets'])])
    seed_artists = {match_key(engine.tracks[i]['artist']) for i in query['seeds']}
    cross = {int(families[i]) for i in query['targets'] if match_key(engine.tracks[i]['artist']) not in seed_artists}
    row = {'user': query['user'], 'candidate_recall': len(set(families[sorted(candidates)]) & targets) / len(targets)}
    for name, scores in {'neural': engine.score(model, features), **{k: signals[k] for k in ('listeners', 'ppr', 'graph')}}.items():
        playlist = playlist_order(engine, scores, candidates, query['seeds'], families)
        ranked = [int(families[i]) for i in playlist]
        row[name] = metrics(ranked, targets)['NDCG@50']
        row[name + '_recall'] = metrics(ranked, targets)['Recall@50']
        if cross:
            row[name + '_cross'] = metrics(ranked, cross)['NDCG@50']
    return row


def aggregate(rows):
    keys = set().union(*(r.keys() for r in rows)) - {'user'}
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in sorted(keys)}


def train_ranker(engine, run, epochs, train_limit, seed):
    rng = np.random.default_rng(seed)
    families = family_index(engine.tracks)
    users = mapped_users(engine.data, engine.meta, 'ranker_train')
    protected = raw_protection(engine.data, engine.meta)
    if set(u for u, _ in users) & set(engine.user_index):
        raise RuntimeError('Ranker labels entered representation graph')
    chosen = rng.permutation(len(users))[:train_limit]
    positives, negatives = [], []
    stats = {'users': len(chosen), 'episodes': 0, 'skipped': 0, 'target_families': 0, 'retrieved_target_families': 0}
    for counter, index in enumerate(chosen):
        uid, known = users[index]
        for discovery in (False, True):
            seeds, targets = episode(known, engine.tracks, families, rng, discovery)
            if not targets:
                stats['skipped'] += 1; continue
            x, candidates, signals = engine.features(seeds, user=uid)
            stats['episodes'] += 1
            stats['target_families'] += len(set(families[sorted(targets)]))
            stats['retrieved_target_families'] += len(set(families[sorted(targets & candidates)]))
            p, n = pair_indices(targets, known | protected[uid], candidates, families, signals['listeners'], rng)
            if not len(p):
                stats['skipped'] += 1; continue
            positives.append(x[p]); negatives.append(x[n])
        if (counter+1) % 100 == 0:
            print(f'Ranker examples: {counter+1}/{len(chosen)} users', flush=True)
    write_json(run / 'sampling.json', stats)
    if not positives:
        raise RuntimeError('No retrievable ranker positives; inspect sampling.json and graph coverage')
    positive = torch.from_numpy(np.concatenate(positives)); negative = torch.from_numpy(np.concatenate(negatives))
    model = DiscoveryRanker(positive.shape[1])
    combined = torch.cat((positive, negative))
    model.mean.copy_(combined.mean(0)); model.scale.copy_(combined.std(0).clamp_min(.05))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    dev_queries = list(evaluation_queries(engine, families))
    if len(dev_queries) < 10:
        raise RuntimeError('Fewer than 10 eligible DEV users')
    # DEV is selection data, including users outside this computational subset.
    dev = [(q, *engine.features(q['seeds'])) for q in dev_queries[:120]]
    best, best_state, best_epoch, stale, curves = -float('inf'), None, 0, 0, []
    for epoch in range(epochs):
        model.train(); losses = []
        for ids in torch.randperm(len(positive)).split(2048):
            loss = F.softplus(model(negative[ids]) - model(positive[ids])).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        rows = [evaluate_one(engine, model, q, x, ids, signals, families) for q, x, ids, signals in dev]
        report = aggregate(rows)
        objective = report['neural'] + report.get('neural_cross', 0.)
        curve = {'epoch': epoch+1, 'loss': float(np.mean(losses)), 'dev': report}
        curves.append(curve); print(json.dumps(curve), flush=True)
        if objective > best:
            best, best_epoch, best_state, stale = objective, epoch+1, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        write_json(run / 'training.json', curves)
        if stale >= 4:
            break
    model.load_state_dict(best_state); model.eval()
    state = {'schema': SCHEMA, 'weights': best_state, 'dimensions': positive.shape[1], 'epoch': best_epoch,
             'status': 'EXPERIMENTAL_DEV_SELECTED', 'implementation': implementation(),
             'artifacts': artifact_hashes(run), 'representation_sha256': fingerprint(run / 'representation.pt')}
    torch.save(state, run / 'ranker.tmp'); (run / 'ranker.tmp').replace(run / 'ranker.pt')
    rows = [evaluate_one(engine, model, q, *engine.features(q['seeds']), families) for q in dev_queries]
    comparisons = {b: paired_interval([r['neural'] for r in rows], [r[b] for r in rows]) for b in ('listeners', 'ppr', 'graph')}
    cross_rows = [r for r in rows if 'neural_cross' in r]
    cross_comparisons = {b: paired_interval([r['neural_cross'] for r in cross_rows], [r[b+'_cross'] for r in cross_rows])
                         for b in ('listeners', 'ppr', 'graph')}
    better = all(c['ci95'][0] > 0 for c in [*comparisons.values(), *cross_comparisons.values()])
    report = {'status': 'DEV_GAIN_OBSERVED_NOT_RELEASE' if better else 'NO_PROVEN_DEV_GAIN',
              'split': 'dev', 'frozen_final_used': False, 'selection_data_reused': True,
              'checkpoint_sha256': fingerprint(run / 'ranker.pt'), 'metrics': aggregate(rows),
              'comparisons': comparisons, 'cross_comparisons': cross_comparisons, 'per_user': rows}
    write_json(run / 'evaluation.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'per_user'}), flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/main')
    parser.add_argument('--run', type=Path, default=ROOT / 'data/honest-v1')
    parser.add_argument('--representation-epochs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--train-users', type=int, default=2400)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check-only', action='store_true', help='Read-only preflight, no graph building or training')
    args = parser.parse_args()
    if min(args.representation_epochs, args.epochs, args.train_users) < 1:
        parser.error('Epochs and train-users must be positive')
    source, run = args.source.resolve(), args.run.resolve()
    data = json.loads((source / 'data/dataset.json').read_text())
    representation, ranker = partition_users(data['users'])
    eligible = [u for u in data['users'] if u['id'] in representation and 5 <= len(u['tracks']) <= 2000]
    core, seen = supported_core(eligible)
    index = set(seen)
    ranker_eligible = sum(u['id'] in ranker and len(set(u['tracks']) & index) >= 5 for u in data['users'])
    dev_eligible = sum(u['split'] == 'dev' and len(set(u['tracks']) & index) >= 5 for u in data['users'])
    preflight = {'representation_users': len(core), 'catalog_tracks': len(seen),
                 'eligible_ranker_users': ranker_eligible, 'eligible_dev_users': dev_eligible,
                 'ranker_graph_user_overlap': 0, 'frozen_final_used': False}
    print(json.dumps(preflight), flush=True)
    if len(seen) < 100 or ranker_eligible < 10 or dev_eligible < 10:
        raise RuntimeError('Insufficient disjoint users/catalog for training')
    if args.check_only:
        return
    if run.exists():
        raise RuntimeError('Run already exists. Preserve old experiments: choose a new --run directory')
    deterministic(args.seed)
    run.mkdir(parents=True)
    write_json(run / 'partition.json', {'schema': SCHEMA, 'representation_users': sorted(representation),
               'ranker_users': sorted(ranker), 'source_dataset_sha256': fingerprint(source / 'data/dataset.json'),
               'original_splits_preserved': True, 'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
    for u in data['users']:
        if u['id'] in ranker:
            u['split'] = 'ranker_train'
    write_json(run / 'data/dataset.json', data)
    # Retain original split provenance and consumption marker; never run release_check.
    for relative in ('data/split_manifest.json', 'reports/final_consumed.lock'):
        path = source / relative
        if path.exists():
            (run / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, run / relative)
    set_run(run); build()
    meta, graph = load_graph(run / 'data')
    embeddings, taste, curves = train_representation(data, meta, graph, args.representation_epochs, args.seed)
    torch.save({'embeddings': embeddings, 'taste': taste.state_dict(), 'vocabulary': [t['id'] for t in meta['tracks']],
                'status': 'TRAINED_NOT_RELEASE', 'implementation': implementation()}, run / 'representation.pt')
    write_json(run / 'representation_training.json', curves)
    engine = engine_from_parts(run, data, meta, graph, embeddings, taste)
    train_ranker(engine, run, args.epochs, args.train_users, args.seed)
    print(f'Inference only: python -m python_mvp.cli --run {run} --mode 2', flush=True)


if __name__ == '__main__':
    main()

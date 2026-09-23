"""Exact seen-catalog evaluation. Frozen final split can be consumed only once."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
try:
    from .config import DATA, MODELS, REPORTS, load_data, write_json, deterministic
    from .build_graph import load_graph
    from .retrieval import diffusion, candidate_range, score_rows
except ImportError:
    from config import DATA, MODELS, REPORTS, load_data, write_json, deterministic
    from build_graph import load_graph
    from retrieval import diffusion, candidate_range, score_rows


def shuffle_edges(edges, seed):
    rng = np.random.default_rng(seed)
    result, present = list(edges), set(edges)
    for _ in range(20 * len(result)):
        a, b = rng.integers(len(result), size=2)
        u, i = result[a]
        v, j = result[b]
        if u == v or i == j or (u, j) in present or (v, i) in present:
            continue
        present.remove((u, i)); present.remove((v, j))
        present.add((u, j)); present.add((v, i))
        result[a], result[b] = (u, j), (v, i)
    return result


def queries(data, meta, split, profile_size=None):
    index = {raw: i for i, raw in enumerate(meta['seen'])}
    for u in data['users']:
        if u['split'] != split:
            continue
        items = sorted(index[t] for t in u['tracks'] if t in index)
        if len(items) < 5:
            continue
        rng = np.random.default_rng(int(hashlib.sha256(u['id'].encode()).hexdigest()[:8], 16))
        items = rng.permutation(items).tolist()
        cut = min(2000, max(2, int(.8 * len(items))))
        if profile_size is not None:
            if len(items) <= profile_size:
                continue
            cut = profile_size
        yield {'user': u['id'], 'seeds': sorted(items[:cut]), 'targets': set(items[cut:]),
               'known': set(items), 'cold': sum(t not in index for t in u['tracks'])}


def metrics(ranked, targets):
    target = set(targets)
    hits = np.array([int(t in target) for t in ranked], dtype=float)
    out = {}
    for k in (10, 50, 100):
        out[f'Recall@{k}'] = float(hits[:k].sum() / max(1, len(target)))
        if k != 100:
            discount = 1 / np.log2(np.arange(min(k, len(hits))) + 2)
            ideal = (1 / np.log2(np.arange(min(k, len(target))) + 2)).sum()
            out[f'NDCG@{k}'] = float((hits[:k] * discount).sum() / ideal) if ideal else 0.
    positions = np.flatnonzero(hits)
    out['MRR'] = float(1 / (positions[0] + 1)) if len(positions) else 0.
    return out


def paired_interval(a, b, seed=42):
    delta = np.asarray(a) - np.asarray(b)
    if len(delta) < 2:
        return {'mean': float(delta.mean()) if len(delta) else 0., 'ci95': [-1., 1.], 'users': len(delta)}
    rng = np.random.default_rng(seed)
    means = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(1000)]
    return {'mean': float(delta.mean()), 'ci95': np.quantile(means, [.025, .975]).tolist(), 'users': len(delta)}


def evaluate(data, meta, graph, split='dev', embeddings=None, taste=None, ranker=None, profile_size=None, cross_artist=False):
    n = len(meta['seen'])
    popularity = np.asarray(graph['user_incidence'].sum(0)).ravel()
    records, cold = [], 0
    for query in queries(data, meta, split, profile_size):
        seeds, target = query['seeds'], query['targets']
        excluded = set(seeds)
        if cross_artist:
            artists = {meta['tracks'][s]['artist'].casefold() for s in seeds}
            excluded |= {i for i, t in enumerate(meta['tracks']) if t['artist'].casefold() in artists}
            target = target - excluded
            if not target:
                continue
        scores = {'popularity': popularity.copy(),
                  'graph': np.asarray(graph['global'][seeds].multiply(graph['confidence'][seeds]).sum(0)).ravel(),
                  'ppr': diffusion(seeds, graph)}
        candidate_ids = []
        with torch.no_grad():
            if embeddings is not None:
                scores['neural'] = (embeddings @ torch.nn.functional.normalize(embeddings[seeds].mean(0), dim=0)).numpy()
            if taste is not None:
                heads, masses, assignment = taste(embeddings, seeds)
                active = sorted(set(assignment.argmax(-1).tolist()))
                scores['taste'] = (embeddings @ heads[active].T).max(-1).values.numpy()
                rows = candidate_range(seeds, heads, masses, assignment, embeddings, graph)
                candidate_ids = list(dict.fromkeys(r['track'] for r in rows if r['track'] not in excluded))
                if ranker is not None:
                    values = score_rows(rows, ranker, embeddings, heads).numpy()
                    final = np.full(n, -np.inf)
                    for row, value in zip(rows, values):
                        final[row['track']] = max(final[row['track']], float(value))
                    scores['final'] = final
        result = {'user': query['user']}
        for method, values in scores.items():
            values = values.copy()
            values[sorted(excluded)] = -np.inf
            ranked = np.lexsort((np.arange(n), -values))
            ranked = [int(t) for t in ranked if np.isfinite(values[t])]
            result[method] = metrics(ranked, target)
        if taste is not None:
            result['candidate'] = {f'CandidateRecall@{k}': len(set(candidate_ids[:k]) & target) / len(target) for k in (100, 500, 1000, 2000)}
        records.append(result)
        cold += query['cold']
    means = {}
    if records:
        for method in records[0]:
            if method != 'user':
                means[method] = {k: float(np.mean([r[method][k] for r in records])) for k in records[0][method]}
    return {'split': split, 'users': len(records), 'cold_target_tracks': cold, 'full_seen_catalog': n,
            'profile_size': profile_size, 'cross_artist': cross_artist, 'metrics': means, 'per_user': records}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['dev', 'shadow'], default='dev')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    deterministic(args.seed)
    try:
        from .training import load_components
    except ImportError:
        from training import load_components
    data, meta, graph, embeddings, taste, ranker = load_components(args.seed, require_ranker=True)
    destination = REPORTS / f'evaluation_{args.split}_{args.seed}.json'
    report = evaluate(data, meta, graph, args.split, embeddings, taste, ranker)
    report['cross_artist'] = evaluate(data, meta, graph, args.split, embeddings, taste, ranker, cross_artist=True)
    report['range_by_profile_size'] = [evaluate(data, meta, graph, args.split, embeddings, taste, profile_size=k)
                                       for k in (5, 20, 50, 200, 500)]
    write_json(destination, report)
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__':
    main()

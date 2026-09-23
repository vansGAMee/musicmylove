"""Same per-interest, human-supported candidate range for training and serving."""
import numpy as np
import torch
from scipy import sparse
try:
    from .config import MIN_USERS, MIN_CONFIDENCE
except ImportError:
    from config import MIN_USERS, MIN_CONFIDENCE


def diffusion(seeds, graph, iterations=12, restart=.35):
    n = graph['global'].shape[0]
    if '_transition' not in graph:
        affinity = (graph['global'] + graph['local']).multiply(graph['confidence'])
        sums = np.asarray(affinity.sum(1)).ravel()
        graph['_transition'] = (sparse.diags(1 / np.maximum(sums, 1e-12)) @ affinity, sums)
    transition, sums = graph['_transition']
    start = np.zeros(n)
    start[sorted(set(seeds))] = 1 / len(set(seeds))
    value = start.copy()
    for _ in range(iterations):
        dangling = value[sums == 0].sum()
        value = restart * start + (1 - restart) * (transition.T @ value + dangling * start)
    return value


def candidate_range(seeds, heads, masses, assignments, embeddings, graph, radius=None):
    settings = graph.get('_range', {'confidence': MIN_CONFIDENCE, 'min_users': MIN_USERS, 'radius': 0.})
    radius = settings['radius'] if radius is None else radius
    seeds = sorted(set(int(i) for i in seeds))
    if assignments.shape[0] != len(seeds):
        raise ValueError('Assignment rows must match sorted unique seeds')
    # Hard memberships are used only to activate supported taste ranges.
    owner = assignments.detach().argmax(-1).cpu().numpy()
    rows = []
    for h in range(len(heads)):
        members = [s for s, a in zip(seeds, owner) if a == h]
        if not members or float(masses[h].detach()) < .01:
            continue
        # Direct independent-human gate; two-hop paths cannot invent confidence.
        evidence = {k: np.asarray(graph[k][members].max(axis=0).toarray()).ravel()
                    for k in ('global', 'local', 'users', 'sessions', 'confidence')}
        eligible = (evidence['users'] >= settings['min_users']) & (evidence['confidence'] >= settings['confidence'])
        eligible[seeds] = False
        candidates = np.flatnonzero(eligible)
        if not len(candidates):
            continue
        proximity = (embeddings[candidates] @ heads[h]).detach().cpu().numpy()
        walk = diffusion(members, graph)
        for c, sim in zip(candidates, proximity):
            if sim < radius or walk[c] <= 0:
                continue
            rows.append({'track': int(c), 'head': h,
                         'retrieval_score': float(sim + walk[c]),
                         'features': [float(evidence['global'][c]), float(evidence['local'][c]),
                                      float(np.log1p(evidence['users'][c])), float(np.log1p(evidence['sessions'][c])),
                                      float(evidence['confidence'][c]), float(masses[h].detach())],
                         'support_users': int(evidence['users'][c]), 'support_sessions': int(evidence['sessions'][c])})
    return sorted(rows, key=lambda r: (-r['retrieval_score'], r['track'], r['head']))


def score_rows(rows, ranker, embeddings, heads):
    if not rows:
        return torch.empty(0)
    candidates = torch.tensor([r['track'] for r in rows], dtype=torch.long)
    head_ids = torch.tensor([r['head'] for r in rows], dtype=torch.long)
    features = torch.tensor([r['features'] for r in rows], dtype=torch.float32)
    return ranker(embeddings[candidates], heads[head_ids], features)

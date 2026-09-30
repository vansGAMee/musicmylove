"""Behavioral checks on real dev libraries; no genre-derived targets."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
try:
    from .evaluate import queries
except ImportError:
    from evaluate import queries


def check_tastes(data, meta, embeddings, model):
    rng = np.random.default_rng(42)
    results, profiles = [], []
    with torch.no_grad():
        for q in queries(data, meta, 'dev'):
            seeds = q['seeds']
            heads, mass, assignment = model(embeddings, seeds)
            shuffled = model(embeddings, rng.permutation(seeds).tolist() + seeds[:1] * 10)
            invariant = torch.equal(heads, shuffled[0]) and torch.equal(mass, shuffled[1])
            active = sorted(set(assignment.argmax(-1).tolist()))
            distinct = True
            if len(active) > 1:
                gram = heads[active] @ heads[active].T
                gram.fill_diagonal_(-1)
                distinct = float(gram.max()) < .95
            stability = None
            if len(seeds) >= 50:
                subset = sorted(rng.choice(seeds, int(.8 * len(seeds)), replace=False).tolist())
                other, _, assign = model(embeddings, subset)
                other_active = sorted(set(assign.argmax(-1).tolist()))
                sim = (heads[active] @ other[other_active].T).numpy()
                left, right = linear_sum_assignment(-sim)
                stability = float(sim[left, right].sum() / max(len(active), len(other_active)))
            results.append({'user': q['user'], 'permutation_duplicates_equal': invariant,
                            'active_heads': len(active), 'distinct_heads': distinct, 'stability_cosine': stability})
            profiles.append(seeds)
        mixed = []
        # Combine disjoint real user libraries; don't prescribe artist/genre assignments.
        for a, b in zip(profiles[::2], profiles[1::2]):
            if len(set(a) & set(b)) > .1 * min(len(a), len(b)):
                continue
            h, m, assign = model(embeddings, sorted(set(a + b)))
            active = sorted(set(assign.argmax(-1).tolist()))
            gram = h[active] @ h[active].T
            gram.fill_diagonal_(-1)
            mixed.append({'heads': len(active), 'distinct': len(active) >= 2 and float(gram.max()) < .95})
    stability = [r['stability_cosine'] for r in results if r['stability_cosine'] is not None]
    passed = (len(results) >= 10 and all(r['permutation_duplicates_equal'] and r['distinct_heads'] for r in results)
              and len(mixed) >= 5 and np.mean([m['distinct'] for m in mixed]) >= .8
              and len(stability) >= 10 and np.mean(stability) >= .8)
    return {'passed': bool(passed), 'profiles': results, 'mixed_profiles': mixed,
            'thresholds': {'minimum_profiles': 10, 'mixed_distinct_fraction': .8, 'stability_cosine': .8}}

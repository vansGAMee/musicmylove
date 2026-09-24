"""Unknown-negative sampling; all known positives always excluded."""
import numpy as np

def sample_negatives(rng, n, known, supported, popularity, positive, similarities, count=32):
    allowed = np.array(sorted(set(range(n)) - set(known) - set(supported)), dtype=int)
    if not len(allowed):
        return allowed
    # Four pools, sampled without replacement overall; never known user positives.
    pools = [allowed, allowed[np.argsort(np.abs(np.log1p(popularity[allowed]) - np.log1p(popularity[positive])))[:max(1, len(allowed)//4)]],
             allowed[np.argsort(similarities[allowed])[len(allowed)//3: max(len(allowed)//3+1, 2*len(allowed)//3)]],
             allowed[np.argsort(-similarities[allowed])[:max(1, len(allowed)//10)]]]
    selected = set()
    for pool in pools:
        pool = np.array(sorted(set(pool.tolist()) - selected), dtype=int)
        if len(pool):
            selected.update(rng.choice(pool, min(max(1, count//4), len(pool)), replace=False).tolist())
    return np.array(sorted(selected), dtype=int)


def protect(seeds, graph):
    # Exclude direct evidence and strong two-hop evidence, not just positive edges.
    supported = graph['confidence'].copy()
    supported.data = (supported.data >= .1).astype(float)
    supported.eliminate_zeros()
    direct = set(supported[seeds].nonzero()[1].tolist())
    strong = graph['confidence'].copy()
    strong.data = (strong.data >= .35).astype(float)
    strong.eliminate_zeros()
    first = strong[seeds].nonzero()[1]
    twohop = set(strong[first].nonzero()[1].tolist()) if len(first) else set()
    return direct | twohop



def profile_episode(known, tracks, rng, discovery=False):
    """Mix library completion with whole-artist holdout, using TRAIN users only."""
    items = sorted(known)
    if discovery:
        groups = {}
        for i in items:
            groups.setdefault(tracks[i]['artist'].casefold(), []).append(i)
        eligible = [a for a in sorted(groups) if len(items) - len(groups[a]) >= 2]
        if eligible:
            artist = eligible[int(rng.integers(len(eligible)))]
            targets = set(groups[artist])
            return sorted(set(items) - targets)[:2000], targets
    items = rng.permutation(items).tolist()
    cut = min(2000, max(2, int(.8 * len(items))))
    return sorted(items[:cut]), set(items[cut:])

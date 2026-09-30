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


def candidate_range(seeds, heads, masses, assignments, embeddings, graph, radius=None,
                    audio_embeddings=None, audio_mask=None, audio_radius=None):
    if audio_embeddings is not None:
        raise ValueError('Audio space is not aligned with graph space; use graph retrieval')
    settings = graph.get('_range', {'confidence': MIN_CONFIDENCE, 'min_users': MIN_USERS, 'radius': 0.})
    radius = settings['radius'] if radius is None else radius
    seeds = sorted(set(int(i) for i in seeds))
    if not seeds or assignments.shape[0] != len(seeds):
        raise ValueError('Assignment rows must match nonempty sorted unique seeds')
    owner = assignments.detach().argmax(-1).cpu().numpy()
    active = sorted(set(owner.tolist()))
    emb = embeddings.detach().cpu()
    sims = (emb @ heads.detach().cpu().T).numpy()
    walk = diffusion(seeds, graph)
    g = np.asarray(graph['global'][seeds].multiply(graph['confidence'][seeds]).sum(0)).ravel()
    local = np.asarray(graph['local'][seeds].sum(0)).ravel()
    allowed = (sims[:, active].max(1) >= radius) & (walk > 0)
    allowed[seeds] = False
    ids = np.flatnonzero(allowed)
    primary = sorted(ids.tolist(), key=lambda c: (-walk[c], c))[:1800]
    evidence, queues = {}, []
    for h in active:
        members = [s for s, a in zip(seeds, owner) if a == h]
        ev = {k: np.asarray(graph[k][members].max(axis=0).toarray()).ravel()
              for k in ('users', 'sessions', 'confidence')}
        evidence[h] = ev
        eligible = (ev['users'] >= settings['min_users']) & (ev['confidence'] >= settings['confidence']) & (sims[:, h] >= radius)
        eligible[seeds] = False
        queues.append(sorted(np.flatnonzero(eligible).tolist(), key=lambda c: (-sims[c, h], c)))
    # Reserve direct per-interest candidates before filling with diffusion.
    # Round-robin retrieval is not a final artist quota or musical score.
    selected = set()
    for position in range(max((len(q) for q in queues), default=0)):
        for q in queues:
            if position < len(q):
                selected.add(q[position])
            if len(selected) >= 200:
                break
        if len(selected) >= 200:
            break
    selected.update(primary)
    rows = []
    for c in sorted(selected, key=lambda c: (-walk[c], c)):
        for h in active:
            # Radius selects the candidate union, not its best head in advance.
            ev = evidence[h]
            rows.append({'track': c, 'head': h, 'retrieval_score': float(walk[c]),
                         'features': [float(g[c]), float(local[c]), float(np.log1p(ev['users'][c])),
                                      float(walk[c]), float(ev['confidence'][c]), float(masses[h].detach())],
                         'support_users': int(ev['users'][c]), 'support_sessions': int(ev['sessions'][c]),
                         'modality': 'graph'})
    return rows


def score_rows(rows, ranker, embeddings, heads, audio_embeddings=None, audio_mask=None, audio_heads=None):
    if not rows:
        return torch.empty(0)
    device = next(ranker.parameters()).device
    candidates = torch.tensor([r['track'] for r in rows], dtype=torch.long, device=device)
    head_ids = torch.tensor([r['head'] for r in rows], dtype=torch.long, device=device)
    features = torch.tensor([r['features'] for r in rows], dtype=torch.float32, device=device)
    cand_graph = embeddings[candidates].to(device)
    head_graph = heads[head_ids].to(device)
    try:
        from .networks import MultimodalRanker
    except ImportError:
        from networks import MultimodalRanker
    if isinstance(ranker, MultimodalRanker) and audio_embeddings is not None and audio_mask is not None:
        cand_audio = audio_embeddings[candidates].to(device)
        cand_mask = audio_mask[candidates].unsqueeze(-1).float().to(device)
        head_aud = audio_heads[head_ids].to(device) if audio_heads is not None else torch.zeros_like(head_graph)
        return ranker(cand_graph, cand_audio, cand_mask, head_graph, head_aud, features)
    return ranker(cand_graph, head_graph, features)

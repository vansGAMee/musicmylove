"""Train-user-only graph with separately normalized user and session associations."""
import argparse
from collections import Counter
import json
import numpy as np
from scipy import sparse
import torch
try:
    from .config import DATA, REPORTS, MIN_USERS, load_data, write_json
except ImportError:
    from config import DATA, REPORTS, MIN_USERS, load_data, write_json


CANARIES = ['Nirvana', 'Metallica', 'Ariana Grande', 'Aphex Twin', 'Joy Division',
            'Miles Davis', 'Black Sabbath', 'Boards of Canada']


def incidence(contexts, n):
    rows, cols = [], []
    for i, context in enumerate(contexts):
        rows.extend([i] * len(context))
        cols.extend(context)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(contexts), n))


def association(matrix, weights=None, minimum_support=3, eligible=None):
    """Blockwise exact co-occurrence; avoid materializing the dense all-pairs product."""
    degree = np.asarray(matrix.sum(0)).ravel()
    sizes = np.asarray(matrix.sum(1)).ravel()
    weight = 1 / np.maximum(sizes - 1, 1)
    if weights is not None:
        weight *= weights
    weighted = sparse.diags(weight) @ matrix
    norms = np.sqrt(np.maximum(np.asarray(matrix.multiply(weighted).sum(0)).ravel(), 1e-12))
    supports, affinities = [], []
    for start in range(0, matrix.shape[1], 128):
        stop = min(start + 128, matrix.shape[1])
        support = (matrix[:, start:stop].T @ matrix).tocsr()
        support.data[support.data < minimum_support] = 0
        support.eliminate_zeros()
        if eligible is not None:
            support = support.multiply(eligible[start:stop]).tocsr()
        mask = support.copy(); mask.data[:] = 1
        affinity = (matrix[:, start:stop].T @ weighted).multiply(mask).tocsr()
        affinity = sparse.diags(1 / norms[start:stop]) @ affinity @ sparse.diags(1 / norms)
        supports.append(support); affinities.append(affinity)
    affinity, support = sparse.vstack(affinities).tocsr(), sparse.vstack(supports).tocsr()
    affinity.setdiag(0); support.setdiag(0)
    affinity.eliminate_zeros(); support.eliminate_zeros()
    return affinity, support, degree


def adjacency(matrix):
    # LGCN normalized bipartite adjacency, with distinct context types.
    adj = sparse.bmat([[None, matrix.T], [matrix, None]], format='coo')
    degree = np.asarray(adj.sum(1)).ravel()
    scale = 1 / np.sqrt(np.maximum(degree, 1e-12))
    adj = (sparse.diags(scale) @ adj @ sparse.diags(scale)).tocoo()
    return torch.sparse_coo_tensor(np.vstack((adj.row, adj.col)),
                                   torch.tensor(adj.data, dtype=torch.float32), adj.shape, check_invariants=True).coalesce()


def load_graph(data_dir=None):
    import os
    from pathlib import Path
    if data_dir is not None:
        base = Path(data_dir)
    else:
        run_env = os.environ.get('MUSICMVP_RUN')
        base = Path(run_env).resolve() / 'data' if run_env else DATA
    meta = json.loads((base / 'graph.json').read_text())
    result = {k: sparse.load_npz(base / (k + '.npz')).tocsr()
              for k in ('global', 'local', 'users', 'sessions', 'confidence', 'user_incidence', 'session_incidence')}
    return meta, result


def supported_core(users, minimum_support=MIN_USERS):
    """Count distinct actual rows, pruning singleton contexts to a fixed point."""
    users = sorted(users, key=lambda u: u['id'])
    if len({u['id'] for u in users}) != len(users):
        raise ValueError('Duplicate user IDs would inflate independent support')
    seen = set(t for u in users for t in u['tracks'])
    while True:
        retained = [u for u in users if len(set(u['tracks']) & seen) >= 2]
        counts = Counter(t for u in retained for t in set(u['tracks']) & seen)
        updated = {t for t, c in counts.items() if c >= minimum_support}
        if updated == seen:
            return retained, sorted(seen)
        users, seen = retained, updated


def build():
    data = load_data()
    # Huge libraries are retained in dataset for positives, excluded from graph fitting.
    train = [u for u in data['users'] if u['split'] == 'train' and 5 <= len(u['tracks']) <= 2000]
    train, seen = supported_core(train)
    if len(seen) < 100:
        raise RuntimeError('DATA_BLOCKED: fewer than 100 tracks supported by >=3 independent train users')
    index = {t: i for i, t in enumerate(seen)}
    libraries, user_ids, sessions, session_weights = [], [], [], []
    for u in train:
        library = sorted({index[t] for t in u['tracks'] if t in index})
        if len(library) < 2:
            continue
        user_ids.append(u['id'])
        libraries.append(library)
        valid = [sorted({index[t] for t in s if t in index}) for s in u['sessions']]
        valid = sorted({tuple(s) for s in valid if 2 <= len(s) <= 80})
        # Repeated identical contexts count once; all sessions from a user saturate.
        for s in valid:
            artists = Counter(data['tracks'][seen[t]]['artist'].casefold() for t in s)
            dominance = max(artists.values()) / len(s)
            sessions.append(s)
            session_weights.append((1 - .8 * dominance) / max(1, len(valid)))
    ui, si = incidence(libraries, len(seen)), incidence(sessions, len(seen))
    if not np.all(np.asarray(ui.sum(0)).ravel() >= MIN_USERS):
        raise RuntimeError('Catalog support differs from actual graph rows')
    global_a, support_u, degree = association(ui)
    eligible = support_u.copy(); eligible.data[:] = 1
    local_a, support_s, _ = association(si, session_weights, minimum_support=1, eligible=eligible)
    # Common popularity alone is insufficient: require excess over independent expectation.
    coo = support_u.tocoo()
    expected = degree[coo.row] * degree[coo.col] / max(1, len(libraries))
    excess = np.maximum(0, (coo.data - expected) / (coo.data + expected + 1e-8))
    reliability = coo.data / (coo.data + 10.)
    confidence = sparse.csr_matrix((excess * reliability, (coo.row, coo.col)), shape=support_u.shape)
    gate = support_u.copy()
    gate.data = (gate.data >= MIN_USERS).astype(float)
    gate.eliminate_zeros()
    confidence = confidence.multiply(gate).tocsr()
    confidence.eliminate_zeros()
    global_a = global_a.multiply(gate).tocsr()
    local_a = local_a.multiply(gate).tocsr()
    graph = {'global': global_a, 'local': local_a, 'users': support_u, 'sessions': support_s,
             'confidence': confidence, 'user_incidence': ui,
             'session_incidence': sparse.diags(np.sqrt(session_weights)) @ si}
    DATA.mkdir(parents=True, exist_ok=True)
    for key, value in graph.items():
        sparse.save_npz(DATA / (key + '.npz'), value.tocsr())
    meta = {'fingerprint': data['fingerprint'], 'seen': seen, 'train_users': user_ids,
            'tracks': [data['tracks'][i] for i in seen], 'raw_tracks': len(data['tracks']),
            'train_seen_tracks': len(seen), 'trained_embeddings': 0, 'ann_indexed_tracks': 0,
            'recommendable_tracks': 0, 'excluded_large_libraries': sum(len(u['tracks']) > 2000 for u in data['users'] if u['split'] == 'train')}
    write_json(DATA / 'graph.json', meta)
    audit = {'fingerprint': data['fingerprint'], 'train_users': len(libraries), 'sessions': len(sessions),
             'tracks': len(seen), 'user_pairs_with_at_least_3_users': support_u.nnz // 2,
             'supported_pairs': confidence.nnz // 2, 'canaries': {}}
    for artist in CANARIES:
        candidates = [i for i, t in enumerate(meta['tracks']) if t['artist'].casefold() == artist.casefold()]
        examples = []
        for i in sorted(candidates, key=lambda t: (-degree[t], t))[:2]:
            row = global_a.getrow(i).multiply(confidence.getrow(i)).tocoo()
            neighbours = sorted(zip(row.col, row.data), key=lambda x: (-x[1], x[0]))[:10]
            examples.append({'track': meta['tracks'][i], 'neighbors': [dict(track=meta['tracks'][j], **{
                key: float(graph[key][i, j]) for key in ('global', 'local', 'users', 'sessions', 'confidence')}) for j, _ in neighbours]})
        audit['canaries'][artist] = examples
    audit['status'] = 'GRAPH_FAIL' if confidence.nnz == 0 else 'AUDIT_REQUIRED'
    write_json(REPORTS / 'graph_audit.json', audit)
    print(json.dumps({k: v for k, v in audit.items() if k != 'canaries'}))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()

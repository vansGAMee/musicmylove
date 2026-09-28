"""Finite preregistered DEV-only radius/confidence search; never pad the range."""
import numpy as np
import torch
try:
    from .artifacts import RangeConfig
    from .config import REPORTS, MIN_CANDIDATE_RECALL, write_json
    from .evaluate import queries
    from .retrieval import candidate_range
except ImportError:
    from artifacts import RangeConfig
    from config import REPORTS, MIN_CANDIDATE_RECALL, write_json
    from evaluate import queries
    from retrieval import candidate_range


def calibrate(data, meta, graph, embeddings, taste, seed):
    examples = list(queries(data, meta, 'dev'))
    if len(examples) < 10:
        raise RuntimeError('RANGE_FAIL: fewer than 10 eligible DEV profiles')
    # Deterministic bounded tuning subset. All DEV users still used by acceptance gates.
    examples = sorted(examples, key=lambda q: q['user'])[:200]
    results = []
    with torch.no_grad():
        encoded = [(q, taste(embeddings, q['seeds'])) for q in examples]
        for confidence in (.2, .35):
            for radius in (-.1, 0., .25):
                setting = RangeConfig(confidence=confidence, radius=radius).to_dict()
                graph['_range'] = setting
                recall, sizes, p50_recall = [], [], []
                for q, (heads, masses, assignments) in encoded:
                    rows = candidate_range(q['seeds'], heads, masses, assignments, embeddings, graph)
                    ids = list(dict.fromkeys(r['track'] for r in rows))
                    rec = len(set(ids[:2000]) & q['targets']) / len(q['targets'])
                    recall.append(rec)
                    sizes.append(len(ids))
                    if len(q['seeds']) >= 50:
                        p50_recall.append(rec)
                mean_p50 = float(np.mean(p50_recall)) if p50_recall else float(np.mean(recall))
                results.append({'config': setting, 'candidate_recall_2000': float(np.mean(recall)),
                                'profile_50_recall_2000': mean_p50,
                                'mean_range_size': float(np.mean(sizes)), 'users': len(examples)})
    admitted = [r for r in results if r['profile_50_recall_2000'] >= MIN_CANDIDATE_RECALL]
    if not admitted:
        write_json(REPORTS / f'range_calibration_{seed}.json', {'split': 'dev', 'trials': results, 'selected': None})
        raise RuntimeError('RANGE_FAIL: no DEV setting reaches required candidate recall')
    selected = min(admitted, key=lambda r: (r['mean_range_size'], -r['candidate_recall_2000']))
    write_json(REPORTS / f'range_calibration_{seed}.json', {'split': 'dev', 'trials': results, 'selected': selected})
    return selected['config']

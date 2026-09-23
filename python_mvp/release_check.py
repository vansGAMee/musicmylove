"""Several seeds, paired user uncertainty, frozen one-shot final verification."""
import argparse
import json
import torch
try:
    from .config import REPORTS, write_json, fingerprint, deterministic
    from .training import load_components, checkpoint, interval
    from .evaluate import evaluate, paired_interval
    from .recommend import recommend
    from .build_graph import CANARIES, adjacency
    from .networks import GraphEncoder
except ImportError:
    from config import REPORTS, write_json, fingerprint, deterministic
    from training import load_components, checkpoint, interval
    from evaluate import evaluate, paired_interval
    from recommend import recommend
    from build_graph import CANARIES, adjacency
    from networks import GraphEncoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['shadow', 'final'], required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    a = p.parse_args()
    seeds = sorted(set(a.seeds))
    if len(seeds) < 3:
        p.error('At least three independent training seeds required')
    components, hashes = {}, {}
    for seed in seeds:
        components[seed] = load_components(seed, require_ranker=True)
        hashes[str(seed)] = {stage: fingerprint(checkpoint(stage, seed)) for stage in ('graph', 'taste', 'ranker', 'shuffled')}
    target = REPORTS / f'release_{a.split}.json'
    if a.split == 'final':
        shadow = json.loads((REPORTS / 'release_shadow.json').read_text())
        if shadow['status'] != 'PASS' or shadow['checkpoints'] != hashes:
            raise RuntimeError('BLOCKED: shadow gates/checkpoint identities must pass first')
        with (REPORTS / 'final_consumed.lock').open('x') as f:
            f.write(json.dumps(hashes))
    results, passed = {}, True
    for seed, (data, meta, graph, embeddings, taste, ranker) in components.items():
        deterministic(seed)
        report = evaluate(data, meta, graph, a.split, embeddings, taste, ranker)
        cross = evaluate(data, meta, graph, a.split, embeddings, taste, ranker, cross_artist=True)
        comparisons = {b: interval(report, 'final', b) for b in ('popularity', 'graph', 'ppr', 'neural', 'taste')}
        random_model = GraphEncoder(len(embeddings), graph['user_incidence'].shape[0], graph['session_incidence'].shape[0])
        with torch.no_grad():
            random_embeddings, _ = random_model(adjacency(graph['user_incidence']), adjacency(graph['session_incidence']))
        shuffled = torch.load(checkpoint('shuffled', seed), map_location='cpu', weights_only=True)
        if shuffled['fingerprint'] != meta['fingerprint']:
            raise RuntimeError('Stale shuffled control')
        controls = {}
        for name, vectors in [('random', random_embeddings), ('shuffled', shuffled['embeddings'])]:
            controls[name] = evaluate(data, meta, graph, a.split, embeddings=vectors)
            comparisons[name] = paired_interval([r['final']['NDCG@50'] for r in report['per_user']],
                                                [r['neural']['NDCG@50'] for r in controls[name]['per_user']])
            comparisons['graph_vs_' + name] = paired_interval([r['neural']['NDCG@50'] for r in report['per_user']],
                                                            [r['neural']['NDCG@50'] for r in controls[name]['per_user']])
        range_groups = [evaluate(data, meta, graph, a.split, embeddings, taste, profile_size=k) for k in (5, 20, 50, 200, 500)]
        seed_pass = (report['users'] >= 30 and cross['users'] >= 30
                     and all(c['ci95'][0] > 0 for c in comparisons.values())
                     and interval(cross, 'final', 'popularity')['ci95'][0] > 0
                     and all(r['users'] >= 10 and r['metrics']['candidate']['CandidateRecall@2000'] >= .6 for r in range_groups))
        canaries = {}
        profiles = [[artist] for artist in CANARIES] + [['Nirvana', 'Joy Division', 'Boards of Canada'], ['Metallica', 'Aphex Twin']]
        for profile in profiles:
            ids = [tid for artist in profile for tid in [t['id'] for t in meta['tracks'] if t['artist'].casefold() == artist.casefold()][:10]]
            if not ids:
                canaries[' + '.join(profile)] = {'status': 'UNRESOLVED'}
                seed_pass = False
                continue
            result = recommend(ids, seed)
            canaries[' + '.join(profile)] = result
            seed_pass &= len(result['top50']) == 50
        passed &= seed_pass
        results[str(seed)] = {'passed': bool(seed_pass), 'evaluation': report, 'cross_artist': cross,
                              'range_groups': range_groups, 'comparisons': comparisons, 'controls': controls, 'canaries': canaries}
    output = {'status': 'PASS' if passed else 'FAIL', 'split': a.split,
              'checkpoints': hashes, 'results': results}
    write_json(target, output)
    print(output['status'])
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()

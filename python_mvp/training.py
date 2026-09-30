"""Training orchestration. All unknown negatives carry reduced loss weight.
Train loss is telemetry; checkpoint selection uses exact dev ranking only.
"""
import argparse
import json
import math
import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F
try:
    from .config import DATA, MODELS, REPORTS, MIN_CANDIDATE_RECALL, deterministic, load_data, write_json, fingerprint
    from .networks import GraphEncoder, MultiInterest, NeuralRanker
    from .build_graph import load_graph, adjacency
    from .evaluate import queries, evaluate, paired_interval, shuffle_edges
    from .retrieval import candidate_range, score_rows
except ImportError:
    from config import DATA, MODELS, REPORTS, MIN_CANDIDATE_RECALL, deterministic, load_data, write_json, fingerprint
    from networks import GraphEncoder, MultiInterest, NeuralRanker
    from build_graph import load_graph, adjacency
    from evaluate import queries, evaluate, paired_interval, shuffle_edges
    from retrieval import candidate_range, score_rows


try:
    from .sampling import sample_negatives, protect, profile_episode
    from .artifacts import contract, validate_contract, RangeConfig
except ImportError:
    from sampling import sample_negatives, protect, profile_episode
    from artifacts import contract, validate_contract, RangeConfig


def checkpoint(stage, seed):
    import os
    from pathlib import Path
    models_dir = (Path(os.environ['MUSICMVP_RUN']).resolve() / 'models') if os.environ.get('MUSICMVP_RUN') else MODELS
    target = models_dir / f'{stage}_{seed}.pt'
    return target


def load_components(seed, require_ranker=False, load_taste=True):
    data = load_data()
    meta, graph = load_graph()
    if meta['fingerprint'] != data['fingerprint']:
        raise RuntimeError('Dataset/graph fingerprint mismatch')
    if not checkpoint('graph', seed).exists():
        raise RuntimeError('MODEL_NOT_TRAINED: no graph checkpoint')
    saved = torch.load(checkpoint('graph', seed), map_location='cpu', weights_only=True)
    meta['_contract'] = contract()
    validate_contract(saved.get('contract'), meta['_contract'])
    if saved['fingerprint'] != meta['fingerprint'] or saved['status'] != 'DEV_PASS':
        raise RuntimeError('GRAPH_ENCODER_FAIL: no accepted checkpoint for this dataset')
    embeddings = saved['embeddings']
    if saved.get('vocabulary') != [t['id'] for t in meta['tracks']] or embeddings.shape != (len(meta['tracks']), 96):
        raise RuntimeError('ARTIFACT_MISMATCH: trained embedding vocabulary/shape differs')
    taste = MultiInterest()
    path = checkpoint('taste', seed)
    if load_taste and path.exists():
        state = torch.load(path, map_location='cpu', weights_only=True)
        validate_contract(state.get('contract'), meta['_contract'])
        if 'range_config' not in state:
            raise RuntimeError('RANGE_NOT_CALIBRATED')
        graph['_range'] = RangeConfig(**state['range_config']).to_dict()
        if state['fingerprint'] != meta['fingerprint'] or state['parent_sha256'] != fingerprint(checkpoint('graph', seed)):
            raise RuntimeError('Stale taste checkpoint')
        if state['status'] != 'DEV_PASS':
            raise RuntimeError('TASTE_FAIL: checkpoint failed dev gates')
        taste.load_state_dict(state['weights'])
        taste.eval()
    else:
        taste = None
    ranker = None
    if require_ranker:
        if taste is None:
            raise RuntimeError('TASTE_FAIL: no trained taste checkpoint')
        state = torch.load(checkpoint('ranker', seed), map_location='cpu', weights_only=True)
        validate_contract(state.get('contract'), meta['_contract'])
        if state['fingerprint'] != meta['fingerprint'] or state['parent_sha256'] != fingerprint(path) or state['status'] != 'DEV_PASS':
            raise RuntimeError('RANKER_FAIL: stale or unaccepted checkpoint')
        ranker = NeuralRanker()
        ranker.load_state_dict(state['weights'])
        ranker.eval()
    return data, meta, graph, embeddings, taste, ranker


def save(stage, seed, model, meta, status, **extra):
    MODELS.mkdir(parents=True, exist_ok=True)
    if '_contract' not in meta:
        meta['_contract'] = contract()
    target = checkpoint(stage, seed)
    tmp = target.with_suffix('.tmp')
    torch.save({'weights': model.state_dict(), 'fingerprint': meta['fingerprint'],
                'status': status, 'seed': seed, 'vocabulary': [t['id'] for t in meta['tracks']], 'contract': meta['_contract'], **extra}, tmp)
    tmp.replace(target)


def interval(report, method, baseline):
    return paired_interval([r[method]['NDCG@50'] for r in report['per_user']],
                           [r[baseline]['NDCG@50'] for r in report['per_user']])


def train_graph(args):
    data, (meta, graph) = load_data(), load_graph()
    if meta['fingerprint'] != data['fingerprint']:
        raise RuntimeError('Stale graph')
    baseline = evaluate(data, meta, graph)
    write_json(REPORTS / 'baselines_dev.json', baseline)
    if baseline['users'] < 10 or max(baseline['metrics']['graph']['Recall@50'], baseline['metrics']['ppr']['Recall@50']) <= baseline['metrics']['popularity']['Recall@50']:
        raise RuntimeError('GRAPH_FAIL: graph baselines do not beat popularity on dev Recall@50')
    ui, si = graph['user_incidence'], graph['session_incidence']
    popularity = np.asarray(ui.sum(0)).ravel()
    n = ui.shape[1]
    curves, evaluations = {}, {}
    # The control is trained too, with both context graphs degree-preserving shuffled.
    for control in ('trained', 'shuffled'):
        deterministic(args.seed)
        rng = np.random.default_rng(args.seed)
        matrices = []
        for matrix in (ui, si):
            if control == 'shuffled':
                coo = matrix.tocoo()
                edges = shuffle_edges(list(zip(coo.row.tolist(), coo.col.tolist())), args.seed)
                # Preserve support degrees exactly; weighted session scale stays per row.
                row_scale = np.asarray(matrix.max(1).toarray()).ravel()
                matrix = sparse.csr_matrix(([row_scale[u] for u, _ in edges], tuple(zip(*edges))), shape=matrix.shape)
            matrices.append(matrix)
        ug, sg = [adjacency(m) for m in matrices]
        model = GraphEncoder(n, ui.shape[0], si.shape[0])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        with torch.no_grad():
            initial, _ = model(ug, sg)
        if control == 'trained':
            evaluations['random'] = evaluate(data, meta, graph, embeddings=initial)
        users = [set(matrices[0].getrow(u).indices.tolist()) for u in range(ui.shape[0])]
        original_known = [set(ui.getrow(u).indices.tolist()) for u in range(ui.shape[0])]
        curves[control], best, best_state, best_embeddings = [], -math.inf, None, None
        for epoch in range(args.epochs):
            loss_total, updates = 0., 0
            for start in range(0, len(users), args.batch_size):
                embeddings, user_vectors = model(ug, sg)
                losses = []
                for u in range(start, min(start + args.batch_size, len(users))):
                    if not users[u]:
                        continue
                    positive = int(rng.choice(sorted(users[u])))
                    sims = (embeddings @ user_vectors[u]).detach().numpy()
                    negative = sample_negatives(rng, n, users[u] | original_known[u], protect([positive], graph), popularity, positive, sims)
                    if not len(negative):
                        continue
                    logits = (embeddings[[positive, *negative]] @ user_vectors[u]) / .1
                    logits = logits + torch.tensor([0.] + [math.log(.25)] * len(negative))
                    losses.append(-F.log_softmax(logits, 0)[0])
                if losses:
                    loss = torch.stack(losses).mean()
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    updates += 1; loss_total += float(loss.detach())
            with torch.no_grad():
                embeddings, _ = model(ug, sg)
            report = evaluate(data, meta, graph, embeddings=embeddings)
            metric = report['metrics']['neural']['NDCG@50']
            curves[control].append({'epoch': epoch + 1, 'loss': loss_total/max(updates, 1), 'updates': updates, 'dev': report['metrics']})
            print(json.dumps({'stage': control, **curves[control][-1]}), flush=True)
            if updates and metric > best:
                best, best_state, best_embeddings = metric, {k: v.detach().clone() for k, v in model.state_dict().items()}, embeddings.detach().clone()
            write_json(REPORTS / f'graph_curves_{args.seed}.json', curves)
        if best_state is None:
            raise RuntimeError('GRAPH_ENCODER_FAIL: no admissible training updates')
        model.load_state_dict(best_state)
        evaluations[control] = evaluate(data, meta, graph, embeddings=best_embeddings)
        best_epoch = curves[control][max(range(len(curves[control])), key=lambda i: curves[control][i]['dev']['neural']['NDCG@50'])]['epoch']
        save('graph' if control == 'trained' else 'shuffled', args.seed, model, meta, 'UNVALIDATED', embeddings=best_embeddings, epoch=best_epoch, metrics=evaluations[control]['metrics'], training_config=vars(args))
    trained = evaluations['trained']
    comparisons = {b: interval(trained, 'neural', b) for b in ('popularity', 'graph', 'ppr')}
    for control in ('random', 'shuffled'):
        comparisons[control] = paired_interval([r['neural']['NDCG@50'] for r in trained['per_user']],
                                               [r['neural']['NDCG@50'] for r in evaluations[control]['per_user']])
    passed = all(c['ci95'][0] > 0 for k, c in comparisons.items() if k in ('popularity', 'random', 'shuffled'))
    passed &= trained['metrics']['neural']['Recall@10'] > trained['metrics']['popularity']['Recall@10']
    write_json(REPORTS / f'graph_comparison_{args.seed}.json', {'status': 'DEV_PASS' if passed else 'GRAPH_ENCODER_FAIL',
                                                            'comparisons': comparisons, 'evaluations': evaluations})
    state = torch.load(checkpoint('graph', args.seed), weights_only=True)
    state['status'] = 'DEV_PASS' if passed else 'GRAPH_ENCODER_FAIL'
    torch.save(state, checkpoint('graph', args.seed))
    if not passed:
        raise RuntimeError('GRAPH_ENCODER_FAIL: see graph_comparison report; do not train downstream stages')


def train_profile(args, stage):
    data, meta, graph, embeddings, taste, _ = load_components(args.seed, load_taste=stage != 'taste')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if stage == 'taste':
        taste = MultiInterest()
        model = taste
        lr = args.lr
    else:
        if taste is None:
            raise RuntimeError('TASTE_FAIL: train taste first')
        for p in taste.parameters():
            p.requires_grad_(False)
        range_report = evaluate(data, meta, graph, embeddings=embeddings, taste=taste, profile_size=50)
        if range_report['users'] < 10:
            range_report = evaluate(data, meta, graph, embeddings=embeddings, taste=taste)
        write_json(REPORTS / f'range_dev_{args.seed}.json', range_report)
        if range_report['users'] < 10:
            raise RuntimeError('RANGE_FAIL: fewer than 10 eligible DEV users')
        if range_report['metrics']['candidate']['CandidateRecall@2000'] < MIN_CANDIDATE_RECALL:
            raise RuntimeError('RANGE_FAIL: do not train ranker; missing candidates cannot be repaired by ranking')
        model = NeuralRanker().to(device)
        taste = taste.to(device)
        embeddings = embeddings.to(device)
        lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3 if stage == 'ranker' else 1e-4)
    rng, best, curves = np.random.default_rng(args.seed), -math.inf, []
    train_queries = list(queries(data, meta, 'train'))
    steps_per_epoch = 1200 if stage == 'ranker' else len(train_queries)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * steps_per_epoch), eta_min=1e-5) if stage == 'ranker' else None
    popularity = np.asarray(graph['user_incidence'].sum(0)).ravel()
    parent = checkpoint('graph' if stage == 'taste' else 'taste', args.seed)

    # Evaluate and save initial baseline checkpoint before training loop
    model.eval()
    init_key = 'taste' if stage == 'taste' else 'final'
    init_report = evaluate(data, meta, graph, embeddings=embeddings, taste=taste, ranker=model if stage == 'ranker' else None)
    best = init_report['metrics'][init_key]['NDCG@50']
    save(stage, args.seed, model.cpu(), meta, 'UNVALIDATED', parent_sha256=fingerprint(parent), epoch=0, metrics=init_report['metrics'], training_config=vars(args))
    if stage == 'ranker':
        model.to(device)
    model.train()
    for epoch in range(args.epochs):
        total, updates = 0., 0
        for qi in rng.permutation(len(train_queries))[:steps_per_epoch]:
            query = train_queries[qi]
            seeds, targets = profile_episode(query['known'], meta['tracks'], rng, discovery=(qi % 2 == 0))
            heads, masses, assignments = taste(embeddings, seeds)
            if stage == 'taste':
                positive = int(rng.choice(sorted(targets)))
                sims = (embeddings @ heads.T).max(-1).values.detach().numpy()
                negative = sample_negatives(rng, len(embeddings), query['known'], protect([positive], graph), popularity, positive, sims)
                if not len(negative):
                    continue
                logits = torch.logsumexp((embeddings[[positive, *negative]] @ heads.T) / .1 + masses.clamp_min(1e-8).log(), -1)
                logits = logits + torch.tensor([0.] + [math.log(.25)] * len(negative))
                loss = -F.log_softmax(logits, 0)[0] + taste.regularization(heads, masses, assignments)
            else:
                rows = candidate_range(seeds, heads.detach().cpu(), masses.detach().cpu(), assignments.detach().cpu(), embeddings.detach().cpu(), graph)
                # Hard negative candidates: true targets retained, user known positives excluded.
                rows = [r for r in rows if r['track'] in targets or r['track'] not in query['known']]
                track_ids = sorted({r['track'] for r in rows})
                positive_mask = torch.tensor([t in targets for t in track_ids], dtype=torch.bool, device=device)
                if not positive_mask.any() or positive_mask.all():
                    continue
                track_to_idx = {t: i for i, t in enumerate(track_ids)}
                row_target_idx = torch.tensor([track_to_idx[r['track']] for r in rows], dtype=torch.long, device=device)
                per_head = score_rows(rows, model, embeddings, heads)
                values = torch.full((len(track_ids),), -math.inf, device=device)
                values = values.scatter_reduce(0, row_target_idx, per_head, reduce='amax')
                neg_lse = torch.logsumexp(values[~positive_mask] + math.log(0.25), dim=0)
                pos_vals = values[positive_mask]
                loss = F.softplus(neg_lse - pos_vals).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += float(loss.detach()); updates += 1
        model.eval()
        report = evaluate(data, meta, graph, embeddings=embeddings, taste=taste, ranker=model if stage == 'ranker' else None)
        key = 'final' if stage == 'ranker' else 'taste'
        score = report['metrics'][key]['NDCG@50']
        curves.append({'epoch': epoch + 1, 'loss': total / max(1, updates), 'updates': updates, 'dev': report['metrics']})
        print(json.dumps(curves[-1]), flush=True)
        if updates and score > best:
            best = score
            save(stage, args.seed, model.cpu(), meta, 'UNVALIDATED', parent_sha256=fingerprint(parent), epoch=epoch + 1, metrics=report['metrics'], training_config=vars(args))
            if stage == 'ranker':
                model.to(device)
        model.train()
        write_json(REPORTS / f'{stage}_curves_{args.seed}.json', curves)
    if best == -math.inf:
        raise RuntimeError(stage.upper() + '_FAIL: no admissible updates')
    state = torch.load(checkpoint(stage, args.seed), weights_only=True)
    model.load_state_dict(state['weights']); model.eval()
    if stage == 'ranker':
        model.to(device)


    if stage == 'taste':
        try:
            from .calibrate_range import calibrate
        except ImportError:
            from calibrate_range import calibrate
        graph['_range'] = calibrate(data, meta, graph, embeddings, taste, args.seed)
        state['range_config'] = graph['_range']
    report = evaluate(data, meta, graph, embeddings=embeddings, taste=taste, ranker=model if stage == 'ranker' else None)
    comparisons = {b: interval(report, key, b) for b in (('popularity', 'graph', 'ppr', 'neural', 'taste') if stage == 'ranker' else ('popularity', 'graph', 'ppr', 'neural'))}
    if stage == 'ranker':
        passed = all(c['ci95'][0] >= 0 for c in comparisons.values())
    else:
        passed = all(c['ci95'][0] >= 0 for k, c in comparisons.items() if k in ('popularity', 'neural'))
    if stage == 'taste':
        try:
            from .taste_checks import check_tastes
        except ImportError:
            from taste_checks import check_tastes
        checks = check_tastes(data, meta, embeddings, taste)
        passed &= checks['passed']
        report['taste_checks'] = checks
        range_groups = [evaluate(data, meta, graph, embeddings=embeddings, taste=taste, profile_size=k) for k in (5, 20, 50, 200, 500)]
        report['range_groups'] = range_groups
        eligible_groups = [r for r in range_groups if r['users'] >= 10 and r['profile_size'] is not None and r['profile_size'] >= 50]
        if eligible_groups:
            passed &= all(r['metrics']['candidate']['CandidateRecall@2000'] >= MIN_CANDIDATE_RECALL for r in eligible_groups)
    else:
        passed &= interval(report, 'final', 'taste')['ci95'][0] >= 0
    passed &= state.get('epoch', 0) > 0
    state['status'] = 'DEV_PASS' if passed else stage.upper() + '_FAIL'
    torch.save(state, checkpoint(stage, args.seed))
    write_json(REPORTS / f'{stage}_dev_{args.seed}.json', {'status': state['status'], 'comparisons': comparisons, 'evaluation': report})
    if not passed:
        raise RuntimeError(state['status'] + ': inspect dev report; no accepted checkpoint')


def main(stage):
    p = argparse.ArgumentParser(description=f'Train own {stage} network; requires real audited data')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--batch-size', type=int, default=128)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        p.error('positive epochs, batch size and learning rate required')
    deterministic(args.seed)
    train_graph(args) if stage == 'graph' else train_profile(args, stage)

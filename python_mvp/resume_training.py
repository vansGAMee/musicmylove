"""Recover ranker training after interruption, keeping the saved representation.

Streams only relevant ranker/DEV data; never rebuilds or trains the graph/taste.
The interrupted legacy ranker has no epoch checkpoint: that stage restarts.
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import torch
from . import honest_training as training
from .config import ROOT, deterministic, fingerprint, write_json
from .build_graph import load_graph
from .networks import MultiInterest
from .discovery_engine import DiscoveryRanker
from .evaluate import paired_interval

SCHEMA = 'compact-ranker-recovery-v1'


def recovery_contract(run):
    return {'schema': SCHEMA, 'implementation': {**training.implementation(),
            'resume_training.py': fingerprint(Path(__file__)),
            'ranker_data.py': fingerprint(ROOT/'ranker_data.py')},
            'artifacts': training.artifact_hashes(run),
            'representation_sha256': fingerprint(run/'representation.pt')}


def verify_recovery(run):
    """Validate additional training provenance when serving a recovered model."""
    run = Path(run)
    saved = json.loads((run/'recovery.json').read_text())
    if saved != recovery_contract(run):
        raise RuntimeError('Recovery inputs/code changed; preserve this run and inspect the mismatch')
    return saved


def validate_checkpoint(run):
    state = torch.load(run/'ranker.pt', map_location='cpu', weights_only=True)
    if (state.get('schema') != training.SCHEMA or state.get('implementation') != training.implementation()
            or state.get('artifacts') != training.artifact_hashes(run)
            or state.get('representation_sha256') != fingerprint(run/'representation.pt')):
        raise RuntimeError('Saved ranker inputs/code changed')
    return state


def load_recovered(run):
    """Inference needs graph/vectors/weights, not millions of raw listen records."""
    run = Path(run)
    verify_recovery(run)
    state = validate_checkpoint(run)
    saved = torch.load(run/'representation.pt', map_location='cpu', weights_only=True)
    meta, graph = load_graph(run/'data')
    if saved['vocabulary'] != [t['id'] for t in meta['tracks']]:
        raise RuntimeError('Representation vocabulary mismatch')
    taste = MultiInterest(); taste.load_state_dict(saved['taste'])
    engine = training.engine_from_parts(run, {}, meta, graph, saved['embeddings'], taste)
    engine.raw_data_path = run/'data/dataset.json'
    model = DiscoveryRanker(state['dimensions']); model.load_state_dict(state['weights']); model.eval()
    return engine, model


def finish_evaluation(engine, model, run):
    """Finish the same DEV report if termination happened after ranker saving."""
    rows = [training.evaluate_one(engine, model, q, *engine.features(q['seeds']), engine.families)
            for q in training.evaluation_queries(engine, engine.families)]
    if len(rows) < 10:
        raise RuntimeError('Fewer than 10 eligible DEV users')
    comparisons = {b: paired_interval([r['neural'] for r in rows], [r[b] for r in rows])
                   for b in ('listeners', 'ppr', 'graph')}
    cross = [r for r in rows if 'neural_cross' in r]
    cross_comparisons = {b: paired_interval([r['neural_cross'] for r in cross], [r[b+'_cross'] for r in cross])
                        for b in ('listeners', 'ppr', 'graph')}
    better = all(c['ci95'][0] > 0 for c in [*comparisons.values(), *cross_comparisons.values()])
    write_json(run/'evaluation.json', dict(status='DEV_GAIN_OBSERVED_NOT_RELEASE' if better else 'NO_PROVEN_DEV_GAIN',
        split='dev', frozen_final_used=False, selection_data_reused=True,
        checkpoint_sha256=fingerprint(run/'ranker.pt'), metrics=training.aggregate(rows),
        comparisons=comparisons, cross_comparisons=cross_comparisons, per_user=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'data/targeted-v1/model')
    parser.add_argument('--check-only', action='store_true', help='Validate saved artifacts; no training or data projection')
    args = parser.parse_args()
    run = args.run.resolve()
    required = ['representation.pt', 'partition.json', 'data/dataset.json', 'data/graph.json']
    if any(not (run/name).is_file() for name in required):
        raise RuntimeError('Incomplete representation: recovery requires saved graph, taste, dataset and partition')
    manifest = run/'recovery.json'
    current = recovery_contract(run)
    if manifest.exists() and json.loads(manifest.read_text()) != current:
        raise RuntimeError('Recovery inputs/code changed; preserve this run and inspect the mismatch')
    state = validate_checkpoint(run) if (run/'ranker.pt').exists() else None
    next_command = 'python -m python_mvp.cli ~/Downloads/liked.json --run '+shlex.quote(str(run))+' --mode 2'
    evaluation = run/'evaluation.json'
    if state is not None and evaluation.exists():
        if json.loads(evaluation.read_text()).get('checkpoint_sha256') != fingerprint(run/'ranker.pt'):
            raise RuntimeError('Evaluation checkpoint changed')
        print('Training already complete. '+next_command)
        return
    partition = json.loads((run/'partition.json').read_text())
    if partition.get('schema') != training.SCHEMA:
        raise RuntimeError('Only disjoint-user honest runs can be recovered')
    config = partition['config']
    epochs, limit, seed = config['epochs'], config['train_users'], config['seed']
    if min(epochs, limit) < 1:
        raise RuntimeError('Invalid original training settings')
    representation = torch.load(run/'representation.pt', map_location='cpu', weights_only=True)
    if representation.get('implementation') != training.implementation():
        raise RuntimeError('Saved representation code changed')
    meta, graph = load_graph(run/'data')
    if representation['vocabulary'] != [t['id'] for t in meta['tracks']]:
        raise RuntimeError('Representation vocabulary mismatch')
    if (set(partition['representation_users']) & set(partition['ranker_users'])
            or not set(meta['train_users']) <= set(partition['representation_users'])):
        raise RuntimeError('Representation/ranker split mismatch')
    print(f"Verified saved representation: {len(meta['tracks'])} tracks; graph and taste will be reused.",flush=True)
    if args.check_only:
        print('Check only: no training or dataset projection performed.')
        return
    if not manifest.exists():
        # Keep the failed attempt's logs without changing any source artifacts.
        backup = run/'recovery-original'; backup.mkdir(exist_ok=True)
        for name in ('training.json','sampling.json'):
            source, target = run/name, backup/name
            if source.exists() and not target.exists():
                shutil.copyfile(source,target)
        write_json(manifest,current)
    from .ranker_data import load_ranker_data
    print('Streaming compact ranker/DEV data; raw sessions are not retained...',flush=True)
    data = load_ranker_data(run/'data/dataset.json', meta)
    ranker_ids = {u['id'] for u in data['users'] if u['split']=='ranker_train'}
    if not ranker_ids <= set(partition['ranker_users']):
        raise RuntimeError('Projected ranker users differ from original partition')
    deterministic(seed)
    taste = MultiInterest(); taste.load_state_dict(representation['taste'])
    engine = training.engine_from_parts(run, data, meta, graph, representation['embeddings'], taste)
    print(f"Compact data: {len(data['tracks'])} relevant raw identities, {len(data['users'])} ranker/DEV users.",flush=True)
    if state is None:
        print('Restarting only the ranker stage; the legacy attempt saved no epoch weights.',flush=True)
        training.train_ranker(engine,run,epochs=epochs,train_limit=limit,seed=seed)
    else:
        model = DiscoveryRanker(state['dimensions']); model.load_state_dict(state['weights']); model.eval()
        print('Saved ranker is ready; finishing DEV report only.',flush=True)
        finish_evaluation(engine,model,run)
    print('Ready. '+next_command)


if __name__ == '__main__':
    main()

"""Reuse immutable graph vectors; train discovery on existing data, without network access."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch
try:
    from .config import ROOT, DIM, fingerprint, write_json
    from .artifacts import contract
except ImportError:
    from config import ROOT, DIM, fingerprint, write_json
    from artifacts import contract


def import_graph(path, meta, current, seed):
    """Explicit representation import, NOT silent acceptance of stale executable models.

    The graph is consumed as a frozen table. Its original training contract remains
    in provenance; new downstream code never reconstructs or trains its encoder.
    """
    state = torch.load(path, map_location='cpu', weights_only=True)
    old = state.get('contract', {})
    for field in ('schema', 'artifacts', 'model'):
        if old.get(field) != current.get(field):
            raise RuntimeError(f'IMPORT_BLOCKED: graph {field} changed')
    vectors = state.get('embeddings')
    if (state.get('status') != 'DEV_PASS' or state.get('seed') != seed
            or state.get('fingerprint') != meta['fingerprint']
            or state.get('vocabulary') != [t['id'] for t in meta['tracks']]
            or not isinstance(vectors, torch.Tensor)
            or vectors.shape != (len(meta['tracks']), DIM)
            or not torch.isfinite(vectors).all()):
        raise RuntimeError('IMPORT_BLOCKED: unaccepted graph, invalid vectors or vocabulary')
    state['import_provenance'] = {'source_sha256': fingerprint(path), 'original_contract': old,
                                 'representation': 'frozen graph vectors; not retrained or revalidated'}
    state['contract'] = current
    return state


def prepare(source, destination, seeds):
    source, destination = source.resolve(), destination.resolve()
    if source == destination:
        raise RuntimeError('Use a separate destination; source experiment is immutable')
    source_contract = contract(source / 'data')
    identity = {'source': str(source), 'contract': source_contract, 'seeds': seeds,
                'graphs': {str(seed): fingerprint(source / 'models' / f'graph_{seed}.pt') for seed in seeds}}
    manifest = destination / 'refresh.json'
    if destination.exists():
        if not manifest.exists() or json.loads(manifest.read_text()) != identity:
            raise RuntimeError('Destination exists with different inputs/code; use a new --run')
        source_lock = source / 'reports' / 'final_consumed.lock'
        if source_lock.exists():
            shutil.copy2(source_lock, destination / 'reports' / source_lock.name)
        return
    meta = json.loads((source / 'data' / 'graph.json').read_text())
    states = {seed: import_graph(source / 'models' / f'graph_{seed}.pt', meta, source_contract, seed) for seed in seeds}
    destination.mkdir(parents=True)
    (destination / 'data').mkdir()
    (destination / 'models').mkdir()
    (destination / 'reports').mkdir()
    # Copy only prepared data, never download archives/audio/cache.
    for name in source_contract['artifacts']:
        shutil.copy2(source / 'data' / name, destination / 'data' / name)
    for seed, state in states.items():
        torch.save(state, destination / 'models' / f'graph_{seed}.pt')
    # A new directory does not create new held-out users. Preserve test consumption.
    lock = source / 'reports' / 'final_consumed.lock'
    if lock.exists():
        shutil.copy2(lock, destination / 'reports' / lock.name)
    write_json(manifest, identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data' / 'main')
    parser.add_argument('--run', type=Path, default=ROOT / 'data' / 'discovery-v1')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    seeds = sorted(set(args.seeds))
    try:
        prepare(args.source, args.run, seeds)
        print(f'Prepared offline experiment: {args.run.resolve()}', flush=True)
        if args.prepare_only:
            return 0
        env = os.environ.copy()
        env['MUSICMVP_RUN'] = str(args.run.resolve())
        for seed in seeds:
            for stage in ('taste', 'ranker'):
                target = args.run / 'models' / f'{stage}_{seed}.pt'
                parent = args.run / 'models' / f'{"graph" if stage == "taste" else "taste"}_{seed}.pt'
                if target.exists():
                    saved = torch.load(target, map_location='cpu', weights_only=True)
                    if (saved.get('status') == 'DEV_PASS'
                            and saved.get('contract') == contract(args.run / 'data')
                            and saved.get('parent_sha256') == fingerprint(parent)
                            and saved.get('training_config', {}).get('epochs') == args.epochs):
                        print(f'Already accepted: {stage}, seed={seed}', flush=True)
                        continue
                subprocess.run([sys.executable, str(ROOT / f'train_{stage}.py'), '--seed', str(seed),
                                '--epochs', str(args.epochs)], env=env, check=True)
            subprocess.run([sys.executable, str(ROOT / 'evaluate.py'), '--seed', str(seed)], env=env, check=True)
        print('DEV finished. Quality is experimental; no frozen test was opened.')
        print(f'MUSICMVP_RUN={args.run.resolve()} python {ROOT / "recommend.py"} tracks.txt --experimental')
        return 0
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'REFRESH_FAILED: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

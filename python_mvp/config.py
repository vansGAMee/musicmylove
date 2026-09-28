"""Offline experiment only; never imported by the production TypeScript app."""
from pathlib import Path
import hashlib
import os
import json
import random
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RUN = Path(os.environ.get('MUSICMVP_RUN', str(ROOT))).resolve()
DATA = RUN / 'data'
MODELS = RUN / 'models'
REPORTS = RUN / 'reports'
DIM = 96
HEADS = 16
SEED = 42
MIN_USERS = 3
MIN_CONFIDENCE = .35
# Fixed initial hypotheses, NOT empirically calibrated quality guarantees.
MIN_CANDIDATE_RECALL = .60


def deterministic(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def set_run(run_dir):
    global RUN, DATA, MODELS, REPORTS
    RUN = Path(run_dir).resolve()
    DATA = RUN / 'data'
    MODELS = RUN / 'models'
    REPORTS = RUN / 'reports'
    os.environ['MUSICMVP_RUN'] = str(RUN)
    import sys
    for mod in list(sys.modules.values()):
        if mod is not None and Path(getattr(mod, '__file__', '') or '').resolve().parent == ROOT:
            if hasattr(mod, 'RUN') and getattr(mod, 'RUN', None) != RUN:
                mod.RUN = RUN
            if hasattr(mod, 'DATA') and getattr(mod, 'DATA', None) != DATA:
                mod.DATA = DATA
            if hasattr(mod, 'MODELS') and getattr(mod, 'MODELS', None) != MODELS:
                mod.MODELS = MODELS
            if hasattr(mod, 'REPORTS') and getattr(mod, 'REPORTS', None) != REPORTS:
                mod.REPORTS = REPORTS
    return RUN


def load_data(data_path=None):
    if data_path is not None:
        path = Path(data_path)
    else:
        run_env = os.environ.get('MUSICMVP_RUN')
        path = (Path(run_env).resolve() / 'data' / 'dataset.json') if run_env else (DATA / 'dataset.json')
    if not path.exists():
        raise RuntimeError(f'DATA_BLOCKED: dataset missing in requested run: {path}')
    return json.loads(path.read_text())


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

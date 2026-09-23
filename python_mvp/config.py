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


def load_data():
    path = DATA / 'dataset.json'
    if not path.exists():
        raise RuntimeError('DATA_BLOCKED: run prepare_data.py on original ListenBrainz JSONL with users and timestamps')
    return json.loads(path.read_text())


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

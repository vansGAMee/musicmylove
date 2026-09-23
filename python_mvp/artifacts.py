"""Fail-closed artifact contracts; never load old, unversioned experiments."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
try:
    from .config import DATA, DIM, HEADS, fingerprint
except ImportError:
    from config import DATA, DIM, HEADS, fingerprint


@dataclass(frozen=True)
class RangeConfig:
    confidence: float = .35
    min_users: int = 3
    radius: float = 0.

    def __post_init__(self):
        if not .1 <= self.confidence <= 1 or self.min_users < 3 or not -1 <= self.radius <= 1:
            raise ValueError('Unsafe range configuration')

    def to_dict(self):
        return asdict(self)


PRODUCTION_MIN_CONFIDENCE = 0.20
PRODUCTION_MIN_RADIUS = -0.10


def validate_production_range(config):
    """Diagnostic smoke configurations (e.g. confidence=0.1, radius=-0.25) are strictly prohibited in production."""
    if isinstance(config, RangeConfig):
        cfg = config.to_dict()
    elif isinstance(config, dict):
        cfg = config
    else:
        raise ValueError("Invalid range configuration type")
    conf = float(cfg.get('confidence', 0.))
    rad = float(cfg.get('radius', 0.))
    users = int(cfg.get('min_users', 0))
    if conf < PRODUCTION_MIN_CONFIDENCE:
        raise RuntimeError(f"DIAGNOSTIC_SMOKE_RANGE_IN_PRODUCTION: confidence {conf} < minimum production threshold {PRODUCTION_MIN_CONFIDENCE}")
    if rad < PRODUCTION_MIN_RADIUS:
        raise RuntimeError(f"DIAGNOSTIC_SMOKE_RANGE_IN_PRODUCTION: radius {rad} < minimum production threshold {PRODUCTION_MIN_RADIUS}")
    if users < 3:
        raise RuntimeError(f"DIAGNOSTIC_SMOKE_RANGE_IN_PRODUCTION: min_users {users} < 3")


def contract(directory=DATA):
    directory = Path(directory)
    paths = [directory / name for name in ('dataset.json', 'split_manifest.json', 'graph.json')]
    paths += sorted(directory.glob('*.npz'))
    sources = [Path(__file__).with_name(name) for name in ('networks.py', 'retrieval.py', 'config.py', 'prepare_data.py', 'build_graph.py', 'training.py', 'evaluate.py', 'sampling.py', 'calibrate_range.py')]
    return {'schema': 2, 'model': {'dimension': DIM, 'heads': HEADS, 'layers': 2},
            'artifacts': {p.name: fingerprint(p) for p in paths},
            'implementation': {p.name: fingerprint(p) for p in sources if p.exists()}}


def validate_contract(saved, current):
    if not saved or saved != current:
        raise RuntimeError('ARTIFACT_MISMATCH: dataset, split, graph, vocabulary, config or implementation changed; use a new experiment')

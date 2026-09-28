"""Independent Release Verification Script.
Does not replace quality evaluation or consume the frozen split again.
Validates frozen quality evidence against current artifacts and exact ordered stability.
Exits with code 0 ONLY on complete RELEASE_PASS across all 3 seeds; otherwise exits 2.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple

try:
    from .config import ROOT, fingerprint
    from .recommend import recommend
except ImportError:
    from config import ROOT, fingerprint
    from recommend import recommend


def check_gate_1_identity_collisions(meta: dict) -> Tuple[bool, str]:
    """Gate 1: Verify zero identity collisions in tracks catalog."""
    tracks = meta.get("tracks", [])
    seen_ids = set()
    collisions = []
    for t in tracks:
        tid = t["id"]
        if tid in seen_ids:
            collisions.append(tid)
        seen_ids.add(tid)

    if collisions:
        return False, f"IDENTITY_FAIL: {len(collisions)} duplicate track IDs: {collisions[:5]}"
    return True, f"PASS (0 collisions across {len(tracks)} tracks)"


def check_gate_2_user_leakage(data: dict) -> Tuple[bool, str]:
    """Gate 2: Verify zero user overlap across train, dev, shadow, and final splits."""
    users_by_split: Dict[str, Set[str]] = {}
    for u in data.get("users", []):
        sp = u.get("split", "unknown")
        users_by_split.setdefault(sp, set()).add(u["id"])

    splits = ["train", "dev", "shadow", "final"]
    overlaps = []
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            inter = users_by_split.get(s1, set()) & users_by_split.get(s2, set())
            if inter:
                overlaps.append(f"{s1} & {s2}: {len(inter)} users")

    if overlaps:
        return False, f"DATA_LEAKAGE: User overlap detected: {'; '.join(overlaps)}"
    total_users = sum(len(s) for s in users_by_split.values())
    return True, f"PASS (0 user leakage across {total_users} users in {len(users_by_split)} splits)"


def check_gate_5_stability(meta: dict, seed: int = 42) -> Tuple[bool, str]:
    sample = [t['id'] for t in meta['tracks'][:20]]
    if len(sample) < 2:
        return False, 'STABILITY_FAIL: insufficient catalog'
    try:
        results = [recommend(lines, seed=seed)['top50']
                   for lines in (sample, list(reversed(sample)), sample + sample)]
        if not results[0] or results[0] != results[1] or results[0] != results[2]:
            return False, 'STABILITY_FAIL: empty or changed ordered output'
        return True, 'PASS: identical ordered tracks and scores'
    except Exception as error:
        return False, f'STABILITY_FAIL: {error}'


def verify_all_release_gates(run_dir: Path, seeds: List[int]) -> bool:
    # Verify existing one-shot evidence; never silently rerun the frozen test.
    try:
        from .config import set_run
        from .training import load_components, checkpoint
    except ImportError:
        from config import set_run
        from training import load_components, checkpoint
    set_run(run_dir)
    try:
        report = json.loads((run_dir / 'reports' / 'release_final.json').read_text())
        if len(set(seeds)) < 3 or report.get('status') != 'PASS' or report.get('split') != 'final':
            raise RuntimeError('No passing final evidence for three seeds')
        for seed in sorted(set(seeds)):
            data, meta, *_ = load_components(seed, require_ranker=True)
            expected = report['checkpoints'][str(seed)]
            if any(fingerprint(checkpoint(stage, seed)) != expected[stage]
                   for stage in ('graph', 'taste', 'ranker', 'shuffled')):
                raise RuntimeError('Checkpoint changed after evaluation')
            if not report['results'][str(seed)]['passed']:
                raise RuntimeError('Seed quality gates failed')
            for check in (check_gate_1_identity_collisions(meta), check_gate_2_user_leakage(data),
                          check_gate_5_stability(meta, seed)):
                if not check[0]:
                    raise RuntimeError(check[1])
        print('RELEASE_PASS: frozen quality evidence, contracts and ordered stability verified')
        return True
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        print(f'RELEASE_FAIL: {error}')
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    default_run = Path(os.environ.get('MUSICMVP_RUN', str(ROOT / 'data' / 'smoke-real')))
    p.add_argument("--run", type=Path, default=default_run)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = p.parse_args()

    success = verify_all_release_gates(args.run.resolve(), args.seeds)
    if not success:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()

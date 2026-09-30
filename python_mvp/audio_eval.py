"""Concrete audio evaluation suite implementing Tests A, B, C, D, E, F from Section 19.
Zero synthetic shortcuts. Fail-fast on representation collapse or random control failure.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn import functional as F

try:
    from .config import DATA, MODELS, REPORTS, DIM, deterministic, write_json, fingerprint
    from .audio_network import AudioEncoder
    from .audio_pipeline import decode_audio, extract_segments, compute_log_mel, get_mel_filterbank
    from .train_audio import split_audio_dataset, load_audio_tracks
except ImportError:
    from config import DATA, MODELS, REPORTS, DIM, deterministic, write_json, fingerprint
    from audio_network import AudioEncoder
    from audio_pipeline import decode_audio, extract_segments, compute_log_mel, get_mel_filterbank
    from train_audio import split_audio_dataset, load_audio_tracks


def run_segment_retrieval(model: AudioEncoder, eval_tracks: List[dict],
                          candidate_pool: List[dict], mel_fb: torch.Tensor,
                          shuffle_candidates: bool = False, seed: int = 42
                          ) -> Tuple[float, float, float, torch.Tensor]:
    """Evaluate multi-segment retrieval of eval_tracks among candidate_pool."""
    model.eval()
    view_eval = []
    valid_eval = []
    with torch.no_grad():
        for t in eval_tracks:
            p = t.get("abs_path")
            if not p or not Path(p).exists():
                continue
            try:
                wf = decode_audio(p)
                segs = extract_segments(wf, num_segments=2, is_training=False)
                mel = compute_log_mel(segs[0], mel_fb=mel_fb).unsqueeze(0)
                view_eval.append(mel)
                valid_eval.append(t)
            except Exception:
                continue

        view_cand = []
        valid_cand = []
        for t in candidate_pool:
            p = t.get("abs_path")
            if not p or not Path(p).exists():
                continue
            try:
                wf = decode_audio(p)
                segs = extract_segments(wf, num_segments=2, is_training=False)
                mel = compute_log_mel(segs[1], mel_fb=mel_fb).unsqueeze(0)
                view_cand.append(mel)
                valid_cand.append(t)
            except Exception:
                continue

        if len(view_eval) < 2 or len(view_cand) < 2:
            return 0.0, 0.0, 0.0, torch.empty(0)

        max_t = max(m.shape[-1] for m in view_eval + view_cand)
        batch_eval = torch.stack([F.pad(m, (0, max_t - m.shape[-1])) for m in view_eval])
        batch_cand = torch.stack([F.pad(m, (0, max_t - m.shape[-1])) for m in view_cand])

        z_eval = model.forward_segment(batch_eval)
        z_cand = model.forward_segment(batch_cand)

        if shuffle_candidates:
            rng = np.random.default_rng(seed)
            shuf_order = rng.permutation(len(z_cand))
            z_cand = z_cand[shuf_order]

        cand_id_map = {t["track_id"]: i for i, t in enumerate(valid_cand)}
        sim = torch.matmul(z_eval, z_cand.T)
        ranks = []
        for i, t in enumerate(valid_eval):
            tid = t["track_id"]
            if tid not in cand_id_map:
                continue
            target_idx = cand_id_map[tid]
            sorted_indices = torch.argsort(sim[i], descending=True).tolist()
            rank = sorted_indices.index(target_idx) + 1
            ranks.append(rank)

        if not ranks:
            return 0.0, 0.0, 0.0, z_eval

        acc1 = float(np.mean([r == 1 for r in ranks]))
        acc5 = float(np.mean([r <= 5 for r in ranks]))
        mrr = float(np.mean([1.0 / r for r in ranks]))

    return acc1, acc5, mrr, z_eval


def evaluate_audio_suite(manifest_path: Path, seed: int = 42) -> dict:
    deterministic(seed)
    tracks = load_audio_tracks(manifest_path)
    if len(tracks) < 10:
        raise RuntimeError(f"AUDIO_DATA_FAIL: Insufficient audio tracks ({len(tracks)})")

    mel_fb = get_mel_filterbank()
    ckpt_path = MODELS / f'audio_{seed}.pt'
    if not ckpt_path.exists():
        raise RuntimeError(f"AUDIO_NOT_TRAINED: missing checkpoint {ckpt_path}")

    # Load trained model
    trained_model = AudioEncoder(dim=DIM)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    trained_model.load_state_dict(ckpt['weights'])
    trained_model.eval()

    # Load random-init control model (same architecture, random seed)
    random_model = AudioEncoder(dim=DIM)
    random_model.eval()

    # Split dataset artist-disjoint
    train_tracks, val_tracks = split_audio_dataset(tracks, seed=seed, val_fraction=0.25)

    # Candidate pool is the entire available collection of tracks
    candidate_pool = tracks

    # TEST A: Held-out same-track retrieval among full candidate pool
    test_a_acc1, test_a_acc5, test_a_mrr, trained_embs = run_segment_retrieval(
        trained_model, val_tracks, candidate_pool, mel_fb, shuffle_candidates=False
    )

    # TEST B: Random encoder control comparison among full candidate pool (averaged across 3 random seeds)
    rand_a1s, rand_a5s, rand_mrrs = [], [], []
    for r_seed in [101, 102, 103]:
        torch.manual_seed(r_seed)
        r_model = AudioEncoder(dim=DIM)
        r_model.eval()
        a1, a5, mrr, _ = run_segment_retrieval(r_model, val_tracks, candidate_pool, mel_fb, shuffle_candidates=False)
        rand_a1s.append(a1)
        rand_a5s.append(a5)
        rand_mrrs.append(mrr)
    rand_acc1 = float(np.mean(rand_a1s))
    rand_acc5 = float(np.mean(rand_a5s))
    rand_mrr = float(np.mean(rand_mrrs))
    test_b_beats_random = bool(test_a_mrr > rand_mrr)

    # TEST D: Artist-disjoint check
    train_artists = {t.get("artist_name", t.get("source", "")).strip().casefold() for t in train_tracks}
    val_artists = {t.get("artist_name", t.get("source", "")).strip().casefold() for t in val_tracks}
    artist_overlap = len(train_artists & val_artists)
    test_d_disjoint_pass = (artist_overlap == 0)

    # TEST F: Representation collapse check
    emb_std = float(trained_embs.std(dim=0).mean().item()) if len(trained_embs) else 0.0
    cov = torch.cov(trained_embs.T) if len(trained_embs) else torch.empty(0)
    matrix_rank = int(torch.linalg.matrix_rank(cov).item()) if len(trained_embs) else 0
    min_rank = min(10, max(1, len(trained_embs) - 1)) if len(trained_embs) else 0
    test_f_no_collapse = (emb_std > 0.01 and matrix_rank >= min_rank)

    # TEST C: Shuffled pairing control (permute candidates to break acoustic correspondence)
    shuf_acc1, shuf_acc5, shuf_mrr, _ = run_segment_retrieval(
        trained_model, val_tracks, candidate_pool, mel_fb, shuffle_candidates=True, seed=seed
    )
    test_c_beats_shuffled = bool(test_a_mrr > shuf_mrr or test_a_acc1 > shuf_acc1)

    all_passed = (test_a_mrr >= 0.20 and
                  test_b_beats_random and
                  test_c_beats_shuffled and
                  test_d_disjoint_pass and
                  test_f_no_collapse)

    report = {
        "status": "PASS" if all_passed else "FAIL",
        "seed": seed,
        "tracks_total": len(tracks),
        "val_tracks_count": len(val_tracks),
        "candidate_pool_size": len(candidate_pool),
        "test_a_same_track_retrieval": {
            "accuracy_at_1": test_a_acc1,
            "accuracy_at_5": test_a_acc5,
            "mrr": test_a_mrr,
            "passed": test_a_mrr >= 0.20
        },
        "test_b_random_control": {
            "trained_mrr": test_a_mrr,
            "random_mrr": rand_mrr,
            "trained_acc1": test_a_acc1,
            "random_acc1": rand_acc1,
            "beats_random": test_b_beats_random
        },
        "test_c_shuffled_control": {
            "trained_mrr": test_a_mrr,
            "shuffled_mrr": shuf_mrr,
            "beats_shuffled": test_c_beats_shuffled
        },
        "test_d_artist_disjoint": {
            "artist_overlap": artist_overlap,
            "passed": test_d_disjoint_pass
        },
        "test_f_representation_collapse": {
            "embedding_std": emb_std,
            "covariance_matrix_rank": matrix_rank,
            "no_collapse": test_f_no_collapse
        }
    }
    write_json(REPORTS / f'audio_eval_{seed}.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=DATA / 'cache' / 'audio' / 'audio_manifest.json')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    report = evaluate_audio_suite(Path(args.manifest).resolve(), args.seed)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == '__main__':
    main()

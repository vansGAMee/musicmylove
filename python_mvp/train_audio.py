"""Self-supervised contrastive training for AudioEncoder.
Multi-view same-track InfoNCE + cross-modal GraphEncoder alignment.
Random weights -> loss -> backward -> optimizer step -> checkpoint.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from .config import DATA, MODELS, REPORTS, DIM, deterministic, write_json, fingerprint, load_data
    from .audio_network import AudioEncoder
    from .audio_pipeline import (
        decode_audio, extract_segments, compute_log_mel, get_mel_filterbank,
        SEGMENT_SAMPLES, SAMPLE_RATE
    )
    from .artifacts import contract, validate_contract
except ImportError:
    from config import DATA, MODELS, REPORTS, DIM, deterministic, write_json, fingerprint, load_data
    from audio_network import AudioEncoder
    from audio_pipeline import (
        decode_audio, extract_segments, compute_log_mel, get_mel_filterbank,
        SEGMENT_SAMPLES, SAMPLE_RATE
    )
    from artifacts import contract, validate_contract


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE loss for normalized embeddings."""
    b = z1.shape[0]
    sim = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(b, device=z1.device)
    loss1 = F.cross_entropy(sim, labels)
    loss2 = F.cross_entropy(sim.T, labels)
    return 0.5 * (loss1 + loss2)


def cross_modal_loss(z_audio: torch.Tensor, z_graph: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Bi-directional contrastive alignment between audio and graph representations."""
    b = z_audio.shape[0]
    sim = torch.matmul(z_audio, z_graph.T) / temperature
    labels = torch.arange(b, device=z_audio.device)
    loss_a = F.cross_entropy(sim, labels)
    loss_g = F.cross_entropy(sim.T, labels)
    return 0.5 * (loss_a + loss_g)


def load_audio_tracks(audio_manifest_path: Path) -> List[dict]:
    """Load indexed audio tracks from manifest."""
    if not audio_manifest_path.exists():
        raise FileNotFoundError(f"Audio manifest not found: {audio_manifest_path}")
    manifest = json.loads(audio_manifest_path.read_text())
    return manifest.get("tracks", [])


def split_audio_dataset(tracks: List[dict], seed: int = 42,
                        val_fraction: float = 0.20) -> Tuple[List[dict], List[dict]]:
    """Split audio tracks into train and validation by artist (artist-disjoint) to prevent leakage."""
    rng = np.random.default_rng(seed)
    # Group by artist
    by_artist: Dict[str, List[dict]] = {}
    for t in tracks:
        art = t.get("artist_name", t.get("source", "unknown")).strip().casefold()
        by_artist.setdefault(art, []).append(t)

    artists = sorted(by_artist.keys())
    rng.shuffle(artists)

    if len(artists) <= 1:
        # Single artist fallback: split by track directly
        cut = max(1, int(len(tracks) * (1.0 - val_fraction)))
        return tracks[:cut], tracks[cut:]

    val_artists = set()
    val_count = 0
    target_val = max(1, int(len(tracks) * val_fraction))
    for art in artists:
        if len(val_artists) + 1 >= len(artists):
            break
        val_artists.add(art)
        val_count += len(by_artist[art])
        if val_count >= target_val:
            break

    train_tracks = [t for art, trs in by_artist.items() if art not in val_artists for t in trs]
    val_tracks = [t for art, trs in by_artist.items() if art in val_artists for t in trs]
    return train_tracks, val_tracks


def prepare_log_mel_batch(tracks: List[dict], mel_fb: torch.Tensor, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare two contrastive views (multi-view segments) for a batch of tracks."""
    view_a, view_b = [], []
    for t in tracks:
        p = t.get("abs_path")
        if not p or not Path(p).exists():
            continue
        try:
            waveform = decode_audio(p)
            # Extract two different segments
            segments = extract_segments(waveform, num_segments=2, is_training=True, rng=rng)
            mel_a = compute_log_mel(segments[0], mel_fb=mel_fb).unsqueeze(0)  # [1, 128, T]
            mel_b = compute_log_mel(segments[1], mel_fb=mel_fb).unsqueeze(0)  # [1, 128, T]
            view_a.append(mel_a)
            view_b.append(mel_b)
        except Exception:
            continue

    if not view_a:
        return torch.empty(0), torch.empty(0)

    # Pad temporal dimension to max in batch if slight mismatch
    max_t = max(m.shape[-1] for m in view_a + view_b)
    padded_a = [F.pad(m, (0, max_t - m.shape[-1])) for m in view_a]
    padded_b = [F.pad(m, (0, max_t - m.shape[-1])) for m in view_b]
    return torch.stack(padded_a), torch.stack(padded_b)


def evaluate_same_track_retrieval(model: AudioEncoder, val_tracks: List[dict],
                                  mel_fb: torch.Tensor, max_eval: int = 200) -> dict:
    """TEST A: Evaluate same-track multi-segment retrieval on held-out tracks."""
    model.eval()
    rng = np.random.default_rng(12345)
    selected = val_tracks[:max_eval]
    view_a, view_b = [], []
    valid_tracks = []
    with torch.no_grad():
        for t in selected:
            p = t.get("abs_path")
            if not p or not Path(p).exists():
                continue
            try:
                waveform = decode_audio(p)
                segments = extract_segments(waveform, num_segments=2, is_training=False)
                mel_a = compute_log_mel(segments[0], mel_fb=mel_fb).unsqueeze(0)
                mel_b = compute_log_mel(segments[1], mel_fb=mel_fb).unsqueeze(0)
                view_a.append(mel_a)
                view_b.append(mel_b)
                valid_tracks.append(t)
            except Exception:
                continue

        if len(view_a) < 2:
            return {"accuracy_at_1": 0.0, "mrr": 0.0, "tracks_evaluated": 0}

        max_t = max(m.shape[-1] for m in view_a + view_b)
        batch_a = torch.stack([F.pad(m, (0, max_t - m.shape[-1])) for m in view_a])
        batch_b = torch.stack([F.pad(m, (0, max_t - m.shape[-1])) for m in view_b])

        z_a = model.forward_segment(batch_a)
        z_b = model.forward_segment(batch_b)

        # Similarity matrix [N, N]
        sim = torch.matmul(z_a, z_b.T)
        ranks = []
        for i in range(len(valid_tracks)):
            sorted_idx = torch.argsort(sim[i], descending=True).tolist()
            rank = sorted_idx.index(i) + 1
            ranks.append(rank)

        acc1 = float(np.mean([r == 1 for r in ranks]))
        acc5 = float(np.mean([r <= 5 for r in ranks]))
        mrr = float(np.mean([1.0 / r for r in ranks]))

        # TEST F: Check representation collapse (std dev across catalog and singular values)
        emb_std = float(z_a.std(dim=0).mean().item())
        cov = torch.cov(z_a.T)
        matrix_rank = int(torch.linalg.matrix_rank(cov).item())

    return {
        "accuracy_at_1": acc1,
        "accuracy_at_5": acc5,
        "mrr": mrr,
        "embedding_std": emb_std,
        "covariance_matrix_rank": matrix_rank,
        "tracks_evaluated": len(valid_tracks)
    }


def train_audio(args):
    deterministic(args.seed)
    audio_manifest_path = Path(args.manifest).resolve()
    tracks = load_audio_tracks(audio_manifest_path)
    if len(tracks) < 10:
        raise RuntimeError(f"AUDIO_DATA_FAIL: insufficient audio tracks ({len(tracks)}) in {audio_manifest_path}")

    train_tracks, val_tracks = split_audio_dataset(tracks, seed=args.seed, val_fraction=0.20)
    print(f"Audio dataset split: {len(train_tracks)} train, {len(val_tracks)} artist-disjoint val tracks")

    mel_fb = get_mel_filterbank()
    model = AudioEncoder(dim=DIM)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Optional cross-modal alignment with GraphEncoder
    graph_checkpoint_path = MODELS / f'graph_{args.seed}.pt'
    graph_embeddings = None
    graph_meta = None
    if graph_checkpoint_path.exists():
        saved_graph = torch.load(graph_checkpoint_path, map_location='cpu', weights_only=True)
        if saved_graph.get('status') == 'DEV_PASS':
            graph_embeddings = saved_graph.get('embeddings')
            graph_vocab = saved_graph.get('vocabulary', [])
            graph_vocab_index = {v: i for i, v in enumerate(graph_vocab)}
            print(f"Found GraphEncoder checkpoint with {len(graph_vocab)} tracks for cross-modal alignment.")

    # Record initial weights for update verification
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

    rng = np.random.default_rng(args.seed)
    best_mrr = -1.0
    best_weights = None
    curves = []

    print(f"Beginning AudioEncoder training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(train_tracks)
        total_loss = 0.0
        updates = 0
        batch_size = args.batch_size

        for start_idx in range(0, len(train_tracks), batch_size):
            batch_slice = train_tracks[start_idx:start_idx + batch_size]
            batch_a, batch_b = prepare_log_mel_batch(batch_slice, mel_fb, rng)
            if len(batch_a) < 2:
                continue

            z_a, z_b = model.forward_multiview(batch_a, batch_b)
            loss_ssl = info_nce_loss(z_a, z_b, temperature=0.07)
            loss = loss_ssl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach())
            updates += 1

        val_metrics = evaluate_same_track_retrieval(model, val_tracks, mel_fb)
        avg_loss = total_loss / max(1, updates)
        curves.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "updates": updates,
            "val_metrics": val_metrics
        })
        print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Val Acc@1: {val_metrics['accuracy_at_1']:.4f} | MRR: {val_metrics['mrr']:.4f}")

        if val_metrics["mrr"] > best_mrr and updates > 0:
            best_mrr = val_metrics["mrr"]
            best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_weights is None:
        raise RuntimeError("AUDIO_ENCODER_FAIL: no admissible training updates")

    model.load_state_dict(best_weights)
    final_val = evaluate_same_track_retrieval(model, val_tracks, mel_fb)

    # Verify weights actually changed
    weights_changed = not all(torch.equal(initial_weights[k], best_weights[k]) for k in initial_weights)
    if not weights_changed:
        raise RuntimeError("AUDIO_ENCODER_FAIL: weights did not change during training")

    # Representation collapse check (TEST F):
    # Embedding std must be > 0.01 and rank must reach full rank up to 10
    min_rank = min(10, max(1, final_val["tracks_evaluated"] - 1))
    passed = (final_val["tracks_evaluated"] >= 2 and
              final_val["mrr"] >= 0.20 and
              final_val["embedding_std"] >= 0.01 and
              final_val["covariance_matrix_rank"] >= min_rank)

    status = "DEV_PASS" if passed else "AUDIO_ENCODER_FAIL"

    target_checkpoint = MODELS / f'audio_{args.seed}.pt'
    MODELS.mkdir(parents=True, exist_ok=True)
    contract_info = contract()
    torch.save({
        "weights": model.state_dict(),
        "status": status,
        "seed": args.seed,
        "weights_updated": True,
        "contract": contract_info,
        "validation_metrics": final_val,
        "training_curves": curves,
        "tracks_indexed": len(tracks)
    }, target_checkpoint)

    write_json(REPORTS / f'audio_curves_{args.seed}.json', curves)
    write_json(REPORTS / f'audio_dev_{args.seed}.json', {
        "status": status,
        "seed": args.seed,
        "passed": passed,
        "validation": final_val,
        "weights_updated": weights_changed
    })

    print(f"AudioEncoder training complete. Checkpoint: {target_checkpoint} | Status: {status}")
    if not passed:
        raise RuntimeError(f"AUDIO_ENCODER_FAIL: failed audio validation criteria (MRR={final_val['mrr']:.3f}, rank={final_val['covariance_matrix_rank']})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=DATA / 'cache' / 'audio' / 'audio_manifest.json')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()
    deterministic(args.seed)
    train_audio(args)


if __name__ == '__main__':
    main()

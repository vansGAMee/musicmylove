#!/usr/bin/env python3
"""
ml/tasteliftnet_embeddings.py
Stage A: Trainable Track Embeddings with InfoNCE Contrastive Loss.
Initializes randomly with Gaussian noise (seed 42), optimizes via AdamW, evaluates on validation split.
Auto-detects GPU (RTX 4060 CUDA) or multi-threaded CPU.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrackEmbeddingModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Proper Gaussian random initialization: Xavier/He equivalent for embedding tables
        # Stddev = sqrt(2 / (V + d))
        std = math.sqrt(2.0 / (vocab_size + embedding_dim))
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=std)
        
    def forward(self, track_indices: torch.Tensor) -> torch.Tensor:
        # Returns L2-normalized embeddings
        emb = self.embeddings(track_indices)
        return F.normalize(emb, p=2, dim=-1)

def build_pairs_from_sessions(sessions: list[list[int]], window_size: int = 5, max_pairs: int = 1500000):
    pairs = []
    for s in sessions:
        if len(s) < 2:
            continue
        n = len(s)
        for i in range(n):
            target = s[i]
            # Context window around target
            left = max(0, i - window_size)
            right = min(n, i + window_size + 1)
            for j in range(left, right):
                if i != j:
                    context = s[j]
                    pairs.append((target, context))
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs

def train_embeddings(
    sessions_path: str,
    output_dir: str = "models",
    embedding_dim: int = 64,
    batch_size: int = 512,
    epochs: int = 5,
    lr: float = 1e-3,
    temperature: float = 0.07,
    seed: int = 42,
    device_str: str = None,
    is_shuffled: bool = False,
    output_name: str = "track_embeddings",
    max_pairs: int = 1500000
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if device.type == "cpu":
        torch.set_num_threads(6)
        print(f"[device] Using CPU with 6 threads")
    else:
        print(f"[device] Using GPU acceleration on {torch.cuda.get_device_name(0)}")

    root = Path(__file__).resolve().parents[1]
    out = root / output_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"[data] Loading sessions from {sessions_path}...")
    data = json.loads((root / sessions_path).read_text())
    
    # Check if split_sessions format (train/val/test) or legacy format
    if "train" in data and "val" in data:
        train_sessions = data["train"]
        val_sessions = data["val"]
        print(f"[data] Detected strict held-out split: Train sessions={len(train_sessions):,}, Val sessions={len(val_sessions):,}")
    else:
        sessions = data.get("sessions", [])
        split_idx = int(len(sessions) * 0.85)
        train_sessions = sessions[:split_idx]
        val_sessions = sessions[split_idx:]
        print(f"[data] Fallback split: Train sessions={len(train_sessions):,}, Val sessions={len(val_sessions):,}")

    if is_shuffled:
        print("[control] Generating SHUFFLED train sessions (preserving marginal track frequency)...")
        rng = random.Random(seed + 999)
        tokens = [tok for s in train_sessions for tok in s]
        rng.shuffle(tokens)
        shuffled_train = []
        cur = 0
        for s in train_sessions:
            L = len(s)
            shuffled_train.append(tokens[cur:cur + L])
            cur += L
        train_sessions = shuffled_train

    # Determine vocab size
    all_sessions_for_vocab = train_sessions + val_sessions
    max_idx = max(max(s) for s in all_sessions_for_vocab if s)
    # Check catalog tracks if available for full vocab size
    catalog_path = root / "data/cache/pipeline/expanded_catalog.json"
    if catalog_path.exists():
        cat_tracks = json.loads(catalog_path.read_text()).get("tracks", [])
        vocab_size = max(len(cat_tracks), max_idx + 1)
    else:
        vocab_size = max_idx + 1
    print(f"[model] Track vocabulary size: {vocab_size:,}, embedding dimension: {embedding_dim}")

    # Generate positive co-occurrence pairs strictly from respective splits
    print("[data] Generating train positive pairs (window=5) strictly from TRAIN sessions...")
    train_pairs = build_pairs_from_sessions(train_sessions, window_size=5, max_pairs=max_pairs)
    random.shuffle(train_pairs)

    print("[data] Generating val positive pairs (window=5) strictly from VAL sessions...")
    val_pairs = build_pairs_from_sessions(val_sessions, window_size=5, max_pairs=100000)
    random.shuffle(val_pairs)

    print(f"[data] Non-leaking dataset: Train pairs={len(train_pairs):,}, Validation pairs={len(val_pairs):,}")

    model = TrackEmbeddingModel(vocab_size, embedding_dim, seed=seed).to(device)
    initial_weights_sample = model.embeddings.weight.data[:5, :5].clone().cpu().numpy().tolist()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    def evaluate_loss(pairs_subset: list[tuple[int, int]], eval_batch_size: int = 1024) -> float:
        model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for i in range(0, min(len(pairs_subset), 30000), eval_batch_size):
                batch = pairs_subset[i:i + eval_batch_size]
                if len(batch) < 8:
                    continue
                targets = torch.tensor([p[0] for p in batch], dtype=torch.long, device=device)
                contexts = torch.tensor([p[1] for p in batch], dtype=torch.long, device=device)

                target_emb = model(targets)    # [B, d]
                context_emb = model(contexts)  # [B, d]

                sim = torch.matmul(target_emb, context_emb.T) / temperature
                labels = torch.arange(len(batch), dtype=torch.long, device=device)
                loss = F.cross_entropy(sim, labels)
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(1, n_batches)

    # Initial loss before any backprop
    initial_val_loss = evaluate_loss(val_pairs)
    print(f"\n[init] Initial Validation Loss (Random Weights before training): {initial_val_loss:.4f}")

    training_curve = []
    validation_curve = []
    best_val_loss = float("inf")
    best_weights = None

    print("\n[training] Commencing AdamW backpropagation...")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_pairs)
        epoch_loss = 0.0
        batches = 0

        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i:i + batch_size]
            if len(batch) < 16:
                continue

            targets = torch.tensor([p[0] for p in batch], dtype=torch.long, device=device)
            contexts = torch.tensor([p[1] for p in batch], dtype=torch.long, device=device)

            optimizer.zero_grad()
            target_emb = model(targets)
            context_emb = model(contexts)

            sim = torch.matmul(target_emb, context_emb.T) / temperature
            labels = torch.arange(len(batch), dtype=torch.long, device=device)

            loss = F.cross_entropy(sim, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

            if batches % 250 == 0:
                print(f"  Epoch {epoch}/{epochs} | Batch {batches} | Step Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / max(1, batches)
        val_loss = evaluate_loss(val_pairs)
        training_curve.append(avg_train_loss)
        validation_curve.append(val_loss)

        print(f"[epoch {epoch}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.embeddings.weight.data.clone().cpu()
            torch.save(model.state_dict(), out / "tasteliftnet_embeddings_best.pt")

    dt = time.time() - t_start
    print(f"\n[training] Completed {epochs} epochs in {dt:.2f}s! Best Validation Loss: {best_val_loss:.4f}")

    # Restore best weights
    model.embeddings.weight.data.copy_(best_weights.to(device))
    final_weights_sample = model.embeddings.weight.data[:5, :5].clone().cpu().numpy().tolist()

    # Save learned embeddings as numpy array and JSON metadata
    all_embeddings = F.normalize(model.embeddings.weight.data, p=2, dim=-1).cpu().numpy().astype(np.float32)
    np.save(out / f"{output_name}.npy", all_embeddings)

    report = {
        "stage": f"Stage A: Learned Track Embeddings ({'Shuffled Control' if is_shuffled else 'Real Human'})",
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "seed": seed,
        "is_shuffled": is_shuffled,
        "epochs": epochs,
        "device": str(device),
        "runtime_sec": round(dt, 2),
        "initial_loss": round(initial_val_loss, 4),
        "best_val_loss": round(best_val_loss, 4),
        "training_loss_curve": [round(x, 4) for x in training_curve],
        "validation_loss_curve": [round(x, 4) for x in validation_curve],
        "weights_sample_before": initial_weights_sample,
        "weights_sample_after": final_weights_sample,
        "weights_updated_verified": initial_weights_sample != final_weights_sample,
    }
    report_name = f"{output_name}_report.json" if is_shuffled else "embeddings_report.json"
    (out / report_name).write_text(json.dumps(report, indent=2))
    print(f"[output] Saved learned embeddings to {out / (output_name + '.npy')} and report to {out / report_name}")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", default="data/cache/pipeline/split_sessions.json")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--pairs", type=int, default=1500000)
    parser.add_argument("--shuffled", action="store_true", help="Train shuffled control model")
    args = parser.parse_args()
    
    # Check if split_sessions exists, otherwise fallback to expanded_sessions
    sess_path = args.sessions
    if not (Path(__file__).resolve().parents[1] / sess_path).exists():
        sess_path = "data/cache/pipeline/expanded_sessions.json"
        
    out_name = "shuffled_track_embeddings" if args.shuffled else "track_embeddings"
    train_embeddings(
        sessions_path=sess_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.dim,
        max_pairs=args.pairs,
        is_shuffled=args.shuffled,
        output_name=out_name
    )

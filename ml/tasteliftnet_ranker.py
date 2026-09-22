#!/usr/bin/env python3
"""
ml/tasteliftnet_ranker.py
Stage B: TasteLiftNet Multi-Head Ranker with Deep Audio Feature Integration.
Trained via Pairwise Ranking Loss on true session continuations vs negative candidates.
Zero label leakage. Deeply integrates Essentia 65-dim audio measurements (MFCC, rhythm, spectral).
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

class MultiHeadTasteAttention(nn.Module):
    def __init__(self, embedding_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.queries = nn.Parameter(torch.randn(num_heads, embedding_dim) / math.sqrt(embedding_dim))

    def forward(self, seed_embeddings: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, M, d = seed_embeddings.shape
        scores = torch.einsum("kd, bmd -> bkm", self.queries, seed_embeddings) / math.sqrt(d)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        heads = torch.einsum("bkm, bmd -> bkd", attn_weights, seed_embeddings)
        heads = F.normalize(heads, p=2, dim=-1)
        return heads, attn_weights

class AudioProjection(nn.Module):
    def __init__(self, input_dim: int = 65, proj_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, proj_dim),
        )
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        proj = self.net(audio_features)
        return F.normalize(proj, p=2, dim=-1)

class TasteLiftNetRanker(nn.Module):
    def __init__(self, embedding_dim: int = 64, num_heads: int = 4, audio_dim: int = 65, audio_proj_dim: int = 16):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.attention = MultiHeadTasteAttention(embedding_dim, num_heads)
        self.audio_proj = AudioProjection(audio_dim, audio_proj_dim)

        # Interactions: 4 head dots + 1 max dot + 1 acoustic similarity = 6
        interaction_dim = num_heads + 1 + 1
        # Evidence: popularity + session_count = 2 (strictly valid catalog metadata, 0 leakage)
        evidence_dim = 2
        # Candidate embedding: 64
        # Audio proj: 16
        # Audio mask: 1
        input_dim = embedding_dim + interaction_dim + audio_proj_dim + 1 + evidence_dim  # 64 + 6 + 16 + 1 + 2 = 89

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(
        self,
        seed_embeddings: torch.Tensor,       # [B, M, d]
        candidate_embeddings: torch.Tensor,   # [B, d]
        candidate_audio: torch.Tensor,        # [B, audio_dim]
        candidate_audio_mask: torch.Tensor,   # [B, 1]
        seed_audio_proj: torch.Tensor,        # [B, audio_proj_dim]
        evidence_features: torch.Tensor,      # [B, 2]
        seed_mask: torch.Tensor = None        # [B, M]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Multi-head taste encoding
        heads, attn = self.attention(seed_embeddings, seed_mask) # [B, K, d]

        # 2. Candidate-head interactions: [B, K]
        c = candidate_embeddings.unsqueeze(1)
        dots = torch.sum(heads * c, dim=-1) # [B, K]
        max_dot, _ = torch.max(dots, dim=-1, keepdim=True) # [B, 1]

        # 3. Audio projection & Acoustic similarity
        cand_audio_proj = self.audio_proj(candidate_audio) * candidate_audio_mask # [B, audio_proj_dim]
        acoustic_sim = torch.sum(cand_audio_proj * seed_audio_proj, dim=-1, keepdim=True) * candidate_audio_mask # [B, 1]

        # 4. Feature Fusion
        interactions = torch.cat([dots, max_dot, acoustic_sim], dim=-1) # [B, K + 2]
        fused = torch.cat([
            candidate_embeddings,
            interactions,
            cand_audio_proj,
            candidate_audio_mask,
            evidence_features
        ], dim=-1) # [B, 89]

        # 5. Scoring MLP
        logits = self.mlp(fused).squeeze(-1) # [B]
        return logits, heads

def build_ranker_dataset(
    sessions: list[list[int]],
    catalog_tracks: list[dict],
    max_examples: int = 100000,
    seed: int = 42
):
    random.seed(seed)
    vocab_size = len(catalog_tracks)
    examples = []

    for s in sessions:
        if len(s) < 3:
            continue
        cut = random.randint(1, min(len(s) - 1, 10))
        seeds = s[:cut]
        pos_candidates = s[cut:]

        for pos in pos_candidates:
            neg_random = random.randint(0, vocab_size - 1)
            while neg_random in s:
                neg_random = random.randint(0, vocab_size - 1)

            examples.append({
                "seeds": seeds,
                "pos": pos,
                "neg": neg_random
            })
            if len(examples) >= max_examples:
                break
        if len(examples) >= max_examples:
            break

    return examples

def train_ranker(
    catalog_path: str = "data/cache/pipeline/expanded_catalog.json",
    sessions_path: str = "data/cache/pipeline/expanded_sessions.json",
    embeddings_path: str = "models/track_embeddings.npy",
    audio_path: str = "data/cache/audio-vectors.json",
    output_dir: str = "models",
    epochs: int = 4,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    device_str: str = None
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
        print(f"[device] Using GPU: {torch.cuda.get_device_name(0)}")

    root = Path(__file__).resolve().parents[1]
    out = root / output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("[data] Loading catalog, embeddings, and sessions...")
    cat_data = json.loads((root / catalog_path).read_text())
    tracks = cat_data.get("tracks", [])
    sessions = json.loads((root / sessions_path).read_text()).get("sessions", [])
    embeddings_np = np.load(root / embeddings_path)
    embeddings_tensor = torch.tensor(embeddings_np, dtype=torch.float32, device=device)

    audio_map = {}
    audio_file = root / audio_path
    if audio_file.exists():
        a_data = json.loads(audio_file.read_text())
        ids = a_data.get("ids", [])
        vecs = a_data.get("vectors", [])
        for mid, vec in zip(ids, vecs):
            if len(vec) == 65:
                audio_map[mid] = vec
    print(f"[data] Audio feature matches: {len(audio_map):,} (65-dim Essentia acoustic measurements)")

    raw_examples = build_ranker_dataset(sessions, tracks, max_examples=80000, seed=seed)
    random.shuffle(raw_examples)
    split_idx = int(len(raw_examples) * 0.85)
    train_ex = raw_examples[:split_idx]
    val_ex = raw_examples[split_idx:]
    print(f"[data] Train examples: {len(train_ex):,}, Validation examples: {len(val_ex):,}")

    model = TasteLiftNetRanker(embedding_dim=64, num_heads=4, audio_dim=65, audio_proj_dim=16).to(device)
    initial_mlp_weight = model.mlp[0].weight.data[:3, :3].clone().cpu().numpy().tolist()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def prepare_batch(batch_items, is_positive: bool):
        B = len(batch_items)
        max_seeds = max(len(x["seeds"]) for x in batch_items)
        max_seeds = min(max_seeds, 15)

        seed_tensor = torch.zeros(B, max_seeds, 64, device=device)
        seed_mask = torch.zeros(B, max_seeds, dtype=torch.bool, device=device)
        seed_audio_list = []
        cand_indices = []
        audio_feats = []
        audio_masks = []
        evidence_feats = []

        for b, item in enumerate(batch_items):
            s_list = item["seeds"][:max_seeds]
            s_audio_vecs = []
            for s_idx, track_id in enumerate(s_list):
                if track_id < len(embeddings_tensor):
                    seed_tensor[b, s_idx] = embeddings_tensor[track_id]
                    seed_mask[b, s_idx] = True
                    t_info = tracks[track_id] if track_id < len(tracks) else {}
                    mbid = t_info.get("mbid", "")
                    if mbid in audio_map:
                        s_audio_vecs.append(audio_map[mbid])

            if s_audio_vecs:
                mean_audio = np.mean(s_audio_vecs, axis=0).tolist()
                seed_audio_list.append(mean_audio)
            else:
                seed_audio_list.append([0.0] * 65)

            c_id = item["pos"] if is_positive else item["neg"]
            cand_indices.append(c_id)

            t_info = tracks[c_id] if c_id < len(tracks) else {}
            mbid = t_info.get("mbid", "")
            if mbid in audio_map:
                audio_feats.append(audio_map[mbid])
                audio_masks.append([1.0])
            else:
                audio_feats.append([0.0] * 65)
                audio_masks.append([0.0])

            pop = t_info.get("popularity", 0.1)
            sess = min(1.0, math.log1p(t_info.get("sessions", 1)) / 10.0)
            evidence_feats.append([pop, sess])

        cand_tensor = embeddings_tensor[cand_indices]
        audio_tensor = torch.tensor(audio_feats, dtype=torch.float32, device=device)
        audio_mask_tensor = torch.tensor(audio_masks, dtype=torch.float32, device=device)
        seed_audio_tensor = torch.tensor(seed_audio_list, dtype=torch.float32, device=device)
        seed_audio_proj = model.audio_proj(seed_audio_tensor)
        evidence_tensor = torch.tensor(evidence_feats, dtype=torch.float32, device=device)

        return seed_tensor, cand_tensor, audio_tensor, audio_mask_tensor, seed_audio_proj, evidence_tensor, seed_mask

    def eval_loss(ex_subset):
        model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for i in range(0, min(len(ex_subset), 10000), batch_size):
                b_items = ex_subset[i:i + batch_size]
                if len(b_items) < 8:
                    continue
                s_t, c_pos, a_t, a_m, s_a_p, ev_pos, s_m = prepare_batch(b_items, is_positive=True)
                pos_logits, heads = model(s_t, c_pos, a_t, a_m, s_a_p, ev_pos, s_m)

                _, c_neg, a_t_n, a_m_n, s_a_p_n, ev_neg, _ = prepare_batch(b_items, is_positive=False)
                neg_logits, _ = model(s_t, c_neg, a_t_n, a_m_n, s_a_p_n, ev_neg, s_m)

                # Pairwise Ranking BCE loss: -log(sigmoid(s_pos - s_neg))
                diff = pos_logits - neg_logits
                loss = F.binary_cross_entropy_with_logits(diff, torch.ones_like(diff))
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(1, n_batches)

    initial_val_loss = eval_loss(val_ex)
    print(f"\n[init] Initial Ranker Pairwise Validation Loss: {initial_val_loss:.4f}")

    training_curve = []
    validation_curve = []
    best_val_loss = float("inf")

    t_start = time.time()
    print("\n[training] Commencing Pairwise Ranker backpropagation...")

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_ex)
        epoch_loss = 0.0
        batches = 0

        for i in range(0, len(train_ex), batch_size):
            b_items = train_ex[i:i + batch_size]
            if len(b_items) < 8:
                continue

            optimizer.zero_grad()
            s_t, c_pos, a_t, a_m, s_a_p, ev_pos, s_m = prepare_batch(b_items, is_positive=True)
            pos_logits, heads = model(s_t, c_pos, a_t, a_m, s_a_p, ev_pos, s_m)

            _, c_neg, a_t_n, a_m_n, s_a_p_n, ev_neg, _ = prepare_batch(b_items, is_positive=False)
            neg_logits, _ = model(s_t, c_neg, a_t_n, a_m_n, s_a_p_n, ev_neg, s_m)

            # Head diversity regularization
            H = F.normalize(heads, dim=-1)
            H_gram = torch.matmul(H, H.transpose(1, 2))
            diag = torch.eye(4, device=device).unsqueeze(0)
            ortho_reg = torch.mean((H_gram - diag) ** 2)

            diff = pos_logits - neg_logits
            loss = F.binary_cross_entropy_with_logits(diff, torch.ones_like(diff)) + 0.05 * ortho_reg

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

            if batches % 100 == 0:
                print(f"  Epoch {epoch}/{epochs} | Batch {batches} | Step Loss: {loss.item():.4f}")

        avg_train = epoch_loss / max(1, batches)
        val_loss = eval_loss(val_ex)
        training_curve.append(avg_train)
        validation_curve.append(val_loss)

        print(f"[epoch {epoch}/{epochs}] Train Loss: {avg_train:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out / "tasteliftnet_ranker_best.pt")

    dt = time.time() - t_start
    print(f"\n[training] Ranker training finished in {dt:.2f}s! Best Validation Loss: {best_val_loss:.4f}")

    model.load_state_dict(torch.load(out / "tasteliftnet_ranker_best.pt"))
    final_mlp_weight = model.mlp[0].weight.data[:3, :3].clone().cpu().numpy().tolist()

    export_weights = {
        "queries": model.attention.queries.data.cpu().numpy().tolist(),
        "audio_proj_0_w": model.audio_proj.net[0].weight.data.cpu().numpy().tolist(),
        "audio_proj_0_b": model.audio_proj.net[0].bias.data.cpu().numpy().tolist(),
        "audio_proj_2_w": model.audio_proj.net[2].weight.data.cpu().numpy().tolist(),
        "audio_proj_2_b": model.audio_proj.net[2].bias.data.cpu().numpy().tolist(),
        "mlp_0_w": model.mlp[0].weight.data.cpu().numpy().tolist(),
        "mlp_0_b": model.mlp[0].bias.data.cpu().numpy().tolist(),
        "mlp_2_w": model.mlp[2].weight.data.cpu().numpy().tolist(),
        "mlp_2_b": model.mlp[2].bias.data.cpu().numpy().tolist(),
        "mlp_4_w": model.mlp[4].weight.data.cpu().numpy().tolist(),
        "mlp_4_b": model.mlp[4].bias.data.cpu().numpy().tolist(),
    }
    (out / "tasteliftnet_weights.json").write_text(json.dumps(export_weights))

    report = {
        "stage": "Stage B: TasteLiftNet Ranker (Pairwise Ranking + Multi-Head Attention + Deep Audio Projection)",
        "num_heads": 4,
        "embedding_dim": 64,
        "audio_dim": 65,
        "epochs": epochs,
        "device": str(device),
        "runtime_sec": round(dt, 2),
        "initial_loss": round(initial_val_loss, 4),
        "best_val_loss": round(best_val_loss, 4),
        "training_loss_curve": [round(x, 4) for x in training_curve],
        "validation_loss_curve": [round(x, 4) for x in validation_curve],
        "weights_sample_before": initial_mlp_weight,
        "weights_sample_after": final_mlp_weight,
        "weights_updated_verified": initial_mlp_weight != final_mlp_weight,
    }
    (out / "ranker_report.json").write_text(json.dumps(report, indent=2))
    print("[output] Saved ranker weights and report to", out)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    train_ranker(epochs=args.epochs, batch_size=args.batch_size)

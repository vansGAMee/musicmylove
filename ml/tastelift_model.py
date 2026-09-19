"""Four learned query heads over an unordered track set; JSON-only deployment.

Training requires PyTorch. Inference math and UTF-8 hashes are exported so the
web application never needs Python, a database, pretrained models, or audio.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import torch
from torch import nn
from torch.nn import functional as F

from ml.tastelift_data import hashed_subword_ids, normalize_text, _atomic_json_write


def bpr_loss(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    return F.softplus(negative - positive).mean()


class TasteLift(nn.Module):
    """Track vectors feed four full-set attention pools, with no positions.

    Track = L2(tanh(W [artist; title; mean(artist grams); mean(title grams)] + b)).
    Heads = L2(softmax(L2(query) dot track / .2) @ tracks).
    Affinity = .1 * logsumexp(softplus(scale) * head dot candidate / .1).
    Lift = affinity - softplus(popularity_weight) * popularity_percentile.
    """

    def __init__(self, vocabulary, tracks, dim=24, buckets=8192, seed=41):
        super().__init__()
        if dim < 4 or buckets < 1:
            raise ValueError("dimension must support four heads and buckets must be positive")
        self.vocabulary = vocabulary
        self.dim, self.buckets, self.seed = dim, buckets, seed
        self.attention_temperature, self.affinity_temperature = 0.2, 0.1
        self.normalization_epsilon = 1e-12
        # fork_rng keeps scratch initialization reproducible without resetting
        # the trainer's dropout/shuffle RNG when reconstructing a checkpoint.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.artist_embedding = nn.Embedding(len(vocabulary["artist"]) + 1, dim, padding_idx=0)
            self.title_embedding = nn.Embedding(len(vocabulary["title"]) + 1, dim, padding_idx=0)
            self.subword_embedding = nn.Embedding(buckets, dim)
            self.track_projection = nn.Linear(dim * 4, dim)
            self.queries = nn.Parameter(torch.empty(4, dim))
            nn.init.orthogonal_(self.queries)
            self.log_score_scale = nn.Parameter(torch.tensor(math.log(math.expm1(4.0))))
            self.log_popularity_weight = nn.Parameter(torch.tensor(math.log(math.expm1(0.3))))
        self.tracks = tracks
        self.track_index = {track["id"]: i for i, track in enumerate(tracks)}
        self._features = [self.metadata_features(track) for track in tracks]
        self.register_buffer("popularity", torch.tensor([track.get("popularity", {}).get("percentile", 0.5) for track in tracks]), persistent=False)
        self.training_summary = {}

    @classmethod
    def from_dataset(cls, dataset, dim=24, buckets=8192, seed=41):
        # Limit exact identities to the training partition, ranked by exposure.
        # Validation/test metadata can only take the OOV subword path unless an
        # identity independently occurred in training; no frozen split tuning.
        counts = Counter(track_id for row in dataset["episodes"] if row["partition"] == "train"
                         for track_id in row["seed_ids"] + [row["positive_id"], row["negative_id"]])
        vocabulary = {}
        for field, cap in (("artist", 4096), ("title", 8192)):
            frequency = Counter()
            for track in dataset["tracks"]:
                if counts[track["id"]]:
                    frequency[normalize_text(track[field])] += counts[track["id"]]
            chosen = sorted(frequency, key=lambda text: (-frequency[text], text))[:cap]
            vocabulary[field] = {text: i + 1 for i, text in enumerate(sorted(chosen))}
        return cls(vocabulary, dataset["tracks"], dim, buckets, seed)

    def metadata_features(self, track):
        artist, title = normalize_text(track["artist"]), normalize_text(track["title"])
        def grams(field, value):
            cached = track.get(f"{field}_subword_ids")
            # Dataset FNV buckets are divisible by model buckets by default.
            if cached is not None and 65536 % self.buckets == 0:
                return sorted({int(index) % self.buckets for index in cached})
            return hashed_subword_ids(value, buckets=self.buckets)
        return (self.vocabulary["artist"].get(artist, 0), self.vocabulary["title"].get(title, 0),
                grams("artist", artist), grams("title", title))

    def _encode_features(self, features):
        device = self.queries.device
        artist = torch.tensor([row[0] for row in features], device=device)
        title = torch.tensor([row[1] for row in features], device=device)
        if self.training:
            # Train the same subword fallback used by novel live metadata.
            artist = artist * (torch.rand(artist.shape, device=device) >= 0.15)
            title = title * (torch.rand(title.shape, device=device) >= 0.15)
        parts = [self.artist_embedding(artist), self.title_embedding(title)]
        for field in (2, 3):
            flattened, offsets = [], [0]
            for row in features:
                flattened.extend(row[field])
                offsets.append(len(flattened))
            parts.append(F.embedding_bag(torch.tensor(flattened, dtype=torch.long, device=device),
                         self.subword_embedding.weight, torch.tensor(offsets, device=device),
                         mode="mean", include_last_offset=True))
        return F.normalize(torch.tanh(self.track_projection(torch.cat(parts, dim=-1))), dim=-1, eps=self.normalization_epsilon)

    def encode_metadata(self, tracks):
        return self._encode_features([self.metadata_features(track) for track in tracks])

    def encode_tracks(self, track_ids):
        unique, inverse = torch.unique(track_ids, sorted=True, return_inverse=True)
        vectors = self._encode_features([self._features[index] for index in unique.tolist()])
        return vectors[inverse]

    def encode_set(self, track_ids, mask):
        if track_ids.ndim != 2 or mask.shape != track_ids.shape:
            raise ValueError("track ids and mask must have matching [batch, seeds] shape")
        counts = mask.sum(dim=1)
        # Serving retains every accepted Task 1 seed (5--500); training episodes
        # remain independently limited to 5--30 in tastelift_data.py.
        if torch.any(counts < 5) or torch.any(counts > 500):
            raise ValueError("each seed set must contain 5 to 500 tracks")
        # Canonical summation order gives exact permutation invariance, even
        # for floating-point addition; masked garbage never reaches embeddings.
        ordered = torch.sort(track_ids.masked_fill(~mask.bool(), len(self.tracks)), dim=1).values
        ordered = ordered[:, :int(counts.max())]
        active = ordered != len(self.tracks)
        vectors = self.encode_tracks(ordered.clamp(max=len(self.tracks) - 1))
        logits = torch.einsum("hd,bsd->bhs", F.normalize(self.queries, dim=-1), vectors) / self.attention_temperature
        attention = torch.softmax(logits.masked_fill(~active[:, None, :], -torch.inf), dim=-1)
        return F.normalize(attention @ vectors, dim=-1, eps=self.normalization_epsilon)

    def encode_seed_vectors(self, track_ids, mask):
        """Return canonically ordered seed vectors and their active mask."""
        if track_ids.ndim != 2 or mask.shape != track_ids.shape:
            raise ValueError("track ids and mask must have matching [batch, seeds] shape")
        counts = mask.sum(dim=1)
        if torch.any(counts < 5) or torch.any(counts > 500):
            raise ValueError("each seed set must contain 5 to 500 tracks")
        ordered = torch.sort(track_ids.masked_fill(~mask.bool(), len(self.tracks)), dim=1).values
        ordered = ordered[:, :int(counts.max())]
        active = ordered != len(self.tracks)
        return self.encode_tracks(ordered.clamp(max=len(self.tracks) - 1)), active, ordered

    def score_embeddings(self, heads, candidates, popularity):
        per_head = torch.einsum("bhd,bcd->bch", heads, candidates) * F.softplus(self.log_score_scale)
        affinity = self.affinity_temperature * torch.logsumexp(per_head / self.affinity_temperature, dim=-1)
        prior = F.softplus(self.log_popularity_weight) * popularity
        return {"per_head": per_head, "affinity": affinity, "popularity_prior": prior, "lift": affinity - prior}

    def score_target_embeddings(self, seed_vectors, seed_mask, candidates):
        """Score each candidate against its learned top-eight seed context."""
        similarities = torch.einsum("bsd,bcd->bcs", seed_vectors, candidates)
        similarities = similarities.masked_fill(~seed_mask[:, None, :], -torch.inf)
        context_size = min(8, similarities.shape[-1])
        top, indices = torch.topk(similarities, k=context_size, dim=-1, largest=True, sorted=True)
        attention = torch.softmax(top / 0.12, dim=-1)
        relevance = (torch.where(torch.isfinite(top), top, torch.zeros_like(top)) * attention).sum(dim=-1) * F.softplus(self.log_score_scale)
        return {"relevance": relevance, "context_seed_indices": indices}

    def score_candidates(self, track_ids, mask, candidate_ids):
        heads = self.encode_set(track_ids, mask)
        seed_vectors, seed_mask, ordered = self.encode_seed_vectors(track_ids, mask)
        candidates = self.encode_tracks(candidate_ids)
        diagnostics = self.score_embeddings(heads, candidates, self.popularity[candidate_ids])
        target = self.score_target_embeddings(seed_vectors, seed_mask, candidates)
        canonical_indices = ordered[:, None, :].expand(-1, candidate_ids.shape[1], -1).gather(2, target["context_seed_indices"])
        return {**diagnostics, "affinity": target["relevance"], "popularity_prior": torch.zeros_like(target["relevance"]),
                "lift": target["relevance"], "context_seed_indices": canonical_indices}

    def diversity_loss(self, heads):
        eye = torch.eye(4, device=heads.device)
        queries = F.normalize(self.queries, dim=-1)
        return ((queries @ queries.T - eye) ** 2).mean() + ((heads @ heads.transpose(-1, -2) - eye) ** 2).mean()

    def loss(self, seeds, mask, positive, negative, diversity_weight=0.02):
        heads = self.encode_set(seeds, mask)
        seed_vectors, seed_mask, _ = self.encode_seed_vectors(seeds, mask)
        if negative.ndim == 1:
            candidates = torch.stack([positive, negative], dim=1)
            scores = self.score_target_embeddings(seed_vectors, seed_mask, self.encode_tracks(candidates))["relevance"]
            bpr = bpr_loss(scores[:, 0], scores[:, 1])
        else:
            candidates = torch.cat([positive.unsqueeze(1), negative], dim=1)
            scores = self.score_target_embeddings(seed_vectors, seed_mask, self.encode_tracks(candidates))["relevance"]
            pos_scores = scores[:, :1]
            neg_scores = scores[:, 1:]
            bpr = F.softplus(neg_scores - pos_scores).mean()
        return bpr + diversity_weight * self.diversity_loss(heads)

    def export_json(self, path):
        payload = {
            "format": "tastelift-v1", "architecture": {"dim": self.dim, "heads": 4,
                "attention_temperature": self.attention_temperature, "affinity_temperature": self.affinity_temperature,
                "normalization_epsilon": self.normalization_epsilon, "pooling": "four-query-attention",
                "track_activation": "tanh", "affinity": "temperature-logsumexp", "seed": self.seed},
            "hashing": {"normalization": "NFKC-lower-whitespace", "encoding": "UTF-8",
                "algorithm": "FNV-1a-32", "offset_basis": 2166136261, "prime": 16777619,
                "buckets": self.buckets, "char_ngram_min": 3, "char_ngram_max": 5,
                "padding": ["^", "$"], "deduplicate": True, "sort": "numeric-ascending", "unicode_units": "codepoints"},
            "vocabulary": self.vocabulary,
            "weights": {name: value.detach().cpu().tolist() for name, value in self.state_dict().items()},
            "popularity": {track["id"]: float(track.get("popularity", {}).get("percentile", 0.0))
                           for track in self.tracks if track.get("popularity", {}).get("percentile", 0.0) > 0},
            "training": self.training_summary,
        }
        _atomic_json_write(Path(path), payload)

    @classmethod
    def from_export(cls, path, tracks):
        payload = json.loads(Path(path).read_text())
        if payload.get("format") != "tastelift-v1":
            raise ValueError("unsupported TasteLift export")
        arch = payload["architecture"]
        model = cls(payload["vocabulary"], tracks, arch["dim"], payload["hashing"]["buckets"], arch["seed"])
        model.load_state_dict({name: torch.tensor(value) for name, value in payload["weights"].items()})
        model.training_summary = payload["training"]
        model.eval()
        return model


def _batch(model, rows, device):
    seeds = torch.zeros((len(rows), max(len(row["seed_ids"]) for row in rows)), dtype=torch.long, device=device)
    mask = torch.zeros_like(seeds, dtype=torch.bool)
    for i, row in enumerate(rows):
        ids = [model.track_index[track] for track in row["seed_ids"]]
        seeds[i, :len(ids)] = torch.tensor(ids, device=device)
        mask[i, :len(ids)] = True
    positive = torch.tensor([model.track_index[row["positive_id"]] for row in rows], device=device)
    negative = torch.tensor([model.track_index[row["negative_id"]] for row in rows], device=device)
    return seeds, mask, positive, negative


def _sample_candidates_32(model, batch, tracks_by_band, seed, epoch, offset, device):
    candidates = torch.zeros((len(batch), 32), dtype=torch.long, device=device)
    for i, row in enumerate(batch):
        band = row.get("positive_band")
        if band is None:
            band = int(model.popularity[model.track_index[row["positive_id"]]].item() * 10)
        band = min(9, max(0, int(band)))
        history_ids = set(row.get("user_history_ids", [])) | set(row.get("seed_ids", [])) | {row.get("positive_id")}
        history_indices = {model.track_index[tid] for tid in history_ids if tid in model.track_index}
        pool = [idx for idx in tracks_by_band.get(band, []) if idx not in history_indices]
        if not pool:
            pool = [idx for idx in range(len(model.tracks)) if idx not in history_indices]
        if not pool:
            pool = [model.track_index.get(row.get("negative_id", row["positive_id"]), 0)]
        rng = random.Random(f"{seed}:{epoch}:{offset}:{i}")
        if len(pool) >= 32:
            sampled = rng.sample(pool, 32)
        else:
            sampled = [pool[rng.randrange(len(pool))] for _ in range(32)]
        candidates[i] = torch.tensor(sampled, dtype=torch.long, device=device)
    return candidates


def _save_checkpoint(path, payload):
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _execution_provenance(require_kitty, environment):
    environment = os.environ if environment is None else environment
    window_id = environment.get("KITTY_WINDOW_ID")
    if require_kitty and not window_id:
        raise ValueError("--require-kitty must run inside a Kitty window")
    return {"runner": "kitty" if window_id else "unverified",
            "kitty_window_id": window_id, "kitty_pid": environment.get("KITTY_PID")}


def train_model(dataset, epochs=40, checkpoint_dir=Path("data/cache/tastelift/checkpoints"),
                batch_size=64, seed=41, resume=False, device=None, max_train_rows=None, learning_rate=0.002,
                patience=8, require_kitty=False, execution_environment=None):
    """Scratch by default; deterministic epoch-boundary resume with optimizer/RNG.

    Only train and validation episodes are consumed. Selection uses validation
    BPR; test episodes never enter gradients, stopping, or hyperparameters.
    """
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    execution = _execution_provenance(require_kitty, execution_environment)
    model = TasteLift.from_dataset(dataset, seed=seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    eligible = lambda row, partition: (row["partition"] == partition
        and row["positive_band"] == row["negative_band"] and row.get("negative_source") == "retrieval")
    rows = [row for row in dataset["episodes"] if eligible(row, "train")]
    validation = [row for row in dataset["episodes"] if eligible(row, "validation")]
    if max_train_rows:
        rows, validation = rows[:max_train_rows], validation[:max_train_rows]
    if not rows:
        raise ValueError("no popularity-matched retrieval hard-negative training pairs")
    # Include metadata and every consumed episode to prevent incompatible resume.
    signature = hashlib.sha256(json.dumps({"tracks": dataset["tracks"], "train": rows,
        "validation": validation}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    config = {"seed": seed, "batch_size": batch_size, "learning_rate": learning_rate,
              "device": device, "threads": torch.get_num_threads(), "signature": signature,
              "patience": patience, "negative_source": "retrieval"}
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    last_path, best_path = directory / "last.pt", directory / "best.pt"
    start, history, best = 0, [], float("inf")
    if resume:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["config"] != config:
            raise ValueError("checkpoint dataset or training configuration mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start, history, best = checkpoint["epoch"], checkpoint["history"], checkpoint["best"]
    tracks_by_band: dict[int, list[int]] = {}
    for i, track in enumerate(model.tracks):
        band = track.get("popularity", {}).get("band")
        if band is None:
            pct = track.get("popularity", {}).get("percentile", 0.5)
            band = min(9, max(0, int(pct * 10)))
        tracks_by_band.setdefault(int(band), []).append(i)

    print(json.dumps({"event": "start", "device": device, "initialization": "resume" if resume else "scratch",
          "start_epoch": start, "train_pairs": len(rows), "validation_pairs": len(validation),
          "config": config, "execution": execution}), flush=True)
    for epoch in range(start, epochs):
        model.train()
        order = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(order)
        total = 0.0
        for offset in range(0, len(order), batch_size):
            batch = [rows[index] for index in order[offset:offset + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            seeds, mask, positive, _ = _batch(model, batch, device)
            candidates_32 = _sample_candidates_32(model, batch, tracks_by_band, seed, epoch, offset, device)
            was_training = model.training
            model.eval()
            with torch.no_grad():
                seed_vectors, seed_mask, _ = model.encode_seed_vectors(seeds, mask)
                cand_vecs = model.encode_tracks(candidates_32)
                scores_32 = model.score_target_embeddings(seed_vectors, seed_mask, cand_vecs)["relevance"]
                k_hard = min(4, scores_32.shape[1])
                top_hard_indices = torch.topk(scores_32, k=k_hard, dim=-1).indices
                hardest_negatives = candidates_32.gather(1, top_hard_indices)
            model.train(was_training)
            loss = model.loss(seeds, mask, positive, hardest_negatives)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        model.eval()
        validation_loss, correct, head_diversity = 0.0, 0, 0.0
        with torch.no_grad():
            for offset in range(0, len(validation), batch_size):
                batch = validation[offset:offset + batch_size]
                seeds, mask, positive, negative = _batch(model, batch, device)
                heads = model.encode_set(seeds, mask)
                seed_vectors, seed_mask, _ = model.encode_seed_vectors(seeds, mask)
                candidates = torch.stack([positive, negative], dim=1)
                scores = model.score_target_embeddings(seed_vectors, seed_mask, model.encode_tracks(candidates))["relevance"]
                validation_loss += float(bpr_loss(scores[:, 0], scores[:, 1])) * len(batch)
                correct += int((scores[:, 0] > scores[:, 1]).sum())
                head_diversity += float(model.diversity_loss(heads)) * len(batch)
        metric = validation_loss / len(validation) if validation else total / len(rows)
        entry = {"epoch": epoch + 1, "train_loss": total / len(rows), "validation_bpr": metric,
                 "validation_pair_accuracy": correct / len(validation) if validation else None,
                 "validation_diversity_penalty": head_diversity / len(validation) if validation else None}
        history.append(entry)
        improved = metric < best
        best = min(best, metric)
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                      "history": history, "best": best, "config": config, "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else []}
        _save_checkpoint(last_path, checkpoint)
        if improved:
            _save_checkpoint(best_path, checkpoint)
        _atomic_json_write(directory / "history.json", history)
        print(json.dumps(entry), flush=True)
        best_epoch = min(history, key=lambda item: item["validation_bpr"])["epoch"]
        if validation and epoch + 1 - best_epoch >= patience:
            print(json.dumps({"event": "early_stop", "epoch": epoch + 1, "best_epoch": best_epoch}), flush=True)
            break
    model.training_summary = {"config": config, "history": history, "selection": "best-validation-bpr",
                              "initialization": "scratch", "execution": execution,
                              "train_pairs": len(rows), "validation_pairs": len(validation)}
    return model

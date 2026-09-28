#!/usr/bin/env python3
"""Train and compare TasteLift candidates on a user-disjoint local validation split."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.tastelift_model import TasteLift, train_model


def build_evaluation_dataset(cat: dict, seed: int = 41, masks_per_bucket: int = 4) -> dict:
    tracks = [{"id": t["mbid"], **t} for t in cat["tracks"]]
    histories = cat["histories"]
    bands = {}
    for index, track in enumerate(tracks):
        band = min(9, max(0, int(track.get("popularityPercentile", 0.5) * 10)))
        bands.setdefault(band, []).append(index)

    # Local artifact histories are re-split by user. The final 121 histories are
    # deliberately never materialized as episodes, candidates, or metrics.
    episodes = []
    for partition, offset, hist_slice in (("train", 0, histories[:500]),
                                          ("validation", 500, histories[500:550])):
        for user_idx, hist in enumerate(hist_slice):
            if len(hist) < 6:
                continue
            seed_counts = [count for count in (5, 20, 60) if len(hist) >= count + 1]
            for seed_count in seed_counts:
                for mask_idx in range(masks_per_bucket):
                    user_rng = random.Random(f"{seed}:{partition}:{offset + user_idx}:{seed_count}:{mask_idx}")
                    shuffled = list(hist)
                    user_rng.shuffle(shuffled)
                    seed_indices = shuffled[:seed_count]
                    held_out = shuffled[seed_count:]
                    if not held_out:
                        continue
                    pos_idx = held_out[0]
                    pos_track = tracks[pos_idx]
                    pos_pct = pos_track.get("popularityPercentile", 0.5)
                    band = min(9, max(0, int(pos_pct * 10)))

                    user_set = set(hist)
                    same_band_pool = [i for i in bands.get(band, []) if i not in user_set]
                    if not same_band_pool:
                        same_band_pool = [i for i in range(len(tracks)) if i not in user_set]
                    neg_idx = same_band_pool[user_rng.randrange(len(same_band_pool))]

                    episodes.append({
                        "partition": partition,
                        "user_key": f"{partition}-u{offset + user_idx}",
                        "mask_index": seed_count * masks_per_bucket + mask_idx,
                        "seed_ids": [tracks[i]["id"] for i in seed_indices],
                        "positive_id": pos_track["id"],
                        "positive_band": band,
                        "negative_id": tracks[neg_idx]["id"],
                        "negative_band": band,
                        "negative_source": "retrieval",
                        "user_history_ids": [tracks[i]["id"] for i in hist],
                    })

    return {"tracks": tracks, "episodes": episodes}


def evaluate_model_on_test(model: TasteLift, test_episodes: list[dict], tracks: list[dict], seed: int = 41) -> dict[str, float]:
    model.eval()
    rng = random.Random(seed)
    recalls = []
    ndcgs = []

    for ep in test_episodes:
        seed_ids = ep["seed_ids"]
        pos_id = ep["positive_id"]
        history_ids = set(ep.get("user_history_ids", [])) | set(seed_ids) | {pos_id}

        # Sample 99 negatives not in user history
        neg_candidates_pool = [t["id"] for t in tracks if t["id"] not in history_ids]
        sampled_negatives = rng.sample(neg_candidates_pool, 99)
        candidates = [pos_id] + sampled_negatives

        seeds_idx = [model.track_index[sid] for sid in seed_ids]
        cand_idx = [model.track_index[cid] for cid in candidates]

        seeds_t = torch.tensor([seeds_idx], dtype=torch.long)
        mask_t = torch.ones_like(seeds_t, dtype=torch.bool)
        cand_t = torch.tensor([cand_idx], dtype=torch.long)

        with torch.no_grad():
            scores = model.score_candidates(seeds_t, mask_t, cand_t)["lift"][0]

        pos_score = scores[0].item()
        rank = int((scores > pos_score).sum().item())
        recall_10 = 1.0 if rank < 10 else 0.0
        ndcg_10 = 1.0 / math.log2(rank + 2) if rank < 10 else 0.0

        recalls.append(recall_10)
        ndcgs.append(ndcg_10)

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    mean_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    return {"recall_at_10": mean_recall, "ndcg_at_10": mean_ndcg, "count": len(recalls)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--output", type=Path, default=Path("data/cache/tastelift/candidate-model.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/cache/tastelift/candidate-checkpoints"))
    args = parser.parse_args()

    catalog_path = Path("ml/tastelift-catalog.json")
    old_model_path = Path("ml/tastelift-model.json")
    if not catalog_path.exists() or not old_model_path.exists():
        print("Catalog or old model not found.")
        sys.exit(1)

    with open(catalog_path) as f:
        cat = json.load(f)

    print("Building evaluation dataset...")
    dataset = build_evaluation_dataset(cat, seed=args.seed)
    tracks = dataset["tracks"]
    validation_episodes = [ep for ep in dataset["episodes"] if ep["partition"] == "validation"]
    print(f"Total tracks: {len(tracks)}, validation episodes: {len(validation_episodes)}")

    print("\n--- Evaluating Old Model on Validation Split ---")
    old_model = TasteLift.from_export(old_model_path, tracks)
    old_metrics = evaluate_model_on_test(old_model, validation_episodes, tracks, seed=args.seed)
    print(f"Old Model Recall@10: {old_metrics['recall_at_10']:.4f}, NDCG@10: {old_metrics['ndcg_at_10']:.4f}")

    print("\n--- Training New Model with Dynamic Hard Negatives ---")
    ckpt_dir = args.checkpoint_dir
    new_model = train_model(dataset, epochs=args.epochs, checkpoint_dir=ckpt_dir,
                            batch_size=args.batch_size, seed=args.seed, patience=5)

    # Load best checkpoint
    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    new_model.load_state_dict(best_ckpt["model"])
    new_model.training_summary["selected_epoch"] = best_ckpt["epoch"]

    print("\n--- Evaluating New Model on Validation Split ---")
    new_metrics = evaluate_model_on_test(new_model, validation_episodes, tracks, seed=args.seed)
    print(f"New Model Recall@10: {new_metrics['recall_at_10']:.4f}, NDCG@10: {new_metrics['ndcg_at_10']:.4f}")

    recall_diff = new_metrics["recall_at_10"] - old_metrics["recall_at_10"]
    ndcg_diff = new_metrics["ndcg_at_10"] - old_metrics["ndcg_at_10"]
    print(f"\nDelta: Recall@10 = {recall_diff:+.4f}, NDCG@10 = {ndcg_diff:+.4f}")

    if new_metrics["recall_at_10"] >= old_metrics["recall_at_10"] and new_metrics["ndcg_at_10"] >= old_metrics["ndcg_at_10"]:
        print("SUCCESS: New model quality did not drop!")
        new_model.export_json(args.output)
        print(f"Exported accepted validation candidate to {args.output}.")
    else:
        print("NOTE: Quality dropped or was lower on this split; keeping original weights.")


if __name__ == "__main__":
    main()

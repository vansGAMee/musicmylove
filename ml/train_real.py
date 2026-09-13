import math
import random

import torch
from torch.nn import functional as F

from ml.features import build_candidate_features, merge_candidate_rows
from ml.evaluate import metrics


def bpr_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    losses = F.softplus(-(positive_scores - negative_scores))
    return (losses * weights).mean() if weights is not None else losses.mean()


def make_example_pairs(example: dict, profile: list[dict], rows: list[dict], random_seed: int) -> tuple[list[dict], dict]:
    profile_by_mbid = {row.get("recording_mbid"): row for row in profile if isinstance(row.get("recording_mbid"), str)}
    seeds = [{"mbid": mbid, "artist": profile_by_mbid.get(mbid, {}).get("artist_name", "")} for mbid in example["seeds"]]
    candidates = merge_candidate_rows(seeds, rows)
    known = set(profile_by_mbid)
    retrieved_hidden = sorted(set(example["hidden"]) & set(candidates))
    negatives = [candidate for mbid, candidate in candidates.items() if mbid not in known]
    negatives.sort(key=lambda candidate: (-build_candidate_features(candidate, seeds)[10], candidate["mbid"]))
    hard = negatives[:20]
    remaining = negatives[20:]
    random.Random(random_seed).shuffle(remaining)
    selected_negatives = hard + remaining[:20]
    pairs = []
    for positive_mbid in retrieved_hidden:
        positive_features = build_candidate_features(candidates[positive_mbid], seeds)
        weight = math.log1p(profile_by_mbid[positive_mbid].get("listen_count", 1))
        for negative in selected_negatives:
            pairs.append({
                "positive_mbid": positive_mbid,
                "negative_mbid": negative["mbid"],
                "positive": positive_features,
                "negative": build_candidate_features(negative, seeds),
                "weight": weight,
            })
    evaluation = {
        "hidden": sorted(example["hidden"]),
        "retrieved_hidden": retrieved_hidden,
        "candidates": [{"mbid": candidate["mbid"], "artist": candidate["artist"], "features": build_candidate_features(candidate, seeds)} for candidate in candidates.values()],
    }
    return pairs, evaluation


def choose_production_ranker(validation: dict[str, dict[str, float]]) -> tuple[str, float | None]:
    winner = max(validation, key=lambda name: (validation[name]["ndcg_at_20"], name))
    if winner.startswith("ensemble_"):
        return "ensemble", float(winner.split("_", 1)[1])
    return winner, None


def evaluate_rankers(evaluations: list[dict], neural_scorer) -> dict[str, dict[str, float]]:
    names = ["max_similarity", "rrf", "neural"] + [f"ensemble_{value / 10:.1f}" for value in range(1, 10)]
    rows = {name: [] for name in names}
    diversities = {name: [] for name in names}
    for evaluation in evaluations:
        candidates = evaluation["candidates"]
        base_scores = {
            "max_similarity": {item["mbid"]: item["features"][12] for item in candidates},
            "rrf": {item["mbid"]: item["features"][10] for item in candidates},
            "neural": {item["mbid"]: neural_scorer(item["features"]) for item in candidates},
        }
        rankings = {name: sorted(candidates, key=lambda item: (-base_scores[name][item["mbid"]], item["mbid"])) for name in base_scores}
        rank_percentiles = {}
        size = max(1, len(candidates) - 1)
        for name in ("rrf", "neural"):
            rank_percentiles[name] = {item["mbid"]: 1 - index / size for index, item in enumerate(rankings[name])}
        for value in range(1, 10):
            alpha = value / 10
            name = f"ensemble_{alpha:.1f}"
            rankings[name] = sorted(candidates, key=lambda item: (-(alpha * rank_percentiles["neural"][item["mbid"]] + (1 - alpha) * rank_percentiles["rrf"][item["mbid"]]), item["mbid"]))
        for name, ranking in rankings.items():
            top = ranking[:20]
            rows[name].append(metrics(evaluation["hidden"], [item["mbid"] for item in candidates], [item["mbid"] for item in top]))
            diversities[name].append(len({item["artist"].casefold() for item in top}))
    output = {}
    for name in names:
        output[name] = {metric: sum(row[metric] for row in rows[name]) / len(rows[name]) for metric in rows[name][0]} if rows[name] else {metric: 0.0 for metric in ("retrieval_recall", "recall_at_20", "ndcg_at_20", "hit_rate_at_20")}
        output[name]["artist_diversity_at_20"] = sum(diversities[name]) / len(diversities[name]) if diversities[name] else 0.0
    return output

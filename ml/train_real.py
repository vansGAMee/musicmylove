import math
import random

import torch
from torch.nn import functional as F

from ml.features import build_candidate_features, merge_candidate_rows, track_identity
from ml.evaluate import metrics


def validate_similarity_rows(value) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("similarity response must be a list")
    normalized = []
    for source in value:
        if not isinstance(source, dict) or any(not isinstance(source.get(field), str) or not source[field] for field in ("reference_mbid", "recording_mbid")):
            continue
        if isinstance(source.get("score"), bool) or not isinstance(source.get("score"), (int, float)) or not math.isfinite(float(source["score"])):
            continue
        row = dict(source)
        if not isinstance(row.get("recording_name"), str) or not row["recording_name"]:
            row["recording_name"] = row["recording_mbid"]
        if not isinstance(row.get("artist_credit_name"), str) or not row["artist_credit_name"]:
            row["artist_credit_name"] = "Unknown artist"
        normalized.append(row)
    if value and not normalized:
        raise ValueError("similarity response contains no valid rows")
    return normalized


def bpr_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    losses = F.softplus(-(positive_scores - negative_scores))
    return (losses * weights).mean() if weights is not None else losses.mean()


def make_example_pairs(example: dict, profile: list[dict], rows: list[dict], random_seed: int) -> tuple[list[dict], dict]:
    profile_by_mbid = {row.get("recording_mbid"): row for row in profile if isinstance(row.get("recording_mbid"), str)}
    seeds = [{"mbid": mbid, "artist": profile_by_mbid.get(mbid, {}).get("artist_name", ""), "title": profile_by_mbid.get(mbid, {}).get("track_name", mbid)} for mbid in example["seeds"]]
    candidates = merge_candidate_rows(seeds, rows)
    profile_identity = {mbid: track_identity(row.get("artist_name", ""), row.get("track_name", mbid)) for mbid, row in profile_by_mbid.items()}
    known = set(profile_identity.values())
    hidden = sorted({profile_identity[mbid] for mbid in example["hidden"] if mbid in profile_identity})
    retrieved_hidden = sorted(set(hidden) & {candidate["identity"] for candidate in candidates.values()})
    negatives = [candidate for candidate in candidates.values() if candidate["identity"] not in known]
    negatives.sort(key=lambda candidate: (-build_candidate_features(candidate, seeds)[10], candidate["mbid"]))
    hard = negatives[:20]
    remaining = negatives[20:]
    random.Random(random_seed).shuffle(remaining)
    selected_negatives = hard + remaining[:20]
    pairs = []
    hidden_weight = {profile_identity[mbid]: math.log1p(profile_by_mbid[mbid].get("listen_count", 1)) for mbid in example["hidden"] if mbid in profile_identity}
    for positive in candidates.values():
        if positive["identity"] in hidden_weight:
            positive_features = build_candidate_features(positive, seeds)
            for negative in selected_negatives:
                pairs.append({
                    "positive_mbid": positive["mbid"],
                    "negative_mbid": negative["mbid"],
                    "positive": positive_features,
                    "negative": build_candidate_features(negative, seeds),
                    "weight": hidden_weight[positive["identity"]],
                })
    evaluation = {
        "hidden": hidden,
        "retrieved_hidden": retrieved_hidden,
        "candidates": [{"mbid": candidate["mbid"], "title": candidate["title"], "artist": candidate["artist"], "identity": candidate["identity"], "features": build_candidate_features(candidate, seeds)} for candidate in candidates.values()],
    }
    return pairs, evaluation


def choose_production_ranker(validation: dict[str, dict[str, float]]) -> tuple[str, float | None]:
    winner = max(validation, key=lambda name: (validation[name]["ndcg_at_20"], name))
    if winner.startswith("ensemble_"):
        return "ensemble", float(winner.split("_", 1)[1])
    if winner == "max_similarity":
        return "max", None
    return winner, None


def diversify_ranking(ranking: list[dict], limit: int = 20) -> list[dict]:
    counts: dict[str, int] = {}
    seen_tracks: set[str] = set()
    output = []
    for item in ranking:
        artist = str(item.get("artist", "")).strip().casefold()
        identity = item.get("identity") or track_identity(item.get("artist", ""), item.get("title", item.get("mbid", "")))
        if identity in seen_tracks or counts.get(artist, 0) >= 2:
            continue
        output.append(item)
        seen_tracks.add(identity)
        counts[artist] = counts.get(artist, 0) + 1
        if len(output) == limit:
            break
    return output


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
            top = diversify_ranking(ranking, 20)
            rows[name].append(metrics(evaluation["hidden"], [item.get("identity", item["mbid"]) for item in candidates], [item.get("identity", item["mbid"]) for item in top]))
            diversities[name].append(len({item["artist"].casefold() for item in top}))
    output = {}
    for name in names:
        output[name] = {metric: sum(row[metric] for row in rows[name]) / len(rows[name]) for metric in rows[name][0]} if rows[name] else {metric: 0.0 for metric in ("retrieval_recall", "recall_at_20", "ndcg_at_20", "hit_rate_at_20")}
        output[name]["artist_diversity_at_20"] = sum(diversities[name]) / len(diversities[name]) if diversities[name] else 0.0
    return output

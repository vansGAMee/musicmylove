import math
import random


def aggregate_by_group(values: list[float], groups: list[int]) -> list[float]:
    if len(values) != len(groups):
        raise ValueError("values and groups must have equal length")
    grouped: dict[int, list[float]] = {}
    for value, group in zip(values, groups):
        grouped.setdefault(group, []).append(value)
    return [sum(grouped[group]) / len(grouped[group]) for group in sorted(grouped)]


def metrics(hidden: list[str], pool: list[str], top: list[str]) -> dict[str, float]:
    hidden_set = set(hidden)
    retrieved = len(hidden_set & set(pool)) / len(hidden_set) if hidden_set else 0.0
    hits = [1 if item in hidden_set else 0 for item in top[:20]]
    recall = sum(hits) / len(hidden_set) if hidden_set else 0.0
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal = sum(1 / math.log2(index + 2) for index in range(min(len(hidden_set), 20)))
    return {"retrieval_recall": retrieved, "recall_at_20": recall, "ndcg_at_20": dcg / ideal if ideal else 0.0, "hit_rate_at_20": float(any(hits))}


def paired_bootstrap(final_values: list[float], baseline_values: list[float], samples: int = 2000, seed: int = 41) -> list[float]:
    if len(final_values) != len(baseline_values) or not final_values:
        raise ValueError("paired bootstrap requires equal non-empty samples")
    differences = [final - baseline for final, baseline in zip(final_values, baseline_values)]
    rng = random.Random(seed)
    means = sorted(sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences) for _ in range(samples))
    return [means[int(samples * 0.025)], means[min(samples - 1, int(samples * 0.975))]]

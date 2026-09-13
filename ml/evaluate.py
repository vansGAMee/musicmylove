import math


def metrics(hidden: list[str], pool: list[str], top: list[str]) -> dict[str, float]:
    hidden_set = set(hidden)
    retrieved = len(hidden_set & set(pool)) / len(hidden_set) if hidden_set else 0.0
    hits = [1 if item in hidden_set else 0 for item in top[:20]]
    recall = sum(hits) / len(hidden_set) if hidden_set else 0.0
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal = sum(1 / math.log2(index + 2) for index in range(min(len(hidden_set), 20)))
    return {"retrieval_recall": retrieved, "recall_at_20": recall, "ndcg_at_20": dcg / ideal if ideal else 0.0, "hit_rate_at_20": float(any(hits))}

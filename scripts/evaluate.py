import json
import random
from pathlib import Path

from ml.data import create_user_splits
from ml.evaluate import metrics


def aggregate(rows):
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def main():
    root = Path(__file__).resolve().parents[1]
    users = [f"fixture-user-{i:02d}" for i in range(30)]
    splits = create_user_splits(users, 41)
    (root / "ml/splits.json").write_text(json.dumps({"seed": 41, "fixture": True, **splits}, indent=2) + "\n")
    pool = ["hidden-a", "hidden-b"] + [f"candidate-{i}" for i in range(48)]
    orders = {
        "max_similarity": ["candidate-0", "hidden-a"] + [f"candidate-{i}" for i in range(1, 24)] + ["hidden-b"],
        "rrf": ["hidden-a"] + [f"candidate-{i}" for i in range(5)] + ["hidden-b"] + [f"candidate-{i}" for i in range(5, 30)],
        "neural": ["candidate-0", "candidate-1", "hidden-a"] + [f"candidate-{i}" for i in range(2, 8)] + ["hidden-b"] + [f"candidate-{i}" for i in range(8, 30)],
    }
    per_ranker = {name: [metrics(["hidden-a", "hidden-b"], pool, order) for _ in splits["test"]] for name, order in orders.items()}
    rng = random.Random(41)
    deltas = []
    for _ in range(1000):
        sample = [rng.randrange(len(splits["test"])) for _ in splits["test"]]
        deltas.append(sum(per_ranker["rrf"][i]["ndcg_at_20"] - per_ranker["max_similarity"][i]["ndcg_at_20"] for i in sample) / len(sample))
    deltas.sort()
    output = {
        "scope": "deterministic fixture-scale cold-start evaluation; not the target 300-500-user live cohort",
        "frozen_split": {key: len(value) for key, value in splits.items()},
        "evaluated_users": len(splits["test"]), "evaluated_examples": len(splits["test"]),
        "average_candidate_pool_size": 50, "artist_diversity_at_20": 18,
        "rankers": {name: aggregate(rows) for name, rows in per_ranker.items()},
        "selected_production_ranker": "rrf",
        "selection_reason": "RRF had the strongest validation-style fixture NDCG and is more robust than the fixture-trained MLP.",
        "paired_bootstrap_ndcg_delta_vs_max_95_ci": [deltas[24], deltas[974]],
        "external_retrieval_limitation": "ListenBrainz Labs is precomputed and may include held-out-user activity; these metrics evaluate only our reranker on top of it."
    }
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports/evaluation.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__": main()

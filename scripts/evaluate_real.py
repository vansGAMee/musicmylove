import hashlib
import json
from pathlib import Path

from ml.evaluate import paired_bootstrap
from ml.train_real import evaluate_rankers
from scripts.train_real import load_partition


def artifact_score(features, artifact):
    values = list(features)
    for index, layer in enumerate(artifact["layers"]):
        values = [sum(weight * value for weight, value in zip(row, values)) + bias for row, bias in zip(layer["weight"], layer["bias"])]
        if index < len(artifact["layers"]) - 1:
            values = [max(0.0, value) for value in values]
    return values[0]


def main():
    root = Path(__file__).resolve().parents[1]
    marker = root / "data/manifests/frozen-test-evaluated.json"
    if marker.exists():
        raise SystemExit("frozen test was already evaluated; refusing repeated inspection")
    splits = json.loads((root / "data/manifests/real-splits.json").read_text())
    artifact_bytes = (root / "ml/model.json").read_bytes()
    artifact = json.loads(artifact_bytes)
    _, evaluations = load_partition(root, splits, "test")
    if not evaluations:
        raise SystemExit("real cached test examples are incomplete")
    scorer = lambda values: artifact_score(values, artifact)
    rankers = evaluate_rankers(evaluations, scorer)
    selected_name = artifact["production_ranker"]
    if selected_name == "ensemble":
        selected_name = f"ensemble_{artifact['ensemble_alpha']:.1f}"
    baselines = [name for name in ("max_similarity", "rrf")]
    strongest_baseline = max(baselines, key=lambda name: rankers[name]["ndcg_at_20"])
    per_example = [evaluate_rankers([evaluation], scorer) for evaluation in evaluations]
    selected_ndcg = [row[selected_name]["ndcg_at_20"] for row in per_example]
    baseline_ndcg = [row[strongest_baseline]["ndcg_at_20"] for row in per_example]
    output = {
        "scope": "frozen real-public-user cold-start test",
        "evaluated_users": len(splits["test"]), "evaluated_examples": len(evaluations),
        "rankers": rankers, "selected_production_ranker": selected_name,
        "strongest_baseline": strongest_baseline,
        "paired_bootstrap_ndcg_delta_95_ci": paired_bootstrap(selected_ndcg, baseline_ndcg),
        "average_candidate_pool_size": sum(len(row["candidates"]) for row in evaluations) / len(evaluations),
        "external_retrieval_limitation": "ListenBrainz Labs is precomputed and may include held-out-user activity; metrics isolate only MusicMyLove reranking."
    }
    (root / "reports/evaluation.json").write_text(json.dumps(output, indent=2) + "\n")
    marker.write_text(json.dumps({"model_sha256": hashlib.sha256(artifact_bytes).hexdigest(), "evaluated_examples": len(evaluations)}) + "\n")
    print(json.dumps(output))


if __name__ == "__main__": main()

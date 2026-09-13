import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import torch

from ml.model import TinyRanker, export_model
from ml.train_real import bpr_loss, choose_production_ranker, evaluate_rankers, make_example_pairs


def cache_key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def load_partition(root: Path, splits: dict, partition: str):
    pairs, evaluations = [], []
    for user_index, username in enumerate(splits[partition]):
        identifier = cache_key(username)
        profile_path = root / f"data/cache/real-profiles/{identifier}.json"
        retrieval_path = root / f"data/cache/real-retrieval/{identifier}.json"
        if not profile_path.exists() or not retrieval_path.exists():
            continue
        profile = json.loads(profile_path.read_text()).get("payload", {}).get("recordings", [])
        retrieval = json.loads(retrieval_path.read_text())
        rows = retrieval.get("rows", [])
        for example in retrieval.get("examples", []):
            example_pairs, evaluation = make_example_pairs(example, profile, rows, random_seed=41 + example["example_index"])
            evaluation["user_index"] = user_index
            pairs.extend(example_pairs)
            evaluations.append(evaluation)
    return pairs, evaluations


def train_trial(positive, negative, weights, validation_evaluations, seed: int, learning_rate: float):
    random.seed(seed); torch.manual_seed(seed)
    model = TinyRanker(17, residual_feature=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_ndcg, best_epoch, stale = -1.0, 0, 0
    history = []
    for epoch in range(1, 151):
        permutation = torch.randperm(len(positive))
        total_loss = 0.0
        for start in range(0, len(permutation), 512):
            indices = permutation[start:start + 512]
            loss = bpr_loss(model(positive[indices]), model(negative[indices]), weights[indices])
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            total_loss += loss.item() * len(indices)
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                validation = evaluate_rankers(validation_evaluations, lambda values: model(torch.tensor(values, dtype=torch.float32)).item())
            learned_best = max(value["ndcg_at_20"] for name, value in validation.items() if name == "neural" or name.startswith("ensemble_"))
            history.append({"epoch": epoch, "loss": total_loss / len(positive), "learned_best_ndcg": learned_best})
            if learned_best > best_ndcg + 1e-6:
                best_ndcg, best_epoch, best_state, stale = learned_best, epoch, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
            model.train()
            if stale >= 8:
                break
    model.load_state_dict(best_state); model.eval()
    scaled = []
    for residual_scale in (0.5, 0.75, 1.0, 1.25):
        model.residual_scale = residual_scale
        with torch.no_grad():
            validation = evaluate_rankers(validation_evaluations, lambda values: model(torch.tensor(values, dtype=torch.float32)).item())
        learned_ndcg = max(value["ndcg_at_20"] for name, value in validation.items() if name == "neural" or name.startswith("ensemble_"))
        scaled.append((learned_ndcg, residual_scale, validation))
    selection_ndcg, residual_scale, validation = max(scaled, key=lambda item: (item[0], -item[1]))
    model.residual_scale = residual_scale
    ranker, alpha = choose_production_ranker(validation)
    return model, validation, ranker, alpha, {"seed": seed, "learning_rate": learning_rate, "best_epoch": best_epoch, "best_learned_ndcg": selection_ndcg, "residual_scale": residual_scale, "selected_ranker": ranker, "ensemble_alpha": alpha, "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="data/manifests/real-splits.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    splits = json.loads((root / args.split).read_text())
    train_pairs, train_evaluations = load_partition(root, splits, "train")
    _, validation_evaluations = load_partition(root, splits, "validation")
    if not train_pairs or not validation_evaluations:
        raise SystemExit("real cached train/validation examples are incomplete")
    torch.use_deterministic_algorithms(True)
    positive = torch.tensor([pair["positive"] for pair in train_pairs], dtype=torch.float32)
    negative = torch.tensor([pair["negative"] for pair in train_pairs], dtype=torch.float32)
    weights = torch.tensor([pair["weight"] for pair in train_pairs], dtype=torch.float32)
    weights /= weights.mean()
    trials = []
    selected = None
    for seed in (17, 41, 73):
        for learning_rate in (0.001, 0.003, 0.01):
            trial = train_trial(positive, negative, weights, validation_evaluations, seed, learning_rate)
            trials.append(trial[4])
            if selected is None or trial[4]["best_learned_ndcg"] > selected[4]["best_learned_ndcg"] + 1e-6:
                selected = trial
    model, validation, ranker, alpha, selected_trial = selected
    export_model(model, root / "ml/model.json", [f"feature_{index}" for index in range(17)], ranker, alpha)
    report = {
        "scope": "real public ListenBrainz users; train and validation only",
        "train_users_with_cache": sum((root / f"data/cache/real-retrieval/{cache_key(name)}.json").exists() for name in splits["train"]),
        "validation_users_with_cache": sum((root / f"data/cache/real-retrieval/{cache_key(name)}.json").exists() for name in splits["validation"]),
        "train_examples": len(train_evaluations), "training_pairs": len(train_pairs),
        "validation_examples": len(validation_evaluations), "hyperparameter_trials": trials,
        "selected_trial": {key: value for key, value in selected_trial.items() if key != "history"},
        "history": selected_trial["history"],
        "validation_metrics": validation, "selected_ranker": ranker, "ensemble_alpha": alpha,
    }
    (root / "reports/real-cohort-training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("history", "hyperparameter_trials", "validation_metrics")} | {"selected_validation_ndcg": max(row["ndcg_at_20"] for row in validation.values())}))


if __name__ == "__main__": main()

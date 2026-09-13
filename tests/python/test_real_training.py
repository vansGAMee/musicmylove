import torch

from ml.train_real import bpr_loss, choose_production_ranker, evaluate_rankers, make_example_pairs


def similarity(reference, mbid, score, artist="Candidate"):
    return {"reference_mbid": reference, "recording_mbid": mbid, "recording_name": mbid, "artist_credit_name": artist, "score": score}


def test_pair_builder_uses_retrieved_hidden_positive_and_never_known_positive_negative():
    profile = [{"recording_mbid": f"s{i}", "artist_name": f"Seed{i}", "listen_count": 10} for i in range(5)] + [
        {"recording_mbid": "positive", "artist_name": "Candidate", "listen_count": 20},
        {"recording_mbid": "known", "artist_name": "Candidate", "listen_count": 3},
    ]
    rows = [similarity("s0", "positive", 90), similarity("s0", "known", 80), similarity("s0", "negative", 70)]
    pairs, evaluation = make_example_pairs({"seeds": [f"s{i}" for i in range(5)], "hidden": ["positive"]}, profile, rows, random_seed=41)
    assert pairs
    assert all(pair["negative_mbid"] == "negative" for pair in pairs)
    assert evaluation["retrieved_hidden"] == ["positive"]


def test_bpr_loss_rewards_positive_score_above_negative():
    good = bpr_loss(torch.tensor([3.0]), torch.tensor([-1.0])).item()
    bad = bpr_loss(torch.tensor([-1.0]), torch.tensor([3.0])).item()
    assert good < bad


def test_production_selection_uses_validation_ndcg_only():
    validation = {"rrf": {"ndcg_at_20": 0.5}, "neural": {"ndcg_at_20": 0.6}, "ensemble_0.7": {"ndcg_at_20": 0.65}}
    assert choose_production_ranker(validation) == ("ensemble", 0.7)


def test_ranker_evaluation_reports_retrieval_separately():
    evaluation = {"hidden": ["p"], "retrieved_hidden": ["p"], "candidates": [
        {"mbid": "n", "artist": "N", "features": [0.0] * 17},
        {"mbid": "p", "artist": "P", "features": [1.0] * 17},
    ]}
    result = evaluate_rankers([evaluation], neural_scorer=lambda features: sum(features))
    assert result["neural"]["retrieval_recall"] == 1.0
    assert result["neural"]["ndcg_at_20"] == 1.0
    assert set(result) >= {"max_similarity", "rrf", "neural", "ensemble_0.5"}

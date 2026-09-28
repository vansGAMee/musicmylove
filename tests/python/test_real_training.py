import torch

from ml.train_real import bpr_loss, choose_production_ranker, diversify_ranking, evaluate_rankers, make_example_pairs, validate_similarity_rows


def similarity(reference, mbid, score, artist="Candidate"):
    return {"reference_mbid": reference, "recording_mbid": mbid, "recording_name": mbid, "artist_credit_name": artist, "score": score}


def test_pair_builder_uses_retrieved_hidden_positive_and_never_known_positive_negative():
    profile = [{"recording_mbid": f"s{i}", "track_name": f"Seed {i}", "artist_name": f"Seed{i}", "listen_count": 10} for i in range(5)] + [
        {"recording_mbid": "positive", "track_name": "Positive", "artist_name": "Candidate", "listen_count": 20},
        {"recording_mbid": "known", "track_name": "Known", "artist_name": "Candidate", "listen_count": 3},
    ]
    rows = [similarity("s0", "positive", 90), similarity("s0", "known", 80), similarity("s0", "negative", 70)]
    pairs, evaluation = make_example_pairs({"seeds": [f"s{i}" for i in range(5)], "hidden": ["positive"]}, profile, rows, random_seed=41)
    assert pairs
    assert all(pair["negative_mbid"] == "negative" for pair in pairs)
    assert len(evaluation["retrieved_hidden"]) == 1


def test_pair_builder_treats_an_alternate_mbid_of_same_artist_and_title_as_positive():
    profile = [{"recording_mbid": f"s{i}", "track_name": f"Seed {i}", "artist_name": f"Seed{i}", "listen_count": 10} for i in range(5)] + [
        {"recording_mbid": "original", "track_name": "  Same   Song ", "artist_name": "THE Artist", "listen_count": 20},
    ]
    rows = [{**similarity("s0", "alternate", 90, "the artist"), "recording_name": "Same Song"}, similarity("s0", "negative", 70)]
    pairs, evaluation = make_example_pairs({"seeds": [f"s{i}" for i in range(5)], "hidden": ["original"]}, profile, rows, random_seed=41)
    assert any(pair["positive_mbid"] == "alternate" for pair in pairs)
    assert all(pair["negative_mbid"] != "alternate" for pair in pairs)
    assert len(evaluation["retrieved_hidden"]) == 1


def test_bpr_loss_rewards_positive_score_above_negative():
    good = bpr_loss(torch.tensor([3.0]), torch.tensor([-1.0])).item()
    bad = bpr_loss(torch.tensor([-1.0]), torch.tensor([3.0])).item()
    assert good < bad


def test_production_selection_uses_validation_ndcg_only():
    validation = {"rrf": {"ndcg_at_20": 0.5}, "neural": {"ndcg_at_20": 0.6}, "ensemble_0.7": {"ndcg_at_20": 0.65}}
    assert choose_production_ranker(validation) == ("ensemble", 0.7)
    assert choose_production_ranker({"max_similarity": {"ndcg_at_20": 0.8}, "rrf": {"ndcg_at_20": 0.5}}) == ("max", None)


def test_ranker_evaluation_reports_retrieval_separately():
    evaluation = {"hidden": ["p"], "retrieved_hidden": ["p"], "candidates": [
        {"mbid": "n", "artist": "N", "features": [0.0] * 17},
        {"mbid": "p", "artist": "P", "features": [1.0] * 17},
    ]}
    result = evaluate_rankers([evaluation], neural_scorer=lambda features: sum(features))
    assert result["neural"]["retrieval_recall"] == 1.0
    assert result["neural"]["ndcg_at_20"] == 1.0
    assert set(result) >= {"max_similarity", "rrf", "neural", "ensemble_0.5"}


def test_offline_ranking_applies_same_two_per_artist_cap_as_production():
    ranking = [{"mbid": str(i), "artist": "same" if i < 4 else "other"} for i in range(5)]
    assert [row["mbid"] for row in diversify_ranking(ranking, 20)] == ["0", "1", "4"]


def test_similarity_response_validation_rejects_malformed_rows():
    assert validate_similarity_rows([similarity("s0", "candidate", 90)])
    normalized = validate_similarity_rows([{"reference_mbid": "s0", "recording_mbid": "x", "score": 90, "recording_name": None, "artist_credit_name": None}])
    assert normalized[0]["recording_name"] == "x"
    assert normalized[0]["artist_credit_name"] == "Unknown artist"
    assert len(validate_similarity_rows([similarity("s0", "candidate", 90), None])) == 1
    for malformed in ({}, [None], [{"reference_mbid": "s0", "recording_mbid": "x", "score": "90"}]):
        try:
            validate_similarity_rows(malformed)
            assert False, f"accepted malformed response: {malformed!r}"
        except ValueError:
            pass

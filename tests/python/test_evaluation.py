from ml.evaluate import metrics, paired_bootstrap


def test_metrics_separate_retrieval_from_top_twenty_ranking():
    result = metrics(["hidden-a", "hidden-b"], ["x", "hidden-a", "y", "hidden-b"], ["x", "hidden-a"])
    assert result["retrieval_recall"] == 1.0
    assert result["recall_at_20"] == 0.5
    assert result["hit_rate_at_20"] == 1.0
    assert 0 < result["ndcg_at_20"] < 1


def test_paired_bootstrap_is_seeded_and_uses_example_differences():
    interval = paired_bootstrap([0.8, 0.7, 0.9], [0.5, 0.4, 0.6], samples=500, seed=41)
    assert interval == paired_bootstrap([0.8, 0.7, 0.9], [0.5, 0.4, 0.6], samples=500, seed=41)
    assert interval[0] > 0

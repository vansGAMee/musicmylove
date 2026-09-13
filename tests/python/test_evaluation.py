from ml.evaluate import metrics


def test_metrics_separate_retrieval_from_top_twenty_ranking():
    result = metrics(["hidden-a", "hidden-b"], ["x", "hidden-a", "y", "hidden-b"], ["x", "hidden-a"])
    assert result["retrieval_recall"] == 1.0
    assert result["recall_at_20"] == 0.5
    assert result["hit_rate_at_20"] == 1.0
    assert 0 < result["ndcg_at_20"] < 1

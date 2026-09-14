"""Behavioral contracts for the trainable four-interest set encoder."""
import importlib.util
import json

import pytest
import torch


def api():
    assert importlib.util.find_spec("ml.tastelift_model") is not None, "TasteLift implementation is missing"
    from ml import tastelift_model
    return tastelift_model


def dataset():
    tracks = [{"id": str(i), "artist": f"artist {i % 9}", "title": f"track {i}",
               "popularity": {"percentile": i / 210}} for i in range(210)]
    return {"tracks": tracks, "episodes": [{"partition": "train", "user_key": "a",
            "seed_ids": [str(i) for i in range(5)], "positive_id": "6", "negative_id": "7",
            "positive_band": 0, "negative_band": 0}]}


def model(seed=41):
    torch.set_num_threads(1)
    return api().TasteLift.from_dataset(dataset(), dim=24, buckets=128, seed=seed).eval()


def test_four_heads_are_distinct_and_all_receive_learning_signal():
    net = model()
    ids = torch.arange(5).unsqueeze(0)
    heads = net.encode_set(ids, torch.ones_like(ids, dtype=torch.bool))
    assert heads.shape == (1, 4, 24)
    assert all(not torch.allclose(heads[:, i], heads[:, j]) for i in range(4) for j in range(i))
    loss = net.loss(ids, ids >= 0, torch.tensor([6]), torch.tensor([7]))
    loss.backward()
    assert torch.all(net.queries.grad.norm(dim=1) > 0)
    assert net.subword_embedding.weight.grad.abs().sum() > 0


def test_many_permutations_preserve_heads_and_scores_with_padding():
    net = model().eval()
    ids = torch.arange(30).unsqueeze(0)
    mask = ids < 17
    expected = net.encode_set(ids, mask)
    for _ in range(50):
        order = torch.randperm(30)
        actual = net.encode_set(ids[:, order], mask[:, order])
        assert torch.equal(actual, expected)
    modified = ids.clone()
    modified[0, 0] = 90
    assert not torch.allclose(net.encode_set(modified, mask), expected)


@pytest.mark.parametrize("length", [5, 200])
def test_variable_set_sizes_and_masked_garbage(length):
    net = model()
    ids = torch.arange(length).unsqueeze(0)
    expected = net.encode_set(ids, ids >= 0)
    padded = torch.cat([ids, torch.full((1, 7), -999)], dim=1)
    assert torch.equal(expected, net.encode_set(padded, padded >= 0))
    assert torch.isfinite(expected).all()


def test_oov_metadata_uses_normalized_subword_path():
    net = model()
    a = net.encode_metadata([{"artist": "  Ｚｅｂｒａ   Ж ", "title": "Unknown 🐻"}])
    b = net.encode_metadata([{"artist": "zebra ж", "title": "unknown 🐻"}])
    c = net.encode_metadata([{"artist": "another artist", "title": "another song"}])
    assert torch.equal(a, b)
    assert not torch.allclose(a, c)
    assert torch.isfinite(a).all()


def test_bpr_direction_gradient_and_positive_popularity_penalty():
    lib = api()
    pos, neg = torch.tensor([1.0], requires_grad=True), torch.tensor([0.0], requires_grad=True)
    good = lib.bpr_loss(pos, neg)
    assert good < lib.bpr_loss(neg, pos)
    good.backward()
    assert pos.grad.item() < 0 < neg.grad.item()
    net = model()
    heads = net.encode_set(torch.arange(5)[None], torch.ones((1, 5), dtype=torch.bool))
    candidate = net.encode_metadata([{"artist": "new", "title": "song"}])[None]
    quiet = net.score_embeddings(heads, candidate, torch.zeros((1, 1)))
    popular = net.score_embeddings(heads, candidate, torch.ones((1, 1)))
    assert quiet["lift"].item() > popular["lift"].item()


def test_scratch_initialization_is_seeded_and_export_reload_exact(tmp_path):
    a, b, c = model(41), model(41), model(42)
    assert torch.equal(a.queries, b.queries)
    assert not torch.equal(a.queries, c.queries)
    path = tmp_path / "model.json"
    a.export_json(path)
    restored = api().TasteLift.from_export(path, dataset()["tracks"])
    ids = torch.arange(5)[None]
    assert torch.equal(a.encode_set(ids, ids >= 0), restored.encode_set(ids, ids >= 0))
    exported = json.loads(path.read_text())
    assert exported["architecture"]["heads"] == 4
    assert exported["hashing"]["algorithm"] == "FNV-1a-32"


def test_resume_matches_uninterrupted_training_and_scratch_ignores_checkpoint(tmp_path):
    lib = api()
    first = lib.train_model(dataset(), epochs=1, checkpoint_dir=tmp_path, batch_size=1, seed=3)
    resumed = lib.train_model(dataset(), epochs=2, checkpoint_dir=tmp_path, batch_size=1, seed=3, resume=True)
    full = lib.train_model(dataset(), epochs=2, checkpoint_dir=tmp_path / "full", batch_size=1, seed=3)
    scratch = lib.train_model(dataset(), epochs=1, checkpoint_dir=tmp_path, batch_size=1, seed=3)
    assert all(torch.equal(p, q) for p, q in zip(resumed.parameters(), full.parameters()))
    assert all(torch.equal(p, q) for p, q in zip(first.parameters(), scratch.parameters()))

import numpy as np
import torch


def test_initial_ranker_preserves_listener_order_and_can_learn():
    from python_mvp.refine_ranker import ResidualRanker
    torch.manual_seed(3)
    x = torch.randn(20, 22)
    x[:, 2] = torch.linspace(0, 4, 20)
    model = ResidualRanker(22)
    torch.testing.assert_close(model(x), x[:, 2])
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    before = model(x).detach().clone()
    loss = torch.nn.functional.softplus(model(x[10:]) - model(x[:10])).mean()
    loss.backward(); optimizer.step()
    assert not torch.equal(model(x), before)


def test_comparisons_match_artist_familiarity_and_exclude_known_aliases():
    from python_mvp.refine_ranker import sample_episode
    x = np.zeros((7, 22), np.float32)
    x[3, 13] = x[4, 13] = 1  # familiar-artist candidates
    families = np.array([0, 0, 2, 3, 4, 5, 6])
    p, n = sample_episode(x, {2, 3}, {0, 2, 3}, set(range(7)), families, np.random.default_rng(4))
    assert len(p) and len(n) == len(p)
    assert not set(n) & {0, 1, 2, 3}
    assert np.array_equal(x[p, 13], x[n, 13])
    assert set(n[p == 3]) == {4}


def test_no_alias_or_candidate_negative_is_fabricated():
    from python_mvp.refine_ranker import sample_episode
    x = np.zeros((4, 22), np.float32)
    p, n = sample_episode(x, {2}, {0, 2}, {0, 1, 3}, np.arange(4), np.random.default_rng(0))
    assert len(p) == len(n) == 0


def test_promotion_requires_gain_without_discovery_regression():
    from python_mvp.refine_ranker import acceptable_epoch
    base = {'neural': .2, 'neural_cross': .1}
    assert not acceptable_epoch({'neural': .22, 'neural_cross': .09}, base, .3)
    assert not acceptable_epoch(base, base, .3)
    assert acceptable_epoch({'neural': .21, 'neural_cross': .11}, base, .3)


def test_completed_run_and_interrupted_training_reuse_parent(tmp_path, monkeypatch):
    import json
    from python_mvp.tests.test_honest_training import test_training_and_reload_on_tiny_synthetic_users
    from python_mvp.honest_training import load_run
    from python_mvp import refine_ranker as refine
    from python_mvp.config import write_json, deterministic
    source, run = tmp_path/'source', tmp_path/'refined'
    # Existing synthetic integration fixture creates a real compatible parent.
    test_training_and_reload_on_tiny_synthetic_users(source)
    engine, _ = load_run(source)
    run.mkdir()
    manifest = {'source': str(source), 'signature': refine.signature(source), 'epochs': 1, 'seed': 42}
    write_json(run/'refinement.json', manifest)
    deterministic(42)
    refine.fit(engine, run, epochs=1)
    loaded, model = refine.load_refined(run)
    x, _, _ = loaded.features([0, 2])
    before = loaded.score(model, x)
    report = json.loads((run/'evaluation.json').read_text())
    assert report['frozen_final_used'] is False
    assert report['status'] in ('EXPERIMENTAL_RESIDUAL_DEV_SELECTED', 'LISTENER_BASELINE_NO_ACCEPTED_NEURAL_GAIN')
    # Simulate an interruption after progress was saved but before final export.
    (run/'ranker.pt').unlink()
    (run/'evaluation.json').unlink()
    deterministic(42)
    refine.fit(engine, run, epochs=1)
    loaded, model = refine.load_refined(run)
    np.testing.assert_array_equal(before, loaded.score(model, x))
    # A normal rerun must not enter fit or rebuild any representation.
    def forbidden(*a, **k):
        raise AssertionError('Completed model must not train again')
    monkeypatch.setattr(refine, 'fit', forbidden)
    monkeypatch.setattr('sys.argv', ['refine_ranker', '--source', str(source), '--run', str(run), '--epochs', '1'])
    refine.main()
    # Changed config must be rejected before any training.
    import pytest
    monkeypatch.setattr('sys.argv', ['refine_ranker', '--source', str(source), '--run', str(run), '--epochs', '2'])
    with pytest.raises(RuntimeError, match='another configuration'):
        refine.main()


def test_cli_prefers_completed_refinement_and_ignores_incomplete_run(tmp_path, monkeypatch):
    from python_mvp import cli
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    (tmp_path/'data/honest-v1').mkdir(parents=True)
    (tmp_path/'data/honest-v1/ranker.pt').touch()
    (tmp_path/'data/refined-v1').mkdir()
    assert cli.default_run().name == 'honest-v1'
    (tmp_path/'data/refined-v1/ranker.pt').touch()
    assert cli.default_run().name == 'refined-v1'

from ml.data import combine_sequential_experiment, create_user_splits, make_pairs


def test_splits_are_deterministic_and_user_disjoint():
    users = [f"user-{i}" for i in range(20)]
    first = create_user_splits(users, 41)
    second = create_user_splits(users, 41)
    assert first == second
    assert not (set(first["train"]) & set(first["validation"]))
    assert not (set(first["train"]) & set(first["test"]))
    assert sorted(first["train"] + first["validation"] + first["test"]) == sorted(users)


def test_pair_builder_never_uses_seed_target_or_known_positive_as_negative():
    positives = {f"p{i}" for i in range(8)}
    pairs = make_pairs([f"p{i}" for i in range(5)], "p5", positives, ["p6", "n1", "p7", "n2"])
    assert {pair[1] for pair in pairs} == {"n1", "n2"}
    assert all(pair[0] == "p5" for pair in pairs)


def test_sequential_experiment_uses_old_users_for_training_and_keeps_new_test_frozen():
    first = {"train": ["a"], "validation": ["b"], "test": ["c"]}
    second = {"train": ["d"], "validation": ["e"], "test": ["f"]}
    assert combine_sequential_experiment(first, second) == {"train": ["a", "b", "c", "d"], "validation": ["e"], "test": ["f"]}
    try:
        combine_sequential_experiment(first, {"train": ["a"], "validation": [], "test": []})
        assert False, "expected overlap rejection"
    except ValueError:
        pass

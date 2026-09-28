import random


def create_user_splits(usernames: list[str], seed: int = 41) -> dict[str, list[str]]:
    ordered = sorted(set(usernames))
    random.Random(seed).shuffle(ordered)
    train_end = int(len(ordered) * 0.7)
    validation_end = train_end + int(len(ordered) * 0.15)
    return {"train": sorted(ordered[:train_end]), "validation": sorted(ordered[train_end:validation_end]), "test": sorted(ordered[validation_end:])}


def combine_sequential_experiment(first: dict[str, list[str]], second: dict[str, list[str]]) -> dict[str, list[str]]:
    first_users = set(first["train"] + first["validation"] + first["test"])
    second_users = set(second["train"] + second["validation"] + second["test"])
    if first_users & second_users:
        raise ValueError("sequential cohorts overlap")
    return {
        "train": sorted(first_users | set(second["train"])),
        "validation": sorted(second["validation"]),
        "test": sorted(second["test"]),
    }


def make_pairs(seeds: list[str], target: str, known_positives: set[str], candidates: list[str]) -> list[tuple[str, str]]:
    if target in seeds:
        raise ValueError("hidden target must not appear among seeds")
    return [(target, candidate) for candidate in candidates if candidate not in known_positives and candidate not in seeds and candidate != target]

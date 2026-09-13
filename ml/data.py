import random


def create_user_splits(usernames: list[str], seed: int = 41) -> dict[str, list[str]]:
    ordered = sorted(set(usernames))
    random.Random(seed).shuffle(ordered)
    train_end = int(len(ordered) * 0.7)
    validation_end = train_end + int(len(ordered) * 0.15)
    return {"train": sorted(ordered[:train_end]), "validation": sorted(ordered[train_end:validation_end]), "test": sorted(ordered[validation_end:])}


def make_pairs(seeds: list[str], target: str, known_positives: set[str], candidates: list[str]) -> list[tuple[str, str]]:
    if target in seeds:
        raise ValueError("hidden target must not appear among seeds")
    return [(target, candidate) for candidate in candidates if candidate not in known_positives and candidate not in seeds and candidate != target]

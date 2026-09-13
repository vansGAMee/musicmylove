import random


def _valid(record):
    return isinstance(record, dict) and isinstance(record.get("recording_mbid"), str) and record["recording_mbid"] and isinstance(record.get("listen_count"), int)


def is_suitable(recordings: list[dict]) -> bool:
    valid = [record for record in recordings if _valid(record)]
    return len(valid) >= 20 and sum(record["listen_count"] >= 2 for record in valid) >= 12


def build_examples(username: str, recordings: list[dict], seed: int) -> list[dict]:
    strong = sorted((record for record in recordings if _valid(record) and record["listen_count"] >= 2), key=lambda row: (-row["listen_count"], row["recording_mbid"]))[:50]
    if len(strong) < 12:
        return []
    mbids = [record["recording_mbid"] for record in strong]
    rng = random.Random(f"{seed}:{username}")
    examples = []
    for index in range(3):
        shuffled = list(mbids)
        rng.shuffle(shuffled)
        examples.append({"example_index": index, "seeds": shuffled[:5], "hidden": shuffled[5:10]})
    return examples


def negatives_from_candidates(known_positives: set[str], candidates: list[str]) -> list[str]:
    return list(dict.fromkeys(candidate for candidate in candidates if candidate not in known_positives))

import random


def _valid(record):
    return isinstance(record, dict) and isinstance(record.get("recording_mbid"), str) and record["recording_mbid"] and isinstance(record.get("listen_count"), int)


def _unique_recordings(recordings: list[dict]) -> list[dict]:
    unique: dict[str, dict] = {}
    for record in recordings:
        if _valid(record) and (record["recording_mbid"] not in unique or record["listen_count"] > unique[record["recording_mbid"]]["listen_count"]):
            unique[record["recording_mbid"]] = record
    return list(unique.values())


def validate_profile_response(value) -> dict:
    if value == {}:
        return {"payload": {"recordings": []}}
    if not isinstance(value, dict) or not isinstance(value.get("payload"), dict) or not isinstance(value["payload"].get("recordings"), list):
        raise ValueError("profile response has an invalid schema")
    normalized = dict(value)
    normalized["payload"] = dict(value["payload"])
    normalized["payload"]["recordings"] = [record for record in value["payload"]["recordings"] if _valid(record)]
    return normalized


def is_suitable(recordings: list[dict]) -> bool:
    valid = _unique_recordings(recordings)
    return len(valid) >= 20 and sum(record["listen_count"] >= 2 for record in valid) >= 12


def build_examples(username: str, recordings: list[dict], seed: int) -> list[dict]:
    strong = sorted((record for record in _unique_recordings(recordings) if record["listen_count"] >= 2), key=lambda row: (-row["listen_count"], row["recording_mbid"]))[:50]
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

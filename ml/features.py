import math


def merge_candidate_rows(seeds: list[dict], rows: list[dict]) -> dict[str, dict]:
    seed_ids = {seed["mbid"] for seed in seeds}
    by_reference: dict[str, list[dict]] = {seed["mbid"]: [] for seed in seeds}
    for row in rows:
        reference = row.get("reference_mbid")
        if reference in by_reference and isinstance(row.get("recording_mbid"), str) and isinstance(row.get("score"), (int, float)):
            by_reference[reference].append(row)
    candidates: dict[str, dict] = {}
    for reference in sorted(by_reference):
        listing = by_reference[reference]
        maximum = max([max(0.0, float(row["score"])) for row in listing] or [0.0])
        for rank, row in enumerate(listing, 1):
            mbid = row["recording_mbid"]
            if mbid in seed_ids:
                continue
            candidate = candidates.setdefault(mbid, {
                "mbid": mbid,
                "title": row.get("recording_name", mbid),
                "artist": row.get("artist_credit_name", "Unknown artist"),
                "evidence": [],
            })
            candidate["evidence"].append({
                "seed_mbid": reference,
                "normalized_similarity": max(0.0, float(row["score"])) / maximum if maximum else 0.0,
                "reciprocal_rank": 1.0 / (60 + rank),
            })
    return candidates


def build_candidate_features(candidate: dict, seeds: list[dict]) -> list[float]:
    similarities = sorted((item["normalized_similarity"] for item in candidate["evidence"]), reverse=True)
    reciprocal = sorted((item["reciprocal_rank"] for item in candidate["evidence"]), reverse=True)
    similarities = (similarities + [0.0] * 5)[:5]
    reciprocal = (reciprocal + [0.0] * 5)[:5]
    positive = [value for value in similarities if value > 0]
    mean = sum(positive) / len(positive) if positive else 0.0
    stddev = math.sqrt(sum((value - mean) ** 2 for value in positive) / len(positive)) if positive else 0.0
    artist = str(candidate.get("artist", "")).strip().casefold()
    same_artist = sum(str(seed.get("artist", "")).strip().casefold() == artist for seed in seeds) / 5
    return [
        *similarities,
        *reciprocal,
        sum(reciprocal),
        len(candidate["evidence"]) / 5,
        similarities[0],
        similarities[1],
        mean,
        stddev,
        same_artist,
    ]

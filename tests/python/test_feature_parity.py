import json
import subprocess

from ml.features import build_candidate_features, merge_candidate_rows


def row(reference, mbid, score, artist="Artist X"):
    return {"reference_mbid": reference, "recording_mbid": mbid, "recording_name": mbid, "artist_credit_name": artist, "score": score}


def test_python_features_match_typescript_contract():
    seeds = [
        {"mbid": "seed-a", "artist": "Artist A"}, {"mbid": "seed-b", "artist": "Artist B"},
        {"mbid": "seed-c", "artist": "Artist C"}, {"mbid": "seed-d", "artist": "Artist D"},
        {"mbid": "seed-e", "artist": "Artist E"},
    ]
    rows = [row("seed-a", "x", 100), row("seed-a", "y", 50), row("seed-b", "y", 20), row("seed-b", "x", 10), row("seed-e", "x", 3)]
    candidate = merge_candidate_rows(seeds, rows)["x"]
    features = build_candidate_features(candidate, seeds)
    assert len(features) == 17
    assert features[:5] == [1, 1, 0.5, 0, 0]
    assert abs(features[10] - (1 / 61 + 1 / 61 + 1 / 62)) < 1e-12
    assert features[11] == 3 / 5


def test_seed_permutation_does_not_change_features():
    seeds = [{"mbid": f"s{i}", "artist": f"A{i}"} for i in range(5)]
    rows = [row(seed["mbid"], "x", index + 1) for index, seed in enumerate(seeds)]
    first = build_candidate_features(merge_candidate_rows(seeds, rows)["x"], seeds)
    reversed_features = build_candidate_features(merge_candidate_rows(list(reversed(seeds)), rows)["x"], list(reversed(seeds)))
    assert first == reversed_features


def test_unicode_identity_and_features_match_typescript(tmp_path):
    seeds = [
        {"mbid": "s0", "title": "ΟΣ", "artist": "STRASSE"},
        *[{"mbid": f"s{i}", "title": f"T{i}", "artist": f"A{i}"} for i in range(1, 5)],
    ]
    rows = [
        {**row("s0", "alternate", 100, "strasse"), "recording_name": "ος"},
        {**row("s0", "candidate", 90, "Straße"), "recording_name": "Else"},
    ]
    candidates = merge_candidate_rows(seeds, rows)
    lists = {seed["mbid"]: [] for seed in seeds}
    for item in rows:
        lists[item["reference_mbid"]].append({"mbid": item["recording_mbid"], "title": item["recording_name"], "artist": item["artist_credit_name"], "score": item["score"]})
    path = tmp_path / "features.json"
    path.write_text(json.dumps({"seeds": seeds, "lists": lists}))
    completed = subprocess.run(["npm", "exec", "tsx", "scripts/ts-feature-parity.ts", str(path)], capture_output=True, text=True, check=True)
    typescript = json.loads(completed.stdout)
    assert typescript["mbids"] == sorted(candidates)
    assert typescript["features"] == {mbid: build_candidate_features(candidate, seeds) for mbid, candidate in candidates.items()}

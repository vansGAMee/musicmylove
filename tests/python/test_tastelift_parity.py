"""The exported PyTorch TasteLift model and production TypeScript agree."""
import json
import subprocess

import torch

from ml.tastelift_model import TasteLift


def _tracks():
    return [
        {"id": "seed-c", "artist": "Björk", "title": "Jóga"},
        {"id": "seed-a", "artist": "  Ｚｅｂｒａ   Ж ", "title": "Unknown 🐻"},
        {"id": "seed-e", "artist": "$uicideboy$", "title": "23"},
        {"id": "seed-b", "artist": "1991", "title": "28"},
        {"id": "seed-d", "artist": "Artist D", "title": "Song D"},
        {"id": "candidate-z", "artist": "Björk", "title": "Hidden Place", "popularity": {"percentile": 0.37}},
        {"id": "candidate-a", "artist": "An OOV Artist", "title": "A Very New Song"},
    ]


def _max_error(actual, expected):
    if isinstance(expected, list):
        return max((_max_error(observed, wanted) for observed, wanted in zip(actual, expected)), default=0.0)
    if isinstance(expected, dict):
        return max((_max_error(actual[key], value) for key, value in expected.items()), default=0.0)
    return abs(actual - expected)


def test_real_export_matches_typescript_heads_scores_and_lift(tmp_path):
    tracks = _tracks()
    model = TasteLift.from_export("ml/tastelift-model.json", tracks).eval()
    ids = torch.tensor([[model.track_index[track["id"]] for track in tracks[:5]]])
    candidate_ids = torch.tensor([[model.track_index[track["id"]] for track in tracks[5:]]])
    with torch.no_grad():
        heads = model.encode_set(ids, torch.ones_like(ids, dtype=torch.bool))
        score = model.score_candidates(ids, torch.ones_like(ids, dtype=torch.bool), candidate_ids)
    expected = {
        "heads": heads[0].tolist(),
        "candidates": [{key: score[key][0, index].tolist() if key == "per_head" else score[key][0, index].item()
                        for key in ("per_head", "affinity", "popularity_prior", "lift")}
                       for index in range(candidate_ids.shape[1])],
    }
    assert expected["candidates"][0]["popularity_prior"] == 0
    payload = {"seeds": tracks[:5], "candidates": [
        {"mbid": track["id"], "artist": track["artist"], "title": track["title"],
         **({"popularityPercentile": track["popularity"]["percentile"]} if "popularity" in track else {})}
        for track in tracks[5:]
    ]}
    path = tmp_path / "tastelift.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        ["npm", "exec", "tsx", "scripts/tastelift-parity.ts", str(path)],
        capture_output=True, text=True, check=True,
    )
    actual = json.loads(completed.stdout)
    assert _max_error(actual, expected) < 2e-6

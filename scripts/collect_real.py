import argparse
import hashlib
import json
import urllib.parse
from pathlib import Path

from ml.api import ApiClient, PermanentApiError
from ml.data import create_user_splits
from ml.examples import build_examples, is_suitable, validate_profile_response
from ml.train_real import validate_similarity_rows

ALGORITHM = "session_based_days_7500_session_300_contribution_5_threshold_15_limit_50_skip_30_top_n_listeners_1000"
MLHD_ALGORITHM = "session_based_mlhd_session_300_contribution_5_threshold_15_limit_50_skip_30"


def key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["profiles", "retrieval", "all"], default="all")
    parser.add_argument("--max-users", type=int, default=400)
    parser.add_argument("--retrieval-source", choices=["listenbrainz", "mlhd"], default="listenbrainz")
    parser.add_argument("--partitions", nargs="+", choices=["train", "validation", "test"], default=["train", "validation", "test"])
    parser.add_argument("--manifest", default="data/manifests/usernames.json")
    parser.add_argument("--split", default="data/manifests/real-splits.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / args.manifest).read_text())
    client = ApiClient()
    profile_dir = root / "data/cache/real-profiles"
    usable = []
    for index, username in enumerate(manifest["usernames"], 1):
        path = profile_dir / f"{key(username)}.json"
        try:
            value = client.cached_json(path, "GET", f"https://api.listenbrainz.org/1/stats/user/{urllib.parse.quote(username, safe='')}/recordings?range=all_time&count=100", validator=validate_profile_response)
            recordings = value.get("payload", {}).get("recordings", [])
            if is_suitable(recordings):
                usable.append({"username": username, "profile": path.name, "recordings": recordings})
        except PermanentApiError as error:
            print(f"profile_error index={index} type={type(error).__name__}", flush=True)
        if index % 25 == 0:
            print(f"profiles={index}/{len(manifest['usernames'])} usable={len(usable)}", flush=True)
    usable = usable[:args.max_users]
    split_path = root / args.split
    split_path.parent.mkdir(parents=True, exist_ok=True)
    if split_path.exists():
        splits = json.loads(split_path.read_text())
    else:
        splits = {"seed": 41, **create_user_splits([item["username"] for item in usable], 41)}
        split_path.write_text(json.dumps(splits, indent=2) + "\n")
    if args.stage == "profiles":
        print(json.dumps({"usable": len(usable), "split_counts": {name: len(splits[name]) for name in ("train", "validation", "test")}})); return
    by_name = {item["username"]: item for item in usable}
    suffix = "" if args.retrieval_source == "listenbrainz" else "-mlhd"
    retrieval_dir = root / f"data/cache/real-retrieval{suffix}"
    retrieval_raw_dir = root / f"data/cache/real-retrieval-raw{suffix}"
    endpoint = "similar-recordings" if args.retrieval_source == "listenbrainz" else "mlhd-similar-recordings"
    algorithm = ALGORITHM if args.retrieval_source == "listenbrainz" else MLHD_ALGORITHM
    completed = 0
    target_count = sum(len(splits[partition]) for partition in args.partitions)
    for partition in args.partitions:
        for username in splits[partition]:
            if username not in by_name:
                continue
            destination = retrieval_dir / f"{key(username)}.json"
            examples = build_examples(username, by_name[username]["recordings"], 41)
            seed_mbids = sorted({mbid for example in examples for mbid in example["seeds"]})
            payload = [{"recording_mbids": seed_mbids, "algorithm": algorithm}]
            legacy_raw = retrieval_raw_dir / f"{key(username)}.json"
            previous_seeds = None
            if destination.exists():
                previous = json.loads(destination.read_text())
                previous_seeds = sorted({mbid for example in previous.get("examples", []) for mbid in example.get("seeds", [])})
            payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
            raw_destination = legacy_raw if legacy_raw.exists() and previous_seeds == seed_mbids else retrieval_raw_dir / f"{key(username)}-{payload_hash}.json"
            try:
                rows = client.cached_json(
                    raw_destination,
                    "POST",
                    f"https://labs.api.listenbrainz.org/{endpoint}/json",
                    payload,
                    validator=validate_similarity_rows,
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".json.tmp")
                temporary.write_text(json.dumps({"partition": partition, "examples": examples, "rows": rows}))
                temporary.replace(destination)
                completed += 1
            except PermanentApiError as error:
                print(f"retrieval_error completed={completed} type={type(error).__name__}", flush=True)
            if completed % 25 == 0:
                print(f"retrieval={completed}/{target_count}", flush=True)
    print(json.dumps({"usable": len(usable), "retrieval_source": args.retrieval_source, "retrieval_users": completed, "split_counts": {name: len(splits[name]) for name in ("train", "validation", "test")}}))


if __name__ == "__main__": main()

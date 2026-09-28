import argparse
import hashlib
import json
from pathlib import Path

from ml.cohort import activity_from_archive, choose_active_users_excluding, discover_latest_archive, download_verified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/manifests/usernames.json")
    parser.add_argument("--exclude")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--source-url")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.source_url:
        url = args.source_url
        dump_id = url.rstrip("/").split("/")[-2]
    else:
        url, dump_id = discover_latest_archive()
    archive = root / "data/downloads" / url.rsplit("/", 1)[-1]
    byte_count, sha256 = download_verified(url, archive)
    try:
        counts = activity_from_archive(archive)
        excluded = set()
        if args.exclude:
            excluded = set(json.loads((root / args.exclude).read_text()).get("usernames", []))
        selected = choose_active_users_excluding(counts, args.count, 20, args.seed, excluded)
        manifest = {
            "source_url": url,
            "dump_id": dump_id,
            "compressed_bytes": byte_count,
            "sha256": sha256,
            "unique_usernames_in_dump": len(counts),
            "users_with_at_least_20_incremental_listens": sum(value >= 20 for value in counts.values()),
            "activity_threshold": 20,
            "sample_seed": args.seed,
            "excluded_users": len(excluded),
            "usernames": selected,
        }
        destination = root / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2) + "\n")
        sample_hash = hashlib.sha256("\n".join(selected).encode()).hexdigest()
        print(json.dumps({key: value for key, value in manifest.items() if key != "usernames"} | {"sampled_users": len(selected), "sample_sha256": sample_hash}))
    finally:
        archive.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

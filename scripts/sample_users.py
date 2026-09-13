import hashlib
import json
from pathlib import Path

from ml.cohort import choose_users, discover_latest_archive, download_verified, usernames_from_archive


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    url, dump_id = discover_latest_archive()
    archive = root / "data/downloads" / url.rsplit("/", 1)[-1]
    byte_count, sha256 = download_verified(url, archive)
    try:
        all_names = usernames_from_archive(archive)
        selected = choose_users(all_names, 500, 41)
        manifest = {
            "source_url": url,
            "dump_id": dump_id,
            "compressed_bytes": byte_count,
            "sha256": sha256,
            "unique_usernames_in_dump": len(all_names),
            "sample_seed": 41,
            "usernames": selected,
        }
        destination = root / "data/manifests/usernames.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2) + "\n")
        sample_hash = hashlib.sha256("\n".join(selected).encode()).hexdigest()
        print(json.dumps({key: value for key, value in manifest.items() if key != "usernames"} | {"sampled_users": len(selected), "sample_sha256": sample_hash}))
    finally:
        archive.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

import argparse
import hashlib
import json
from pathlib import Path

from ml.data import combine_sequential_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    first = json.loads((root / args.first).read_text())
    second = json.loads((root / args.second).read_text())
    combined = combine_sequential_experiment(first, second)
    destination = root / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(combined, indent=2) + "\n")
    hashes = {name: hashlib.sha256("\n".join(values).encode()).hexdigest() for name, values in combined.items()}
    print(json.dumps({"counts": {name: len(values) for name, values in combined.items()}, "hashes": hashes}))


if __name__ == "__main__": main()

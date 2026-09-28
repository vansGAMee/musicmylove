import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.tastelift_catalog import build_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the train-only TasteLift neural retrieval catalog.")
    parser.add_argument("--dataset", default="data/cache/tastelift/dataset.json")
    parser.add_argument("--model", default="ml/tastelift-model.json")
    parser.add_argument("--manifest", default="data/manifests/real-splits-final.json")
    parser.add_argument("--output", default="ml/tastelift-catalog.json")
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = build_catalog(root, root / args.dataset, root / args.model, root / args.manifest, root / args.output, args.batch_size)
    print(json.dumps({"tracks": len(result["tracks"]), "dimensions": result["dimensions"], "maximum_error": result["quantization"]["maximumError"]}))


if __name__ == "__main__":
    main()

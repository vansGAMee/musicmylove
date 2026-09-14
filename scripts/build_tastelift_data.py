import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.tastelift_data import DEFAULT_MASKS_PER_USER, build_dataset, write_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline deterministic TasteLift episodes from existing caches.")
    parser.add_argument("--output", default="data/cache/tastelift/dataset.json")
    parser.add_argument("--checkpoint", default="data/cache/tastelift/dataset.checkpoint.json")
    parser.add_argument("--masks-per-user", type=int, default=DEFAULT_MASKS_PER_USER)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset = build_dataset(root, masks_per_user=args.masks_per_user, seed=args.seed, checkpoint_path=root / args.checkpoint, checkpoint_every=args.checkpoint_every)
    write_dataset(root / args.output, dataset)
    print(json.dumps(dataset["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()

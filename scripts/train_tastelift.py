#!/usr/bin/env python3
"""Train TasteLift from local episodes; full runs belong in detached Kitty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ml.tastelift_model import train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/cache/tastelift/dataset.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/cache/tastelift/checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("ml/tastelift-model.json"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-train-rows", type=int)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    before = args.dataset.stat()
    dataset = json.loads(args.dataset.read_text())
    after = args.dataset.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("dataset changed while loading; retry after atomic publication")
    if dataset.get("version") != 2:
        raise ValueError("TasteLift requires rebuilt dataset version 2")
    model = train_model(dataset, epochs=args.epochs, checkpoint_dir=args.checkpoint_dir,
                        batch_size=args.batch_size, seed=args.seed, resume=args.resume,
                        device=args.device, max_train_rows=args.max_train_rows, patience=args.patience)
    best = torch.load(args.checkpoint_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    model.training_summary["selected_epoch"] = best["epoch"]
    model.export_json(args.output)
    print(json.dumps({"event": "exported", "path": str(args.output), "selected_epoch": best["epoch"],
                      "bytes": args.output.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()

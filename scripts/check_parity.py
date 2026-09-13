import json
import random
import subprocess
import tempfile
from pathlib import Path

from ml.model import TinyRanker, export_model


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_path = root / "ml" / "model.json"
    if not model_path.exists():
        import torch
        torch.manual_seed(13)
        export_model(TinyRanker(), model_path, [f"feature_{i}" for i in range(17)], "rrf")
    artifact = json.loads(model_path.read_text())
    rng = random.Random(19)
    vectors = [[rng.random() for _ in range(17)] for _ in range(32)]
    values = vectors
    for index, layer in enumerate(artifact["layers"]):
        values = [[sum(w * x for w, x in zip(row, vector)) + bias for row, bias in zip(layer["weight"], layer["bias"])] for vector in values]
        if index < len(artifact["layers"]) - 1:
            values = [[max(0.0, value) for value in vector] for vector in values]
    python_scores = [value[0] for value in values]
    with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
        json.dump({"model": artifact, "vectors": vectors}, handle)
        handle.flush()
        output = subprocess.check_output(["npx", "tsx", "scripts/ts-parity.ts", handle.name], cwd=root, text=True)
    ts_scores = json.loads(output)
    maximum = max(abs(a - b) for a, b in zip(python_scores, ts_scores))
    if maximum > 1e-9:
        raise SystemExit(f"parity failed: {maximum}")
    print(f"parity vectors=32 max_abs_error={maximum:.3e}")


if __name__ == "__main__":
    main()

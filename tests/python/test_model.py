import json
from pathlib import Path

import torch

from ml.model import TinyRanker, export_model


def test_export_preserves_forward_scores(tmp_path: Path):
    torch.manual_seed(7)
    model = TinyRanker(17)
    vector = torch.linspace(0, 1, 17)
    expected = model(vector).item()
    output = tmp_path / "model.json"
    export_model(model, output, [f"f{i}" for i in range(17)], "rrf")
    artifact = json.loads(output.read_text())
    values = vector.tolist()
    for index, layer in enumerate(artifact["layers"]):
        values = [sum(w * x for w, x in zip(row, values)) + bias for row, bias in zip(layer["weight"], layer["bias"])]
        if index < len(artifact["layers"]) - 1:
            values = [max(0.0, value) for value in values]
    assert abs(values[0] - expected) < 1e-7

import json
from pathlib import Path

import torch
from torch import nn


class TinyRanker(nn.Module):
    def __init__(self, feature_count: int = 17):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def export_model(model: TinyRanker, path: Path, feature_names: list[str], production_ranker: str, ensemble_alpha: float | None = None) -> None:
    layers = []
    for module in model.network:
        if isinstance(module, nn.Linear):
            layers.append({
                "weight": module.weight.detach().double().tolist(),
                "bias": module.bias.detach().double().tolist(),
            })
    artifact = {
        "version": 1,
        "feature_names": feature_names,
        "activation": "relu",
        "production_ranker": production_ranker,
        "ensemble_alpha": ensemble_alpha,
        "layers": layers,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n")


def load_model(path: Path, dtype=torch.float64) -> tuple[TinyRanker, dict]:
    artifact = json.loads(path.read_text())
    model = TinyRanker(len(artifact["feature_names"])).to(dtype=dtype)
    linear_layers = [module for module in model.network if isinstance(module, nn.Linear)]
    if len(linear_layers) != len(artifact["layers"]):
        raise ValueError("artifact layer count mismatch")
    with torch.no_grad():
        for module, layer in zip(linear_layers, artifact["layers"]):
            module.weight.copy_(torch.tensor(layer["weight"], dtype=dtype))
            module.bias.copy_(torch.tensor(layer["bias"], dtype=dtype))
    model.eval()
    return model, artifact

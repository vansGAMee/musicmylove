import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from ml.model import TinyRanker, export_model


def main():
    random.seed(41); torch.manual_seed(41)
    model = TinyRanker(17)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    positives = torch.tensor([[min(1, random.random() + 0.35) for _ in range(17)] for _ in range(400)])
    negatives = torch.tensor([[random.random() * 0.55 for _ in range(17)] for _ in range(400)])
    last = 0.0
    for _ in range(80):
        loss = F.softplus(-(model(positives) - model(negatives))).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step(); last = loss.item()
    root = Path(__file__).resolve().parents[1]
    export_model(model, root / "ml/model.json", [f"feature_{i}" for i in range(17)], "rrf")
    print(json.dumps({"examples": 400, "epochs": 80, "final_bpr_loss": last}))


if __name__ == "__main__": main()

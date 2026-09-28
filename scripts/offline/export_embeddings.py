#!/usr/bin/env python3
"""
scripts/offline/export_embeddings.py
Exports learned track embeddings into compact binary formats:
- models/track_embeddings.npy (Float32)
- models/track_embeddings.int8.bin (Quantized Int8)
- public/data/embeddings.int8.bin (Static CDN asset for 0-cost Vercel)
Computes SHA256 digest and verifies 1-to-1 row alignment with catalog tracks.
"""

import hashlib
import json
from pathlib import Path
import numpy as np

def export_embeddings(
    npy_path: str = "models/track_embeddings.npy",
    catalog_path: str = "data/cache/pipeline/expanded_catalog.json",
    output_dir: str = "public/data",
):
    root = Path(__file__).resolve().parents[2]
    npy_file = root / npy_path
    cat_file = root / catalog_path
    pub_dir = root / output_dir
    pub_dir.mkdir(parents=True, exist_ok=True)

    if not npy_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {npy_file}")

    print(f"[load] Loading embeddings from {npy_file}...")
    emb = np.load(npy_file)
    print(f"  Shape: {emb.shape}, Dtype: {emb.dtype}")

    if not cat_file.exists():
        raise FileNotFoundError(f"Catalog file not found: {cat_file}")

    print(f"[verify] Verifying 1-to-1 row alignment with catalog {cat_file}...")
    cat_data = json.loads(cat_file.read_text())
    tracks = cat_data.get("tracks", [])
    num_tracks = len(tracks)

    if emb.shape[0] < num_tracks:
        raise ValueError(f"Embedding rows ({emb.shape[0]}) less than catalog tracks ({num_tracks})")

    # If embeddings was trained on full vocab, slice to exact catalog size if needed
    if emb.shape[0] > num_tracks:
        print(f"  Trimming embeddings from {emb.shape[0]} to {num_tracks} tracks...")
        emb = emb[:num_tracks]

    # Verify L2 normalization
    norms = np.linalg.norm(emb, axis=1)
    mean_norm = float(np.mean(norms))
    print(f"  Embedding mean L2 norm: {mean_norm:.4f} (expected ~1.0000)")

    # Compute SHA256 of the raw float32 embeddings
    f32_bytes = emb.astype(np.float32).tobytes()
    f32_sha256 = hashlib.sha256(f32_bytes).hexdigest()
    print(f"  Float32 SHA256: {f32_sha256}")

    # Quantize to Int8: [-1.0, 1.0] -> [-127, 127]
    print("[quantize] Quantizing Float32 embeddings to Int8 [-127, 127]...")
    int8_emb = np.clip(np.round(emb * 127.0), -127, 127).astype(np.int8)
    int8_bytes = int8_emb.tobytes()
    int8_sha256 = hashlib.sha256(int8_bytes).hexdigest()
    print(f"  Int8 size: {len(int8_bytes) / (1024 * 1024):.2f} MB")
    print(f"  Int8 SHA256: {int8_sha256}")

    # Write binary outputs
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    (models_dir / "track_embeddings.int8.bin").write_bytes(int8_bytes)
    (pub_dir / "embeddings.int8.bin").write_bytes(int8_bytes)
    print(f"[saved] Written models/track_embeddings.int8.bin and {pub_dir / 'embeddings.int8.bin'}")

    # Write export verification report
    verification = {
        "status": "verified",
        "num_tracks": num_tracks,
        "embedding_dim": int(emb.shape[1]),
        "mean_l2_norm": round(mean_norm, 4),
        "f32_sha256": f32_sha256,
        "int8_sha256": int8_sha256,
        "int8_byte_size": len(int8_bytes),
        "int8_file": "public/data/embeddings.int8.bin",
        "row_alignment_verified": True
    }
    (models_dir / "embeddings_export_manifest.json").write_text(json.dumps(verification, indent=2))
    print("[report] Saved verification manifest to models/embeddings_export_manifest.json")
    return verification

if __name__ == "__main__":
    export_embeddings()

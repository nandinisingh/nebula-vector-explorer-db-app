from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DIMENSIONS = [f"dim_{i}" for i in range(256)]
REQUIRED_FILES = ["item_embeddings.csv", "user_embeddings.csv", "item_metadata.csv", "user_metadata.csv",
                  "item_embeddings_3d.csv", "user_embeddings_3d.csv", "evaluation_results_per_user.csv",
                  "evaluation_metrics_aggregate.json"]


def model_dir(model_id: str = "model_1") -> Path:
    return DATA_ROOT / model_id


def available_models() -> list[str]:
    if not DATA_ROOT.exists():
        return []
    return sorted(path.name for path in DATA_ROOT.iterdir() if path.is_dir() and all((path / name).exists() for name in REQUIRED_FILES))


def load_data(model_id: str = "model_1") -> dict:
    root = model_dir(model_id)
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {', '.join(missing)} in {root}. Run `python generate_synthetic_data.py` from the embedding_explorer directory.")
    items = pd.read_csv(root / "item_metadata.csv", low_memory=False)
    users = pd.read_csv(root / "user_metadata.csv")
    coords = pd.read_csv(root / "item_embeddings_3d.csv")
    user_coords = pd.read_csv(root / "user_embeddings_3d.csv")
    item_emb = pd.read_csv(root / "item_embeddings.csv", dtype={"item_id": "category"})
    user_emb = pd.read_csv(root / "user_embeddings.csv", dtype={"user_id": "category"})
    if not set(["item_id", "x", "y", "z"]).issubset(coords) or not set(DIMENSIONS).issubset(item_emb):
        raise ValueError("Projection or item embedding file has an invalid schema (expected 256 dimensions and x/y/z coordinates).")
    if item_emb["item_id"].duplicated().any() or items["item_id"].duplicated().any():
        raise ValueError("Item IDs must be unique in embeddings and metadata.")
    if set(item_emb.item_id.astype(str)) != set(items.item_id.astype(str)):
        raise ValueError("Item IDs do not match between embeddings and metadata.")
    if not set(["user_id", "x", "y", "z"]).issubset(user_coords):
        raise ValueError("User projection file has an invalid schema (expected user_id and x/y/z coordinates).")
    if set(user_coords.user_id.astype(str)) != set(users.user_id.astype(str)):
        raise ValueError("User IDs do not match between user metadata and 3D projection.")
    if coords.item_id.duplicated().any() or set(coords.item_id.astype(str)) != set(items.item_id.astype(str)):
        raise ValueError("Item IDs do not match between item metadata and 3D projection.")
    if set(user_emb.user_id.astype(str)) != set(users.user_id.astype(str)):
        raise ValueError("User IDs do not match between user metadata and embeddings.")
    evaluation = pd.read_csv(root / "evaluation_results_per_user.csv")
    aggregate = json.loads((root / "evaluation_metrics_aggregate.json").read_text())
    # Keep only low-dimensional data in the plotted frame; embeddings stay available for cosine search.
    plotted = coords.merge(items, on="item_id", validate="one_to_one")
    users = users.merge(user_coords, on="user_id", validate="one_to_one")
    return {"model_id": model_id, "items": plotted, "users": users, "item_embeddings": item_emb.set_index("item_id")[DIMENSIONS].to_numpy(dtype=np.float32),
            "item_ids": item_emb.item_id.astype(str).to_numpy(), "user_embeddings": user_emb.set_index("user_id")[DIMENSIONS].to_numpy(dtype=np.float32),
            "user_ids": user_emb.user_id.astype(str).to_numpy(), "evaluation": evaluation, "aggregate": aggregate}

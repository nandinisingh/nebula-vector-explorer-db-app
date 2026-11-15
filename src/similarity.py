from __future__ import annotations

import numpy as np


def _rank(query: np.ndarray, matrix: np.ndarray, ids: np.ndarray, top_k: int, exclude: int | None = None):
    scores = matrix @ query.astype(np.float32)
    if exclude is not None:
        scores[exclude] = -np.inf
    take = min(top_k, len(scores))
    ix = np.argpartition(scores, -take)[-take:]
    ix = ix[np.argsort(scores[ix])[::-1]]
    return [(str(ids[i]), float(scores[i])) for i in ix]


def compute_item_neighbors(item_id, embeddings, ids, top_k=20):
    lookup = {str(value): i for i, value in enumerate(ids)}
    index = lookup[str(item_id)]
    return _rank(embeddings[index], embeddings, ids, top_k, index)


def compute_user_item_ranking(user_id, user_embeddings, user_ids, item_embeddings, item_ids, top_k=50):
    lookup = {str(value): i for i, value in enumerate(user_ids)}
    return _rank(user_embeddings[lookup[str(user_id)]], item_embeddings, item_ids, top_k)

# Recommendation Embedding Explorer

A local Streamlit tool for exploring synthetic user and item embeddings, cosine neighbors, and evaluation results. The demo includes two models over the same 100,000 items and 1,000 users with 256-dimensional normalized vectors.

## Setup

```bash
cd nebula-vector-explorer-db-app
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python generate_synthetic_data.py
streamlit run src/app.py
```

Generation is deterministic (seed 42) and writes eight files into each of `data/model_1/` and `data/model_2/`. Generated datasets are intentionally not stored in Git: the two 100,000 × 256 item-vector CSVs exceed GitHub’s per-file size limit. Generate them locally before launching the app. The interactive 3D projections use broad, overlapping synthetic neighborhoods with uneven spread; coordinates are generated without fitting UMAP to 100,000 vectors so setup stays practical. Run `python generate_synthetic_data.py --projections-only` to refresh just the illustrative 3D layouts without rewriting the 256D embeddings or evaluation files. Model 2 is a controlled synthetic variant, not a real production model. Both folders use the same catalog and users.

## Use

- **Item space:** Explore the full catalog, color by item type or metadata, filter popular versus long-tail items, select an item, and inspect metadata plus 20 nearest neighbors. The 3D dense-region control frames the central 98% on each axis; turn it off to include sparse tails. “Surprise me” opens a random item. Show user points and select a user to inspect their top-20 embedding candidates.
- **Compare models:** Compare aligned 3D projections, movement distribution, and changes in an item's top-20 neighbors. Alignment is for visual comparison; cosine rankings are computed in each model's original 256D space.
- **Model quality:** Compare the radar profile across user cohorts and inspect recommendation exposure by item metadata. The metrics in both models are synthetic demo values.
- Evaluation values are supplied by generated files; the app does not compute metrics from raw interaction logs.

Similarity scores are computed in the original 256D space. The 3D plot is only an exploratory projection and global distances are distorted.

## Data files

`item_embeddings.csv`, `user_embeddings.csv`, `item_metadata.csv`, `user_metadata.csv`, `item_embeddings_3d.csv`, `user_embeddings_3d.csv`, `evaluation_results_per_user.csv`, and `evaluation_metrics_aggregate.json`.

The loader validates required files, dimensionality, unique IDs, and item/user ID alignment. To add a model, place a complete `model_3/` directory under `data/` with the same schema, then refresh the app. The model selectors discover complete directories automatically.

## Metric notes

The synthetic quality dashboard reports relevance (NDCG@50 and Recall@50), freshness (released 2022 or later in this synthetic dataset), diversity (genre plus artist/show variety), discovery (genres absent from the user's synthetic recent history), localization (preferred-language match), and popularity calibration (closeness to a synthetic per-user popularity target). These values are examples for interface development, not evidence of model performance. For real evaluation, use a consistent time-based test set and candidate eligibility rules, and document the exact definition and denominator for every metric.

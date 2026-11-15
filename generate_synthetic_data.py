"""Create a reproducible synthetic recommendation embedding dataset."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "model_1"
OUT_2 = ROOT / "data" / "model_2"
N_ITEMS, N_USERS, DIMS = 100_000, 1_000, 256
SEED = 42
MUSIC_GENRES = ["pop", "rock", "hip-hop", "electronic", "jazz", "classical", "folk", "country"]
PODCAST_TOPICS = ["news", "comedy", "science", "history", "business", "health"]
THEME_ANCHORS = {
    "culture": np.array([-3.2, .1, .2], dtype="float32"),
    "technology": np.array([2.8, 2.0, .1], dtype="float32"),
    "arts": np.array([.3, 3.4, 2.0], dtype="float32"),
    "heritage": np.array([-2.1, -2.6, 2.2], dtype="float32"),
    "wellbeing": np.array([2.2, -2.4, -2.1], dtype="float32"),
}


def unit_rows(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype("float32")


def projection_parameters(vocab: list[str], theme_by_genre: dict[str, str], rng: np.random.Generator):
    """Spread related themes across a loose 3D landscape with overlapping genre centers."""
    anchors = {name: point + rng.normal(0, .28, 3).astype("float32") for name, point in THEME_ANCHORS.items()}
    genre_offsets = {name: rng.normal(0, .58, 3).astype("float32") for name in vocab}
    rotations = {}
    for name in vocab:
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        rotations[name] = rotation.astype("float32")
    return anchors, genre_offsets, rotations


def project_labels(labels: np.ndarray, vocab: list[str], theme_by_genre: dict[str, str], params: tuple,
                   rng: np.random.Generator, noise_scale: float = 1.0) -> np.ndarray:
    anchors, genre_offsets, rotations = params
    result = np.empty((len(labels), 3), dtype="float32")
    for genre in vocab:
        ix = np.flatnonzero(labels == genre)
        if not len(ix):
            continue
        base = anchors[theme_by_genre[genre]] + genre_offsets[genre]
        noise = rng.normal(size=(len(ix), 3)).astype("float32") * np.array([1.05, .76, .58], dtype="float32")
        noise = (noise @ rotations[genre]).astype("float32") * noise_scale
        # A small broad tail makes boundaries imperfect and leaves occasional
        # cross-neighborhood points for open-ended inspection.
        tails = rng.random(len(ix)) < .07
        noise[tails] += rng.normal(0, 1.45, (int(tails.sum()), 3)).astype("float32")
        result[ix] = base + noise
    result[:, 1] += .12 * np.sin(result[:, 0] * .55)
    result[:, 2] += .10 * np.cos(result[:, 1] * .5)
    return result


def write_projection_only() -> None:
    """Regenerate only the illustrative 3D layout without rewriting embeddings/evaluation files."""
    vocab = MUSIC_GENRES + PODCAST_TOPICS
    theme_by_genre = {"pop": "culture", "rock": "culture", "hip-hop": "culture", "electronic": "technology",
                      "jazz": "culture", "classical": "arts", "folk": "heritage", "country": "heritage",
                      "news": "culture", "comedy": "culture", "science": "technology", "history": "heritage",
                      "business": "technology", "health": "wellbeing"}
    rng = np.random.default_rng(SEED + 1000)
    items = pd.read_csv(OUT / "item_metadata.csv", usecols=["item_id", "genre"])
    users = pd.read_csv(OUT / "user_metadata.csv", usecols=["user_id", "preferred_genre"])
    params = projection_parameters(vocab, theme_by_genre, rng)
    xyz = project_labels(items.genre.to_numpy(), vocab, theme_by_genre, params, rng)
    user_xyz = project_labels(users.preferred_genre.to_numpy(), vocab, theme_by_genre, params, rng, noise_scale=.78)
    pd.DataFrame({"item_id": items.item_id, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
                  "projection_method": "synthetic_overlapping_theme_layout_3d"}).to_csv(OUT / "item_embeddings_3d.csv", index=False, float_format="%.5g")
    pd.DataFrame({"user_id": users.user_id, "x": user_xyz[:, 0], "y": user_xyz[:, 1], "z": user_xyz[:, 2],
                  "projection_method": "synthetic_overlapping_theme_layout_3d"}).to_csv(OUT / "user_embeddings_3d.csv", index=False, float_format="%.5g")
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    offset = np.array([.9, -.6, .45], dtype="float32")
    model2_xyz = (xyz @ rotation).astype("float32") * 1.08 + offset + rng.normal(0, .3, xyz.shape).astype("float32")
    model2_user_xyz = (user_xyz @ rotation).astype("float32") * 1.08 + offset + rng.normal(0, .28, user_xyz.shape).astype("float32")
    pd.DataFrame({"item_id": items.item_id, "x": model2_xyz[:, 0], "y": model2_xyz[:, 1], "z": model2_xyz[:, 2],
                  "projection_method": "synthetic_overlapping_theme_layout_3d"}).to_csv(OUT_2 / "item_embeddings_3d.csv", index=False, float_format="%.5g")
    pd.DataFrame({"user_id": users.user_id, "x": model2_user_xyz[:, 0], "y": model2_user_xyz[:, 1], "z": model2_user_xyz[:, 2],
                  "projection_method": "synthetic_overlapping_theme_layout_3d"}).to_csv(OUT_2 / "user_embeddings_3d.csv", index=False, float_format="%.5g")
    print("Updated illustrative 3D projection files for Model 1 and Model 2.")


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    n_music = int(N_ITEMS * 0.7)
    n_podcast = N_ITEMS - n_music
    types = np.array(["music"] * n_music + ["podcast"] * n_podcast)
    genres = np.concatenate([rng.choice(MUSIC_GENRES, n_music), rng.choice(PODCAST_TOPICS, n_podcast)])
    ids = np.array([f"music_{i:06d}" for i in range(1, n_music + 1)] + [f"podcast_{i:06d}" for i in range(1, n_podcast + 1)])
    vocab = MUSIC_GENRES + PODCAST_TOPICS
    theme_by_genre = {"pop": "culture", "rock": "culture", "hip-hop": "culture", "electronic": "technology",
                      "jazz": "culture", "classical": "arts", "folk": "heritage", "country": "heritage",
                      "news": "culture", "comedy": "culture", "science": "technology", "history": "heritage",
                      "business": "technology", "health": "wellbeing"}
    themes = list(dict.fromkeys(theme_by_genre[name] for name in vocab))
    theme_vectors = unit_rows(rng.normal(size=(len(themes), DIMS)).astype("float32"))
    theme_vector_by_name = dict(zip(themes, theme_vectors))
    centers = unit_rows(np.stack([theme_vector_by_name[theme_by_genre[name]] + rng.normal(0, .035, DIMS) for name in vocab]).astype("float32"))
    centers_by_genre = {name: centers[i] for i, name in enumerate(vocab)}
    item_vecs = np.empty((N_ITEMS, DIMS), dtype="float32")
    for name in vocab:
        ix = np.flatnonzero(genres == name)
        # Keep a preference signal while allowing meaningful within-topic variation.
        item_vecs[ix] = unit_rows(centers_by_genre[name] + rng.normal(0, 0.075, (len(ix), DIMS)).astype("float32"))

    years = np.where(types == "music", rng.integers(1960, 2025, N_ITEMS), rng.integers(2015, 2025, N_ITEMS))
    popularity = rng.beta(2.2, 4.5, N_ITEMS).astype("float32")
    energy = rng.beta(2.2, 2.0, N_ITEMS).astype("float32")
    valence = rng.beta(2.1, 2.1, N_ITEMS).astype("float32")
    duration = np.where(types == "music", rng.integers(120, 480, N_ITEMS), rng.integers(600, 5400, N_ITEMS))
    tempo = np.where(types == "music", rng.integers(60, 201, N_ITEMS), np.nan)
    acoustic = np.where(types == "music", rng.beta(2, 3, N_ITEMS), np.nan)
    language = rng.choice(["English", "Swedish", "Spanish", "French", "German"], N_ITEMS, p=[.58, .13, .11, .1, .08])
    artist_ids = np.where(types == "music", rng.choice([f"artist_{i:04d}" for i in range(1, 3001)], N_ITEMS), "")
    show_ids = np.where(types == "podcast", rng.choice([f"show_{i:04d}" for i in range(1, 1601)], N_ITEMS), "")
    creators = np.where(types == "music", artist_ids, show_ids)
    era = np.where(years < 1980, "1960s–70s", np.where(years < 2000, "1980s–90s", np.where(years < 2015, "2000s–14", "2015+")))
    item_names = [f"{g.title()} {t.title()} {i:06d}" for i, (g, t) in enumerate(zip(genres, types), 1)]
    metadata = pd.DataFrame({"item_id": ids, "item_name": item_names, "item_type": types, "genre": genres,
                             "release_year": years, "release_era": era, "popularity_score": popularity,
                             "duration_seconds": duration, "energy": energy, "valence": valence,
                             "tempo": tempo, "acousticness": acoustic, "language": language,
                             "artist_id": artist_ids, "podcast_show_id": show_ids, "creator_id": creators})
    metadata.to_csv(OUT / "item_metadata.csv", index=False)
    emb = pd.DataFrame(item_vecs, columns=[f"dim_{i}" for i in range(DIMS)])
    emb.insert(0, "item_id", ids)
    emb.to_csv(OUT / "item_embeddings.csv", index=False, float_format="%.6g")

    user_ids = np.array([f"user_{i:04d}" for i in range(1, N_USERS + 1)])
    preferred = rng.choice(vocab, N_USERS)
    user_vecs = unit_rows(np.stack([centers_by_genre[g] for g in preferred]) + rng.normal(0, 0.09, (N_USERS, DIMS)).astype("float32"))
    user_emb = pd.DataFrame(user_vecs, columns=[f"dim_{i}" for i in range(DIMS)])
    user_emb.insert(0, "user_id", user_ids)
    user_emb.to_csv(OUT / "user_embeddings.csv", index=False, float_format="%.6g")
    preferred_language = rng.choice(["English", "Swedish", "Spanish", "French", "German"], N_USERS, p=[.58, .13, .11, .1, .08])
    user_meta = pd.DataFrame({"user_id": user_ids, "user_segment": rng.choice(["casual", "regular", "power_user"], N_USERS, p=[.5, .35, .15]),
                              "signup_year": rng.integers(2018, 2025, N_USERS),
                              "primary_preference": np.where(np.isin(preferred, MUSIC_GENRES), "music", "podcast"),
                              "preferred_genre": preferred, "preferred_language": preferred_language,
                              "region": np.where(preferred_language == "Swedish", "Nordics", np.where(preferred_language == "Spanish", "LatAm", "Global")),
                              "popularity_target": rng.uniform(.25, .65, N_USERS).round(3)})
    user_meta.to_csv(OUT / "user_metadata.csv", index=False)

    # Synthetic, overlapping 3D layout; not a fitted dimensionality reduction.
    projection_rng = np.random.default_rng(SEED + 1000)
    projection_params = projection_parameters(vocab, theme_by_genre, projection_rng)
    xyz = project_labels(genres, vocab, theme_by_genre, projection_params, projection_rng)
    pd.DataFrame({"item_id": ids, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "projection_method": "synthetic_umap_layout_3d"}).to_csv(OUT / "item_embeddings_3d.csv", index=False, float_format="%.5g")

    user_xyz = project_labels(preferred, vocab, theme_by_genre, projection_params, projection_rng, noise_scale=.78)
    pd.DataFrame({"user_id": user_ids, "x": user_xyz[:, 0], "y": user_xyz[:, 1], "z": user_xyz[:, 2],
                  "projection_method": "synthetic_umap_layout_3d"}).to_csv(OUT / "user_embeddings_3d.csv", index=False, float_format="%.5g")

    # Model 2 is a controlled synthetic variant over the same catalog and users.
    OUT_2.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(OUT_2 / "item_metadata.csv", index=False)
    user_meta.to_csv(OUT_2 / "user_metadata.csv", index=False)
    model2_items = unit_rows(item_vecs + rng.normal(0, .035, item_vecs.shape).astype("float32"))
    model2_users = unit_rows(user_vecs + rng.normal(0, .035, user_vecs.shape).astype("float32"))
    for filename, key, vectors in (("item_embeddings.csv", "item_id", model2_items), ("user_embeddings.csv", "user_id", model2_users)):
        frame = pd.DataFrame(vectors, columns=[f"dim_{i}" for i in range(DIMS)])
        frame.insert(0, key, ids if key == "item_id" else user_ids)
        frame.to_csv(OUT_2 / filename, index=False, float_format="%.6g")
    rotation, _ = np.linalg.qr(projection_rng.normal(size=(3, 3)))
    offset = np.array([.9, -.6, .45], dtype="float32")
    model2_xyz = (xyz @ rotation) * 1.08 + offset + projection_rng.normal(0, .3, xyz.shape)
    model2_user_xyz = (user_xyz @ rotation) * 1.08 + offset + projection_rng.normal(0, .28, user_xyz.shape)
    pd.DataFrame({"item_id": ids, "x": model2_xyz[:, 0], "y": model2_xyz[:, 1], "z": model2_xyz[:, 2],
                  "projection_method": "synthetic_umap_layout_3d"}).to_csv(OUT_2 / "item_embeddings_3d.csv", index=False, float_format="%.5g")
    pd.DataFrame({"user_id": user_ids, "x": model2_user_xyz[:, 0], "y": model2_user_xyz[:, 1], "z": model2_user_xyz[:, 2],
                  "projection_method": "synthetic_umap_layout_3d"}).to_csv(OUT_2 / "user_embeddings_3d.csv", index=False, float_format="%.5g")

    # Per-user demo outcomes intentionally trade a little relevance for greater variety.
    item_by_genre = {g: np.flatnonzero(genres == g) for g in vocab}
    by_model = {}
    for model_name, relevant_at_top, secondary_count in (("model_1", 6, 8), ("model_2", 5, 20)):
        per_user = []
        for uid, pref, preferred_lang, popularity_target in zip(user_ids, preferred, preferred_language, user_meta.popularity_target):
            history_primary = rng.choice(item_by_genre[pref], size=12, replace=False)
            history_secondary_pool = np.flatnonzero(~np.isin(genres, [pref]))
            history_secondary = rng.choice(history_secondary_pool, size=6, replace=False)
            history_idx = np.concatenate([history_primary, history_secondary])
            history_genres = set(genres[history_idx].tolist())
            pool = item_by_genre[pref]
            truth_pool = pool[~np.isin(pool, history_idx)]
            truth_idx = rng.choice(truth_pool, size=10, replace=False)
            relevant = truth_idx[:relevant_at_top]
            primary_pool = pool[~np.isin(pool, np.concatenate([history_idx, truth_idx]))]
            secondary_pool = np.flatnonzero(~np.isin(genres, [pref]))
            secondary_pool = secondary_pool[~np.isin(secondary_pool, np.concatenate([history_idx, truth_idx]))]
            extra_secondary = rng.choice(secondary_pool, size=secondary_count, replace=False)
            extra_primary = rng.choice(primary_pool, size=50 - relevant_at_top - secondary_count, replace=False)
            pred_idx = np.concatenate([relevant, extra_primary, extra_secondary])
            rng.shuffle(pred_idx[:relevant_at_top])
            truth_set = set(ids[truth_idx].tolist())
            pred = ids[pred_idx].tolist()
            hits = [j for j, item in enumerate(pred) if item in truth_set]
            dcg = sum(1 / np.log2(rank + 2) for rank in hits)
            ideal = sum(1 / np.log2(rank + 2) for rank in range(len(truth_set)))
            pred_genres = genres[pred_idx]
            pred_creators = creators[pred_idx]
            diversity = .55 * (len(set(pred_genres)) / min(8, len(pred_genres))) + .45 * (len(set(pred_creators)) / len(pred_creators))
            discovery = sum(g not in history_genres for g in pred_genres) / len(pred_genres)
            fresh = float(np.mean(years[pred_idx] >= 2022))
            localization = float(np.mean(language[pred_idx] == preferred_lang))
            pop_calibration = max(0.0, 1.0 - abs(float(popularity[pred_idx].mean()) - float(popularity_target)))
            per_user.append({"user_id": uid, "ndcg_at_50": dcg / ideal, "precision_at_50": len(hits) / 50,
                             "recall_at_50": len(hits) / len(truth_set), "freshness": fresh,
                             "diversity": float(diversity), "discovery": float(discovery),
                             "localization": localization, "popularity_calibration": pop_calibration,
                             "ground_truth_items": json.dumps(ids[truth_idx].tolist()),
                             "top_k_predicted_items": json.dumps(pred),
                             "user_history_items": json.dumps(ids[history_idx].tolist())})
        eval_df = pd.DataFrame(per_user)
        out_dir = OUT if model_name == "model_1" else OUT_2
        eval_df.to_csv(out_dir / "evaluation_results_per_user.csv", index=False)
        descriptions = {"ndcg_at_50": "NDCG at 50; relevance with rank discount.",
                       "precision_at_50": "Share of top-50 recommendations in the reference set.",
                       "recall_at_50": "Share of reference items recovered in the top 50.",
                       "freshness": "Share of recommendations released in 2022 or later.",
                       "diversity": "Combined genre variety and artist/show variety in the list.",
                       "discovery": "Share of recommended genres not present in recent user history.",
                       "localization": "Share of recommendations matching the user's preferred language.",
                       "popularity_calibration": "Closeness of mean recommendation popularity to the user's target mix."}
        metrics = [{"metric_name": key, "metric_value": float(eval_df[key].mean()), "description": desc}
                   for key, desc in descriptions.items()]
        (out_dir / "evaluation_metrics_aggregate.json").write_text(json.dumps({"model_id": model_name, "metrics": metrics}, indent=2))
        by_model[model_name] = eval_df
    print(f"Generated Model 1 and Model 2 datasets under {OUT.parent}")


if __name__ == "__main__":
    if "--projections-only" in sys.argv[1:]:
        write_projection_only()
    else:
        main()

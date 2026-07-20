"""
Model training and evaluation stage.

Trains two recommendation models:
  1. Collaborative Filtering (Truncated SVD) over the user-item interaction
     matrix (implicit clickstream signals + explicit ratings).
  2. Content-Based Filtering using item feature vectors (category, price,
     sentiment, popularity, rating) with cosine similarity.

Both are evaluated with Precision@K, Recall@K and NDCG@K on a held-out
split, and logged to MLflow for full experiment tracking.

Usage:
    python src/models/train_model.py
"""
import os
import sys

import mlflow
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from common import get_logger, PROJECT_ROOT  # noqa: E402

logger = get_logger("model_training")

FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.replace(os.sep, '/')}")
mlflow.set_experiment("recomart_recommendation")

N_COMPONENTS = 20
TOP_K = 10
RANDOM_STATE = 42

EVENT_WEIGHTS = {"view": 1, "click": 2, "add_to_cart": 3, "wishlist": 3, "purchase": 5}


def build_interaction_matrix():
    clickstream = pd.read_csv(os.path.join(PROCESSED_DIR, "clickstream_clean.csv"))
    transactions = pd.read_csv(os.path.join(PROCESSED_DIR, "transactions_clean.csv"))

    clickstream["weight"] = clickstream["event_type"].map(EVENT_WEIGHTS).fillna(1)
    implicit = clickstream.groupby(["user_id", "product_id"])["weight"].sum().reset_index()

    explicit = transactions.dropna(subset=["rating"])[["user_id", "product_id", "rating"]].copy()
    explicit = explicit.rename(columns={"rating": "weight"})
    explicit["weight"] = explicit["weight"] * 2  # up-weight explicit ratings

    combined = pd.concat([implicit, explicit], ignore_index=True)
    combined = combined.groupby(["user_id", "product_id"])["weight"].sum().reset_index()

    users = sorted(combined["user_id"].unique())
    items = sorted(combined["product_id"].unique())
    user_idx = {u: i for i, u in enumerate(users)}
    item_idx = {p: i for i, p in enumerate(items)}

    rows = combined["user_id"].map(user_idx)
    cols = combined["product_id"].map(item_idx)
    matrix = csr_matrix(
        (combined["weight"].astype(float), (rows, cols)), shape=(len(users), len(items))
    )
    return matrix, users, items, user_idx, item_idx, combined


def train_test_split_matrix(combined: pd.DataFrame, test_size=0.2):
    """Leave-out split: for each user, hold out a fraction of their interactions for testing."""
    train_rows, test_rows = [], []
    for _, group in combined.groupby("user_id"):
        if len(group) < 2:
            train_rows.append(group)
            continue
        tr, te = train_test_split(group, test_size=test_size, random_state=RANDOM_STATE)
        train_rows.append(tr)
        test_rows.append(te)
    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=combined.columns)
    return train_df, test_df


def matrix_from_df(df, users, items, user_idx, item_idx):
    rows = df["user_id"].map(user_idx)
    cols = df["product_id"].map(item_idx)
    return csr_matrix(
        (df["weight"].astype(float), (rows, cols)), shape=(len(users), len(items))
    )


def precision_recall_ndcg_at_k(test_matrix, scores, k=TOP_K):
    """Compute mean Precision@K, Recall@K, NDCG@K across all users with test interactions."""
    precisions, recalls, ndcgs = [], [], []
    test_csr = test_matrix.tocsr()

    for u in range(test_matrix.shape[0]):
        actual_items = set(test_csr[u].indices)
        if not actual_items:
            continue

        top_k_items = np.argsort(-scores[u])[:k]
        hits = [1 if item in actual_items else 0 for item in top_k_items]

        precision = sum(hits) / k
        recall = sum(hits) / len(actual_items)

        dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        ideal_hits = sorted(hits, reverse=True)
        idcg = sum(h / np.log2(i + 2) for i, h in enumerate(ideal_hits)) or 1.0
        ndcg = dcg / idcg

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)

    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "users_evaluated": len(precisions),
    }


def build_content_scores(users, items, user_idx, item_idx, train_matrix):
    """Content-based filtering: recommend items similar to those each user interacted with."""
    item_features_path = os.path.join(FEATURES_DIR, "item_features.csv")
    item_features = pd.read_csv(item_features_path)

    feature_cols = ["price_norm", "category_encoded", "avg_rating_item",
                    "sentiment_score", "popularity_score"]
    for col in feature_cols:
        if col not in item_features.columns:
            item_features[col] = 0.0

    item_features = item_features.dropna(subset=["product_id"])
    item_features[feature_cols] = item_features[feature_cols].fillna(0.0)

    scaler = MinMaxScaler()
    item_features[feature_cols] = scaler.fit_transform(item_features[feature_cols])

    item_feature_matrix = np.zeros((len(items), len(feature_cols)))
    for _, row in item_features.iterrows():
        pid = row["product_id"]
        if pid in item_idx:
            item_feature_matrix[item_idx[pid]] = row[feature_cols].values

    item_sim = cosine_similarity(item_feature_matrix)

    train_csr = train_matrix.tocsr()
    scores = np.zeros((len(users), len(items)))
    for u in range(len(users)):
        interacted = train_csr[u].indices
        if len(interacted) == 0:
            continue
        weights = np.array(train_csr[u].data)
        weighted_sim = weights.dot(item_sim[interacted])
        scores[u] = weighted_sim

    scores[train_matrix.nonzero()] = -np.inf
    return scores, item_feature_matrix, item_sim


def run():
    logger.info("Starting model training run")
    matrix, users, items, user_idx, item_idx, combined = build_interaction_matrix()
    train_df, test_df = train_test_split_matrix(combined)

    train_matrix = matrix_from_df(train_df, users, items, user_idx, item_idx)
    test_matrix = matrix_from_df(test_df, users, items, user_idx, item_idx)

    n_components = min(N_COMPONENTS, min(train_matrix.shape) - 1)
    results = {}

    # ---- Model 1: Collaborative Filtering (Truncated SVD) ----
    with mlflow.start_run(run_name="svd_collaborative_filtering") as run_ctx:
        mlflow.log_param("model_type", "collaborative_filtering")
        mlflow.log_param("algorithm", "TruncatedSVD")
        mlflow.log_param("n_components", n_components)
        mlflow.log_param("top_k", TOP_K)
        mlflow.log_param("n_users", len(users))
        mlflow.log_param("n_items", len(items))
        mlflow.log_param("test_size", 0.2)

        model = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        user_factors = model.fit_transform(train_matrix)
        item_factors = model.components_.T

        scores = user_factors.dot(item_factors.T)
        scores[train_matrix.nonzero()] = -np.inf

        cf_metrics = precision_recall_ndcg_at_k(test_matrix, scores, k=TOP_K)
        logger.info("Collaborative filtering metrics: %s", cf_metrics)

        mlflow.log_metrics({
            "precision_at_k": cf_metrics["precision_at_k"],
            "recall_at_k": cf_metrics["recall_at_k"],
            "ndcg_at_k": cf_metrics["ndcg_at_k"],
            "explained_variance_ratio": float(model.explained_variance_ratio_.sum()),
        })

        model_path = os.path.join(MODELS_DIR, "svd_model.npz")
        np.savez(
            model_path,
            user_factors=user_factors,
            item_factors=item_factors,
            users=np.array(users, dtype=object),
            items=np.array(items, dtype=object),
        )
        mlflow.log_artifact(model_path)

        logger.info(
            "SUCCESS: collaborative filtering model logged to MLflow (run_id=%s)",
            run_ctx.info.run_id,
        )
        results["collaborative"] = {"run_id": run_ctx.info.run_id, **cf_metrics}

    # ---- Model 2: Content-Based Filtering ----
    with mlflow.start_run(run_name="content_based_filtering") as run_ctx:
        mlflow.log_param("model_type", "content_based_filtering")
        mlflow.log_param("algorithm", "cosine_similarity")
        mlflow.log_param("features_used", "price_norm,category_encoded,avg_rating_item,sentiment_score,popularity_score")
        mlflow.log_param("top_k", TOP_K)
        mlflow.log_param("n_users", len(users))
        mlflow.log_param("n_items", len(items))
        mlflow.log_param("test_size", 0.2)

        cb_scores, item_feature_matrix, item_sim = build_content_scores(
            users, items, user_idx, item_idx, train_matrix
        )

        cb_metrics = precision_recall_ndcg_at_k(test_matrix, cb_scores, k=TOP_K)
        logger.info("Content-based filtering metrics: %s", cb_metrics)

        mlflow.log_metrics({
            "precision_at_k": cb_metrics["precision_at_k"],
            "recall_at_k": cb_metrics["recall_at_k"],
            "ndcg_at_k": cb_metrics["ndcg_at_k"],
        })

        cb_model_path = os.path.join(MODELS_DIR, "content_model.npz")
        np.savez(
            cb_model_path,
            item_feature_matrix=item_feature_matrix,
            item_sim=item_sim,
            users=np.array(users, dtype=object),
            items=np.array(items, dtype=object),
        )
        mlflow.log_artifact(cb_model_path)

        logger.info(
            "SUCCESS: content-based model logged to MLflow (run_id=%s)",
            run_ctx.info.run_id,
        )
        results["content_based"] = {"run_id": run_ctx.info.run_id, **cb_metrics}

    logger.info("All models trained. Results: %s", results)
    return results


if __name__ == "__main__":
    run()

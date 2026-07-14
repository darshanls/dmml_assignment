"""
Model training and evaluation stage.

Trains a collaborative-filtering recommendation model using Truncated SVD
matrix factorization over the user-item interaction matrix (implicit
feedback derived from clickstream + explicit ratings from transactions).

Evaluates with Precision@K, Recall@K and NDCG@K on a held-out split, and
logs parameters/metrics/artifacts to MLflow for full experiment tracking.

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
from sklearn.model_selection import train_test_split

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


def run():
    logger.info("Starting model training run")
    matrix, users, items, user_idx, item_idx, combined = build_interaction_matrix()
    train_df, test_df = train_test_split_matrix(combined)

    train_matrix = matrix_from_df(train_df, users, items, user_idx, item_idx)
    test_matrix = matrix_from_df(test_df, users, items, user_idx, item_idx)

    n_components = min(N_COMPONENTS, min(train_matrix.shape) - 1)

    with mlflow.start_run(run_name="svd_collaborative_filtering") as run_ctx:
        mlflow.log_param("n_components", n_components)
        mlflow.log_param("top_k", TOP_K)
        mlflow.log_param("n_users", len(users))
        mlflow.log_param("n_items", len(items))
        mlflow.log_param("test_size", 0.2)

        model = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        user_factors = model.fit_transform(train_matrix)
        item_factors = model.components_.T

        scores = user_factors.dot(item_factors.T)
        # Mask out items the user already interacted with in the training set
        scores[train_matrix.nonzero()] = -np.inf

        metrics = precision_recall_ndcg_at_k(test_matrix, scores, k=TOP_K)
        logger.info("Evaluation metrics: %s", metrics)

        mlflow.log_metrics({
            "precision_at_k": metrics["precision_at_k"],
            "recall_at_k": metrics["recall_at_k"],
            "ndcg_at_k": metrics["ndcg_at_k"],
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
            "SUCCESS: model trained and logged to MLflow (run_id=%s), metrics=%s",
            run_ctx.info.run_id, metrics,
        )
        return {"run_id": run_ctx.info.run_id, **metrics}


if __name__ == "__main__":
    run()

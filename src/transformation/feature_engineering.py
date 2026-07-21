"""
Feature engineering and transformation stage.

Builds features suitable for recommendation algorithms:
    - User activity frequency (events per active day)
    - Average rating per user / per item
    - Item co-occurrence / similarity features (Jaccard over user sets)
Loads dimension/fact tables + features into a SQLite warehouse
(warehouse/recomart.db) using the schema defined in schema.sql.

Usage:
    python src/transformation/feature_engineering.py
"""
import itertools
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from common import get_logger, PROJECT_ROOT  # noqa: E402

logger = get_logger("transformation")

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
WAREHOUSE_DIR = os.path.join(PROJECT_ROOT, "warehouse")
DB_PATH = os.path.join(WAREHOUSE_DIR, "recomart.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(WAREHOUSE_DIR, exist_ok=True)


def _load_processed():
    clickstream = pd.read_csv(os.path.join(PROCESSED_DIR, "clickstream_clean.csv"))
    transactions = pd.read_csv(os.path.join(PROCESSED_DIR, "transactions_clean.csv"))
    products = pd.read_csv(os.path.join(PROCESSED_DIR, "products_clean.csv"))
    sentiment = pd.read_csv(os.path.join(PROCESSED_DIR, "sentiment_clean.csv"))
    clickstream["event_timestamp"] = pd.to_datetime(clickstream["event_timestamp"])
    transactions["transaction_timestamp"] = pd.to_datetime(transactions["transaction_timestamp"])
    return clickstream, transactions, products, sentiment


def build_user_features(clickstream: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    events_per_user = clickstream.groupby("user_id").size().rename("total_events")

    active_days = clickstream.groupby("user_id")["event_timestamp"].apply(
        lambda s: s.dt.date.nunique()
    ).rename("active_days")

    purchases_per_user = transactions.groupby("user_id").size().rename("total_purchases")
    spend_per_user = transactions.groupby("user_id")["total_amount"].sum().rename("total_spend")
    avg_rating_user = transactions.groupby("user_id")["rating"].mean().rename("avg_rating_given")
    last_active = clickstream.groupby("user_id")["event_timestamp"].max().rename("last_active_at")

    # --- New: view count and purchase-to-view conversion ratio ---------------
    views_per_user = (
        clickstream[clickstream["event_type"] == "view"]
        .groupby("user_id")
        .size()
        .rename("total_views")
    )

    # --- New: unique sessions per user ---------------------------------------
    sessions_per_user = (
        clickstream.groupby("user_id")["session_id"].nunique().rename("unique_sessions")
    )

    # --- New: average spend per transaction ----------------------------------
    avg_spend = transactions.groupby("user_id")["total_amount"].mean().rename("avg_spend_per_txn")

    users = pd.concat(
        [events_per_user, active_days, purchases_per_user, spend_per_user,
         avg_rating_user, last_active, views_per_user, sessions_per_user, avg_spend],
        axis=1,
    ).reset_index()

    users["active_days"] = users["active_days"].replace(0, 1)
    users["activity_frequency"] = (users["total_events"] / users["active_days"]).round(3)
    users = users.drop(columns=["active_days"])
    users = users.fillna({"total_purchases": 0, "total_spend": 0.0, "total_views": 0,
                          "unique_sessions": 0, "avg_spend_per_txn": 0.0})

    # Purchase-to-view conversion ratio (avoid div-by-zero)
    users["purchase_to_view_ratio"] = np.where(
        users["total_views"] > 0,
        (users["total_purchases"] / users["total_views"]).round(4),
        0.0,
    )

    # Recency score: days since last active (higher = less recent)
    ref_date = clickstream["event_timestamp"].max()
    users["days_since_last_active"] = (
        (ref_date - pd.to_datetime(users["last_active_at"]))
        .dt.total_seconds()
        .div(86400)
        .round(2)
    )
    return users


def build_item_features(
    clickstream: pd.DataFrame, transactions: pd.DataFrame, products: pd.DataFrame, sentiment: pd.DataFrame
) -> pd.DataFrame:
    # The product catalog from FakeStore already has `rating_count` / `rating_rate` columns.
    # Rename them so they don't collide with the transaction-derived `rating_count` feature.
    products = products.copy()
    for col in ["rating_count", "rating_rate"]:
        if col in products.columns:
            products = products.rename(columns={col: f"api_{col}"})

    interactions_per_item = clickstream.groupby("product_id").size().rename("total_interactions")
    purchases_per_item = transactions.groupby("product_id").size().rename("total_purchases")
    avg_rating_item = transactions.groupby("product_id")["rating"].mean().rename("avg_rating_item")
    rating_count_item = transactions.groupby("product_id")["rating"].count().rename("rating_count")

    # --- New: Bayesian weighted rating (shrink towards global mean) ----------
    global_mean = transactions["rating"].mean()
    C = 10  # confidence weight (minimum ratings equivalent)
    bayesian_avg = transactions.groupby("product_id").apply(
        lambda g: (C * global_mean + g["rating"].sum()) / (C + g["rating"].count())
    ).rename("weighted_avg_rating")

    # --- New: view-to-purchase conversion per item ---------------------------
    views_per_item = (
        clickstream[clickstream["event_type"] == "view"]
        .groupby("product_id")
        .size()
        .rename("total_views")
    )

    # --- New: distinct users who interacted with each item -------------------
    unique_users_item = (
        clickstream.groupby("product_id")["user_id"].nunique().rename("unique_users")
    )

    items = products.merge(
        interactions_per_item, left_on="product_id", right_index=True, how="left"
    ).merge(
        purchases_per_item, left_on="product_id", right_index=True, how="left"
    ).merge(
        avg_rating_item, left_on="product_id", right_index=True, how="left"
    ).merge(
        rating_count_item, left_on="product_id", right_index=True, how="left"
    ).merge(
        bayesian_avg, left_on="product_id", right_index=True, how="left"
    ).merge(
        views_per_item, left_on="product_id", right_index=True, how="left"
    ).merge(
        unique_users_item, left_on="product_id", right_index=True, how="left"
    ).merge(
        sentiment[["product_id", "sentiment_score", "popularity_score"]], on="product_id", how="left"
    )

    items[["total_interactions", "total_purchases", "total_views",
           "unique_users", "rating_count"]] = items[
        ["total_interactions", "total_purchases", "total_views",
         "unique_users", "rating_count"]
    ].fillna(0)
    items["avg_rating_item"] = items["avg_rating_item"].fillna(items["avg_rating_item"].mean())
    items["weighted_avg_rating"] = items["weighted_avg_rating"].fillna(global_mean)

    # Item-level conversion: purchases / views (avoid div-by-zero)
    items["item_conversion_rate"] = np.where(
        items["total_views"] > 0,
        (items["total_purchases"] / items["total_views"]).round(4),
        0.0,
    )
    return items


def build_cooccurrence(clickstream: pd.DataFrame, top_n_users: int = 500) -> pd.DataFrame:
    """Item-item co-occurrence + Jaccard similarity based on shared user sets."""
    user_items = clickstream.groupby("user_id")["product_id"].apply(lambda s: set(s)).to_dict()

    item_users = {}
    for user, items in user_items.items():
        for item in items:
            item_users.setdefault(item, set()).add(user)

    rows = []
    for item_a, item_b in itertools.combinations(sorted(item_users.keys()), 2):
        users_a, users_b = item_users[item_a], item_users[item_b]
        intersection = len(users_a & users_b)
        if intersection == 0:
            continue
        union = len(users_a | users_b)
        jaccard = intersection / union if union else 0.0
        rows.append({
            "product_id_a": item_a,
            "product_id_b": item_b,
            "cooccurrence_count": intersection,
            "similarity_score": round(jaccard, 4),
        })
    return pd.DataFrame(rows)


def load_into_warehouse(users, items, cooccurrence, clickstream, transactions):
    conn = sqlite3.connect(DB_PATH)
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())

        users_cols = ["user_id", "total_events", "total_purchases", "total_spend",
                      "avg_rating_given", "last_active_at", "total_views",
                      "unique_sessions", "avg_spend_per_txn", "activity_frequency",
                      "purchase_to_view_ratio", "days_since_last_active"]
        users[users_cols].to_sql("dim_users", conn, if_exists="append", index=False)

        items_cols = ["product_id", "title", "category", "price", "price_norm",
                      "category_encoded", "avg_rating_item", "rating_count",
                      "weighted_avg_rating", "total_interactions", "total_purchases",
                      "total_views", "unique_users", "item_conversion_rate",
                      "sentiment_score", "popularity_score"]
        items[items_cols].to_sql("dim_items", conn, if_exists="append", index=False)

        cooccurrence.to_sql("item_cooccurrence", conn, if_exists="append", index=False)

        clickstream[["event_id", "user_id", "product_id", "event_type",
                     "event_type_encoded", "device", "event_timestamp"]].to_sql(
            "fact_interactions", conn, if_exists="append", index=False
        )
        transactions[["transaction_id", "user_id", "product_id", "quantity", "unit_price",
                      "total_amount", "rating", "transaction_timestamp"]].to_sql(
            "fact_transactions", conn, if_exists="append", index=False
        )
        conn.commit()
    finally:
        conn.close()


def run():
    logger.info("Starting feature engineering / transformation run")
    clickstream, transactions, products, sentiment = _load_processed()

    users = build_user_features(clickstream, transactions)
    items = build_item_features(clickstream, transactions, products, sentiment)
    cooccurrence = build_cooccurrence(clickstream)

    users.to_csv(os.path.join(FEATURES_DIR, "user_features.csv"), index=False)
    items.to_csv(os.path.join(FEATURES_DIR, "item_features.csv"), index=False)
    cooccurrence.to_csv(os.path.join(FEATURES_DIR, "item_cooccurrence.csv"), index=False)

    load_into_warehouse(users, items, cooccurrence, clickstream, transactions)

    logger.info(
        "SUCCESS: features written to %s and loaded into warehouse at %s "
        "(users=%d, items=%d, item-pairs=%d)",
        FEATURES_DIR, DB_PATH, len(users), len(items), len(cooccurrence),
    )
    return {"users": len(users), "items": len(items), "pairs": len(cooccurrence)}


if __name__ == "__main__":
    run()

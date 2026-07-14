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

    users = pd.concat(
        [events_per_user, active_days, purchases_per_user, spend_per_user, avg_rating_user, last_active],
        axis=1,
    ).reset_index()

    users["active_days"] = users["active_days"].replace(0, 1)
    users["activity_frequency"] = (users["total_events"] / users["active_days"]).round(3)
    users = users.drop(columns=["active_days"])
    users = users.fillna({"total_purchases": 0, "total_spend": 0.0})
    return users


def build_item_features(
    clickstream: pd.DataFrame, transactions: pd.DataFrame, products: pd.DataFrame, sentiment: pd.DataFrame
) -> pd.DataFrame:
    interactions_per_item = clickstream.groupby("product_id").size().rename("total_interactions")
    purchases_per_item = transactions.groupby("product_id").size().rename("total_purchases")
    avg_rating_item = transactions.groupby("product_id")["rating"].mean().rename("avg_rating_item")

    items = products.merge(
        interactions_per_item, left_on="product_id", right_index=True, how="left"
    ).merge(
        purchases_per_item, left_on="product_id", right_index=True, how="left"
    ).merge(
        avg_rating_item, left_on="product_id", right_index=True, how="left"
    ).merge(
        sentiment[["product_id", "sentiment_score", "popularity_score"]], on="product_id", how="left"
    )

    items[["total_interactions", "total_purchases"]] = items[["total_interactions", "total_purchases"]].fillna(0)
    items["avg_rating_item"] = items["avg_rating_item"].fillna(items["avg_rating_item"].mean())
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
                      "avg_rating_given", "activity_frequency", "last_active_at"]
        users[users_cols].to_sql("dim_users", conn, if_exists="append", index=False)

        items_cols = ["product_id", "title", "category", "price", "price_norm",
                      "category_encoded", "avg_rating_item", "total_interactions",
                      "total_purchases", "sentiment_score", "popularity_score"]
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

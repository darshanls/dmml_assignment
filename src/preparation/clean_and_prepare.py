"""
Data preparation stage: clean raw ingested data, handle missing values,
encode categorical attributes, normalize numeric variables, and produce
EDA summary plots. Writes prepared datasets to data/processed/.

Usage:
    python src/preparation/clean_and_prepare.py
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from common import get_logger, RAW_DATA_DIR, PROJECT_ROOT  # noqa: E402

logger = get_logger("preparation")

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "reports", "eda_plots")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def _load_all(source: str, data_type: str, ext="csv") -> pd.DataFrame:
    pattern = os.path.join(RAW_DATA_DIR, source, data_type, "*", f"*.{ext}")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No raw files found for {source}/{data_type}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def clean_clickstream(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="event_id")
    df = df.dropna(subset=["user_id", "product_id", "event_type"])
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce")
    df = df.dropna(subset=["event_timestamp"])

    le_event = LabelEncoder()
    le_device = LabelEncoder()
    df["event_type_encoded"] = le_event.fit_transform(df["event_type"])
    df["device_encoded"] = le_device.fit_transform(df["device"])
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="transaction_id")
    df = df.dropna(subset=["user_id", "product_id", "unit_price", "quantity"])

    # Missing user-item ratings: keep row but flag as implicit feedback (no explicit rating)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["has_explicit_rating"] = df["rating"].notna().astype(int)

    # Clip out-of-range ratings defensively (schema says 1-5)
    df.loc[(df["rating"] < 1) | (df["rating"] > 5), "rating"] = np.nan

    df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"], errors="coerce")
    df = df.dropna(subset=["transaction_timestamp"])

    scaler = MinMaxScaler()
    df[["unit_price_norm", "total_amount_norm"]] = scaler.fit_transform(
        df[["unit_price", "total_amount"]]
    )
    return df


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="product_id")
    df = df.dropna(subset=["product_id", "title", "price", "category"])

    le_category = LabelEncoder()
    df["category_encoded"] = le_category.fit_transform(df["category"])

    scaler = MinMaxScaler()
    df[["price_norm"]] = scaler.fit_transform(df[["price"]])
    return df


def clean_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="product_id")
    df = df.dropna(subset=["product_id"])
    df["sentiment_score"] = df["sentiment_score"].fillna(0.0)
    df["popularity_score"] = df["popularity_score"].fillna(df["popularity_score"].median())
    return df


def run_eda(clickstream: pd.DataFrame, transactions: pd.DataFrame, products: pd.DataFrame):
    # 1. Interaction distribution by event type
    plt.figure(figsize=(6, 4))
    clickstream["event_type"].value_counts().plot(kind="bar", color="steelblue")
    plt.title("Clickstream Event Type Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "event_type_distribution.png"))
    plt.close()

    # 2. Item popularity (top 15 products by interaction count)
    plt.figure(figsize=(7, 4))
    clickstream["product_id"].value_counts().head(15).plot(kind="bar", color="darkorange")
    plt.title("Top 15 Products by Interaction Count")
    plt.xlabel("product_id")
    plt.ylabel("Interactions")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "item_popularity.png"))
    plt.close()

    # 3. User-item interaction sparsity
    n_users = clickstream["user_id"].nunique()
    n_items = clickstream["product_id"].nunique()
    n_interactions = len(clickstream.drop_duplicates(subset=["user_id", "product_id"]))
    sparsity = 1 - (n_interactions / (n_users * n_items))

    plt.figure(figsize=(5, 4))
    plt.bar(["Observed", "Sparsity"], [1 - sparsity, sparsity], color=["green", "lightgray"])
    plt.title(f"User-Item Matrix Sparsity ({sparsity:.4%})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sparsity.png"))
    plt.close()

    # 4. Rating distribution
    plt.figure(figsize=(5, 4))
    transactions["rating"].dropna().value_counts().sort_index().plot(kind="bar", color="purple")
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rating_distribution.png"))
    plt.close()

    # 5. Category price heatmap-style summary
    plt.figure(figsize=(6, 4))
    cat_price = products.groupby("category")["price"].mean().sort_values()
    cat_price.plot(kind="barh", color="teal")
    plt.title("Average Price by Category")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "avg_price_by_category.png"))
    plt.close()

    logger.info(
        "EDA complete: users=%d items=%d interactions=%d sparsity=%.4f",
        n_users, n_items, n_interactions, sparsity,
    )
    return {"n_users": n_users, "n_items": n_items, "n_interactions": n_interactions, "sparsity": sparsity}


def run():
    logger.info("Starting data preparation run")

    clickstream = clean_clickstream(_load_all("clickstream", "events"))
    transactions = clean_transactions(_load_all("transactions", "purchases"))
    products = clean_products(_load_all("products", "catalog"))
    sentiment = clean_sentiment(_load_all("sentiment", "scores"))

    clickstream.to_csv(os.path.join(PROCESSED_DIR, "clickstream_clean.csv"), index=False)
    transactions.to_csv(os.path.join(PROCESSED_DIR, "transactions_clean.csv"), index=False)
    products.to_csv(os.path.join(PROCESSED_DIR, "products_clean.csv"), index=False)
    sentiment.to_csv(os.path.join(PROCESSED_DIR, "sentiment_clean.csv"), index=False)

    stats = run_eda(clickstream, transactions, products)
    logger.info(
        "SUCCESS: prepared datasets written to %s (rows: clickstream=%d, transactions=%d, products=%d, sentiment=%d)",
        PROCESSED_DIR, len(clickstream), len(transactions), len(products), len(sentiment),
    )
    return stats


if __name__ == "__main__":
    run()

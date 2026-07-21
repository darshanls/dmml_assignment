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
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from common import get_logger, RAW_DATA_DIR, PROJECT_ROOT  # noqa: E402

logger = get_logger("preparation")

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "reports", "eda_plots")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def _load_latest(source: str, data_type: str, ext="csv") -> pd.DataFrame:
    """
    Load the latest ingested file for a source/data_type from the most
    recent date partition. Files are timestamped, so lexicographic sort
    picks the most recent run and avoids stacking multiple same-day
    batches as duplicate rows.
    """
    pattern = os.path.join(RAW_DATA_DIR, source, data_type, "*", f"*.{ext}")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No raw files found for {source}/{data_type}")
    return pd.read_csv(files[-1])


def clean_clickstream(df: pd.DataFrame) -> pd.DataFrame:
    rows_before = len(df)
    df = df.drop_duplicates(subset="event_id")
    logger.info("Clickstream: dropped %d duplicate event_id rows", rows_before - len(df))

    df = df.dropna(subset=["user_id", "product_id", "event_type"])
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce")
    df = df.dropna(subset=["event_timestamp"])

    # Normalise unexpected event_type values to closest valid value or drop
    valid_events = {"view", "click", "add_to_cart", "wishlist", "purchase"}
    case_map = {v.lower(): v for v in valid_events}  # e.g. "click" -> "click"
    df["event_type"] = df["event_type"].str.strip().str.lower().map(
        lambda x: case_map.get(x, None)
    )
    n_invalid_evt = df["event_type"].isna().sum()
    if n_invalid_evt:
        logger.info("Clickstream: dropping %d rows with invalid event_type", n_invalid_evt)
    df = df.dropna(subset=["event_type"])

    le_event = LabelEncoder()
    le_device = LabelEncoder()
    df["event_type_encoded"] = le_event.fit_transform(df["event_type"])
    df["device_encoded"] = le_device.fit_transform(df["device"])

    logger.info("Clickstream: %d -> %d rows after cleaning", rows_before, len(df))
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    rows_before = len(df)
    df = df.drop_duplicates(subset="transaction_id")
    logger.info("Transactions: dropped %d duplicate transaction_id rows", rows_before - len(df))

    df = df.dropna(subset=["user_id", "product_id"])

    # Coerce non-numeric values in numeric columns to NaN
    for col in ["unit_price", "quantity", "total_amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["unit_price", "quantity"])

    # Remove rows with negative or zero prices / quantities
    n_neg_price = (df["unit_price"] <= 0).sum()
    n_bad_qty = (df["quantity"] <= 0).sum()
    if n_neg_price:
        logger.info("Transactions: removing %d rows with non-positive unit_price", n_neg_price)
    if n_bad_qty:
        logger.info("Transactions: removing %d rows with non-positive quantity", n_bad_qty)
    df = df[(df["unit_price"] > 0) & (df["quantity"] > 0)]

    # Recalculate total_amount to be consistent after cleaning
    df["total_amount"] = (df["quantity"] * df["unit_price"]).round(2)

    # Missing user-item ratings: keep row but flag as implicit feedback (no explicit rating)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["has_explicit_rating"] = df["rating"].notna().astype(int)

    # Clip out-of-range ratings defensively (schema says 1-5)
    n_oor = ((df["rating"] < 1) | (df["rating"] > 5)).sum()
    if n_oor:
        logger.info("Transactions: nullifying %d out-of-range ratings", n_oor)
    df.loc[(df["rating"] < 1) | (df["rating"] > 5), "rating"] = np.nan

    df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"], errors="coerce")
    n_bad_ts = df["transaction_timestamp"].isna().sum()
    if n_bad_ts:
        logger.info("Transactions: dropping %d rows with bad timestamps", n_bad_ts)
    df = df.dropna(subset=["transaction_timestamp"])

    scaler = MinMaxScaler()
    df[["unit_price_norm", "total_amount_norm"]] = scaler.fit_transform(
        df[["unit_price", "total_amount"]]
    )
    logger.info("Transactions: %d -> %d rows after cleaning", rows_before, len(df))
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

    # 6. Device × Event-type heatmap
    plt.figure(figsize=(7, 4))
    ct = pd.crosstab(clickstream["device"], clickstream["event_type"])
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd")
    plt.title("Interactions: Device × Event Type")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "device_event_heatmap.png"))
    plt.close()

    # 7. Product price distribution histogram
    plt.figure(figsize=(6, 4))
    products["price"].plot(kind="hist", bins=20, color="mediumseagreen", edgecolor="white")
    plt.title("Product Price Distribution")
    plt.xlabel("Price ($)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "price_distribution.png"))
    plt.close()

    # 8. User activity distribution (events per user)
    plt.figure(figsize=(6, 4))
    user_event_counts = clickstream.groupby("user_id").size()
    user_event_counts.plot(kind="hist", bins=30, color="cornflowerblue", edgecolor="white")
    plt.title("User Activity Distribution (Events per User)")
    plt.xlabel("Number of Events")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "user_activity_distribution.png"))
    plt.close()

    # 9. Transactions per hour of day
    plt.figure(figsize=(6, 4))
    transactions["hour"] = transactions["transaction_timestamp"].dt.hour
    transactions["hour"].value_counts().sort_index().plot(kind="bar", color="salmon")
    plt.title("Transactions by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "transactions_by_hour.png"))
    plt.close()
    transactions.drop(columns=["hour"], inplace=True, errors="ignore")

    # 10. Spend distribution (unit_price) box plot by category
    if "category" in products.columns:
        merged = transactions.merge(
            products[["product_id", "category"]], on="product_id", how="left"
        )
        if not merged["category"].isna().all():
            plt.figure(figsize=(8, 4))
            merged.boxplot(column="unit_price", by="category", rot=45)
            plt.title("Unit Price Distribution by Category")
            plt.suptitle("")
            plt.ylabel("Unit Price ($)")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "price_by_category_boxplot.png"))
            plt.close()

    logger.info(
        "EDA complete: users=%d items=%d interactions=%d sparsity=%.4f",
        n_users, n_items, n_interactions, sparsity,
    )
    return {"n_users": n_users, "n_items": n_items, "n_interactions": n_interactions, "sparsity": sparsity}


def run():
    logger.info("Starting data preparation run")

    clickstream = clean_clickstream(_load_latest("clickstream", "events"))
    transactions = clean_transactions(_load_latest("transactions", "purchases"))
    products = clean_products(_load_latest("products", "catalog"))
    sentiment = clean_sentiment(_load_latest("sentiment", "scores"))

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

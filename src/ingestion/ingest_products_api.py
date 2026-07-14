"""
Ingest product catalog data from the FakeStore REST API
(https://fakestoreapi.com/products) into the raw data lake.

Usage:
    python src/ingestion/ingest_products_api.py
"""
import json
import os
import sys
from datetime import datetime

import pandas as pd
import requests

sys.path.append(os.path.dirname(__file__))
from common import get_logger, partitioned_path, network_retry

logger = get_logger("ingest.products_api")

API_URL = "https://fakestoreapi.com/products"
TIMEOUT_SECONDS = 10


@network_retry
def fetch_products():
    """Call the product catalog REST API and return the parsed JSON payload."""
    response = requests.get(API_URL, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    if not payload:
        raise ConnectionError("Empty payload from product catalog API")
    return payload


def _flatten(product: dict) -> dict:
    rating = product.get("rating", {}) or {}
    return {
        "product_id": product.get("id"),
        "title": product.get("title"),
        "price": product.get("price"),
        "category": product.get("category"),
        "description": product.get("description"),
        "image": product.get("image"),
        "rating_rate": rating.get("rate"),
        "rating_count": rating.get("count"),
    }


def run():
    logger.info("Starting product catalog API ingestion run")
    out_dir = partitioned_path(source="products", data_type="catalog")
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    try:
        raw_payload = fetch_products()

        # Persist raw JSON response for full lineage/auditability
        raw_json_path = os.path.join(out_dir, f"products_raw_{ts}.json")
        with open(raw_json_path, "w", encoding="utf-8") as f:
            json.dump(raw_payload, f, indent=2)

        df = pd.DataFrame([_flatten(p) for p in raw_payload])
        out_csv_path = os.path.join(out_dir, f"products_{ts}.csv")
        df.to_csv(out_csv_path, index=False)

        logger.info("SUCCESS: wrote %d products to %s", len(df), out_csv_path)
        return out_csv_path
    except Exception as exc:
        logger.error("FAILED product API ingestion: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    run()

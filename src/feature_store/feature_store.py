"""
Lightweight custom Feature Store for RecoMart.

Reads feature metadata from feature_registry.yaml and serves feature values
from the SQLite warehouse (warehouse/recomart.db) for both:
    - Training  : get_training_features(entity, feature_view)  -> full DataFrame
    - Inference : get_online_features(entity, entity_ids, feature_view) -> row(s)

This mirrors the core capability of a production feature store (e.g. Feast):
a declarative feature registry + consistent offline/online retrieval, without
the operational overhead of standing up Feast's infra for this POC.
"""
import os
import sqlite3
import sys

import pandas as pd
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from common import get_logger, PROJECT_ROOT  # noqa: E402

logger = get_logger("feature_store")

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "feature_registry.yaml")
DB_PATH = os.path.join(PROJECT_ROOT, "warehouse", "recomart.db")


class FeatureStore:
    def __init__(self, registry_path: str = REGISTRY_PATH, db_path: str = DB_PATH):
        with open(registry_path, "r", encoding="utf-8") as f:
            self.registry = yaml.safe_load(f)
        self.db_path = db_path

    def list_feature_views(self):
        return list(self.registry["feature_views"].keys())

    def describe(self, feature_view: str) -> dict:
        return self.registry["feature_views"][feature_view]

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def get_training_features(self, feature_view: str) -> pd.DataFrame:
        """Offline retrieval: full feature table for model training."""
        meta = self.describe(feature_view)
        table = meta["source_table"]
        conn = self._connect()
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        finally:
            conn.close()
        logger.info("Retrieved %d rows for training feature_view='%s'", len(df), feature_view)
        return df

    def get_online_features(self, feature_view: str, entity_ids: list) -> pd.DataFrame:
        """Online retrieval: feature row(s) for specific entity id(s) at inference time."""
        meta = self.describe(feature_view)
        table = meta["source_table"]
        entity_key = self.registry["entities"][meta["entity"]]["join_key"]

        placeholders = ",".join(["?"] * len(entity_ids))
        query = f"SELECT * FROM {table} WHERE {entity_key} IN ({placeholders})"
        conn = self._connect()
        try:
            df = pd.read_sql_query(query, conn, params=entity_ids)
        finally:
            conn.close()
        logger.info(
            "Retrieved %d online rows for feature_view='%s', entity_ids=%s",
            len(df), feature_view, entity_ids,
        )
        return df


def demo():
    """Sample feature retrieval demonstration for both training and inference."""
    fs = FeatureStore()
    print("Available feature views:", fs.list_feature_views())

    print("\n-- Training retrieval: user_features_v1 (head) --")
    train_df = fs.get_training_features("user_features_v1")
    print(train_df.head())

    sample_ids = train_df["user_id"].head(3).tolist() if not train_df.empty else []
    print("\n-- Online (inference) retrieval for users:", sample_ids, "--")
    online_df = fs.get_online_features("user_features_v1", sample_ids)
    print(online_df)


if __name__ == "__main__":
    demo()

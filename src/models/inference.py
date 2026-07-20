"""
Inference interface for the trained recommendation models.

Supports two model types:
  - collaborative: loads models/svd_model.npz (user/item latent factors)
  - content_based: loads models/content_model.npz (item-item cosine similarity)

Returns the top-N recommended product IDs for a given user.

Usage:
    python src/models/inference.py --user_id U0001 --top_n 5
    python src/models/inference.py --user_id U0001 --top_n 5 --model_type content_based
"""
import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from common import get_logger, PROJECT_ROOT  # noqa: E402

logger = get_logger("model_inference")
SVD_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "svd_model.npz")
CB_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "content_model.npz")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


class Recommender:
    """Collaborative filtering recommender using SVD latent factors."""
    def __init__(self, model_path: str = SVD_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")
        data = np.load(model_path, allow_pickle=True)
        self.user_factors = data["user_factors"]
        self.item_factors = data["item_factors"]
        self.users = list(data["users"])
        self.items = list(data["items"])
        self.user_to_idx = {u: i for i, u in enumerate(self.users)}

    def recommend(self, user_id: str, top_n: int = 5):
        if user_id not in self.user_to_idx:
            logger.warning("Unknown user_id=%s; returning empty recommendation list", user_id)
            return []
        u_idx = self.user_to_idx[user_id]
        scores = self.user_factors[u_idx].dot(self.item_factors.T)
        top_idx = np.argsort(-scores)[:top_n]
        return [(self.items[i], float(scores[i])) for i in top_idx]


class ContentBasedRecommender:
    """Content-based recommender using item-item cosine similarity."""
    def __init__(self, model_path: str = CB_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")
        data = np.load(model_path, allow_pickle=True)
        self.item_sim = data["item_sim"]
        self.users = list(data["users"])
        self.items = list(data["items"])
        self.user_to_idx = {u: i for i, u in enumerate(self.users)}
        self.item_to_idx = {p: i for i, p in enumerate(self.items)}
        self._load_user_interactions()

    def _load_user_interactions(self):
        """Load user interaction history from processed clickstream for scoring."""
        import pandas as pd
        cs_path = os.path.join(PROCESSED_DIR, "clickstream_clean.csv")
        if os.path.exists(cs_path):
            cs = pd.read_csv(cs_path)
            self.user_items = cs.groupby("user_id")["product_id"].apply(set).to_dict()
        else:
            self.user_items = {}

    def recommend(self, user_id: str, top_n: int = 5):
        if user_id not in self.user_to_idx:
            logger.warning("Unknown user_id=%s; returning empty recommendation list", user_id)
            return []
        interacted = self.user_items.get(user_id, set())
        interacted_idx = [self.item_to_idx[p] for p in interacted if p in self.item_to_idx]
        if not interacted_idx:
            logger.warning("No interaction history for user_id=%s; returning empty list", user_id)
            return []
        scores = np.mean(self.item_sim[interacted_idx], axis=0)
        for idx in interacted_idx:
            scores[idx] = -np.inf
        top_idx = np.argsort(-scores)[:top_n]
        return [(self.items[i], float(scores[i])) for i in top_idx]


def main():
    parser = argparse.ArgumentParser(description="Get top-N recommendations for a user")
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--model_type", choices=["collaborative", "content_based"],
                        default="collaborative",
                        help="Model type: collaborative (SVD) or content_based (cosine similarity)")
    args = parser.parse_args()

    if args.model_type == "content_based":
        recommender = ContentBasedRecommender()
    else:
        recommender = Recommender()

    recs = recommender.recommend(args.user_id, args.top_n)
    logger.info("Top-%d %s recommendations for %s: %s",
                args.top_n, args.model_type, args.user_id, recs)
    print(f"Top-{args.top_n} {args.model_type} recommendations for {args.user_id}:")
    for product_id, score in recs:
        print(f"  product_id={product_id}  score={score:.4f}")


if __name__ == "__main__":
    main()

"""
Inference interface for the trained SVD recommendation model.

Loads models/svd_model.npz (user/item latent factors) and returns the
top-N recommended product IDs for a given user.

Usage:
    python src/models/inference.py --user_id U0001 --top_n 5
"""
import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from common import get_logger, PROJECT_ROOT  # noqa: E402

logger = get_logger("model_inference")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "svd_model.npz")


class Recommender:
    def __init__(self, model_path: str = MODEL_PATH):
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


def main():
    parser = argparse.ArgumentParser(description="Get top-N recommendations for a user")
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()

    recommender = Recommender()
    recs = recommender.recommend(args.user_id, args.top_n)
    logger.info("Top-%d recommendations for %s: %s", args.top_n, args.user_id, recs)
    print(f"Top-{args.top_n} recommendations for {args.user_id}:")
    for product_id, score in recs:
        print(f"  product_id={product_id}  score={score:.4f}")


if __name__ == "__main__":
    main()

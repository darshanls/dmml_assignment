"""
Ingest product sentiment/popularity scores from an external scoring service.

NOTE: For this POC there is no free, key-less sentiment API that maps to our
FakeStore product catalog, so the HTTP call is simulated locally
(`_call_sentiment_service`). The function signature, retry/error-handling and
logging pattern are identical to a real REST integration (e.g. AWS
Comprehend, a review-sentiment microservice) -- swapping in a real endpoint
only requires changing this one function.

Usage:
    python src/ingestion/ingest_sentiment_api.py
"""
import os
import random
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.append(os.path.dirname(__file__))
from common import get_logger, partitioned_path, network_retry

logger = get_logger("ingest.sentiment_api")

NUM_PRODUCTS = 20
FAILURE_RATE = 0.15  # simulate occasional transient API failures


@network_retry
def _call_sentiment_service(product_id: int) -> dict:
    """Simulated call to an external sentiment/popularity scoring API."""
    time.sleep(0.01)  # simulate network latency
    if random.random() < FAILURE_RATE:
        raise ConnectionError(f"Transient error scoring product_id={product_id}")
    return {
        "product_id": product_id,
        "sentiment_score": round(random.uniform(-1, 1), 3),
        "popularity_score": round(random.uniform(0, 1), 3),
        "scored_at": datetime.utcnow().isoformat(),
    }


def run():
    logger.info("Starting sentiment/popularity API ingestion run")
    results, failures = [], []
    for product_id in range(1, NUM_PRODUCTS + 1):
        try:
            results.append(_call_sentiment_service(product_id))
        except Exception as exc:
            logger.warning("Giving up on product_id=%s after retries: %s", product_id, exc)
            failures.append(product_id)

    df = pd.DataFrame(results)
    out_dir = partitioned_path(source="sentiment", data_type="scores")
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_path = os.path.join(out_dir, f"sentiment_{ts}.csv")
    df.to_csv(out_path, index=False)

    if failures:
        logger.error("Ingestion completed WITH %d failures: %s", len(failures), failures)
    logger.info("SUCCESS: wrote %d sentiment/popularity records to %s", len(df), out_path)
    return out_path


if __name__ == "__main__":
    run()

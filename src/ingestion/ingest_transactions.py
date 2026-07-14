"""
Ingest transactional purchase history (simulated OLTP export) into the raw
data lake, partitioned by source/type/date.

Usage:
    python src/ingestion/ingest_transactions.py
"""
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.append(os.path.dirname(__file__))
from common import get_logger, partitioned_path, network_retry
from data_generator import generate_transactions_batch

logger = get_logger("ingest.transactions")


@network_retry
def fetch_transactions(n: int = 400):
    """Pull the latest completed transactions from the source OLTP system."""
    txns = generate_transactions_batch(n)
    if not txns:
        raise ConnectionError("Empty response from transactions source")
    return txns


def run(n_txns: int = 400):
    logger.info("Starting transactions ingestion run (n=%d)", n_txns)
    try:
        txns = fetch_transactions(n_txns)
        df = pd.DataFrame(txns)

        out_dir = partitioned_path(source="transactions", data_type="purchases")
        filename = f"transactions_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.csv"
        out_path = os.path.join(out_dir, filename)
        df.to_csv(out_path, index=False)

        logger.info("SUCCESS: wrote %d rows to %s", len(df), out_path)
        return out_path
    except Exception as exc:
        logger.error("FAILED transactions ingestion: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    run()

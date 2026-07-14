"""
Ingest web/mobile clickstream events (simulated source system) into the raw
data lake, partitioned by source/type/date.

Usage:
    python src/ingestion/ingest_clickstream.py
"""
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.append(os.path.dirname(__file__))
from common import get_logger, partitioned_path, network_retry
from data_generator import generate_clickstream_batch

logger = get_logger("ingest.clickstream")


@network_retry
def fetch_clickstream_events(n: int = 2000):
    """Pull the latest clickstream events from the source system."""
    events = generate_clickstream_batch(n)
    if not events:
        raise ConnectionError("Empty response from clickstream source")
    return events


def run(n_events: int = 2000):
    logger.info("Starting clickstream ingestion run (n=%d)", n_events)
    try:
        events = fetch_clickstream_events(n_events)
        df = pd.DataFrame(events)

        out_dir = partitioned_path(source="clickstream", data_type="events")
        filename = f"clickstream_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.csv"
        out_path = os.path.join(out_dir, filename)
        df.to_csv(out_path, index=False)

        logger.info("SUCCESS: wrote %d rows to %s", len(df), out_path)
        return out_path
    except Exception as exc:
        logger.error("FAILED clickstream ingestion: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    run()

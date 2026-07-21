"""
Synthetic source-system simulator for RecoMart.

In a real deployment, clickstream events would arrive from Kafka/Segment and
transactions from an OLTP database (e.g. Postgres). For this POC we simulate
those source systems so the ingestion scripts have something realistic to
pull from on every scheduled run (mirrors an incremental/periodic pull).

Product IDs are aligned with the FakeStore API product catalog (IDs 1-20)
so that clickstream/transaction events reference real ingested product data.
"""
import random
from datetime import datetime, timedelta

from faker import Faker

fake = Faker()
random.seed()  # non-deterministic per run to mimic new incoming data

NUM_USERS = 500
NUM_PRODUCTS = 20  # matches FakeStore API catalog size
EVENT_TYPES = ["view", "click", "add_to_cart", "wishlist", "purchase"]

_USER_IDS = [f"U{str(i).zfill(4)}" for i in range(1, NUM_USERS + 1)]
_PRODUCT_IDS = list(range(1, NUM_PRODUCTS + 1))

# ---------------------------------------------------------------------------
# Dirty-data injection ratios (keep small so the pipeline still runs end-to-end
# but large enough to be visible in quality reports).
# ---------------------------------------------------------------------------
_DIRTY_RATIO = 0.04           # ~4 % of rows will carry some data-quality issue
_DUPLICATE_RATIO = 0.02       # ~2 % exact duplicate rows


def generate_clickstream_batch(n: int = 2000):
    """Simulate a batch of new clickstream events since the last ingestion run.

    Includes a small fraction of realistic dirty rows to exercise downstream
    validation / cleaning stages:
      - Missing user_id or product_id
      - Invalid / malformed event_timestamp
      - Unexpected event_type values
      - Exact duplicate rows
    """
    now = datetime.utcnow()
    rows = []
    for _ in range(n):
        row = {
            "event_id": fake.uuid4(),
            "user_id": random.choice(_USER_IDS),
            "product_id": random.choice(_PRODUCT_IDS),
            "event_type": random.choices(
                EVENT_TYPES, weights=[0.55, 0.25, 0.10, 0.05, 0.05]
            )[0],
            "device": random.choice(["web", "android", "ios"]),
            "session_id": fake.uuid4(),
            "event_timestamp": (now - timedelta(seconds=random.randint(0, 3600))).isoformat(),
        }

        # --- inject dirty values on a small % of rows ----------------------
        if random.random() < _DIRTY_RATIO:
            defect = random.choice(["missing_user", "missing_product",
                                    "bad_timestamp", "bad_event_type"])
            if defect == "missing_user":
                row["user_id"] = None
            elif defect == "missing_product":
                row["product_id"] = None
            elif defect == "bad_timestamp":
                row["event_timestamp"] = random.choice(
                    ["not-a-date", "31/13/2025", "", "NaT"]
                )
            elif defect == "bad_event_type":
                row["event_type"] = random.choice(["CLICK", "browse", "unknown", ""])

        rows.append(row)

    # --- inject exact duplicate rows --------------------------------------
    n_dups = max(1, int(n * _DUPLICATE_RATIO))
    dups = [dict(rows[random.randint(0, len(rows) - 1)]) for _ in range(n_dups)]
    rows.extend(dups)

    random.shuffle(rows)
    return rows


def generate_transactions_batch(n: int = 400):
    """Simulate a batch of new completed purchase transactions.

    Includes realistic dirty rows to exercise validation / cleaning:
      - Out-of-range ratings (0, 6, -1, 10)
      - Negative or zero unit_price
      - Missing user_id or product_id
      - quantity = 0 or negative
      - Malformed timestamps
      - Duplicate transaction_id rows
      - Non-numeric rating values (e.g. "five")
    """
    now = datetime.utcnow()
    rows = []
    for _ in range(n):
        qty = random.randint(1, 4)
        unit_price = round(random.uniform(5, 500), 2)
        row = {
            "transaction_id": fake.uuid4(),
            "user_id": random.choice(_USER_IDS),
            "product_id": random.choice(_PRODUCT_IDS),
            "quantity": qty,
            "unit_price": unit_price,
            "total_amount": round(qty * unit_price, 2),
            "rating": random.choices(
                [None, 1, 2, 3, 4, 5], weights=[0.5, 0.03, 0.05, 0.12, 0.15, 0.15]
            )[0],
            "transaction_timestamp": (now - timedelta(seconds=random.randint(0, 86400))).isoformat(),
        }

        # --- inject dirty values on a small % of rows ----------------------
        if random.random() < _DIRTY_RATIO:
            defect = random.choice(["bad_rating", "neg_price", "missing_user",
                                    "bad_qty", "bad_timestamp", "text_rating"])
            if defect == "bad_rating":
                row["rating"] = random.choice([0, -1, 6, 10])  # outside 1-5
            elif defect == "neg_price":
                row["unit_price"] = round(random.uniform(-50, 0), 2)
                row["total_amount"] = round(row["quantity"] * row["unit_price"], 2)
            elif defect == "missing_user":
                row["user_id"] = None
            elif defect == "bad_qty":
                row["quantity"] = random.choice([0, -1, -3])
                row["total_amount"] = round(row["quantity"] * row["unit_price"], 2)
            elif defect == "bad_timestamp":
                row["transaction_timestamp"] = random.choice(
                    ["invalid", "2025-13-40T00:00:00", ""]
                )
            elif defect == "text_rating":
                row["rating"] = random.choice(["five", "good", "N/A"])

        rows.append(row)

    # --- inject exact duplicate rows (same transaction_id) ----------------
    n_dups = max(1, int(n * _DUPLICATE_RATIO))
    dups = [dict(rows[random.randint(0, len(rows) - 1)]) for _ in range(n_dups)]
    rows.extend(dups)

    random.shuffle(rows)
    return rows

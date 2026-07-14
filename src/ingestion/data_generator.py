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


def generate_clickstream_batch(n: int = 2000):
    """Simulate a batch of new clickstream events since the last ingestion run."""
    now = datetime.utcnow()
    rows = []
    for _ in range(n):
        rows.append({
            "event_id": fake.uuid4(),
            "user_id": random.choice(_USER_IDS),
            "product_id": random.choice(_PRODUCT_IDS),
            "event_type": random.choices(
                EVENT_TYPES, weights=[0.55, 0.25, 0.10, 0.05, 0.05]
            )[0],
            "device": random.choice(["web", "android", "ios"]),
            "session_id": fake.uuid4(),
            "event_timestamp": (now - timedelta(seconds=random.randint(0, 3600))).isoformat(),
        })
    return rows


def generate_transactions_batch(n: int = 400):
    """Simulate a batch of new completed purchase transactions."""
    now = datetime.utcnow()
    rows = []
    for _ in range(n):
        qty = random.randint(1, 4)
        unit_price = round(random.uniform(5, 500), 2)
        rows.append({
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
        })
    return rows

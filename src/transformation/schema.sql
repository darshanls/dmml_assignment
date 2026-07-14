-- RecoMart Feature Warehouse Schema (SQLite)
-- Loaded by src/transformation/feature_engineering.py into warehouse/recomart.db

DROP TABLE IF EXISTS dim_users;
CREATE TABLE dim_users (
    user_id             TEXT PRIMARY KEY,
    total_events        INTEGER,
    total_purchases     INTEGER,
    total_spend         REAL,
    avg_rating_given    REAL,
    activity_frequency  REAL,   -- events per active day
    last_active_at      TEXT
);

DROP TABLE IF EXISTS dim_items;
CREATE TABLE dim_items (
    product_id          INTEGER PRIMARY KEY,
    title                TEXT,
    category             TEXT,
    price                REAL,
    price_norm           REAL,
    category_encoded     INTEGER,
    avg_rating_item      REAL,
    total_interactions   INTEGER,
    total_purchases      INTEGER,
    sentiment_score      REAL,
    popularity_score     REAL
);

DROP TABLE IF EXISTS fact_interactions;
CREATE TABLE fact_interactions (
    event_id             TEXT PRIMARY KEY,
    user_id              TEXT,
    product_id           INTEGER,
    event_type           TEXT,
    event_type_encoded   INTEGER,
    device               TEXT,
    event_timestamp      TEXT,
    FOREIGN KEY (user_id) REFERENCES dim_users(user_id),
    FOREIGN KEY (product_id) REFERENCES dim_items(product_id)
);

DROP TABLE IF EXISTS fact_transactions;
CREATE TABLE fact_transactions (
    transaction_id       TEXT PRIMARY KEY,
    user_id               TEXT,
    product_id            INTEGER,
    quantity               INTEGER,
    unit_price             REAL,
    total_amount           REAL,
    rating                 REAL,
    transaction_timestamp  TEXT,
    FOREIGN KEY (user_id) REFERENCES dim_users(user_id),
    FOREIGN KEY (product_id) REFERENCES dim_items(product_id)
);

DROP TABLE IF EXISTS item_cooccurrence;
CREATE TABLE item_cooccurrence (
    product_id_a         INTEGER,
    product_id_b          INTEGER,
    cooccurrence_count    INTEGER,
    similarity_score       REAL,
    PRIMARY KEY (product_id_a, product_id_b)
);

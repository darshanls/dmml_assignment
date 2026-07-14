# Feature Store - Metadata Documentation (Task 7)

The feature registry lives at `src/feature_store/feature_registry.yaml` and
is served by `src/feature_store/feature_store.py` (`FeatureStore` class),
which reads the registry and retrieves feature values from the SQLite
warehouse (`warehouse/recomart.db`) for both **offline (training)** and
**online (inference)** use cases.

## Entities

| Entity | Join key    |
|--------|-------------|
| user   | `user_id`   |
| item   | `product_id`|

## Feature View: `user_features_v1` (source table: `dim_users`)

| Feature              | Type  | Transformation                                          |
|----------------------|-------|-----------------------------------------------------------|
| `total_events`        | int   | count(clickstream events) grouped by `user_id`            |
| `total_purchases`      | int   | count(transactions) grouped by `user_id`                  |
| `total_spend`          | float | sum(`total_amount`) grouped by `user_id`                    |
| `avg_rating_given`      | float | mean(`rating`) grouped by `user_id`, explicit ratings only  |
| `activity_frequency`    | float | `total_events / distinct_active_days`                       |

## Feature View: `item_features_v1` (source table: `dim_items`)

| Feature              | Type  | Transformation                                     |
|-----------------------|-------|-------------------------------------------------------|
| `price_norm`           | float | min-max normalized `price`                             |
| `category_encoded`      | int   | label-encoded `category`                                |
| `avg_rating_item`        | float | mean(`rating`) grouped by `product_id`                    |
| `total_interactions`      | int   | count(clickstream events) grouped by `product_id`          |
| `sentiment_score`          | float | external sentiment API score, range [-1, 1]                |
| `popularity_score`          | float | external popularity API score, range [0, 1]                |

## Feature View: `item_similarity_v1` (source table: `item_cooccurrence`)

| Feature            | Type  | Transformation                                         |
|---------------------|-------|-----------------------------------------------------------|
| `similarity_score`   | float | Jaccard(users(item_a), users(item_b)) over co-occurring users |

## Versioned retrieval

`feature_registry.yaml` has a top-level `version` field. Each feature view
name is suffixed with `_v1`; a future breaking change to a transformation
(e.g. switching `activity_frequency` to a decayed/weighted formula) should be
released as `user_features_v2` so historical training runs referencing `_v1`
remain reproducible.

## Sample retrieval demonstration

```powershell
python src/feature_store/feature_store.py
```

This prints the list of registered feature views, retrieves the full
`user_features_v1` table (training/offline use), and retrieves feature rows
for 3 sample `user_id`s (inference/online use) -- demonstrating that both
paths resolve through the same registry and warehouse table.

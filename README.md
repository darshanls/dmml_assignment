# RecoMart - End-to-End Data Management Pipeline for a Recommendation System

A modular, orchestrated data pipeline that ingests, validates, prepares,
transforms, and models RecoMart's e-commerce user-behavior data to power a
personalized product recommendation engine.

## 1. Problem Statement

**Business problem**: RecoMart wants to increase conversion rate and
cross-selling by recommending relevant products to each user based on their
browsing/purchase behavior, product catalog attributes, and external
sentiment/popularity signals.

**Data sources**:

| Source        | Type              | Attributes                                                       |
|----------------|-------------------|--------------------------------------------------------------------|
| Clickstream     | Simulated event stream (CSV) | `event_id`, `user_id`, `product_id`, `event_type`, `device`, `session_id`, `event_timestamp` |
| Transactions      | Simulated OLTP export (CSV)  | `transaction_id`, `user_id`, `product_id`, `quantity`, `unit_price`, `total_amount`, `rating`, `transaction_timestamp` |
| Product catalog    | REST API (FakeStore API, live) | `product_id`, `title`, `price`, `category`, `description`, `image`, `rating_rate`, `rating_count` |
| Sentiment/popularity | External scoring API (simulated) | `product_id`, `sentiment_score` [-1,1], `popularity_score` [0,1] |

**Pipeline outputs**:
- Clean datasets for EDA → `data/processed/`
- Engineered features for collaborative/content-based models → `data/features/`, `warehouse/recomart.db`
- Deployable model + inference interface → `models/svd_model.npz`, `src/models/inference.py`

**Evaluation metrics**: Precision@K, Recall@K, NDCG@K (K=10), tracked in MLflow.

## 2. Architecture

```
Ingestion  ─▶  Raw Storage  ─▶  Validation  ─▶  Preparation  ─▶  Transformation  ─▶  Feature Store  ─▶  Model Training  ─▶  Inference
 (4 sources)    (partitioned)    (quality       (cleaning,        (feature           (versioned          (SVD +           (top-N
                                  checks +        encoding,          engineering        registry +           MLflow            recs)
                                  PDF report)      normalization,    → SQLite            offline/online       tracking)
                                                   EDA plots)        warehouse)          retrieval)
```

Orchestrated end-to-end with **Prefect** (`src/orchestration/pipeline_flow.py`).

## 3. Project Structure

```
DMML/
├── data/
│   ├── raw/            # partitioned raw ingestion output (source/type/date)
│   ├── processed/       # cleaned/encoded/normalized datasets
│   └── features/         # engineered feature tables (CSV mirror of warehouse)
├── warehouse/
│   └── recomart.db        # SQLite feature warehouse (see schema.sql)
├── models/
│   └── svd_model.npz        # trained collaborative-filtering model
├── mlruns/                    # MLflow experiment tracking store
├── logs/
│   └── ingestion.log            # rotating ingestion logs
├── reports/
│   ├── data_quality_report.json
│   ├── Data_Quality_Report.pdf
│   └── eda_plots/*.png
├── notebooks/
│   └── eda.ipynb                  # cleaning + EDA walkthrough (executed, with outputs)
├── docs/
│   ├── STORAGE_STRUCTURE.md         # Task 3 deliverable
│   ├── FEATURE_METADATA.md            # Task 7 deliverable
│   └── VERSIONING.md                    # Task 8 deliverable
└── src/
    ├── ingestion/        (Task 2)  ingest_clickstream.py, ingest_transactions.py, ingest_products_api.py, ingest_sentiment_api.py, data_generator.py, common.py
    ├── validation/         (Task 4)  validate_data.py, generate_quality_report.py
    ├── preparation/          (Task 5)  clean_and_prepare.py
    ├── transformation/         (Task 6)  feature_engineering.py, schema.sql
    ├── feature_store/            (Task 7)  feature_store.py, feature_registry.yaml
    ├── models/                     (Task 9)  train_model.py, inference.py
    └── orchestration/                (Task 10) pipeline_flow.py
```

## 4. Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

> `requirements.txt` pins `griffe<1.0` and `pydantic==2.7.4` — these are
> required for compatibility with `prefect==2.19.4` on newer Python/pydantic
> environments; see comments in `requirements.txt`.

## 5. Running the Pipeline

**End-to-end (recommended)**:
```powershell
.venv\Scripts\python src\orchestration\pipeline_flow.py
```

**Stage-by-stage**:
```powershell
.venv\Scripts\python src\ingestion\ingest_clickstream.py
.venv\Scripts\python src\ingestion\ingest_transactions.py
.venv\Scripts\python src\ingestion\ingest_products_api.py
.venv\Scripts\python src\ingestion\ingest_sentiment_api.py

.venv\Scripts\python src\validation\validate_data.py
.venv\Scripts\python src\validation\generate_quality_report.py

.venv\Scripts\python src\preparation\clean_and_prepare.py
.venv\Scripts\python src\transformation\feature_engineering.py

.venv\Scripts\python src\feature_store\feature_store.py    # sample retrieval demo

.venv\Scripts\python src\models\train_model.py
.venv\Scripts\python src\models\inference.py --user_id U0001 --top_n 5
```

**View MLflow experiment tracking UI**:
```powershell
.venv\Scripts\python -m mlflow ui --backend-store-uri mlruns
```

**Run the EDA notebook**:
```powershell
.venv\Scripts\python -m jupyter notebook notebooks\eda.ipynb
```

## 6. Data Quality & Validation (Task 4)

`src/validation/validate_data.py` checks each ingested dataset for missing
values, duplicate primary keys, schema mismatches, and range/format
violations (e.g. `rating` must be 1-5, `sentiment_score` must be in [-1,1]).
Results are written to `reports/data_quality_report.json` and rendered to
`reports/Data_Quality_Report.pdf`.

## 7. Feature Engineering (Task 6)

`src/transformation/feature_engineering.py` computes:
- **User features**: `total_events`, `total_purchases`, `total_spend`, `avg_rating_given`, `activity_frequency`
- **Item features**: `price_norm`, `category_encoded`, `avg_rating_item`, `total_interactions`, `sentiment_score`, `popularity_score`
- **Item-item co-occurrence/similarity**: Jaccard similarity over shared user interaction sets

...and loads them into the SQLite warehouse per `src/transformation/schema.sql`.

## 8. Feature Store (Task 7)

See `docs/FEATURE_METADATA.md` and `src/feature_store/`. A lightweight
custom feature store (`FeatureStore` class) serves both offline (training)
and online (inference) retrieval from the same versioned registry.

## 9. Model Training & Evaluation (Task 9)

`src/models/train_model.py` builds a weighted user-item interaction matrix
(implicit clickstream signals + explicit ratings), trains a **Truncated SVD**
collaborative-filtering model, evaluates **Precision@10 / Recall@10 / NDCG@10**
on a per-user held-out split, and logs parameters/metrics/model artifacts to
**MLflow** (`mlruns/`). `src/models/inference.py` provides the deployable
top-N recommendation interface.

## 10. Orchestration (Task 10)

`src/orchestration/pipeline_flow.py` defines a **Prefect** `@flow` with
`@task`-decorated stages (ingest → validate → prepare → transform →
feature-store → train), each with automatic retries and structured logging.
Console/log output from a full run is captured in `logs/ingestion.log` and
the terminal output shown during development (see project report for
screenshots).

## 11. Versioning & Lineage (Task 8)

See `docs/VERSIONING.md` for the Git + DVC workflow (raw/processed data and
model artifacts are `.gitignore`'d and intended to be tracked via `dvc add`
+ `.dvc` pointer files committed to Git).

## 12. Notes & Limitations (POC scope)

- Clickstream and transaction data are **synthetically generated** per run
  (`src/ingestion/data_generator.py`) to simulate a live source system,
  since no production Kafka/OLTP system is available for this POC.
- Product catalog ingestion calls the **real** FakeStore REST API
  (`https://fakestoreapi.com/products`).
- Sentiment/popularity ingestion simulates an external scoring API call
  (with retry/error-handling identical to a real integration) since no
  free key-less sentiment API matches the product catalog.
- Git/DVC repo initialization was left as a manual step (see `docs/VERSIONING.md`)
  to avoid mutating global git config without explicit user consent.

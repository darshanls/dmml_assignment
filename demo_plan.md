---
title: RecoMart Pipeline — Demo Walkthrough Plan
description: Detailed step-by-step script for recording the 5-10 min end-to-end project demo video
---

# RecoMart Pipeline — Demo Walkthrough Plan

This document is a script for recording the required 5-10 minute demo video
showing the complete end-to-end execution of the RecoMart data management
pipeline. Follow the sections in order; each includes what to say and what
commands/screens to show.

**Total target runtime: 7-9 minutes.**

---

## 0. Pre-recording checklist

- [ ] `.venv` created and `requirements.txt` installed.
- [ ] Terminal font size increased for readability on screen.
- [ ] Close unrelated windows/tabs.
- [ ] Have a SQLite viewer (e.g. DB Browser for SQLite, or plain `sqlite3` CLI)
      ready to open `warehouse\recomart.db`.
- [ ] Screen recorder ready (OBS / Windows Game Bar / Loom).

---

## 1. Introduction (0:00 - 0:30)

**Say:**
"This is RecoMart's end-to-end data management pipeline for a product
recommendation system. RecoMart is an e-commerce startup that wants to
increase conversion and cross-selling by personalizing recommendations from
four data sources: clickstream logs, transaction history, a product catalog
REST API, and an external sentiment/popularity scoring API. The pipeline
covers all 10 required stages: ingestion, raw storage, validation,
preparation, feature engineering, a feature store, versioning, model
training/evaluation, and orchestration — end to end, re-runnable with one
command, and fully logged for auditability."

**Show:** `README.md` — scroll to the Problem Statement (Section 1) showing
the data sources table and evaluation metrics, the architecture diagram
(Section 2), and the project structure tree (Section 3).

---

## 2. Environment Setup (0:30 - 0:50) — *skip if already installed, just mention it*

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**Say:** "Dependencies are pinned in `requirements.txt` — pandas, scikit-learn,
MLflow, Prefect, and DVC, among others. The whole pipeline runs from a single
virtual environment with no external infrastructure required, which keeps it
reproducible on any machine."

---

## 3. Data Ingestion — Task 2 & 3 (0:50 - 1:50)

```powershell
.venv\Scripts\python src\ingestion\ingest_clickstream.py
.venv\Scripts\python src\ingestion\ingest_transactions.py
.venv\Scripts\python src\ingestion\ingest_products_api.py
.venv\Scripts\python src\ingestion\ingest_sentiment_api.py
```

**Say:** "Four sources are ingested, covering the two required data types
and more: simulated clickstream and transaction logs mimicking a live
Kafka/OLTP feed, the live FakeStore REST API for the product catalog, and a
retry-capable sentiment/popularity scoring client. Every ingestion script
wraps its fetch call with `tenacity`-based exponential-backoff retries, logs
success/failure to both console and a rotating log file, and writes into a
partitioned raw data lake keyed by source, type, and date — so this is
truly automated and periodic, not a one-off script."

**Show:**
- Console output showing `SUCCESS: wrote N rows to ...` lines.
- `logs\ingestion.log` (rotating log file) — open in a text editor.
- File Explorer: `data\raw\<source>\<type>\<date>\` partitioned folder tree.

---

## 4. Data Validation — Task 4 (1:50 - 2:40)

```powershell
.venv\Scripts\python src\validation\validate_data.py
.venv\Scripts\python src\validation\generate_quality_report.py
```

**Say:** "Because the data generator deliberately injects ~4% dirty rows
and ~2% duplicates — missing IDs, malformed timestamps, out-of-range
ratings, negative prices, non-numeric values like 'five' in the rating
column — the validation stage now has realistic issues to surface.
`validate_data.py` runs a comprehensive check suite: missing-value
percentages per column, duplicate rows and duplicate primary keys, schema
checks, range/format violations, timestamp format validation that catches
unparseable dates, allowed-value checks that flag invalid event types or
device names, and non-numeric detection in numeric fields. Every dataset
gets a status of OK, WARN, FAIL, or MISSING, and the results are written
to a machine-readable JSON report before being rendered into the PDF
you're seeing now."

**Show:** Open `reports\Data_Quality_Report.pdf` — scroll through the
per-dataset status, row counts, and any flagged issues.

---

## 5. Data Preparation & EDA — Task 5 (2:40 - 3:40)

```powershell
.venv\Scripts\python src\preparation\clean_and_prepare.py
```

**Say:** "This stage handles all the dirty data the generator injected.
Clickstream cleaning drops duplicate event IDs, normalises invalid event
types, and removes rows with malformed timestamps. Transaction cleaning
coerces text values in numeric columns, removes rows with non-positive
prices or quantities, nullifies out-of-range ratings, recalculates
`total_amount`, and drops unparseable timestamps — all with before/after
row-count logging so you can see exactly what was cleaned. Categorical
attributes are label-encoded and numeric fields are min-max normalized.
The EDA now produces 10 plots covering interaction distributions, item
popularity, rating distribution, user-item sparsity, plus new ones:
device-by-event-type heatmap, price distribution histogram, user activity
distribution, transactions by hour of day, and unit price box plot by
category."

**Show:**
- `data\processed\*.csv` files in File Explorer.
- Open plots in `reports\eda_plots\` — show both the original plots
  (event type distribution, item popularity, sparsity, rating distribution)
  and the new ones (device_event_heatmap, price_distribution,
  user_activity_distribution, transactions_by_hour,
  price_by_category_boxplot).
- Optionally open `notebooks\eda.ipynb` in Jupyter to show the executed
  notebook walkthrough.

---

## 6. Feature Engineering & Transformation — Task 6 (3:40 - 4:30)

```powershell
.venv\Scripts\python src\transformation\feature_engineering.py
```

**Say:** "Features are engineered here to directly support both
collaborative and content-based recommendation models. User-side features
now include total events, total purchases, total spend, average rating
given, activity frequency, plus new ones: total views, unique sessions,
average spend per transaction, purchase-to-view conversion ratio, and a
recency score (days since last active). Item-side features include
normalized price, encoded category, average rating received, and now also
rating count, a Bayesian weighted average rating that shrinks towards the
global mean to handle items with few ratings, total views per item, unique
users per item, an item-level conversion rate (purchases divided by views),
and the external sentiment/popularity scores. There's also the item-item
co-occurrence table with Jaccard similarity. All of this is loaded into a
SQLite warehouse using a declarative star-schema defined in `schema.sql`,
with dimension tables for users and items and fact tables for interactions
and transactions."

**Show:** Open `warehouse\recomart.db` in a SQLite viewer — show
`dim_users`, `dim_items`, `item_cooccurrence`, `fact_interactions`,
`fact_transactions` tables with sample rows.

---

## 7. Feature Store — Task 7 (4:30 - 5:10)

```powershell
.venv\Scripts\python src\feature_store\feature_store.py
```

**Say:** "Rather than standing up the full operational overhead of Feast
for this POC, we built a lightweight custom feature store that mirrors its
core contract: a declarative `feature_registry.yaml` documents every feature
name, its source table, data type, and exact transformation logic, and is
versioned with a top-level `version` field plus `_v1` suffixes on each
feature view — so a future breaking change would ship as `_v2` without
breaking reproducibility of past training runs. The `FeatureStore` class
exposes `get_training_features` for full-table offline retrieval and
`get_online_features` for row-level online retrieval by entity ID, both
resolving through the same registry and warehouse tables, which is exactly
what's demonstrated in the console output here."

**Show:** Console output listing registered feature views and the sample
training/online retrieval demo. Briefly show `docs\FEATURE_METADATA.md`.

---

## 8. Data Versioning & Lineage — Task 8 (5:10 - 5:50)

```powershell
git log --oneline
dvc list . --dvc-only
```

**Say:** "Code is versioned in Git; raw data, processed data, engineered
features, the SQLite warehouse, and both trained model artifacts are
versioned via DVC, with `.dvc` pointer files — each storing an MD5 hash of
the tracked content — committed to Git. That means every Git commit can
reproduce the exact dataset and model snapshot used at that point in time,
and `git log -p` on a `.dvc` file gives a full lineage of every version.
We're also tracking MLflow's run metadata — metrics, params, and tags — in
Git directly, since those are small text files, while excluding the
duplicate model-artifact copies MLflow keeps internally, since those are
already versioned once via DVC."

**Show:** Git commit history, and the `.dvc` pointer files (`data\raw.dvc`,
`warehouse\recomart.db.dvc`, `models\svd_model.npz.dvc`) — briefly open one
to show the MD5 hash. Reference `docs\VERSIONING.md`.

---

## 9. Model Training & Evaluation — Task 9 (5:50 - 7:00)

```powershell
.venv\Scripts\python src\models\train_model.py
```

**Say:** "Two models are trained: (1) a Truncated SVD collaborative-filtering
model on a weighted user-item interaction matrix combining implicit clickstream
signals and explicit ratings, and (2) a content-based filtering model using
item features (price, category, sentiment, popularity) with cosine similarity.
Both are evaluated with Precision@10, Recall@10, and NDCG@10 on a per-user
held-out split, and each run is tracked separately in MLflow."

```powershell
.venv\Scripts\python -m mlflow ui --backend-store-uri mlruns
```

**Show:** Open `http://127.0.0.1:5000` in a browser — show the
`recomart_recommendation` experiment with both runs:
`svd_collaborative_filtering` and `content_based_filtering`, their
logged params and metrics (`precision_at_k`, `recall_at_k`, `ndcg_at_k`).

Then demo inference with both models:

```powershell
.venv\Scripts\python src\models\inference.py --user_id <a real user_id> --top_n 5
.venv\Scripts\python src\models\inference.py --user_id <a real user_id> --top_n 5 --model_type content_based
```

**Say:** "This is the deployable inference interface — given a user ID and
a model type flag, it returns the top-N recommended product IDs with scores.
Both collaborative and content-based models are supported behind the same
CLI, loading their respective serialized `.npz` artifacts and scoring in
milliseconds, which is exactly the kind of interface a recommendation API
endpoint would wrap."

---

## 10. Pipeline Orchestration — Task 10 (7:00 - 8:15)

```powershell
.venv\Scripts\python src\orchestration\pipeline_flow.py
```

**Say:** "Finally, the entire pipeline — four parallel ingestion tasks,
then validation, preparation, transformation, feature store, and training —
is orchestrated as a single Prefect flow. Each stage is a `@task` with
automatic retries and structured logging, and Prefect's dependency graph
ensures validation only runs after all four ingestion tasks succeed, and
each downstream stage only runs after its upstream stage completes. This is
the DAG the assignment asks for: ingestion → validation → preparation →
transformation → feature store → model training, fully automatable and
schedulable via a Prefect deployment."

**Show:** Console output as the flow executes: 4 parallel ingestion tasks,
followed by validate → prepare → transform → feature_store → train. Point
out the `retries=2, retry_delay_seconds=5` config in
`src\orchestration\pipeline_flow.py` and the final "Pipeline finished. Model
result: ..." line.

---

## 11. Wrap-up (8:15 - 8:45)

**Say:** "That's the full pipeline — from raw multi-source ingestion through
validation, preparation, feature engineering, a versioned feature store, two
trained and evaluated recommendation models, and a single-command Prefect
orchestration, all fully versioned with Git and DVC. Every stage is modular,
independently runnable, logged, and retry-safe, which is what lets this POC
scale toward a production data platform for RecoMart."

**Show:** Scroll through `reports\RecoMart_Project_Report.pdf` — Problem
Statement, Objectives, Methodology, Results, and Conclusion/Future Scope
sections.

**End recording.**

---

## Post-recording

1. Upload the recording to Google Drive, set sharing to "Anyone with the
   link", and copy the link.
2. Zip project deliverables (source code, datasets, trained model, docs) and
   upload to Google Drive similarly.
3. Provide both links so `src\report\generate_project_report.py` can be
   updated (`GDRIVE_VIDEO_LINK`, `GDRIVE_ZIP_LINK`) and the final PDF
   regenerated.

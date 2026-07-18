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
recommendation system. It covers ingestion, storage, validation, preparation,
feature engineering, a feature store, model training, and orchestration."

**Show:** `README.md` — scroll to the architecture diagram (Section 2) and
the project structure tree (Section 3).

---

## 2. Environment Setup (0:30 - 0:50) — *skip if already installed, just mention it*

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**Say:** "Dependencies are pinned in `requirements.txt` — pandas, scikit-learn,
MLflow, Prefect, DVC, etc."

---

## 3. Data Ingestion — Task 2 & 3 (0:50 - 1:50)

```powershell
.venv\Scripts\python src\ingestion\ingest_clickstream.py
.venv\Scripts\python src\ingestion\ingest_transactions.py
.venv\Scripts\python src\ingestion\ingest_products_api.py
.venv\Scripts\python src\ingestion\ingest_sentiment_api.py
```

**Say:** "Four sources are ingested: simulated clickstream and transaction
logs, the live FakeStore REST API for the product catalog, and a simulated
sentiment/popularity scoring API — all with retry logic via `tenacity` and
structured logging."

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

**Say:** "Validation checks missing values, duplicate primary keys, schema
mismatches, and range checks — e.g. rating must be 1-5, sentiment score
between -1 and 1."

**Show:** Open `reports\Data_Quality_Report.pdf` — scroll through the
per-dataset status, row counts, and any flagged issues.

---

## 5. Data Preparation & EDA — Task 5 (2:40 - 3:40)

```powershell
.venv\Scripts\python src\preparation\clean_and_prepare.py
```

**Say:** "This stage deduplicates, handles missing values, label-encodes
categorical fields, min-max normalizes numeric fields, and produces EDA
plots covering interaction distributions, item popularity, and sparsity."

**Show:**
- `data\processed\*.csv` files in File Explorer.
- Open plots in `reports\eda_plots\` (event type distribution, item
  popularity, sparsity, rating distribution, avg price by category).
- Optionally open `notebooks\eda.ipynb` in Jupyter to show the executed
  notebook walkthrough.

---

## 6. Feature Engineering & Transformation — Task 6 (3:40 - 4:30)

```powershell
.venv\Scripts\python src\transformation\feature_engineering.py
```

**Say:** "Features are engineered here: user activity frequency, average
rating per user/item, and item-item Jaccard similarity based on
co-occurring users — then loaded into a SQLite warehouse per `schema.sql`."

**Show:** Open `warehouse\recomart.db` in a SQLite viewer — show
`dim_users`, `dim_items`, `item_cooccurrence`, `fact_interactions`,
`fact_transactions` tables with sample rows.

---

## 7. Feature Store — Task 7 (4:30 - 5:10)

```powershell
.venv\Scripts\python src\feature_store\feature_store.py
```

**Say:** "A lightweight custom feature store reads a declarative
`feature_registry.yaml` and serves both offline (training) and online
(inference) retrieval from the same warehouse tables."

**Show:** Console output listing registered feature views and the sample
training/online retrieval demo. Briefly show `docs\FEATURE_METADATA.md`.

---

## 8. Data Versioning & Lineage — Task 8 (5:10 - 5:50)

```powershell
git log --oneline
dvc list . --dvc-only
```

**Say:** "Code is versioned in Git; raw/processed data and model artifacts
are versioned via DVC, with `.dvc` pointer files committed to Git so every
commit can reproduce the exact dataset/model snapshot used."

**Show:** Git commit history, and the `.dvc` pointer files (`data\raw.dvc`,
`warehouse\recomart.db.dvc`, `models\svd_model.npz.dvc`) — briefly open one
to show the MD5 hash. Reference `docs\VERSIONING.md`.

---

## 9. Model Training & Evaluation — Task 9 (5:50 - 7:00)

```powershell
.venv\Scripts\python src\models\train_model.py
```

**Say:** "A Truncated SVD collaborative-filtering model is trained on a
weighted user-item interaction matrix combining implicit clickstream
signals and explicit ratings. It's evaluated with Precision@10, Recall@10,
and NDCG@10 on a per-user held-out split, and every run is tracked in
MLflow."

```powershell
.venv\Scripts\python -m mlflow ui --backend-store-uri mlruns
```

**Show:** Open `http://127.0.0.1:5000` in a browser — show the
`recomart_recommendation` experiment, the `svd_collaborative_filtering` run,
its logged params (`n_components`, `top_k`, etc.) and metrics
(`precision_at_k`, `recall_at_k`, `ndcg_at_k`).

Then demo inference:

```powershell
.venv\Scripts\python src\models\inference.py --user_id <a real user_id> --top_n 5
```

**Say:** "This is the deployable inference interface — given a user ID, it
returns the top-N recommended product IDs with scores."

---

## 10. Pipeline Orchestration — Task 10 (7:00 - 8:15)

```powershell
.venv\Scripts\python src\orchestration\pipeline_flow.py
```

**Say:** "Finally, the entire pipeline — ingestion, validation, preparation,
transformation, feature store, and training — is orchestrated as a single
Prefect flow with per-task retries and structured logging, so it can be
scheduled and monitored end-to-end."

**Show:** Console output as the flow executes: 4 parallel ingestion tasks,
followed by validate → prepare → transform → feature_store → train. Point
out the `retries=2, retry_delay_seconds=5` config in
`src\orchestration\pipeline_flow.py` and the final "Pipeline finished. Model
result: ..." line.

---

## 11. Wrap-up (8:15 - 8:45)

**Say:** "That's the full pipeline — from raw multi-source ingestion through
to a trained, evaluated, and deployable recommendation model, fully
versioned and orchestrated."

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

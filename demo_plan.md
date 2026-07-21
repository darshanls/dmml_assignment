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
"This is RecoMart — an end-to-end data pipeline for product recommendations.
It ingests four data sources, validates, cleans, engineers features, trains
two models, and is fully orchestrated with Prefect — all re-runnable with
one command."

**Show:** `README.md` — scroll to the Problem Statement (Section 1) showing
the data sources table and evaluation metrics, the architecture diagram
(Section 2), and the project structure tree (Section 3).

---

## 2. Environment Setup (0:30 - 0:50) — *skip if already installed, just mention it*

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**Say:** "All dependencies are pinned in `requirements.txt`. The pipeline
runs from a single virtual environment — no external infrastructure needed."

---

## 3. Run the Full Pipeline — Tasks 2-9 (0:50 - 2:00)

```powershell
.venv\Scripts\python src\orchestration\pipeline_flow.py
```

**Say:** "One Prefect flow executes the whole pipeline: four parallel
ingestion tasks, then validation, cleaning + EDA, feature engineering,
feature store demo, and model training — all with retries and dependency
ordering. The per-stage commands below are the same logic exposed as
individual scripts, but you do not need to run them for the demo."

**Show:** Terminal output scrolling through each task and ending with
`Pipeline finished. Model result: ...`.

---

## 4. Data Ingestion Output — Task 2 & 3 (2:00 - 2:20)

**Say:** "Four sources are ingested inside the single flow: simulated
clickstream and transactions, the live FakeStore product API, and a
sentiment scoring client. Each script has retry logic, structured logging,
and writes to a date-partitioned raw data lake."

**Show:**
- Console output showing `SUCCESS: wrote N rows to ...` lines.
- `logs\ingestion.log` (rotating log file) — open in a text editor.
- File Explorer: `data\raw\<source>\<type>\<date>\` partitioned folder tree.

---

## 5. Data Validation — Task 4 (2:20 - 2:50)

**Say:** "The generator injects ~4% dirty rows — missing IDs, bad
timestamps, negative prices, text in numeric fields — so validation has
real issues to catch. It checks nulls, duplicates, schema, range
violations, timestamp formats, allowed values, and non-numeric entries.
Results go to a JSON report and this PDF."

**Show:** Open `reports\Data_Quality_Report.pdf` — scroll through the
per-dataset status, row counts, and any flagged issues.

---

## 6. Data Preparation & EDA — Task 5 (2:50 - 3:20)

**Say:** "Cleaning handles all injected dirty data — drops duplicates,
coerces bad types, removes invalid rows — with before/after logging.
Categoricals are label-encoded, numerics min-max normalized. EDA produces
10 plots including heatmaps, distributions, and sparsity analysis."

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

## 7. Feature Engineering & Transformation — Task 6 (3:20 - 3:50)

**Say:** "User features include activity frequency, spend, recency, and
purchase-to-view conversion. Item features include Bayesian weighted
ratings, conversion rates, and sentiment scores. Plus item-item
co-occurrence with Jaccard similarity. Everything loads into a SQLite
warehouse using a star-schema."

**Show:** Open `warehouse\recomart.db` in a SQLite viewer — show
`dim_users`, `dim_items`, `item_cooccurrence`, `fact_interactions`,
`fact_transactions` tables with sample rows.

---

## 8. Feature Store — Task 7 (3:50 - 4:10)

**Say:** "A lightweight custom feature store with a versioned YAML
registry. It supports offline retrieval for training and online retrieval
by entity ID, both backed by the same warehouse tables."

**Show:** Console output listing registered feature views and the sample
training/online retrieval demo. Briefly show `docs\FEATURE_METADATA.md`.

---

## 9. Data Versioning & Lineage — Task 8 (4:10 - 4:30)

```powershell
git log --oneline
dvc list . --dvc-only
```

**Say:** "Code lives in Git; data and model artifacts are versioned via
DVC with `.dvc` pointer files committed to Git. Every commit reproduces
the exact dataset and model snapshot. MLflow run metadata is also tracked
in Git."

**Show:** Git commit history, and the `.dvc` pointer files (`data\raw.dvc`,
`warehouse\recomart.db.dvc`, `models\svd_model.npz.dvc`) — briefly open one
to show the MD5 hash. Reference `docs\VERSIONING.md`.

---

## 10. Model Training & Evaluation — Task 9 (4:30 - 5:30)

**Say:** "Two models: SVD collaborative filtering on a weighted interaction
matrix, and content-based filtering with cosine similarity. Both evaluated
with Precision@10, Recall@10, NDCG@10 and logged to MLflow."

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

**Say:** "The inference CLI takes a user ID and returns top-N recommendations
with scores — works for both model types."

---

## 11. Pipeline Orchestration Summary — Task 10 (5:30 - 5:50)

**Say:** "The whole pipeline is orchestrated as a single Prefect flow with
retries and dependency ordering: ingest → validate → prepare → transform
→ feature store → train. Fully schedulable via a Prefect deployment.
Each stage is also independently runnable for debugging or targeted
re-runs."

**Show:** Open `src\orchestration\pipeline_flow.py` and point out the
`@flow`, `@task` decorators and the `retries=2, retry_delay_seconds=5`
config. Reference the final `Pipeline finished. Model result: ...` line
in the terminal output.

---

## 12. Wrap-up (5:50 - 6:20)

**Say:** "That's the full pipeline — ingestion through model training,
versioned with Git and DVC, orchestrated with Prefect. Every stage is
modular, logged, and retry-safe."

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

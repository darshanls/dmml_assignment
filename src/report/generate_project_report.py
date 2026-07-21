"""
Generates the final submission PDF report (reports/RecoMart_Project_Report.pdf)
combining problem statement, objectives, methodology, implementation details,
results/screenshots, and conclusion, per the assignment's submission format.

Run this LAST, after the full pipeline (src/orchestration/pipeline_flow.py)
has executed at least once, so the data-quality report, EDA plots, and model
metrics referenced here actually exist.

Usage:
    python src/report/generate_project_report.py
"""
import glob
import json
import os
import re
import sqlite3
import sys

from fpdf import FPDF

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from common import get_logger, PROJECT_ROOT  # noqa: E402

logger = get_logger("report")

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
PLOTS_DIR = os.path.join(REPORTS_DIR, "eda_plots")
DB_PATH = os.path.join(PROJECT_ROOT, "warehouse", "recomart.db")

# ---- Fill these in before final submission ----
TEAM_MEMBERS = [
    ("Name", "Roll No / ID"),
    ("DARSHAN LS", "2025AA05828"),
    ("SARDESAI SHRISHTI SANJAY", "2025AA05830"),
    ("SUNIL GUPTA", "2025AA05833"),
    ("ANEESH MURALI", "2025AA05834"),
    ("MARISETTI SANTHOSH", "2025AA05836")
]
GDRIVE_VIDEO_LINK = "<PASTE GOOGLE DRIVE VIDEO WALKTHROUGH LINK HERE>"
GDRIVE_ZIP_LINK = "<PASTE GOOGLE DRIVE PROJECT ZIP LINK HERE>"
# -------------------------------------------------


class ReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 8, "RecoMart - End-to-End Data Management Pipeline",
                  align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C", new_x="LMARGIN", new_y="NEXT")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(20, 20, 90)
        self.ln(4)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.set_draw_color(20, 20, 90)
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(3)

    def body_text(self, text, size=10):
        self.set_font("Helvetica", "", size)
        self.multi_cell(0, 5.5, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def bullet_list(self, items, size=10):
        self.set_font("Helvetica", "", size)
        for item in items:
            self.multi_cell(0, 5.5, f"  -  {item}", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)


def _fetch_model_metrics():
    """
    Pull the latest MLflow run metrics from the mlruns file store, if present.
    Returns a dict keyed by run name (e.g. 'svd_collaborative_filtering',
    'content_based_filtering') -> {metric_name: value}, so metrics from
    different runs that share the same metric name (precision_at_k, etc.)
    do not overwrite each other.
    """
    mlruns_dir = os.path.join(PROJECT_ROOT, "mlruns")
    run_dirs = glob.glob(os.path.join(mlruns_dir, "*", "*"))
    # Process oldest -> newest so that, per run name, the most recent run's
    # metrics are what ends up in the returned dict (later writes win).
    run_dirs.sort(key=lambda d: os.path.getmtime(d))
    metrics_by_run = {}
    for run_dir in run_dirs:
        metrics_dir = os.path.join(run_dir, "metrics")
        if not os.path.isdir(metrics_dir):
            continue

        run_name_path = os.path.join(run_dir, "tags", "mlflow.runName")
        if os.path.exists(run_name_path):
            with open(run_name_path, "r", encoding="utf-8") as fh:
                run_name = fh.read().strip()
        else:
            run_name = os.path.basename(run_dir)

        run_metrics = {}
        for f in glob.glob(os.path.join(metrics_dir, "*")):
            name = os.path.basename(f)
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    last_line = fh.readlines()[-1]
                value = float(last_line.strip().split(" ")[1])
                run_metrics[name] = value
            except Exception:
                continue
        if run_metrics:
            metrics_by_run[run_name] = run_metrics
    return metrics_by_run


def _fetch_warehouse_stats():
    if not os.path.exists(DB_PATH):
        return {}
    conn = sqlite3.connect(DB_PATH)
    try:
        stats = {}
        for table in ["dim_users", "dim_items", "fact_interactions", "fact_transactions", "item_cooccurrence"]:
            try:
                stats[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except Exception:
                stats[table] = "N/A"
        return stats
    finally:
        conn.close()


def build_report():
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ---- Title Page ----
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 22)
    pdf.multi_cell(0, 12, "End-to-End Data Management Pipeline\nfor a Recommendation System", align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, "RecoMart Data Platform Team", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Team Members", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    for row in TEAM_MEMBERS:
        pdf.cell(0, 7, "   " + "  |  ".join(row), align="C", new_x="LMARGIN", new_y="NEXT")

    # ---- 1. Problem Statement ----
    pdf.add_page()
    pdf.section_title("1. Problem Statement")
    pdf.body_text(
        "RecoMart, an e-commerce startup, wants to build a data-driven recommendation "
        "engine to improve customer engagement and cross-selling conversion. User "
        "behavior is captured across multiple heterogeneous sources -- web/mobile "
        "clickstream logs, transactional purchase history, product catalog metadata, "
        "and external sentiment/popularity signals -- and must be continuously "
        "ingested, validated, curated into features, and used to train/update a "
        "personalized recommendation model."
    )

    # ---- 2. Objectives ----
    pdf.section_title("2. Objectives")
    pdf.bullet_list([
        "Ingest at least two heterogeneous data types (CSV-based interaction logs, REST API product catalog) on an automated, retryable schedule.",
        "Store raw data in a partitioned local data lake (source / type / date).",
        "Profile and validate data quality (missing values, duplicates, schema, range checks) with an automated report.",
        "Clean, encode, and normalize data; conduct EDA on interaction distributions, item popularity, and sparsity.",
        "Engineer recommendation-ready features (activity frequency, average ratings, item co-occurrence) into a structured warehouse.",
        "Stand up a lightweight, versioned feature store serving both training and inference.",
        "Version raw/processed data and track transformation lineage.",
        "Train and evaluate both collaborative-filtering (SVD) and content-based (cosine similarity) recommendation models (Precision@K, Recall@K, NDCG@K), tracked via MLflow.",
        "Orchestrate the full pipeline end-to-end with automated retries and logging.",
    ])

    # ---- 3. Methodology / Pipeline ----
    pdf.section_title("3. Methodology / Pipeline")
    pdf.body_text(
        "The pipeline is implemented as 6 modular stages, each independently runnable "
        "and orchestrated end-to-end with Prefect:\n\n"
        "1) Ingestion  ->  2) Raw Storage  ->  3) Validation  ->  4) Preparation (Cleaning + EDA)\n"
        "->  5) Transformation (Feature Engineering into SQLite warehouse)\n"
        "->  6) Feature Store retrieval  ->  7) Model Training & Evaluation (MLflow)."
    )
    pdf.body_text(
        "Data sources: (a) clickstream events and (b) transaction/purchase records are "
        "simulated per run to mimic a live Kafka/OLTP source system; (c) product catalog "
        "is fetched live from the FakeStore REST API (https://fakestoreapi.com/products); "
        "(d) product sentiment/popularity scores are retrieved via a retry-capable "
        "scoring-service client (simulated locally, same integration pattern as a real API)."
    )

    # ---- 4. Implementation Details ----
    pdf.section_title("4. Implementation Details")
    pdf.bullet_list([
        "Ingestion: src/ingestion/*.py - tenacity-based exponential-backoff retries, rotating file logs (logs/ingestion.log), partitioned raw storage.",
        "Validation: src/validation/validate_data.py - pandas-based schema/null/duplicate/range checks -> JSON + PDF quality report.",
        "Preparation: src/preparation/clean_and_prepare.py - dedup, missing-value handling, LabelEncoder/MinMaxScaler, EDA plots.",
        "Transformation: src/transformation/feature_engineering.py + schema.sql - user/item/co-occurrence features loaded into SQLite warehouse (warehouse/recomart.db).",
        "Feature Store: src/feature_store/feature_store.py + feature_registry.yaml - versioned, declarative feature metadata with offline/online retrieval.",
        "Modeling: src/models/train_model.py - (1) Truncated SVD collaborative filtering over a weighted implicit+explicit interaction matrix, (2) content-based filtering via item-feature cosine similarity; src/models/inference.py - top-N recommendation interface supporting both model types.",
        "Orchestration: src/orchestration/pipeline_flow.py - Prefect @flow/@task DAG with retries and structured logging.",
        "Versioning: docs/VERSIONING.md - Git + DVC workflow for data/model lineage.",
    ])

    # ---- 5. Results and Output Screenshots ----
    pdf.add_page()
    pdf.section_title("5. Results and Output")

    warehouse_stats = _fetch_warehouse_stats()
    if warehouse_stats:
        pdf.body_text("Warehouse table row counts (latest pipeline run):")
        pdf.bullet_list([f"{table}: {count}" for table, count in warehouse_stats.items()])

    metrics_by_run = _fetch_model_metrics()
    if metrics_by_run:
        pdf.body_text("Model evaluation metrics (MLflow-tracked, latest run per model):")
        for run_name, run_metrics in metrics_by_run.items():
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, run_name, new_x="LMARGIN", new_y="NEXT")
            pdf.bullet_list([f"{name}: {value:.4f}" for name, value in run_metrics.items()])

    quality_json = os.path.join(REPORTS_DIR, "data_quality_report.json")
    if os.path.exists(quality_json):
        with open(quality_json, "r", encoding="utf-8") as f:
            report = json.load(f)
        pdf.body_text("Data quality validation summary:")
        pdf.bullet_list([
            f"{ds['dataset']}: status={ds['status']}, rows={ds.get('row_count', 'N/A')}, issues={len(ds.get('issues', []))}"
            for ds in report.get("datasets", [])
        ])

    pdf.body_text("EDA Summary Plots (all plots generated by clean_and_prepare.py):")
    plot_files = sorted(glob.glob(os.path.join(PLOTS_DIR, "*.png")))
    for fpath in plot_files:
        fname = os.path.basename(fpath)
        title = re.sub(r"[-_]+", " ", fname.replace(".png", "")).title()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        try:
            pdf.image(fpath, w=170)
        except Exception as exc:
            logger.warning("Could not embed plot %s: %s", fname, exc)

    pdf.body_text(
        "Other generated artifacts available in the project root: "
        "reports/Data_Quality_Report.pdf (validation results), "
        "warehouse/recomart.db (SQLite feature warehouse), "
        "data/features/*.csv (engineered features), and mlruns/ "
        "(MLflow experiment tracking)."
    )

    # ---- 6. Conclusion and Future Scope ----
    pdf.add_page()
    pdf.section_title("6. Conclusion and Future Scope")
    pdf.body_text(
        "This POC demonstrates a fully modular, end-to-end data management pipeline "
        "covering ingestion, validation, preparation, feature engineering, a versioned "
        "feature store, model training/evaluation, and orchestration -- aligned with "
        "modern data-stack and MLOps practices. Both the SVD collaborative-filtering "
        "model and the content-based cosine-similarity model achieve measurable "
        "Precision@10 / Recall@10 / NDCG@10 on held-out interactions, and the pipeline "
        "is fully re-runnable end-to-end via a single Prefect flow."
    )
    pdf.body_text("Future scope:")
    pdf.bullet_list([
        "Replace simulated clickstream/transaction sources with a real Kafka topic / OLTP CDC feed.",
        "Extend content-based filtering with product text embeddings (title/description) and hybrid blending with the collaborative model.",
        "Migrate raw storage to a real cloud data lake (S3/ADLS) with lifecycle policies.",
        "Adopt Feast for a production-grade online feature store with a low-latency serving layer.",
        "Schedule the Prefect flow on a deployment/work-pool for true periodic automation with alerting on failures.",
        "A/B test the recommendation model in a live RecoMart storefront and track online CTR/conversion uplift.",
    ])

    # ---- Submission Links ----
    pdf.section_title("7. Submission Links")
    pdf.body_text(f"Video walkthrough (Google Drive): {GDRIVE_VIDEO_LINK}")
    pdf.body_text(f"Project deliverables .zip (Google Drive): {GDRIVE_ZIP_LINK}")

    out_path = os.path.join(REPORTS_DIR, "RecoMart_Project_Report.pdf")
    pdf.output(out_path)
    logger.info("Project report PDF written to %s", out_path)
    return out_path


if __name__ == "__main__":
    build_report()

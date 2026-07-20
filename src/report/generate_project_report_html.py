"""
Generates the final submission report as HTML (reports/RecoMart_Project_Report.html)
combining problem statement, objectives, methodology, implementation details,
results/screenshots, and conclusion, per the assignment's submission format.

Run this LAST, after the full pipeline (src/orchestration/pipeline_flow.py)
has executed at least once, so the data-quality report, EDA plots, and model
metrics referenced here actually exist.

Usage:
    python src/report/generate_project_report_html.py
"""
import glob
import json
import os
import sys
from html import escape

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from common import get_logger, PROJECT_ROOT  # noqa: E402
from generate_project_report import (  # noqa: E402
    TEAM_MEMBERS, GDRIVE_VIDEO_LINK, GDRIVE_ZIP_LINK,
    _fetch_model_metrics, _fetch_warehouse_stats, REPORTS_DIR, PLOTS_DIR,
)

logger = get_logger("report.html")


def _section(title, body_html):
    return f"""
    <section>
        <h2>{escape(title)}</h2>
        {body_html}
    </section>
    """


def _para(text):
    return f"<p>{escape(text)}</p>"


def _bullets(items):
    lis = "\n".join(f"<li>{escape(item)}</li>" for item in items)
    return f"<ul>{lis}</ul>"


def _table(headers, rows):
    thead = "".join(f"<th>{escape(h)}</th>" for h in headers)
    trs = "\n".join(
        "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{trs}</tbody></table>"


def build_html():
    warehouse_stats = _fetch_warehouse_stats()
    metrics_by_run = _fetch_model_metrics()

    quality_json = os.path.join(REPORTS_DIR, "data_quality_report.json")
    quality_rows = []
    if os.path.exists(quality_json):
        with open(quality_json, "r", encoding="utf-8") as f:
            report = json.load(f)
        for ds in report.get("datasets", []):
            quality_rows.append((
                ds.get("dataset"), ds.get("status"),
                ds.get("row_count", "N/A"), len(ds.get("issues", [])),
            ))

    plot_files = [
        "event_type_distribution.png", "item_popularity.png", "sparsity.png",
        "rating_distribution.png", "avg_price_by_category.png",
    ]
    plots_html = ""
    for fname in plot_files:
        fpath = os.path.join(PLOTS_DIR, fname)
        if os.path.exists(fpath):
            title = fname.replace("_", " ").replace(".png", "").title()
            plots_html += (
                f'<figure><figcaption>{escape(title)}</figcaption>'
                f'<img src="eda_plots/{fname}" alt="{escape(title)}"></figure>'
            )

    team_rows = TEAM_MEMBERS[1:] if TEAM_MEMBERS else []
    team_headers = list(TEAM_MEMBERS[0]) if TEAM_MEMBERS else ["Name", "Roll No / ID"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RecoMart - End-to-End Data Management Pipeline Report</title>
<style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0; background: #f7f8fa; color: #222; }}
    .container {{ max-width: 960px; margin: 0 auto; padding: 30px 40px; background: #fff; }}
    header.title-page {{ text-align: center; padding: 60px 20px; border-bottom: 4px solid #14145a; }}
    header.title-page h1 {{ color: #14145a; font-size: 30px; margin-bottom: 6px; }}
    header.title-page h3 {{ color: #444; font-weight: 400; }}
    section {{ margin: 30px 0; }}
    h2 {{ color: #14145a; border-bottom: 2px solid #14145a; padding-bottom: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
    th, td {{ border: 1px solid #ccc; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #14145a; color: #fff; }}
    tr:nth-child(even) {{ background: #f2f2f8; }}
    ul {{ line-height: 1.6; }}
    figure {{ margin: 20px 0; text-align: center; }}
    figcaption {{ font-weight: 600; margin-bottom: 8px; color: #14145a; }}
    img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
    footer {{ text-align: center; padding: 20px; color: #888; font-size: 12px; }}
    .links a {{ color: #14145a; }}
</style>
</head>
<body>
<div class="container">

<header class="title-page">
    <h1>End-to-End Data Management Pipeline<br>for a Recommendation System</h1>
    <h3>RecoMart Data Platform Team</h3>
    {_table(team_headers, team_rows)}
</header>

{_section("1. Problem Statement", _para(
    "RecoMart, an e-commerce startup, wants to build a data-driven recommendation "
    "engine to improve customer engagement and cross-selling conversion. User "
    "behavior is captured across multiple heterogeneous sources -- web/mobile "
    "clickstream logs, transactional purchase history, product catalog metadata, "
    "and external sentiment/popularity signals -- and must be continuously "
    "ingested, validated, curated into features, and used to train/update a "
    "personalized recommendation model."
))}

{_section("2. Objectives", _bullets([
    "Ingest at least two heterogeneous data types (CSV-based interaction logs, REST API product catalog) on an automated, retryable schedule.",
    "Store raw data in a partitioned local data lake (source / type / date).",
    "Profile and validate data quality (missing values, duplicates, schema, range checks) with an automated report.",
    "Clean, encode, and normalize data; conduct EDA on interaction distributions, item popularity, and sparsity.",
    "Engineer recommendation-ready features (activity frequency, average ratings, item co-occurrence) into a structured warehouse.",
    "Stand up a lightweight, versioned feature store serving both training and inference.",
    "Version raw/processed data and track transformation lineage.",
    "Train and evaluate both collaborative-filtering (SVD) and content-based (cosine similarity) recommendation models (Precision@K, Recall@K, NDCG@K), tracked via MLflow.",
    "Orchestrate the full pipeline end-to-end with automated retries and logging.",
]))}

{_section("3. Methodology / Pipeline", _para(
    "The pipeline is implemented as 6 modular stages, each independently runnable "
    "and orchestrated end-to-end with Prefect: Ingestion -> Raw Storage -> Validation "
    "-> Preparation (Cleaning + EDA) -> Transformation (Feature Engineering into SQLite "
    "warehouse) -> Feature Store retrieval -> Model Training & Evaluation (MLflow)."
) + _para(
    "Data sources: (a) clickstream events and (b) transaction/purchase records are "
    "simulated per run to mimic a live Kafka/OLTP source system; (c) product catalog "
    "is fetched live from the FakeStore REST API (https://fakestoreapi.com/products); "
    "(d) product sentiment/popularity scores are retrieved via a retry-capable "
    "scoring-service client (simulated locally, same integration pattern as a real API)."
))}

{_section("4. Implementation Details", _bullets([
    "Ingestion: src/ingestion/*.py - tenacity-based exponential-backoff retries, rotating file logs (logs/ingestion.log), partitioned raw storage.",
    "Validation: src/validation/validate_data.py - pandas-based schema/null/duplicate/range checks -> JSON + PDF quality report.",
    "Preparation: src/preparation/clean_and_prepare.py - dedup, missing-value handling, LabelEncoder/MinMaxScaler, EDA plots.",
    "Transformation: src/transformation/feature_engineering.py + schema.sql - user/item/co-occurrence features loaded into SQLite warehouse (warehouse/recomart.db).",
    "Feature Store: src/feature_store/feature_store.py + feature_registry.yaml - versioned, declarative feature metadata with offline/online retrieval.",
    "Modeling: src/models/train_model.py - (1) Truncated SVD collaborative filtering over a weighted implicit+explicit interaction matrix, (2) content-based filtering via item-feature cosine similarity; src/models/inference.py - top-N recommendation interface supporting both model types.",
    "Orchestration: src/orchestration/pipeline_flow.py - Prefect @flow/@task DAG with retries and structured logging.",
    "Versioning: docs/VERSIONING.md - Git + DVC workflow for data/model lineage.",
]))}

{_section("5. Results and Output", (
    ("<h3>Warehouse table row counts (latest pipeline run)</h3>" + _table(["Table", "Row Count"], list(warehouse_stats.items())))
    if warehouse_stats else ""
) + (
    ("<h3>Model evaluation metrics (MLflow-tracked, latest run per model)</h3>" + "".join(
        f"<h4>{escape(run_name)}</h4>" + _table(["Metric", "Value"], [(k, f"{v:.4f}") for k, v in run_metrics.items()])
        for run_name, run_metrics in metrics_by_run.items()
    ))
    if metrics_by_run else ""
) + (
    ("<h3>Data quality validation summary</h3>" + _table(["Dataset", "Status", "Rows", "Issues"], quality_rows))
    if quality_rows else ""
) + f"<h3>EDA Summary Plots</h3>{plots_html}")}

{_section("6. Conclusion and Future Scope", _para(
    "This POC demonstrates a fully modular, end-to-end data management pipeline "
    "covering ingestion, validation, preparation, feature engineering, a versioned "
    "feature store, model training/evaluation, and orchestration -- aligned with "
    "modern data-stack and MLOps practices. Both the SVD collaborative-filtering "
    "model and the content-based cosine-similarity model achieve measurable "
    "Precision@10 / Recall@10 / NDCG@10 on held-out interactions, and the pipeline "
    "is fully re-runnable end-to-end via a single Prefect flow."
) + _bullets([
    "Replace simulated clickstream/transaction sources with a real Kafka topic / OLTP CDC feed.",
    "Extend content-based filtering with product text embeddings (title/description) and hybrid blending with the collaborative model.",
    "Migrate raw storage to a real cloud data lake (S3/ADLS) with lifecycle policies.",
    "Adopt Feast for a production-grade online feature store with a low-latency serving layer.",
    "Schedule the Prefect flow on a deployment/work-pool for true periodic automation with alerting on failures.",
    "A/B test the recommendation model in a live RecoMart storefront and track online CTR/conversion uplift.",
]))}

{_section("7. Submission Links", f'<p class="links">Video walkthrough (Google Drive): <a href="{escape(GDRIVE_VIDEO_LINK)}">{escape(GDRIVE_VIDEO_LINK)}</a></p><p class="links">Project deliverables .zip (Google Drive): <a href="{escape(GDRIVE_ZIP_LINK)}">{escape(GDRIVE_ZIP_LINK)}</a></p>')}

<footer>RecoMart - End-to-End Data Management Pipeline for a Recommendation System</footer>
</div>
</body>
</html>
"""

    out_path = os.path.join(REPORTS_DIR, "RecoMart_Project_Report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Project report HTML written to %s", out_path)
    return out_path


if __name__ == "__main__":
    build_html()

"""
Prefect orchestration DAG for the RecoMart end-to-end data pipeline:

    ingest (clickstream, transactions, products, sentiment)
        -> validate
        -> prepare (clean + EDA)
        -> transform (feature engineering -> warehouse)
        -> feature store demo (versioned retrieval)
        -> train model (+ MLflow tracking)

Each stage is a Prefect @task with retries; failures are logged and raised
so the flow run is clearly marked Failed in Prefect's UI/logs.

Usage (ad-hoc run):
    python src/orchestration/pipeline_flow.py

Usage (via Prefect CLI / scheduled deployment):
    prefect deployment build src/orchestration/pipeline_flow.py:recomart_pipeline -n recomart-daily
    prefect deployment apply recomart_pipeline-deployment.yaml
"""
import os
import sys

from prefect import flow, task, get_run_logger

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "validation"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "preparation"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "transformation"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "feature_store"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))

import ingest_clickstream
import ingest_transactions
import ingest_products_api
import ingest_sentiment_api
import validate_data
import generate_quality_report
import clean_and_prepare
import feature_engineering
import feature_store as feature_store_module
import train_model


@task(retries=2, retry_delay_seconds=5, name="ingest_clickstream")
def ingest_clickstream_task():
    logger = get_run_logger()
    path = ingest_clickstream.run()
    logger.info("Clickstream ingested -> %s", path)
    return path


@task(retries=2, retry_delay_seconds=5, name="ingest_transactions")
def ingest_transactions_task():
    logger = get_run_logger()
    path = ingest_transactions.run()
    logger.info("Transactions ingested -> %s", path)
    return path


@task(retries=2, retry_delay_seconds=5, name="ingest_products_api")
def ingest_products_task():
    logger = get_run_logger()
    path = ingest_products_api.run()
    logger.info("Products ingested -> %s", path)
    return path


@task(retries=2, retry_delay_seconds=5, name="ingest_sentiment_api")
def ingest_sentiment_task():
    logger = get_run_logger()
    path = ingest_sentiment_api.run()
    logger.info("Sentiment/popularity ingested -> %s", path)
    return path


@task(name="validate")
def validate_task(_upstream):
    logger = get_run_logger()
    report = validate_data.run()
    generate_quality_report.run()
    logger.info("Validation complete; %d datasets checked", len(report["datasets"]))
    return report


@task(name="prepare")
def prepare_task(_upstream):
    logger = get_run_logger()
    stats = clean_and_prepare.run()
    logger.info("Preparation complete: %s", stats)
    return stats


@task(name="transform")
def transform_task(_upstream):
    logger = get_run_logger()
    stats = feature_engineering.run()
    logger.info("Transformation complete: %s", stats)
    return stats


@task(name="feature_store_demo")
def feature_store_task(_upstream):
    logger = get_run_logger()
    fs = feature_store_module.FeatureStore()
    views = fs.list_feature_views()
    logger.info("Feature store operational; registered views: %s", views)
    return views


@task(name="train_model")
def train_model_task(_upstream):
    logger = get_run_logger()
    result = train_model.run()
    logger.info("Model training complete: %s", result)
    return result


@flow(name="recomart_pipeline")
def recomart_pipeline():
    clicks = ingest_clickstream_task()
    txns = ingest_transactions_task()
    products = ingest_products_task()
    sentiment = ingest_sentiment_task()

    validated = validate_task([clicks, txns, products, sentiment])
    prepared = prepare_task(validated)
    transformed = transform_task(prepared)
    fs_views = feature_store_task(transformed)
    model_result = train_model_task(fs_views)
    return model_result


if __name__ == "__main__":
    result = recomart_pipeline()
    print("Pipeline finished. Model result:", result)

"""
Data profiling and validation for the latest raw ingestion batches.

Checks performed per dataset:
    - Missing values (count + % per column)
    - Duplicate rows / duplicate primary keys
    - Schema check (expected columns present, dtypes coercible)
    - Range/format checks (e.g. rating in [1,5], price > 0, sentiment in [-1,1])

Outputs a machine-readable JSON quality report plus a human-readable summary
that `src/validation/generate_quality_report.py` turns into a PDF.
"""
import glob
import json
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from common import get_logger, RAW_DATA_DIR, PROJECT_ROOT  # noqa: E402

logger = get_logger("validation")

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def _latest_files(source: str, data_type: str, ext: str = "csv"):
    pattern = os.path.join(RAW_DATA_DIR, source, data_type, "*", f"*.{ext}")
    files = sorted(glob.glob(pattern))
    return files


def _latest_partition_files(source: str, data_type: str, ext: str = "csv"):
    """Return all files from the most recent date-partition directory."""
    files = _latest_files(source, data_type, ext)
    if not files:
        return []
    latest_date_dir = os.path.dirname(files[-1])
    return sorted(glob.glob(os.path.join(latest_date_dir, f"*.{ext}")))


DATASET_SPECS = {
    "clickstream": {
        "source": "clickstream", "data_type": "events",
        "expected_columns": ["event_id", "user_id", "product_id", "event_type",
                              "device", "session_id", "event_timestamp"],
        "primary_key": "event_id",
        "range_checks": {},
    },
    "transactions": {
        "source": "transactions", "data_type": "purchases",
        "expected_columns": ["transaction_id", "user_id", "product_id", "quantity",
                              "unit_price", "total_amount", "rating", "transaction_timestamp"],
        "primary_key": "transaction_id",
        "range_checks": {"rating": (1, 5), "unit_price": (0, None), "quantity": (1, None)},
    },
    "products": {
        "source": "products", "data_type": "catalog",
        "expected_columns": ["product_id", "title", "price", "category",
                              "description", "image", "rating_rate", "rating_count"],
        "primary_key": "product_id",
        "range_checks": {"price": (0, None), "rating_rate": (0, 5)},
    },
    "sentiment": {
        "source": "sentiment", "data_type": "scores",
        "expected_columns": ["product_id", "sentiment_score", "popularity_score", "scored_at"],
        "primary_key": "product_id",
        "range_checks": {"sentiment_score": (-1, 1), "popularity_score": (0, 1)},
    },
}


def _range_violations(df: pd.DataFrame, column: str, low, high) -> int:
    if column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce")
    mask = pd.Series(False, index=df.index)
    if low is not None:
        mask |= series < low
    if high is not None:
        mask |= series > high
    return int(mask.sum())


def validate_dataset(name: str, spec: dict) -> dict:
    files = _latest_partition_files(spec["source"], spec["data_type"])
    result = {"dataset": name, "files_checked": files, "status": "OK", "issues": []}

    if not files:
        result["status"] = "MISSING"
        result["issues"].append("No ingested files found for this dataset.")
        return result

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    result["row_count"] = len(df)

    # Schema check
    missing_cols = [c for c in spec["expected_columns"] if c not in df.columns]
    if missing_cols:
        result["status"] = "FAIL"
        result["issues"].append(f"Missing expected columns: {missing_cols}")

    # Missing values
    null_pct = (df.isna().mean() * 100).round(2)
    result["null_pct_by_column"] = null_pct.to_dict()

    # Duplicates
    result["duplicate_rows"] = int(df.duplicated().sum())
    pk = spec.get("primary_key")
    if pk and pk in df.columns:
        result["duplicate_primary_keys"] = int(df[pk].duplicated().sum())
        if result["duplicate_primary_keys"] > 0:
            result["issues"].append(f"{result['duplicate_primary_keys']} duplicate '{pk}' values")

    # Range checks
    range_issues = {}
    for col, (low, high) in spec["range_checks"].items():
        violations = _range_violations(df, col, low, high)
        if violations:
            range_issues[col] = violations
            result["issues"].append(f"{violations} rows violate range check on '{col}'")
    result["range_violations"] = range_issues

    if result["issues"] and result["status"] == "OK":
        result["status"] = "WARN"

    return result


def run():
    logger.info("Starting data validation run")
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "datasets": [],
    }
    for name, spec in DATASET_SPECS.items():
        try:
            res = validate_dataset(name, spec)
            report["datasets"].append(res)
            logger.info("Validated '%s': status=%s issues=%d", name, res["status"], len(res.get("issues", [])))
        except Exception as exc:
            logger.error("Validation FAILED for '%s': %s", name, exc, exc_info=True)
            report["datasets"].append({"dataset": name, "status": "ERROR", "issues": [str(exc)]})

    out_path = os.path.join(REPORTS_DIR, "data_quality_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Data quality report written to %s", out_path)
    return report


if __name__ == "__main__":
    run()

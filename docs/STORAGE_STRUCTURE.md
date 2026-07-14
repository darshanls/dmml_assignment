# Raw Data Storage Structure (Task 3)

RecoMart's raw data lake is a local filesystem directory (`data/raw/`) that
mirrors a cloud object-store layout (e.g. an S3 bucket), so migrating to
AWS S3 / Azure Blob later only requires swapping the base path/URI used by
`src/ingestion/common.py::partitioned_path`.

## Partition scheme

```
data/raw/<source>/<data_type>/<YYYY-MM-DD>/<filename>
```

- **source** — origin system: `clickstream`, `transactions`, `products`, `sentiment`
- **data_type** — logical dataset within that source: `events`, `purchases`, `catalog`, `scores`
- **YYYY-MM-DD** — ingestion run date partition (enables efficient date-range reads and lifecycle/retention policies)
- **filename** — timestamped file, e.g. `clickstream_20260714T123912.csv`, so multiple ingestion runs on the same day never collide

## Example tree (after one ingestion run per source)

```
data/raw/
├── clickstream/
│   └── events/
│       └── 2026-07-14/
│           └── clickstream_20260714T123912.csv
├── transactions/
│   └── purchases/
│       └── 2026-07-14/
│           └── transactions_20260714T123927.csv
├── products/
│   └── catalog/
│       └── 2026-07-14/
│           ├── products_raw_20260714T123932.json   # full raw API response (lineage/audit)
│           └── products_20260714T123932.csv         # flattened tabular extract
└── sentiment/
    └── scores/
        └── 2026-07-14/
            └── sentiment_20260714T123940.csv
```

## Rationale

- **Partition by source + type** so downstream jobs (validation, preparation)
  can selectively glob only the datasets they need:
  `data/raw/<source>/<data_type>/*/*.csv`.
- **Partition by date** so the pipeline supports both full-history
  backfills and incremental/daily processing.
- **Raw JSON retained alongside flattened CSV** for the REST API source
  (products) to preserve full lineage back to the exact API payload.

## Migrating to cloud storage (AWS S3 example)

Only `RAW_DATA_DIR` in `src/ingestion/common.py` needs to change, e.g.:

```python
RAW_DATA_DIR = "s3://recomart-data-lake/raw"
```

combined with `s3fs`/`boto3` writes in place of local `to_csv` calls. The
partition scheme (`source/data_type/date/`) stays identical, matching
common data-lake/Hive-style partitioning conventions.

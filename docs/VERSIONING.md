# Data Versioning & Lineage Workflow (Task 8)

RecoMart's pipeline separates **code** (versioned in Git) from **data/model
artifacts** (versioned in DVC), with DVC pointer files (`.dvc`) checked into
Git so every commit can reproduce the exact dataset/model used.

> NOTE: For this POC environment, Git/DVC initialization was intentionally
> left to be run by the developer (`git init`, `dvc init`) rather than
> executed automatically, to avoid mutating global user/repo state without
> explicit consent. The steps below are the exact workflow to run.

## One-time setup

```powershell
git init
git add .gitignore requirements.txt src/ docs/ notebooks/ reports/
git commit -m "Initial commit: RecoMart pipeline source code"

dvc init
dvc remote add -d local_remote ../dvc_storage   # or an S3/Azure/GCS remote
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

## Versioning raw and processed data

Every ingestion/preparation/transformation run produces new files under
`data/raw/`, `data/processed/`, `data/features/`, and `warehouse/`. These are
excluded from Git (see `.gitignore`) and tracked by DVC instead:

```powershell
dvc add data/raw
dvc add data/processed
dvc add data/features
dvc add warehouse/recomart.db
dvc add models/svd_model.npz

git add data/raw.dvc data/processed.dvc data/features.dvc warehouse/recomart.db.dvc models/svd_model.npz.dvc
git commit -m "Data snapshot: <date> - <n> clickstream events, <n> transactions"
dvc push
```

Each `.dvc` pointer file stores an MD5 hash of the tracked data, so
`git log -p data/raw.dvc` gives a full lineage of every dataset version
alongside the exact commit/date/author that produced it.

## Metadata tracked per version

| Metadata field        | Where it's captured                                             |
|------------------------|-------------------------------------------------------------------|
| Data source            | Partitioned folder path: `data/raw/<source>/<type>/<date>/`      |
| Ingestion date          | Date partition + filename timestamp                              |
| Applied transformations | `docs/FEATURE_METADATA.md` + `src/feature_store/feature_registry.yaml` |
| Git commit / DVC hash    | `.dvc` pointer files + `git log`                                 |
| Model run metadata       | MLflow run ID, params, metrics (`mlruns/`)                       |

## Alternative: Git LFS

If DVC is unavailable, the same raw/processed CSV/JSON/SQLite files can be
tracked with Git LFS instead:

```powershell
git lfs install
git lfs track "data/**/*.csv" "data/**/*.json" "warehouse/*.db" "models/*.npz"
git add .gitattributes
git commit -m "Track data/model artifacts with Git LFS"
```

## Reproducing a historical run

```powershell
git checkout <commit_sha>
dvc checkout
python src/orchestration/pipeline_flow.py
```

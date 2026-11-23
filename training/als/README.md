# ALS Training Container — Agent Platform Custom Job

Trains **ALS (retrieval)** + **LightGBM LambdaMART ranker (ranking)**, evaluates, writes artifacts.

## Layout

```
training/als/
├── Dockerfile
├── run_job.py
├── job.env.example
└── README.md
```

## Build

From repo root:

```bash
docker build -f training/als/Dockerfile -t recsys-als-training:latest .
```

Push:

```bash
docker tag recsys-als-training:latest your-registry.io/recsys-als-training:v1
docker push your-registry.io/recsys-als-training:v1
```

## Run locally

```bash
mkdir -p artifacts/job_output

docker run --rm \
  -e DATA_DIR=/data \
  -e OUTPUT_DIR=/output \
  -v "$(pwd)/data:/data:ro" \
  -v "$(pwd)/artifacts/job_output:/output" \
  recsys-als-training:latest
```

Skip evaluation:

```bash
docker run --rm \
  -v "$(pwd)/data:/data:ro" \
  -v "$(pwd)/artifacts/job_output:/output" \
  recsys-als-training:latest \
  --skip-eval
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSACTIONS_PATH` | — | Full path to training CSV (local or `gs://...`) |
| `OUTPUT_URI` | — | Output root for models, mappings, metrics (local or `gs://...`) |
| `DATA_DIR` | `/data` | Legacy: `{DATA_DIR}/raw/transactions_train.csv` |
| `OUTPUT_DIR` | `/output` | Legacy: artifacts under this directory |
| `CONFIG_PATH` | `/app/configs/als.yaml` | Training config (override via `--config`) |

For Agent Platform, set `TRANSACTIONS_PATH` and `OUTPUT_URI` to `gs://` URIs. See [`job.env.example`](job.env.example) and [`custom_job.yaml`](custom_job.yaml).

During evaluation, the job logs progress every 30 minutes (configurable) and after each user chunk. Tune in `configs/als.yaml`:

```yaml
evaluation:
  progress_log_interval_sec: 1800  # heartbeat every 30 min
  user_chunk_size: 5000
```

## Output

```
/output/
├── als/als_model.pkl
├── als/ranker/ranker.pkl
├── mappings/
├── config.yaml
└── metrics.json
```

## Agent Platform Custom Job

Submit with GCS paths (no `/gcs/` fuse mount required):

```bash
gcloud config configurations activate recsys

gcloud ai custom-jobs create \
  --region=asia-south1 \
  --display-name=recsys-als-training \
  --config=training/als/custom_job.yaml
```

Or set env vars in the console / YAML:

```yaml
env:
  - name: TRANSACTIONS_PATH
    value: gs://rec-sys-507218-data/data/raw/transactions_train.csv
  - name: OUTPUT_URI
    value: gs://rec-sys-507218-data/artifacts/job_output
```

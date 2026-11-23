# Dataproc Serverless for Spark — distributed ALS

Spark-based distributed ALS training and sharded evaluation via **Dataproc Serverless for Spark** (no cluster to create or delete).

Unlike the Vertex / Cloud Run containers (single-node `implicit`), this uses **PySpark MLlib ALS** with eval sharded across Spark executors.

## Layout

```
training/distributed/dataproc/
├── als_spark_job.py       # PySpark entrypoint (train + distributed eval)
├── spark_common.py        # Split, mappings, metrics helpers
├── submit_job.sh          # Stage code to GCS + submit Serverless batch
├── job.env.example
└── README.md
```

## Quick start

```bash
# 1. Enable API (once)
gcloud services enable dataproc.googleapis.com

# 2. Configure paths
cp training/distributed/dataproc/job.env.example training/distributed/dataproc/job.env

# 3. Submit train + eval
chmod +x training/distributed/dataproc/submit_job.sh
./training/distributed/dataproc/submit_job.sh

# Train only (skip eval)
./training/distributed/dataproc/submit_job.sh --skip-eval

# 4. Monitor
gcloud dataproc batches describe recsys-als-batch --region=asia-south1
```

Upload local config if not on GCS:

```bash
gsutil cp configs/als.yaml gs://rec-sys-507218-data/configs/als.yaml
```

Or set in `job.env`: `CONFIG_PATH=configs/als.yaml` (uploaded automatically).

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATAPROC_BATCH` | `recsys-als-batch` | Serverless batch name |
| `EVAL_PARTITIONS` | `15` | Eval user shards across executors |
| `MAX_EVAL_USERS` | `0` | Cap eval users (`0` = all) |
| `SPARK_PROPERTIES` | see `submit_job.sh` | JVM stack + shuffle partitions (avoids `StackOverflowError` on large ALS fits) |
| `OUTPUT_URI` | GCS path | Model + mappings + metrics |

## Output (GCS)

```
${OUTPUT_URI}/
├── als/spark_als_model/     # Spark MLlib ALS model
├── mappings/
│   ├── user_mapping.parquet
│   └── item_mapping.parquet
└── metrics_spark_als.json   # Distributed eval metrics
```

## vs single-node `implicit` ALS

| | Vertex / Cloud Run | Dataproc Serverless |
|---|---|---|
| Product | Vertex AI Custom Training | Dataproc Serverless for Spark |
| Engine | `implicit` | Spark MLlib ALS |
| Training | Single node | Distributed (auto-scaled) |
| Model format | `als_model.pkl` | `spark_als_model/` |

The LambdaMART ranker is **not** trained here. Use the Vertex ALS container for ranker training if needed.

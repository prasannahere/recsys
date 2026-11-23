# Hybrid Training Container — Agent Platform Custom Job

Trains the **Hybrid neural retriever**, evaluates against **Popularity** and **ALS** baselines under the same temporal split, and writes artifacts.

## Layout

```
training/hybrid/
├── Dockerfile
├── run_job.py
├── custom_job.yaml
├── job.env.example
└── README.md
```

## Build

From repo root:

```bash
docker build -f training/hybrid/Dockerfile -t recsys-hybrid-training:latest .
```

Push:

```bash
docker tag recsys-hybrid-training:latest your-registry.io/recsys-hybrid-training:v1
docker push your-registry.io/recsys-hybrid-training:v1
```

## Run locally

Mount both `transactions_train.csv` and `articles.csv` under `data/raw/`:

```bash
mkdir -p artifacts/job_output

docker run --rm \
  -e DATA_DIR=/data \
  -e OUTPUT_DIR=/output \
  -v "$(pwd)/data:/data:ro" \
  -v "$(pwd)/artifacts/job_output:/output" \
  recsys-hybrid-training:latest
```

Skip evaluation (train Hybrid only):

```bash
docker run --rm \
  -v "$(pwd)/data:/data:ro" \
  -v "$(pwd)/artifacts/job_output:/output" \
  recsys-hybrid-training:latest \
  --skip-eval
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSACTIONS_PATH` | — | Full path to training CSV (local or `gs://...`) |
| `ARTICLES_PATH` | — | Full path to articles CSV (local or `gs://...`) |
| `OUTPUT_URI` | — | Output root for models, mappings, metrics (local or `gs://...`) |
| `DATA_DIR` | `/data` | Legacy: `{DATA_DIR}/raw/transactions_train.csv` and `articles.csv` |
| `OUTPUT_DIR` | `/output` | Legacy: artifacts under this directory |
| `CONFIG_PATH` | `/app/configs/hybrid.yaml` | Hybrid training config |
| `ALS_CONFIG_PATH` | `/app/configs/als.yaml` | ALS config used during comparison eval |

For Agent Platform, set `TRANSACTIONS_PATH`, `ARTICLES_PATH`, and `OUTPUT_URI` to `gs://` URIs. See [`job.env.example`](job.env.example) and [`custom_job.yaml`](custom_job.yaml).

## Output

```
/output/
├── hybrid/
│   ├── hybrid_model.pt
│   ├── categorical_encoders.pkl
│   ├── item_features.pkl
│   ├── history_items.npy
│   ├── history_lengths.npy
│   ├── item_index.faiss
│   └── item_index.item_indices.npy
├── mappings/
├── config.yaml
├── als_config.yaml
└── metrics_hybrid.json
```

## Agent Platform Custom Job

Submit with GCS paths:

```bash
gcloud config configurations activate recsys

gcloud ai custom-jobs create \
  --region=asia-south1 \
  --display-name=recsys-hybrid-training \
  --config=training/hybrid/custom_job.yaml
```

Or set env vars in the console / YAML:

```yaml
env:
  - name: TRANSACTIONS_PATH
    value: gs://rec-sys-507218-data/data/raw/transactions_train.csv
  - name: ARTICLES_PATH
    value: gs://rec-sys-507218-data/data/raw/articles.csv
  - name: OUTPUT_URI
    value: gs://rec-sys-507218-data/artifacts/job_output
```

The job uses `n2-highmem-32` by default (PyTorch + FAISS on CPU). For GPU training, add an accelerator and use a CUDA-enabled base image.

## CLI equivalent

The container mirrors the local pipeline:

```bash
python -m src.pipelines.train_hybrid --config configs/hybrid.yaml
python -m src.pipelines.evaluate_hybrid --config configs/hybrid.yaml [--load-model]
python -m src.pipelines.evaluate_hybrid --config configs/hybrid.yaml --with-ranker [--load-ranker]
python -m src.pipelines.train_ranker --retriever hybrid --config configs/hybrid.yaml [--load-retrieval]
python -m src.pipelines.compare_retrievers --retrievers popularity,als,hybrid
```

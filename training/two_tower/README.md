# Pure Two-Tower Training Container

Trains **ID-only two-tower** retrieval + **Two-Tower LambdaMART ranker**, evaluates retrieval vs reranked.

No `articles.csv` required — unlike the Hybrid retriever.

## Build & push

From repo root:

```bash
docker build -f training/two_tower/Dockerfile -t recsys-two-tower-training:latest .

docker tag recsys-two-tower-training:latest \
  asia-south1-docker.pkg.dev/rec-sys-507218/recsys/recsys-two-tower-training:v1
docker push asia-south1-docker.pkg.dev/rec-sys-507218/recsys/recsys-two-tower-training:v1
```

## Submit (Cloud Run GPU — recommended)

Uses **L4 GPU** in `asia-southeast1` (Cloud Run L4 in `asia-south1` is invitation-only).
Training-only by default (`--skip-eval`, 1h timeout).

```bash
gcloud config configurations activate recsys

# Build & push image (if changed)
./training/scripts/build_image.sh two_tower v1

# Deploy job + start execution
./training/scripts/submit_cloudrun_job.sh two_tower
```

Or step by step:

```bash
gcloud run jobs replace training/two_tower/cloudrun_job.yaml --region=asia-southeast1
gcloud run jobs execute recsys-two-tower-training --region=asia-southeast1 --wait
```

## Submit (Vertex AI Custom Job)

Requires **training** L4 quota in `asia-south1` (separate from serving quota).

```bash
gcloud ai custom-jobs create \
  --region=asia-south1 \
  --display-name=recsys-two-tower-training \
  --config=training/two_tower/custom_job.yaml
```

Edit `training/two_tower/custom_job.yaml` if you need different GCS paths or image tag:

```yaml
env:
  - name: TRANSACTIONS_PATH
    value: gs://rec-sys-507218-data/data/raw/transactions_train.csv
  - name: OUTPUT_URI
    value: gs://rec-sys-507218-data/artifacts/two_tower_job_output
  - name: CONFIG_PATH
    value: /app/configs/two_tower.yaml
```

Skip evaluation (train only):

```bash
# Not supported via custom_job.yaml env today — pass as container args if needed:
# containerSpec.args: ["--skip-eval"]
```

## Run locally

```bash
docker run --rm \
  -e DATA_DIR=/data \
  -e OUTPUT_DIR=/output \
  -v "$(pwd)/data:/data:ro" \
  -v "$(pwd)/artifacts/job_output:/output" \
  recsys-two-tower-training:latest
```

## Output

```
/output/two_tower/
├── two_tower_model.pt
├── item_index.faiss
├── item_index.item_indices.npy
└── ranker/ranker.pkl
/output/metrics_two_tower.json
```

## CLI

```bash
python -m src.pipelines.train_two_tower --config configs/two_tower.yaml
python -m src.pipelines.evaluate_two_tower --config configs/two_tower.yaml [--load-model]
python -m src.pipelines.evaluate_two_tower --config configs/two_tower.yaml --with-ranker [--load-ranker]
python -m src.pipelines.train_ranker --retriever two_tower --config configs/two_tower.yaml [--load-retrieval]
python -m src.pipelines.compare_retrievers --retrievers popularity,als,two_tower
```

## Hybrid vs Two-Tower

| | Two-Tower | Hybrid |
|---|-----------|--------|
| User signal | customer ID only | ID + purchase history |
| Item signal | article ID only | ID + article metadata |
| Data deps | transactions only | transactions + articles.csv |

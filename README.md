# H&M Recommendation System — ALS Retrieval Baseline

Implicit-feedback **Alternating Least Squares (ALS)** candidate retrieval for the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) dataset.

This repo implements **candidate retrieval** for H&M with four retrievers:

- **Popularity** — global purchase counts
- **ALS** — implicit matrix factorization
- **Two-Tower** — pure ID-only dual encoder (collaborative filtering via embeddings)
- **Hybrid** — dual encoder with purchase history + article metadata

## What ALS Does

ALS learns latent user and article embeddings from collaborative interaction patterns (who bought what). No article or customer metadata is used.

Prediction score:

```
score(u, i) = user_embedding[u] · item_embedding[i]
```

The model is trained on a sparse user × item interaction matrix with implicit-feedback confidence weighting:

```
C_ui = 1 + alpha * r_ui
```

where `r_ui` is the aggregated interaction weight (binary, count, or recency-weighted).

## Role in Architecture

```
All training articles
    → ALS retrieval (Top-K)
    → LambdaMART ranker (per retriever: als / two_tower / hybrid)
    → final recommendations
```

ALS narrows the item catalog from ~100K articles to K candidates (default K=200) per user.

## ALS Limitations

- **User cold start:** new users with no training history cannot receive personalized ALS candidates.
- **Item cold start:** new articles absent from training interactions cannot be retrieved.
- **No metadata:** product type, color, department, etc. are ignored.
- **Opaque factors:** latent dimensions are not directly interpretable.

## Project Structure

```
recsys/
├── configs/als.yaml          # All hyperparameters
├── data/raw/                 # Source CSVs
├── data/mappings/            # user/item ID mappings (generated)
├── artifacts/als/            # Saved model (generated)
├── src/
│   ├── data/                 # load, temporal split, mappings
│   ├── features/             # interaction weighting, CSR matrix
│   ├── retrieval/            # ALS + popularity baseline
│   ├── evaluation/           # Recall, MAP, NDCG metrics
│   └── pipelines/            # train and evaluate CLIs
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

### macOS OpenMP Note

The `implicit` library uses OpenMP for fast ALS training. On macOS you may need:

```bash
brew install libomp
export OPENBLAS_NUM_THREADS=1
```

If installation fails, try:

```bash
pip install implicit --no-binary :all:
```

## Training container (Custom Job)

```bash
docker build -f training/als/Dockerfile -t recsys-als-training:latest .

docker run --rm \
  -v "$(pwd)/data:/data:ro" \
  -v "$(pwd)/artifacts/job_output:/output" \
  recsys-als-training:latest
```

See [training/als/README.md](training/als/README.md) for push, env vars, and job spec.

## Usage

### Notebook walkthrough

Both notebooks train **two models** and evaluate three systems:

| Model | Role |
|-------|------|
| ALS | Retrieval — generates Top-K candidates |
| LambdaMART ranker | Ranking — LightGBM LambdaMART rescoring per retriever |
| Popularity | Baseline — non-personalized comparison |

```bash
jupyter notebook notebooks/als_retrieval_pipeline.ipynb
jupyter notebook notebooks/independent_als_retrieval_pipeline.ipynb
```

### Train ALS

```bash
python -m src.pipelines.train_als --config configs/als.yaml
```

### Evaluate (train + evaluate, or load saved model)

```bash
# ALS only (includes Popularity baseline)
python -m src.pipelines.evaluate_als --config configs/als.yaml
python -m src.pipelines.evaluate_als --config configs/als.yaml --load-model
python -m src.pipelines.evaluate_als --config configs/als.yaml --with-ranker [--load-ranker]

# Single retriever
python -m src.pipelines.evaluate_two_tower --config configs/two_tower.yaml [--load-model]
python -m src.pipelines.evaluate_hybrid --config configs/hybrid.yaml [--load-model]

# Train ranker (independent per retriever, LightGBM LambdaMART)
python -m src.pipelines.train_ranker --retriever als --config configs/als.yaml [--load-retrieval]
python -m src.pipelines.train_ranker --retriever two_tower --config configs/two_tower.yaml
python -m src.pipelines.train_ranker --retriever hybrid --config configs/hybrid.yaml

# Multi-retriever comparison (shared temporal split)
python -m src.pipelines.compare_retrievers \
  --config configs/als.yaml \
  --retrievers popularity,als,two_tower,hybrid \
  [--load-als] [--load-two-tower] [--load-hybrid]
```

## Configuration

Edit [`configs/als.yaml`](configs/als.yaml) to change hyperparameters. Copy the file for experiment runs:

```bash
cp configs/als.yaml configs/als_factors256.yaml
# edit factors: 256
python -m src.pipelines.evaluate_als --config configs/als_factors256.yaml
```

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `split.validation_days` | 7 | Last N days held out for validation |
| `interaction_weighting.method` | recency | binary, count, or recency |
| `als.factors` | 128 | Latent dimensionality |
| `als.alpha` | 20 | Implicit confidence scale |
| `retrieval.candidate_k` | 200 | Candidates per user |

## Expected Outputs

**Logs:** dataset stats, temporal split info, CSR matrix stats, training time, sample recommendations.

**Artifacts:**
- `artifacts/als/als_model.pkl`
- `artifacts/als/config.yaml`
- `data/mappings/user_mapping.parquet`
- `data/mappings/item_mapping.parquet`

**Evaluation table:**

```
Model       Recall@50  Recall@100  Recall@200  MAP@12  NDCG@12
Popularity  ...
ALS         ...
```

Plus diagnostics: evaluable user %, cold-start users, unseen-item interactions, candidate coverage.

## Temporal Evaluation

- **Training:** all transactions strictly before the validation window.
- **Validation:** last `validation_days` calendar days.
- **Metrics:** computed only for validation users present in training (cold-start users reported separately).
- **Two recall variants:**
  - `recall_all_gt@K` — all validation purchases in denominator
  - `recall_retrievable@K` — excludes items never seen in training

## Next Experiments

After the baseline runs successfully, compare:

1. `als.factors`: 64, 128, 256
2. `als.regularization`: 0.01, 0.05, 0.1
3. `als.alpha`: 10, 20, 40
4. `interaction_weighting.method`: binary, count, recency

Track **Recall@100** and **MAP@12** across runs.

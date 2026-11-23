# ALS Vertex Job: What Is Running and Why It Is Slow

This document describes what the `recsys-als-training` Vertex AI custom job does, what phase it is likely in after many hours of runtime, and the specific code paths that cause long wall-clock times.

## Job Overview

| Property | Value |
|----------|-------|
| Container | `asia-south1-docker.pkg.dev/rec-sys-507218/recsys/recsys-als-training:v4` |
| Entrypoint | `training/als/run_job.py` |
| Config | `configs/als.yaml` |
| Data | `gs://rec-sys-507218-data/data/raw/transactions_train.csv` (~31.8M rows) |
| Machine (typical) | `n1-standard-8` or `n1-highmem-32`, 1 worker, no GPU used for ALS |
| Region | `asia-south1` |

The job name suggests "ALS training," but the container runs a **full pipeline**: data load → ALS fit → LightGBM ranker → **full offline evaluation** → write artifacts to GCS.

Vertex does **not** pass `--skip-eval` by default (`training/als/custom_job.yaml`). The Cloud Run variant does skip eval intentionally.

---

## Pipeline Phases (In Order)

```
run_job.py
  │
  ├─ 1. prepare_training_data()          Load CSV from GCS, temporal split, ID mappings
  ├─ 2. prepare_als_matrix()             Build sparse user×item CSR matrix
  ├─ 3. train_als_model()                implicit ALS (20 iterations, 128 factors)
  ├─ 4. als_model.save()                 Write als_model.pkl to GCS
  ├─ 5. train_als_ranker()               Reload data, load ALS, train LightGBM ranker
  ├─ 6. save_ranker()                    Write ranker.pkl
  │
  └─ 7. Evaluation (unless --skip-eval)  ← dominant cost after ~1–2 hours
       ├─ For each chunk of 5,000 eval users:
       │    ├─ generate_candidates()      Score all items per user, take top-200
       │    ├─ add_features_to_candidates()  Row-by-row Python feature engineering
       │    └─ ranker.rerank_candidates()    LambdaMART reranking
       ├─ Popularity baseline candidates
       ├─ compute metrics @ K = [12, 50, 100, 200]
       └─ Write metrics.json to GCS
```

### Expected duration by phase

| Phase | Typical duration | Notes |
|-------|------------------|-------|
| Data load + matrix build | Minutes | Full in-memory load of 31.8M rows |
| ALS training (20 iter) | Minutes | Capped to 1 BLAS thread (see below) |
| Ranker training | Minutes | Data loaded **a second time** inside `train_als_ranker()` |
| **Offline evaluation** | **Many hours** | Hundreds of thousands of users × slow Python paths |

---

## What Is Likely Happening Right Now (15+ Hour Jobs)

If a job has been running for ~15 hours with these symptoms:

- **CPU flat at ~20–25%** on an 8-vCPU machine (~1.5–2 cores active)
- **RAM climbing steadily** (e.g. 30% → 50%+ over many hours)
- Status still shows **"Training"**

then ALS training has almost certainly **already finished**. The job is most likely in **Phase 7: offline evaluation**, specifically inside the per-chunk loop in `training/als/run_job.py`:

```python
for chunk_idx, user_chunk in enumerate(iter_chunks(eval_user_indices, user_chunk_size)):
  chunk_cands = generate_candidates(...)       # slow
  chunk_featured = add_features_to_candidates(...)  # very slow
  ranked_frames.append(ranker.rerank_candidates(...))
  als_candidate_frames.append(chunk_cands)       # memory grows each chunk
```

### How to confirm in Vertex logs

Look for these log lines (in order):

1. `"ALS training completed in X seconds"` — ALS is done
2. `"Starting evaluation for N users (chunk_size=5000, ...)"` — eval has started
3. `"Eval chunk users A-B complete (M candidates)"` — eval progressing chunk by chunk

Progress heartbeats are logged only every **30 minutes** (`progress_log_interval_sec: 1800` in `configs/als.yaml`), so long quiet periods in logs are normal even when work is slow.

---

## Why Evaluation Dominates Wall Time

### Scale of the eval workload

- **Validation window:** last 7 days of transactions (`validation_days: 7`)
- **Eval users:** validation users who also appear in training — typically **hundreds of thousands**
- **Candidates per user:** 200 (`candidate_k: 200`)
- **Chunk size:** 5,000 users per iteration
- **Catalog size:** ~105K items (H&M articles)

Per chunk of 5,000 users, the pipeline may process up to **1,000,000 candidate rows** (5,000 × 200).

---

## Root Causes of Lag

### 1. Full-catalog scoring per user (unnecessary work)

**File:** `src/retrieval/als/model.py` — `recommend_batch()`

For each user, the code computes dot products against **every item** in the catalog (~105K), then selects top-200:

```python
for user_idx in user_indices:
    scores = item_factors @ self.model.user_factors[user_idx]  # all ~105K items
    ...
    top = np.argpartition(-scores, k)[:k]
```

Only top-200 are needed, but all ~105K items are scored. The single-user path uses `implicit`'s optimized `recommend()`; the batch eval path does not.

**Impact:** O(eval_users × num_items × factors) instead of O(eval_users × k × factors).

---

### 2. O(n) ID lookups inside Python loops

**File:** `src/data/mappings.py`

`IdMapping.to_idx()` and `to_original()` filter the entire mapping DataFrame on every call:

```python
def to_idx(self, original_id: str) -> int | None:
    row = self.forward.filter(pl.col("original_id") == original_id)
```

These are called in:

| Location | Calls per eval chunk (5K users) |
|----------|----------------------------------|
| `generate_candidates()` | ~200 per user → ~1M lookups |
| `add_features_to_candidates()` | 2 per candidate row → ~2M lookups |
| `get_eval_user_indices()` | 1 per eval user (once at start) |

With ~1.37M users in the mapping, each lookup scans up to 1.37M rows. Millions of such lookups in pure Python make eval orders of magnitude slower than necessary.

**Fix direction:** dict-backed O(1) lookups.

---

### 3. Row-by-row feature engineering in Python

**File:** `src/ranking/als/features.py` — `add_features_to_candidates()`

```python
for row in candidates.iter_rows(named=True):
    user_idx = user_map.to_idx(customer_id)
    item_idx = item_map.to_idx(article_id)
    rows.append({...})
```

Each candidate row is processed individually. `item_popularity()` and `user_activity()` are also recomputed from the full training DataFrame on every chunk.

**Fix direction:** Polars joins and vectorized feature columns.

---

### 4. Memory accumulation during eval (explains rising RAM)

**File:** `training/als/run_job.py`

```python
als_candidate_frames: list[pl.DataFrame] = []
ranked_frames: list[pl.DataFrame] = []

for ...:
    als_candidate_frames.append(chunk_cands)
    ranked_frames.append(ranker.rerank_candidates(chunk_featured))

als_candidates = pl.concat(als_candidate_frames)
ranked_candidates = pl.concat(ranked_frames)
```

Every chunk's results are held in memory until the final `pl.concat()`. As more chunks complete, RAM usage grows linearly — matching the steady RAM climb seen on long-running jobs.

For ~300K eval users × 200 candidates × multiple DataFrames, peak memory can reach tens of GB before concat.

---

### 5. Data loaded and processed twice

**Files:** `training/als/run_job.py`, `src/ranking/train.py`

`run_job.py` calls `prepare_training_data()` and `prepare_als_matrix()` once for ALS. Then `train_als_ranker(load_retrieval=True)` calls both again internally:

```python
train_df, _val_df, ... = prepare_training_data(config)
matrix = prepare_als_matrix(...)
```

This duplicates:

- GCS download of the full CSV
- Polars parse of 31.8M rows
- Temporal split and mapping build
- Interaction aggregation and CSR construction

**Impact:** Extra minutes at job start; not the main cause of 15-hour runs, but wasteful.

---

### 6. Single-threaded BLAS (underutilizes CPU)

**File:** `src/pipelines/common.py` — `setup_logging()`

```python
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
```

This limits `implicit` ALS training, NumPy matmuls, and LightGBM's BLAS backend to **one thread**. On an 8- or 32-vCPU machine, ALS fit uses a fraction of available cores.

**Observed symptom:** CPU ~23% on 8 vCPUs ≈ ~1.8 cores active — consistent with 1 BLAS thread plus Python overhead.

**Impact:** Slows ALS and ranker training, but eval (Python loops + Polars filters) is still the dominant bottleneck on long jobs.

---

### 7. No GPU acceleration for ALS path

The Docker image installs CUDA PyTorch, but ALS uses the CPU-only `implicit` library. The Vertex job spec has no GPU. GPU is unused for this pipeline.

---

## Resource Utilization Interpretation

| Metric | Pattern | Meaning |
|--------|---------|---------|
| CPU ~20–25% flat | Low, steady | Sequential Python work + 1 BLAS thread; not CPU-bound across all cores |
| RAM linear growth | Steady increase over hours | Eval chunks accumulating in `als_candidate_frames` / `ranked_frames` |
| Long runtime with "Training" status | 10–20+ hours | Vertex labels the whole custom job as "Training"; eval is included |

---

## Configuration Reference

From `configs/als.yaml`:

```yaml
als:
  factors: 128
  iterations: 20
  regularization: 0.05
  alpha: 20

retrieval:
  candidate_k: 200

evaluation:
  ks: [12, 50, 100, 200]
  user_chunk_size: 5000
  progress_log_interval_sec: 1800   # 30-minute log heartbeat
```

---

## Immediate Mitigations

### Training only (skip eval)

Resubmit with `--skip-eval` to match the Cloud Run behavior. Expected wall time: **tens of minutes**, not hours.

```bash
# In container args or custom_job.yaml containerSpec.args:
--skip-eval
```

Or set environment variable `SKIP_EVAL=1`.

### Use Dataproc for distributed ALS

The Spark path (`training/distributed/dataproc/submit_job.sh`) trains MLlib ALS in a distributed setting. It is a separate code path without the ranker, but eval can be sharded across partitions.

---

## Recommended Code Fixes (Longer Term)

| Priority | Change | File(s) |
|----------|--------|---------|
| High | Use `implicit` `recommend()` / batch APIs instead of full-catalog matmul | `src/retrieval/als/model.py` |
| High | Dict-backed O(1) `IdMapping` lookups | `src/data/mappings.py` |
| High | Vectorize `add_features_to_candidates()` with Polars joins | `src/ranking/als/features.py` |
| Medium | Stream or spill eval chunks to disk instead of in-memory concat | `training/als/run_job.py` |
| Medium | Reuse prepared data in `train_als_ranker()` when called from `run_job.py` | `src/ranking/train.py` |
| Low | Raise `OPENBLAS_NUM_THREADS` after ALS fit (e.g. 4–8) | `src/pipelines/common.py` |
| Low | Remove unused CUDA PyTorch from ALS Dockerfile | `training/als/Dockerfile` |

---

## Related Files

| File | Role |
|------|------|
| `training/als/run_job.py` | Vertex container entrypoint |
| `training/als/custom_job.yaml` | Vertex job spec |
| `training/als/cloudrun_job.yaml` | Cloud Run variant (skips eval) |
| `configs/als.yaml` | Hyperparameters and eval settings |
| `src/retrieval/als/model.py` | ALS wrapper and batch scoring |
| `src/retrieval/als/retrieve.py` | Candidate generation for eval |
| `src/ranking/als/features.py` | Ranker feature engineering |
| `src/data/mappings.py` | ID mapping (current bottleneck) |
| `src/pipelines/common.py` | Data prep, default OpenBLAS threads when env unset |
| `docs/openblas-threading.md` | BLAS threading, Dockerfile pin, Vertex CPU usage |
| `training/distributed/dataproc/submit_job.sh` | Alternative Spark/Dataproc path |

---

## Summary

**What is happening:** After the first 1–2 hours, a long-running `recsys-als-training` job is almost certainly in **offline evaluation**, not ALS matrix factorization. It iterates over validation users in chunks of 5,000, scoring candidates, engineering features row-by-row, reranking, and accumulating results in memory.

**Why it is slow:** Evaluation uses brute-force full-catalog scoring per user, O(n) Polars filter lookups inside tight Python loops, and in-memory accumulation of all chunk results. CPU stays low because work is sequential Python, not parallelized across cores. RAM rises steadily as each eval chunk is appended to in-memory lists.

**Fastest workaround today:** Resubmit with `--skip-eval` if you only need trained model artifacts, not metrics.

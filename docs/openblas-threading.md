# OpenBLAS Threading (`OPENBLAS_NUM_THREADS`)

This document explains what BLAS is, how many threads the ALS training pipeline uses on Vertex AI, why `OPENBLAS_NUM_THREADS` is often set to `1`, and when it should be higher.

---

## What is BLAS?

**BLAS** (Basic Linear Algebra Subprograms) is a standard library for fast vector and matrix math (multiplies, dot products, etc.).

This project uses **OpenBLAS** (via NumPy and LightGBM). The environment variable `OPENBLAS_NUM_THREADS` controls how many CPU cores OpenBLAS may use **for a single linear-algebra operation**.

| Setting | Meaning |
|---------|---------|
| `OPENBLAS_NUM_THREADS=1` | One matrix operation uses **one CPU core** |
| `OPENBLAS_NUM_THREADS=8` | One operation may split work across **up to 8 cores** |

### Where BLAS is used in this pipeline

| Step | Uses OpenBLAS? |
|------|----------------|
| NumPy (`einsum`, dot products in feature engineering) | Yes |
| LightGBM ranker training | Yes (internally) |
| `implicit` ALS | Partly (also uses its own C++/OpenMP code) |
| Polars | Mostly its own engine, not BLAS |
| Python eval metric loops | No |

---

## Current configuration in this repo

### 1. Dockerfile (Vertex / Cloud Run container)

```dockerfile
# training/als/Dockerfile
ENV OPENBLAS_NUM_THREADS=1
```

The container **starts with BLAS pinned to 1** for the entire job lifetime unless overridden at runtime.

Same pattern exists in `training/hybrid/Dockerfile` and `training/two_tower/Dockerfile`.

### 2. `setup_logging()` (local runs)

```python
# src/pipelines/common.py
if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = str(min(8, os.cpu_count() or 4))
```

If the env var is **not** already set (typical local dev), default is **`min(8, cpu_count)`** — e.g. **8** on an 8-core machine.

On Vertex, the Dockerfile has already set `1`, so `setup_logging()` does **not** change it.

### 3. ALS fit only (temporary override)

```python
# src/retrieval/als/model.py — ALSModel.fit()
prev_threads = os.environ.get("OPENBLAS_NUM_THREADS")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
try:
    self.model.fit(user_item_matrix, show_progress=True)
finally:
  # restore prev_threads
```

During ALS matrix factorization, BLAS is **forced to 1** even if the environment had a higher value. After `fit()` completes, the previous value is restored.

### 4. ALS retrieval (separate from BLAS)

```python
# src/retrieval/als/model.py — recommend_batch()
max_workers = min(8, len(user_indices), os.cpu_count() or 4)
```

Top-K candidate generation uses a **ThreadPoolExecutor** with up to **8 workers** calling `implicit`'s `recommend()`. This is independent of `OPENBLAS_NUM_THREADS`.

---

## Effective thread counts on Vertex AI

Assuming `n1-standard-8` (8 vCPUs) and the ALS Docker image as built today:

| Component | Threads |
|-----------|---------|
| OpenBLAS / NumPy / LightGBM linear algebra | **1** (Dockerfile) |
| `implicit` ALS `fit()` | BLAS **1** + native OpenMP inside `implicit` |
| `recommend_batch()` | **Up to 8** (ThreadPoolExecutor, capped) |
| Polars | Own thread pool (separate from OpenBLAS) |
| Python eval / metrics | Mostly **single-threaded** |

On `n1-highmem-32` (32 vCPUs): BLAS is still **1**; `recommend_batch` is still capped at **8**.

This matches observed Vertex metrics: **~20–25% CPU** on an 8-vCPU machine (~1.5–2 cores active) during slow eval phases.

---

## Why was BLAS set to 1?

### Thread oversubscription

If every library spawns N threads on an 8-core machine:

```
implicit ALS     → 8 threads
OpenBLAS         → 8 threads
LightGBM         → 8 threads
Polars           → more threads
ThreadPoolExecutor → 8 threads (recommend)
─────────────────────────────────
Total            → 32+ threads competing for 8 CPUs
```

That causes context-switching overhead and can be **slower** than using fewer threads deliberately.

The original fix: **pin OpenBLAS to 1** during ALS training so `implicit` owns parallelism without fighting BLAS.

### The mistake: global pin for the whole job

The Dockerfile sets `OPENBLAS_NUM_THREADS=1` for the **entire container**, not only during ALS `fit()`. Phases that would benefit from more BLAS threads still run at 1:

| Phase | BLAS=1 OK? |
|-------|------------|
| ALS `fit()` | Yes (also forced in `model.py`) |
| LightGBM ranker training | **No** — would benefit from more threads |
| NumPy `einsum` in features | **No** — would benefit from more threads |
| Offline eval | Minor gain (eval is mostly Python/Polars) |

So the machine is underused for ranker training and vectorized feature math, not because BLAS > 1 is bad in general, but because **1 was applied globally**.

---

## What “BLAS > 1” means

**BLAS > 1** means allowing OpenBLAS to use multiple cores (e.g. `OPENBLAS_NUM_THREADS=8`) for matrix operations.

- **BLAS = 1:** one worker does the whole matrix multiply.
- **BLAS = 8:** up to eight workers split the same multiply.

It does **not** mean “use more users” or “more candidates” — only **more CPU parallelism for linear algebra**.

---

## Recommended configuration

Scope BLAS=1 to ALS fit only; use more threads elsewhere:

```
Job start:       OPENBLAS_NUM_THREADS=8   (Dockerfile or setup_logging)
ALS fit():       force to 1               (already in src/retrieval/als/model.py)
After ALS fit:   restore to 8             (automatic via finally block)
```

| Phase | Suggested BLAS threads (8-vCPU VM) |
|-------|-------------------------------------|
| ALS training | **1** |
| Ranker training | **8** |
| Feature engineering (`einsum`) | **8** |
| Eval | **8** (small impact; eval is not BLAS-bound) |

### Practical changes (if tuning performance)

1. Change `training/als/Dockerfile`: `OPENBLAS_NUM_THREADS=8` (or remove the line and rely on `setup_logging()`).
2. Keep `ALSModel.fit()` override at `1` — do not remove it.
3. Optionally align `recommend_batch` `max_workers` with vCPU count (still cap to avoid oversubscription with Polars).

Rebuild and push a new image tag after Dockerfile changes.

---

## Quick reference

| Environment | Effective `OPENBLAS_NUM_THREADS` |
|-------------|----------------------------------|
| Vertex ALS container (current Dockerfile) | **1** (whole job) |
| Local run (no env set) | **min(8, cpu_count)** |
| During `ALSModel.fit()` only | **1** (always, then restored) |
| `recommend_batch()` parallelism | **Up to 8** (not BLAS) |

---

## Related files

| File | Role |
|------|------|
| `training/als/Dockerfile` | Sets `OPENBLAS_NUM_THREADS=1` in container |
| `src/pipelines/common.py` | Default `min(8, cpu_count)` if env unset |
| `src/retrieval/als/model.py` | BLAS=1 during fit; ThreadPoolExecutor for recommend |
| `docs/als-vertex-job-performance.md` | Eval slowness and pipeline overview |

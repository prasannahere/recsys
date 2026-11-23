FROM python:3.11-slim-bookworm

# Cloud Run mounts NVIDIA driver libs at /usr/local/nvidia/lib64 at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-training.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r requirements-training.txt

COPY configs/ configs/
COPY src/ src/
COPY training/two_tower/run_job.py training/two_tower/run_job.py

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    CONFIG_PATH=/app/configs/two_tower.yaml

ENTRYPOINT ["python", "/app/training/two_tower/run_job.py"]

#!/usr/bin/env bash
# Submit ALS job to Dataproc Serverless for Spark.
#
# Usage (from repo root):
#   ./training/distributed/dataproc/submit_job.sh
#   ./training/distributed/dataproc/submit_job.sh --skip-eval

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck disable=SC1091
if [[ -f "${SCRIPT_DIR}/job.env" ]]; then
  source "${SCRIPT_DIR}/job.env"
elif [[ -f "${SCRIPT_DIR}/.env" ]]; then
  source "${SCRIPT_DIR}/.env"
else
  source "${SCRIPT_DIR}/job.env.example"
fi

PROJECT="${GCP_PROJECT:?}"
REGION="${DATAPROC_REGION:-asia-south1}"
BATCH="${DATAPROC_BATCH:-recsys-als-batch}"
TRANSACTIONS_PATH="${TRANSACTIONS_PATH:?}"
OUTPUT_URI="${OUTPUT_URI:?}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/als.yaml}"
CODE_STAGING_URI="${CODE_STAGING_URI:?}"
EVAL_PARTITIONS="${EVAL_PARTITIONS:-15}"
MAX_EVAL_USERS="${MAX_EVAL_USERS:-0}"
SPARK_PROPERTIES="${SPARK_PROPERTIES:-spark.executor.extraJavaOptions=-Xss8m,spark.driver.extraJavaOptions=-Xss8m,spark.sql.shuffle.partitions=200}"

SKIP_EVAL_FLAG=""
if [[ "${1:-}" == "--skip-eval" ]]; then
  SKIP_EVAL_FLAG="--skip-eval"
fi

echo "Staging PySpark files to ${CODE_STAGING_URI}..."
gsutil -m cp \
  "${SCRIPT_DIR}/als_spark_job.py" \
  "${SCRIPT_DIR}/spark_common.py" \
  "${CODE_STAGING_URI}/"

if [[ "${CONFIG_PATH}" != gs://* ]]; then
  gsutil cp "${CONFIG_PATH}" "${CODE_STAGING_URI}/als.yaml"
  REMOTE_CONFIG="${CODE_STAGING_URI}/als.yaml"
else
  REMOTE_CONFIG="${CONFIG_PATH}"
fi

MAIN_PY="${CODE_STAGING_URI}/als_spark_job.py"
COMMON_PY="${CODE_STAGING_URI}/spark_common.py"

echo "Submitting Dataproc Serverless batch ${BATCH} in ${REGION}..."

gcloud dataproc batches submit pyspark "${MAIN_PY}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --batch="${BATCH}" \
  --py-files="${COMMON_PY}" \
  --properties="${SPARK_PROPERTIES}" \
  -- \
  --transactions-path="${TRANSACTIONS_PATH}" \
  --output-uri="${OUTPUT_URI}" \
  --config-path="${REMOTE_CONFIG}" \
  --eval-partitions="${EVAL_PARTITIONS}" \
  --max-eval-users="${MAX_EVAL_USERS}" \
  ${SKIP_EVAL_FLAG}

echo "Batch submitted: ${BATCH}"
echo "Monitor: gcloud dataproc batches describe ${BATCH} --region=${REGION}"

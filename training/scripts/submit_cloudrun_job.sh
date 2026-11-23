#!/usr/bin/env bash
# Deploy and execute a Cloud Run GPU training job (L4).
#
# Usage (from repo root):
#   ./training/scripts/submit_cloudrun_job.sh two_tower
#   ./training/scripts/submit_cloudrun_job.sh two_tower --deploy-only
#   ./training/scripts/submit_cloudrun_job.sh two_tower --execute-only

set -euo pipefail

JOB="${1:?job name required: als | hybrid | two_tower}"
MODE="${2:-}"
REGION="${CLOUDRUN_REGION:-asia-southeast1}"
CONFIG="training/${JOB}/cloudrun_job.yaml"

case "${JOB}" in
  als|hybrid|two_tower) ;;
  *)
    echo "Unknown job: ${JOB}" >&2
    exit 1
    ;;
esac

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 1
fi

JOB_NAME="recsys-${JOB//_/-}-training"

deploy() {
  echo "Deploying ${JOB_NAME} to Cloud Run (${REGION}, L4 GPU)..."
  gcloud run jobs replace "${CONFIG}" --region="${REGION}"
}

execute() {
  echo "Starting execution of ${JOB_NAME}..."
  gcloud run jobs execute "${JOB_NAME}" --region="${REGION}" --wait
}

case "${MODE}" in
  --deploy-only)
    deploy
    ;;
  --execute-only)
    execute
    ;;
  "")
    deploy
    execute
    ;;
  *)
    echo "Unknown option: ${MODE} (use --deploy-only or --execute-only)" >&2
    exit 1
    ;;
esac

echo "Monitor:"
echo "  gcloud run jobs executions list --job=${JOB_NAME} --region=${REGION}"
echo "  gcloud run jobs executions describe EXECUTION --region=${REGION}"

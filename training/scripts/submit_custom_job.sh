#!/usr/bin/env bash
# Submit a Vertex AI Custom Job (GPU) in asia-south1.
#
# Usage (from repo root):
#   ./training/scripts/submit_custom_job.sh two_tower
#   ./training/scripts/submit_custom_job.sh hybrid
#   ./training/scripts/submit_custom_job.sh als

set -euo pipefail

JOB="${1:?job name required: als | hybrid | two_tower}"
REGION="${VERTEX_REGION:-asia-south1}"
CONFIG="training/${JOB}/custom_job.yaml"

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

DISPLAY_NAME="recsys-${JOB//_/-}-training"

echo "Submitting ${DISPLAY_NAME} to Vertex AI (${REGION})..."
gcloud ai custom-jobs create \
  --region="${REGION}" \
  --display-name="${DISPLAY_NAME}" \
  --config="${CONFIG}"

echo "Monitor:"
echo "  gcloud ai custom-jobs list --region=${REGION}"
echo "  gcloud ai custom-jobs stream-logs JOB_ID --region=${REGION}"

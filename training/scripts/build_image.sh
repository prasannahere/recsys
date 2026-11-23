#!/usr/bin/env bash
# Build and push a training image for Cloud Run GPU (linux/amd64).
#
# Usage (from repo root):
#   ./training/scripts/build_image.sh als v1
#   ./training/scripts/build_image.sh hybrid v1
#   ./training/scripts/build_image.sh two_tower v1

set -euo pipefail

JOB="${1:?job name required: als | hybrid | two_tower}"
TAG="${2:-v1}"
PROJECT="${GCP_PROJECT:-rec-sys-507218}"
AR_REGION="${AR_REGION:-asia-south1}"
REPO="${AR_REPO:-recsys}"

case "${JOB}" in
  als)
    IMAGE_NAME="recsys-als-training"
    DOCKERFILE="training/als/Dockerfile"
    ;;
  hybrid)
    IMAGE_NAME="recsys-hybrid-training"
    DOCKERFILE="training/hybrid/Dockerfile"
    ;;
  two_tower)
    IMAGE_NAME="recsys-two-tower-training"
    DOCKERFILE="training/two_tower/Dockerfile"
    ;;
  *)
    echo "Unknown job: ${JOB}" >&2
    exit 1
    ;;
esac

IMAGE="${AR_REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${IMAGE_NAME}:${TAG}"

docker build --platform linux/amd64 -f "${DOCKERFILE}" -t "${IMAGE}" .
docker push "${IMAGE}"

echo "Pushed ${IMAGE}"
echo "Submit Cloud Run GPU job (L4, asia-southeast1):"
echo "  ./training/scripts/submit_cloudrun_job.sh ${JOB}"
echo "Or Vertex AI Custom Job:"
echo "  ./training/scripts/submit_custom_job.sh ${JOB}"

#!/usr/bin/env bash
# Deploy the crop detection API to Google Cloud Run.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-asia-south1}"
SERVICE="${SERVICE:-crop-detection-api}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/ml/${SERVICE}:latest"

gcloud artifacts repositories describe ml --location "$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create ml --repository-format=docker --location "$REGION"

gcloud builds submit --tag "$IMAGE" --file deployment/Dockerfile .

gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 10

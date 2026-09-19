#!/usr/bin/env bash
# Deploy the crop detection API to AWS ECS Fargate via ECR.
set -euo pipefail

AWS_REGION="${AWS_REGION:-ap-south-1}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REPO="${REPO:-crop-detection-api}"
IMAGE="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO}:latest"

aws ecr describe-repositories --repository-names "$REPO" --region "$AWS_REGION" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$REPO" --region "$AWS_REGION"

aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build -f deployment/Dockerfile -t "$IMAGE" .
docker push "$IMAGE"

aws ecs update-service --cluster "${CLUSTER:-ml-cluster}" \
  --service "${SERVICE:-crop-detection-api}" --force-new-deployment --region "$AWS_REGION"

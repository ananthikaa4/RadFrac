#!/bin/bash
# RADFRAC EC2 startup script - Amazon Linux 2023

set -euxo pipefail

REGION="ap-south-1"
ACCOUNT_ID="407525819889"
IMAGE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/radfrac:latest"

# Install and start Docker
dnf install -y docker
systemctl enable --now docker

# Authenticate to ECR using the EC2 instance IAM role
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin \
      "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# Pull the published RADFRAC image
docker pull "$IMAGE"

# Persist runtime data across container restarts
docker volume create radfrac-runtime

# Start RADFRAC
docker run -d \
  --name radfrac \
  --restart unless-stopped \
  -p 80:8501 \
  -p 127.0.0.1:8000:8000 \
  -v radfrac-runtime:/app/runtime \
  "$IMAGE"
#!/usr/bin/env bash
set -e
IMAGE="YOUR_DOCKERHUB_USERNAME/nilechat-runpod:v1"
docker build -t "$IMAGE" .
docker push "$IMAGE"
echo "Pushed: $IMAGE"

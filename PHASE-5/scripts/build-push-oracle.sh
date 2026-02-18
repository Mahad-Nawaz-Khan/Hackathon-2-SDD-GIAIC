#!/bin/bash
# Build and push images to Oracle Container Image Registry (OCIR)
# Usage: ./scripts/build-push-oracle.sh <region> <tenancy-namespace>

set -e

# Configuration - UPDATE THESE
REGION="${1:-ap-mumbai-1}"  # Change to your region
TENANCY_NAMESPACE="${2}"     # Get with: oci os namespace get

if [ -z "$TENANCY_NAMESPACE" ]; then
    echo "Error: Tenancy namespace required"
    echo "Usage: $0 <region> <tenancy-namespace>"
    echo "Get namespace: oci os namespace get"
    exit 1
fi

REGISTRY="${REGION}.ocir.io"
REPOSITORY_PREFIX="${REGISTRY}/${TENANCY_NAMESPACE}"

echo "=== Building images for Oracle OKE ==="
echo "Registry: ${REGISTRY}"
echo "Repository prefix: ${REPOSITORY_PREFIX}"

# Build backend (multi-arch for ARM)
echo ""
echo "=== Building backend ==="
docker buildx build --platform linux/amd64,linux/arm64 \
    -t "${REPOSITORY_PREFIX}/todo-chatbot-backend:latest" \
    -t "${REPOSITORY_PREFIX}/todo-chatbot-backend:v1.0.0" \
    --push \
    "./backend"

# Build frontend with environment variables
echo ""
echo "=== Building frontend ==="
docker buildx build --platform linux/amd64,linux/arm64 \
    -t "${REPOSITORY_PREFIX}/todo-chatbot-frontend:latest" \
    -t "${REPOSITORY_PREFIX}/todo-chatbot-frontend:v1.0.0" \
    --build-arg NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY="${NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY}" \
    --build-arg NEXT_PUBLIC_API_URL="https://api.todo-app.cloud" \
    --push \
    "./frontend"

echo ""
echo "=== Images pushed successfully ==="
echo "Backend: ${REPOSITORY_PREFIX}/todo-chatbot-backend:latest"
echo "Frontend: ${REPOSITORY_PREFIX}/todo-chatbot-frontend:latest"
echo ""
echo "Update values-prod.yaml with these image URLs!"

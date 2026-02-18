#!/usr/bin/env bash
# docker-buildx.sh - Set up Docker Buildx for multi-arch builds
# Supports: linux/amd64, linux/arm64 (required for Oracle ARM Free Tier)
set -euo pipefail

BUILDER_NAME="todo-chatbot-builder"
PLATFORMS="linux/amd64,linux/arm64"

echo "=== Docker Buildx Multi-Arch Setup ==="

# Check Docker is available
if ! command -v docker &> /dev/null; then
  echo "ERROR: Docker is not installed"
  exit 1
fi

# Check buildx is available
if ! docker buildx version &> /dev/null; then
  echo "ERROR: Docker Buildx is not available. Install Docker Desktop or buildx plugin."
  exit 1
fi

# Create or use existing builder
if docker buildx inspect "${BUILDER_NAME}" &> /dev/null; then
  echo "Using existing builder: ${BUILDER_NAME}"
  docker buildx use "${BUILDER_NAME}"
else
  echo "Creating new builder: ${BUILDER_NAME}"
  docker buildx create --name "${BUILDER_NAME}" --use --platform "${PLATFORMS}"
fi

# Bootstrap the builder
docker buildx inspect --bootstrap

echo ""
echo "Builder '${BUILDER_NAME}' is ready for platforms: ${PLATFORMS}"
echo ""
echo "Build commands:"
echo "  Backend:  docker buildx build --platform ${PLATFORMS} -t ghcr.io/OWNER/todo-chatbot-backend:latest --push ./backend"
echo "  Frontend: docker buildx build --platform ${PLATFORMS} -t ghcr.io/OWNER/todo-chatbot-frontend:latest --push ./frontend"

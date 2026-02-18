#!/usr/bin/env bash
# deploy-local.sh - Deploy Todo Chatbot to local Minikube
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NAMESPACE="todo-app"

echo "=== Todo Chatbot - Local Deployment (Minikube) ==="

# Check prerequisites
for cmd in minikube kubectl helm dapr docker; do
  if ! command -v "$cmd" &> /dev/null; then
    echo "ERROR: $cmd is not installed"
    exit 1
  fi
done

# Step 1: Start Minikube (if not running)
if ! minikube status | grep -q "Running"; then
  echo "[1/7] Starting Minikube..."
  minikube start --cpus=4 --memory=8192 --driver=docker
else
  echo "[1/7] Minikube already running"
fi

# Enable ingress addon
minikube addons enable ingress

# Step 2: Install Dapr
echo "[2/7] Installing Dapr..."
if ! dapr status -k 2>/dev/null | grep -q "dapr-operator"; then
  dapr init -k --wait
else
  echo "  Dapr already installed"
fi

# Step 3: Create namespace
echo "[3/7] Creating namespace..."
kubectl apply -f "$PROJECT_DIR/infra/namespace.yaml"

# Step 4: Deploy infrastructure (Redpanda + Redis)
echo "[4/7] Deploying infrastructure..."
helm repo add redpanda https://charts.redpanda.com 2>/dev/null || true
helm repo add bitnami https://charts.bitnami.com/bitnami 2>/dev/null || true
helm repo update

helm upgrade --install redpanda redpanda/redpanda \
  -f "$PROJECT_DIR/infra/local/redpanda/values.yaml" \
  --namespace "$NAMESPACE" --wait --timeout 5m

helm upgrade --install redis bitnami/redis \
  -f "$PROJECT_DIR/infra/local/redis/values.yaml" \
  --namespace "$NAMESPACE" --wait --timeout 3m

# Step 5: Deploy Dapr components
echo "[5/7] Deploying Dapr components..."
kubectl apply -f "$PROJECT_DIR/infra/local/dapr/components/" -n "$NAMESPACE"

# Step 6: Build images (using Minikube's Docker daemon)
echo "[6/7] Building images..."
eval $(minikube docker-env)

docker build -t hackathon-backend:latest "$PROJECT_DIR/backend/"
docker build -t hackathon-frontend:latest "$PROJECT_DIR/frontend/"

# Step 7: Deploy application
echo "[7/7] Deploying application..."
helm upgrade --install todo-chatbot "$PROJECT_DIR/todo-chatbot" \
  -f "$PROJECT_DIR/todo-chatbot/values-local.yaml" \
  --namespace "$NAMESPACE" --wait --timeout 5m

echo ""
echo "=== Deployment Complete ==="
echo ""
kubectl get pods -n "$NAMESPACE"
echo ""
echo "Access the application:"
echo "  1. Add to /etc/hosts: $(minikube ip) todo-chatbot.local api.todo-chatbot.local"
echo "  2. Or run: minikube tunnel (in a separate terminal)"
echo "  3. Open: http://todo-chatbot.local"
echo ""
echo "Dapr Dashboard: dapr dashboard -k"

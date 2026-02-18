#!/usr/bin/env bash
# deploy-prod.sh - Deploy Todo Chatbot to Oracle OKE
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NAMESPACE="todo-app"
KAFKA_NAMESPACE="kafka"
IMAGE_TAG="${1:-latest}"

echo "=== Todo Chatbot - Production Deployment (Oracle OKE) ==="
echo "Image tag: $IMAGE_TAG"

# Check prerequisites
for cmd in kubectl helm dapr; do
  if ! command -v "$cmd" &> /dev/null; then
    echo "ERROR: $cmd is not installed"
    exit 1
  fi
done

# Verify cluster connection
if ! kubectl cluster-info &> /dev/null; then
  echo "ERROR: Cannot connect to Kubernetes cluster. Check KUBECONFIG."
  exit 1
fi

# Step 1: Create namespaces
echo "[1/8] Creating namespaces..."
kubectl apply -f "$PROJECT_DIR/infra/namespace.yaml"
kubectl create namespace "$KAFKA_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Step 2: Install NGINX Ingress
echo "[2/8] Installing NGINX Ingress Controller..."
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx 2>/dev/null || true
helm repo update
helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
  -f "$PROJECT_DIR/infra/nginx-ingress/values.yaml" \
  -n ingress-nginx --create-namespace --wait --timeout 5m

# Step 3: Install Cert-Manager
echo "[3/8] Installing Cert-Manager..."
helm repo add jetstack https://charts.jetstack.io 2>/dev/null || true
helm repo update
helm upgrade --install cert-manager jetstack/cert-manager \
  --namespace cert-manager --create-namespace \
  --set installCRDs=true --wait --timeout 5m
kubectl apply -f "$PROJECT_DIR/infra/prod/cert-manager/"

# Step 4: Install Strimzi Kafka
echo "[4/8] Installing Strimzi Kafka..."
helm repo add strimzi https://strimzi.io/charts/ 2>/dev/null || true
helm repo update
helm upgrade --install strimzi strimzi/strimzi-kafka-operator \
  --namespace "$KAFKA_NAMESPACE" --wait --timeout 5m

echo "  Deploying Kafka cluster (this may take a few minutes)..."
kubectl apply -f "$PROJECT_DIR/infra/prod/strimzi/kafka-cluster.yaml" -n "$KAFKA_NAMESPACE"

# Wait for Kafka to be ready
echo "  Waiting for Kafka to be ready..."
kubectl wait kafka/todo-kafka --for=condition=Ready --timeout=300s -n "$KAFKA_NAMESPACE" 2>/dev/null || \
  echo "  WARNING: Kafka may still be starting. Check: kubectl get kafka -n $KAFKA_NAMESPACE"

kubectl apply -f "$PROJECT_DIR/infra/prod/strimzi/kafka-topic.yaml" -n "$KAFKA_NAMESPACE"

# Step 5: Install Dapr
echo "[5/8] Installing Dapr..."
if ! dapr status -k 2>/dev/null | grep -q "dapr-operator"; then
  dapr init -k --wait
else
  echo "  Dapr already installed"
fi

# Step 6: Deploy Dapr components
echo "[6/8] Deploying Dapr components..."
kubectl apply -f "$PROJECT_DIR/infra/prod/dapr/components/" -n "$NAMESPACE"

# Step 7: Verify secrets exist
echo "[7/8] Verifying secrets..."
for secret in neon-credentials clerk-secrets ai-api-keys; do
  if ! kubectl get secret "$secret" -n "$NAMESPACE" &> /dev/null; then
    echo "  WARNING: Secret '$secret' not found. Create it before the app starts."
    echo "  See: infra/prod/secrets-template.yaml"
  else
    echo "  Secret '$secret' exists"
  fi
done

# Step 8: Deploy application
echo "[8/8] Deploying application..."
helm upgrade --install todo-chatbot "$PROJECT_DIR/todo-chatbot" \
  -f "$PROJECT_DIR/todo-chatbot/values-prod.yaml" \
  --set backend.image.tag="$IMAGE_TAG" \
  --set frontend.image.tag="$IMAGE_TAG" \
  --namespace "$NAMESPACE" --wait --timeout 5m

echo ""
echo "=== Deployment Complete ==="
echo ""
kubectl get pods -n "$NAMESPACE"
echo ""
echo "Ingress IP:"
kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "  Pending..."
echo ""
echo "TLS Certificate:"
kubectl get certificate -n "$NAMESPACE" 2>/dev/null || echo "  Not yet provisioned"

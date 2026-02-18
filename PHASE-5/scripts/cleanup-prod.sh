#!/usr/bin/env bash
# cleanup-prod.sh - Remove all production OKE resources
set -euo pipefail

NAMESPACE="todo-app"
KAFKA_NAMESPACE="kafka"

echo "=== Cleanup Production Deployment ==="
echo "WARNING: This will remove ALL resources from OKE."
read -p "Continue? (y/N): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
  echo "Aborted."
  exit 0
fi

echo "Removing application..."
helm uninstall todo-chatbot -n "$NAMESPACE" 2>/dev/null || true

echo "Removing Dapr components..."
kubectl delete -f infra/prod/dapr/components/ -n "$NAMESPACE" 2>/dev/null || true

echo "Removing Dapr..."
dapr uninstall -k 2>/dev/null || true

echo "Removing Kafka..."
kubectl delete -f infra/prod/strimzi/ -n "$KAFKA_NAMESPACE" 2>/dev/null || true
helm uninstall strimzi -n "$KAFKA_NAMESPACE" 2>/dev/null || true

echo "Removing Cert-Manager..."
kubectl delete -f infra/prod/cert-manager/ 2>/dev/null || true
helm uninstall cert-manager -n cert-manager 2>/dev/null || true

echo "Removing NGINX Ingress..."
helm uninstall ingress-nginx -n ingress-nginx 2>/dev/null || true

echo "Removing namespaces..."
kubectl delete namespace "$NAMESPACE" 2>/dev/null || true
kubectl delete namespace "$KAFKA_NAMESPACE" 2>/dev/null || true
kubectl delete namespace ingress-nginx 2>/dev/null || true
kubectl delete namespace cert-manager 2>/dev/null || true

echo ""
echo "Cleanup complete."

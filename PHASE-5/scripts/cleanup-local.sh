#!/usr/bin/env bash
# cleanup-local.sh - Remove all local Minikube resources
set -euo pipefail

NAMESPACE="todo-app"

echo "=== Cleanup Local Deployment ==="

echo "Removing Helm releases..."
helm uninstall todo-chatbot -n "$NAMESPACE" 2>/dev/null || true
helm uninstall redis -n "$NAMESPACE" 2>/dev/null || true
helm uninstall redpanda -n "$NAMESPACE" 2>/dev/null || true

echo "Removing Dapr..."
dapr uninstall -k 2>/dev/null || true

echo "Removing namespace..."
kubectl delete namespace "$NAMESPACE" 2>/dev/null || true

echo "Stopping Minikube..."
minikube stop 2>/dev/null || true

echo ""
echo "Cleanup complete. Run 'minikube delete' to fully remove the cluster."

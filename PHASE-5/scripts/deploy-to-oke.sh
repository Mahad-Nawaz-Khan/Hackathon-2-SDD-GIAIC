#!/bin/bash
# Deploy Todo Chatbot to Oracle OKE
# Prerequisites: kubectl configured with OKE cluster access

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Deploying Todo Chatbot to Oracle OKE ==="

# Create namespace
echo ""
echo "=== Creating namespace ==="
kubectl create namespace todo-app --dry-run=client -o yaml | kubectl apply -f -

# Create secrets (UPDATE VALUES BEFORE RUNNING!)
echo ""
echo "=== Creating secrets ==="
echo "WARNING: Update secret values in scripts/create-secrets.sh first!"
read -p "Have you updated the secret values? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please update scripts/create-secrets.sh with your actual values"
    exit 1
fi

# Apply secrets
"${SCRIPT_DIR}/create-secrets.sh"

# Install NGINX Ingress Controller
echo ""
echo "=== Installing NGINX Ingress Controller ==="
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
    --namespace ingress-nginx --create-namespace \
    --set controller.service.annotations."service\.beta\.kubernetes\.io/oci-load-balancer-shape"="flexible" \
    --set controller.service.annotations."service\.beta\.kubernetes\.io/oci-load-balancer-shape-flex-min"="10" \
    --set controller.service.annotations."service\.beta\.kubernetes\.io/oci-load-balancer-shape-flex-max"="100"

# Install Cert-Manager
echo ""
echo "=== Installing Cert-Manager ==="
helm repo add jetstack https://charts.jetstack.io
helm repo update
helm upgrade --install cert-manager jetstack/cert-manager \
    --namespace cert-manager --create-namespace \
    --set installCRDs=true \
    --set extraArgs[0]=--enable-certificate-owner-ref=true

# Wait for cert-manager to be ready
kubectl rollout status deployment/cert-manager -n cert-manager --timeout=120s

# Install Dapr
echo ""
echo "=== Installing Dapr ==="
helm repo add dapr https://dapr.github.io/helm-charts
helm repo update
helm upgrade --install dapr dapr/dapr \
    --namespace dapr-system --create-namespace \
    --set global.mtls.enabled=true \
    --set global.logLevel=info

# Wait for Dapr to be ready
kubectl rollout status deployment/dapr-operator -n dapr-system --timeout=120s

# Install Redis
echo ""
echo "=== Installing Redis ==="
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
helm upgrade --install redis bitnami/redis \
    --namespace todo-app \
    --set auth.enabled=false \
    --set architecture=standalone

# Install Strimzi Kafka
echo ""
echo "=== Installing Strimzi Kafka ==="
kubectl create namespace kafka --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f 'https://strimzi.io/install/latest?namespace=kafka' -n kafka

# Wait for Strimzi operator
kubectl rollout status deployment/strimzi-cluster-operator -n kafka --timeout=120s

# Deploy Kafka cluster
echo ""
echo "=== Deploying Kafka cluster ==="
kubectl apply -f "${PROJECT_ROOT}/infra/prod/strimzi/kafka-cluster.yaml" -n kafka

# Wait for Kafka to be ready (this takes a while)
echo "Waiting for Kafka cluster (this may take 5-10 minutes)..."
kubectl wait kafka/todo-chatbot-kafka --for=condition=Ready -n kafka --timeout=600s

# Create Kafka topic
kubectl apply -f "${PROJECT_ROOT}/infra/prod/strimzi/kafka-topic.yaml" -n kafka

# Deploy Dapr components
echo ""
echo "=== Deploying Dapr components ==="
kubectl apply -f "${PROJECT_ROOT}/infra/prod/dapr/components/" -n todo-app

# Deploy Cert-Manager issuer and certificate
echo ""
echo "=== Configuring TLS ==="
kubectl apply -f "${PROJECT_ROOT}/infra/prod/cert-manager/issuer.yaml"
kubectl apply -f "${PROJECT_ROOT}/infra/prod/cert-manager/certificate.yaml"

# Deploy the application with Helm
echo ""
echo "=== Deploying Todo Chatbot application ==="
helm upgrade --install todo-chatbot "${PROJECT_ROOT}/todo-chatbot" \
    --namespace todo-app \
    --values "${PROJECT_ROOT}/todo-chatbot/values-prod.yaml"

# Wait for deployments
echo ""
echo "=== Waiting for deployments ==="
kubectl rollout status deployment/todo-chatbot-backend -n todo-app --timeout=180s
kubectl rollout status deployment/todo-chatbot-frontend -n todo-app --timeout=180s

# Show status
echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Get your Load Balancer IP:"
echo "  kubectl get svc -n ingress-nginx"
echo ""
echo "Configure DNS:"
echo "  todo.todo-app.cloud    → <Load Balancer IP>"
echo "  api.todo-app.cloud → <Load Balancer IP>"
echo ""
echo "Check application status:"
echo "  kubectl get pods -n todo-app"
echo "  kubectl get ingress -n todo-app"

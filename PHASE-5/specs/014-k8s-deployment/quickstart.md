# Quickstart: Kubernetes Deployment

**Feature**: 014-k8s-deployment
**Date**: 2026-02-18

## Prerequisites

### Required Tools

| Tool | Version | Purpose | Install |
|------|---------|---------|---------|
| Docker | 24+ | Container runtime | [docker.com](https://docs.docker.com/get-docker/) |
| kubectl | 1.28+ | Kubernetes CLI | `brew install kubectl` |
| Helm | 3.12+ | Package manager | `brew install helm` |
| Dapr CLI | 1.12+ | Dapr management | `brew install dapr/tap/dapr-cli` |
| Minikube | 1.31+ | Local Kubernetes | `brew install minikube` |

### Verify Installations

```bash
docker --version
kubectl version --client
helm version
dapr --version
minikube version
```

---

## Local Deployment (Minikube)

### Step 1: Start Minikube

```bash
# Start with sufficient resources
minikube start --cpus=4 --memory=8192 --driver=docker

# Enable ingress addon
minikube addons enable ingress

# Verify cluster
kubectl get nodes
```

### Step 2: Install Dapr

```bash
# Initialize Dapr on Kubernetes
dapr init -k

# Wait for Dapr to be ready
kubectl get pods -n dapr-system -w

# Verify Dapr installation
dapr status -k
```

### Step 3: Deploy Infrastructure

```bash
# Add Helm repositories
helm repo add redpanda https://charts.redpanda.com
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Deploy Redpanda (Kafka-compatible)
helm install redpanda redpanda/redpanda \
  -f infra/local/redpanda/values.yaml \
  --namespace todo-app --create-namespace

# Deploy Redis
helm install redis bitnami/redis \
  -f infra/local/redis/values.yaml \
  --namespace todo-app

# Wait for infrastructure pods
kubectl get pods -n todo-app -w
```

### Step 4: Deploy Dapr Components

```bash
# Apply Dapr component configurations
kubectl apply -f infra/local/dapr/components/ -n todo-app

# Verify components
dapr components -k -n todo-app
```

### Step 5: Build and Deploy Application

```bash
# Build images locally (point Docker to Minikube's Docker daemon)
eval $(minikube docker-env)

# Build backend
cd backend
docker build -t hackathon-backend:latest .

# Build frontend
cd ../frontend
docker build -t hackathon-frontend:latest .

# Deploy with Helm
cd ..
helm install todo-chatbot ./todo-chatbot \
  -f todo-chatbot/values-local.yaml \
  --namespace todo-app

# Verify deployment
kubectl get pods -n todo-app
kubectl get ingress -n todo-app
```

### Step 6: Access Application

```bash
# Get Minikube IP
minikube ip

# Add to /etc/hosts (replace 192.168.49.2 with your minikube ip)
echo "192.168.49.2 todo-chatbot.local" | sudo tee -a /etc/hosts
echo "192.168.49.2 api.todo-chatbot.local" | sudo tee -a /etc/hosts

# Or use minikube tunnel (in separate terminal)
minikube tunnel

# Access application
open http://todo-chatbot.local
```

### Step 7: Verify Dapr Integration

```bash
# Check Dapr dashboard
dapr dashboard -k

# Check Dapr sidecar logs
kubectl logs -n todo-app deployment/todo-chatbot-backend -c daprd

# Verify pub/sub connectivity
kubectl exec -n todo-app deployment/todo-chatbot-backend -- curl -s http://localhost:3500/v1.0/healthz
```

---

## Production Deployment (Oracle OKE)

### Step 1: Configure kubectl for OKE

```bash
# Option A: Use Oracle Cloud CLI
oci ce cluster create-kubeconfig \
  --cluster-id <cluster-ocid> \
  --file $HOME/.kube/config \
  --region <region>

# Option B: Use provided kubeconfig
export KUBECONFIG=/path/to/oke-kubeconfig

# Verify connection
kubectl get nodes
```

### Step 2: Create Namespace and Secrets

```bash
# Create namespace
kubectl create namespace todo-app

# Create secrets (use your actual values)
kubectl create secret generic neon-credentials \
  --from-literal=connection-string='postgresql://...' \
  -n todo-app

kubectl create secret generic clerk-secrets \
  --from-literal=secret-key='sk_test_...' \
  --from-literal=issuer='https://...' \
  -n todo-app

kubectl create secret generic ai-api-keys \
  --from-literal=gemini-key='AIza...' \
  --from-literal=z-ai-key='...' \
  -n todo-app
```

### Step 3: Install NGINX Ingress Controller

```bash
# Add NGINX Helm repo
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

# Install NGINX Ingress
helm install ingress-nginx ingress-nginx/ingress-nginx \
  -f infra/nginx-ingress/values.yaml \
  -n ingress-nginx --create-namespace

# Get external IP
kubectl get svc -n ingress-nginx
```

### Step 4: Install Cert-Manager

```bash
# Add Jetstack Helm repo
helm repo add jetstack https://charts.jetstack.io
helm repo update

# Install Cert-Manager
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager --create-namespace \
  --set installCRDs=true

# Apply Let's Encrypt issuer
kubectl apply -f infra/prod/cert-manager/issuer.yaml
```

### Step 5: Install Strimzi Kafka Operator

```bash
# Add Strimzi Helm repo
helm repo add strimzi https://strimzi.io/charts/
helm repo update

# Install Strimzi operator
helm install strimzi strimzi/strimzi-kafka-operator \
  --namespace kafka --create-namespace

# Deploy Kafka cluster
kubectl apply -f infra/prod/strimzi/kafka-cluster.yaml -n kafka

# Wait for Kafka to be ready (takes a few minutes)
kubectl get kafka -n kafka -w

# Deploy topics
kubectl apply -f infra/prod/strimzi/kafka-topic.yaml -n kafka
```

### Step 6: Install Dapr

```bash
# Initialize Dapr on OKE
dapr init -k

# Apply production Dapr components
kubectl apply -f infra/prod/dapr/components/ -n todo-app
```

### Step 7: Configure DNS

```bash
# Get ingress external IP
INGRESS_IP=$(kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Configure DNS A record:
# todo.yourdomain.com -> $INGRESS_IP
# api.todo.yourdomain.com -> $INGRESS_IP
```

### Step 8: Deploy Application

```bash
# Deploy with Helm
helm upgrade --install todo-chatbot ./todo-chatbot \
  -f todo-chatbot/values-prod.yaml \
  --set backend.image.repository=ghcr.io/your-org/todo-chatbot-backend \
  --set backend.image.tag=latest \
  --set frontend.image.repository=ghcr.io/your-org/todo-chatbot-frontend \
  --set frontend.image.tag=latest \
  --namespace todo-app

# Verify deployment
kubectl get pods -n todo-app
kubectl get ingress -n todo-app

# Check TLS certificate
kubectl get certificate -n todo-app
```

### Step 9: Verify Production Deployment

```bash
# Check application health
curl https://todo.yourdomain.com/api/v1/health

# Check Dapr sidecar
kubectl logs -n todo-app deployment/todo-chatbot-backend -c daprd --tail=100

# Check Kafka connectivity
kubectl exec -n kafka -it todo-kafka-kafka-0 -- \
  bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
```

---

## CI/CD Deployment

### GitHub Actions Setup

1. **Add secrets to GitHub repository**:
   - `OKE_KUBECONFIG`: Base64-encoded kubeconfig file
   - `GITHUB_TOKEN`: Automatically provided

2. **Push to main branch** triggers:
   - Multi-arch Docker build
   - GHCR image push
   - Helm deployment to OKE

```bash
# Encode kubeconfig for GitHub secret
cat ~/.kube/config | base64 -w 0
```

### Manual Deployment via GitHub Actions

1. Go to Actions tab in GitHub
2. Select "Deploy to OKE" workflow
3. Click "Run workflow"

---

## Troubleshooting

### Common Issues

**Pod stuck in Pending**:
```bash
kubectl describe pod <pod-name> -n todo-app
# Check events for scheduling issues
```

**Dapr sidecar not starting**:
```bash
kubectl logs <pod-name> -c daprd -n todo-app
# Check Dapr configuration
dapr status -k
```

**Kafka connection refused**:
```bash
# Verify Kafka is running
kubectl get kafka -n kafka

# Check Kafka logs
kubectl logs -n kafka deployment/strimzi-cluster-operator
```

**TLS certificate not issued**:
```bash
# Check Cert-Manager logs
kubectl logs -n cert-manager deployment/cert-manager

# Check certificate status
kubectl describe certificate todo-tls -n todo-app
```

**Ingress not accessible**:
```bash
# Check ingress controller
kubectl get svc -n ingress-nginx

# Check ingress configuration
kubectl describe ingress -n todo-app
```

### Useful Commands

```bash
# View all resources in namespace
kubectl get all -n todo-app

# View Dapr components
dapr components -k -n todo-app

# Port forward for local debugging
kubectl port-forward -n todo-app svc/todo-chatbot-backend 8000:8000

# View resource usage
kubectl top pods -n todo-app

# Execute shell in pod
kubectl exec -it -n todo-app deployment/todo-chatbot-backend -- /bin/bash
```

---

## Cleanup

### Local (Minikube)

```bash
# Delete Helm releases
helm uninstall todo-chatbot -n todo-app
helm uninstall redis -n todo-app
helm uninstall redpanda -n todo-app

# Delete Dapr
dapr uninstall -k

# Stop and delete Minikube
minikube stop
minikube delete
```

### Production (OKE)

```bash
# Delete application
helm uninstall todo-chatbot -n todo-app

# Delete Kafka
kubectl delete -f infra/prod/strimzi/ -n kafka
helm uninstall strimzi -n kafka

# Delete Dapr
dapr uninstall -k

# Delete namespace (removes all resources)
kubectl delete namespace todo-app
kubectl delete namespace kafka
```

# Troubleshooting Guide

## Local (Minikube)

### Pod stuck in Pending

```bash
kubectl describe pod <pod-name> -n todo-app
# Common causes:
# - Insufficient resources: increase minikube --cpus/--memory
# - Image not found: run eval $(minikube docker-env) before building
```

### Dapr sidecar not starting

```bash
# Check Dapr status
dapr status -k

# Check sidecar logs
kubectl logs <pod-name> -c daprd -n todo-app

# Reinstall Dapr
dapr uninstall -k && dapr init -k --wait
```

### Kafka (Redpanda) connection refused

```bash
# Check Redpanda pod
kubectl get pods -n todo-app | grep redpanda

# Check Redpanda logs
kubectl logs -n todo-app -l app.kubernetes.io/name=redpanda

# Verify broker address
kubectl exec -n todo-app deployment/redpanda -- rpk cluster info
```

### Ingress not accessible

```bash
# Verify ingress addon
minikube addons enable ingress

# Check ingress
kubectl get ingress -n todo-app

# Use minikube tunnel (alternative to /etc/hosts)
minikube tunnel
```

### Images not found

```bash
# Point Docker to Minikube's daemon
eval $(minikube docker-env)

# Rebuild images
docker build -t hackathon-backend:latest ./backend
docker build -t hackathon-frontend:latest ./frontend

# Verify
docker images | grep hackathon
```

## Production (Oracle OKE)

### Pod CrashLoopBackOff

```bash
# Check pod logs
kubectl logs <pod-name> -n todo-app --previous

# Common causes:
# - Missing secrets: check kubectl get secrets -n todo-app
# - Database unreachable: verify Neon connection string
# - Port conflict: check container port matches service
```

### Strimzi Kafka not ready

```bash
# Check Kafka status
kubectl get kafka -n kafka

# Check operator logs
kubectl logs -n kafka deployment/strimzi-cluster-operator --tail=100

# Check Kafka pod events
kubectl describe pod todo-kafka-kafka-0 -n kafka

# Common: Wait longer. Strimzi can take 5-10 minutes.
```

### TLS certificate not issued

```bash
# Check certificate status
kubectl describe certificate todo-tls -n todo-app

# Check cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager --tail=100

# Check challenges
kubectl get challenges -n todo-app

# Common causes:
# - DNS not pointing to ingress IP
# - HTTP-01 challenge blocked by firewall
# - Rate limit exceeded (use staging issuer for testing)
```

### Resource limits exceeded

```bash
# Check node resources
kubectl top nodes

# Check pod resources
kubectl top pods -n todo-app

# If OOM killed, increase memory limits in values-prod.yaml
```

### Secrets missing

```bash
# List secrets
kubectl get secrets -n todo-app

# Create missing secrets (see infra/prod/secrets-template.yaml)
kubectl create secret generic neon-credentials \
  --from-literal=connection-string='postgresql://...' -n todo-app
```

## CI/CD (GitHub Actions)

### Build fails on arm64

```bash
# Ensure QEMU is set up in workflow
# Check: docker/setup-qemu-action@v3 is present

# Verify base images support arm64:
# python:3.11-slim -> OK
# node:20-alpine -> OK
```

### GHCR push fails

```bash
# Verify workflow permissions:
# Settings > Actions > General > Workflow permissions > Read and write

# Check GITHUB_TOKEN permissions in workflow:
# permissions:
#   packages: write
```

### Deploy fails

```bash
# Verify OKE_KUBECONFIG secret is set and valid
# Re-encode: cat ~/.kube/config | base64 -w 0

# Test locally first:
# export KUBECONFIG=path/to/oke-kubeconfig
# kubectl cluster-info
```

## Useful Commands

```bash
# View all resources
kubectl get all -n todo-app

# Dapr dashboard
dapr dashboard -k

# Port forward for debugging
kubectl port-forward -n todo-app svc/todo-chatbot-backend 8000:8000

# Execute shell in pod
kubectl exec -it -n todo-app deployment/todo-chatbot-backend -- /bin/bash

# View events
kubectl get events -n todo-app --sort-by='.lastTimestamp'

# Resource usage
kubectl top pods -n todo-app
```

# Production Deployment Status

## Deployed Application

| Service | URL | Status |
|---------|-----|--------|
| Frontend | https://todo.todo-app.cloud | ✅ Live |
| Backend API | https://api.todo-app.cloud | ✅ Live |
| API Docs | https://api.todo-app.cloud/docs | ✅ Available |

---

## Infrastructure

| Component | Service | Details |
|-----------|---------|---------|
| **Kubernetes** | Oracle OKE | Region: ap-mumbai-1 |
| **Container Registry** | Oracle OCIR | ap-mumbai-1.ocir.io/ax12345 |
| **Database** | Neon PostgreSQL | ap-southeast-1 |
| **Kafka** | Strimzi | Self-hosted on OKE |
| **Cache** | Redis | Self-hosted on OKE |
| **Service Mesh** | Dapr | v1.16.9 |
| **Ingress** | NGINX | Load Balancer |
| **TLS** | Cert-Manager + Let's Encrypt | Auto-renewal |

---

## Container Images

| Image | Registry | Tag |
|-------|----------|-----|
| todo-chatbot-backend | ap-mumbai-1.ocir.io/ax12345/todo-chatbot-backend | v1.0.0 |
| todo-chatbot-frontend | ap-mumbai-1.ocir.io/ax12345/todo-chatbot-frontend | v1.0.0 |

---

## Kubernetes Resources

```bash
# Namespace
kubectl get namespace todo-app

# Deployments
kubectl get deployments -n todo-app
# NAME                      READY   UP-TO-DATE   AVAILABLE   AGE
# todo-chatbot-backend      1/1     1            1           24h
# todo-chatbot-frontend     1/1     1            1           24h

# Services
kubectl get services -n todo-app
# NAME                     TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)
# todo-chatbot-backend     ClusterIP   10.96.255.61     <none>        8000/TCP
# todo-chatbot-frontend    ClusterIP   10.104.31.239    <none>        3000/TCP
# todo-backend-dapr        ClusterIP   None             <none>        80/TCP,50001/TCP
# todo-frontend-dapr       ClusterIP   None             <none>        80/TCP,50001/TCP

# Ingress
kubectl get ingress -n todo-app
# NAME               CLASS   HOSTS                   ADDRESS         PORTS     AGE
# todo-chatbot       nginx   todo.todo-app.cloud     129.153.45.123  80, 443   24h
# todo-chatbot-api   nginx   api.todo-app.cloud      129.153.45.123  80, 443   24h
```

---

## Dapr Components

| Component | Type | Purpose |
|-----------|------|---------|
| pubsub | pubsub.kafka | Event streaming (task events, reminders) |
| statestore | state.redis | Task state caching |

---

## Event Topics

| Topic | Purpose | Consumers |
|-------|---------|-----------|
| task-events | Task CRUD events | Audit service, Notification service |
| reminders | Scheduled reminders | Reminder processor |
| task-updates | Task state changes | WebSocket updates |

---

## Secrets (Kubernetes)

```bash
kubectl get secrets -n todo-app
# NAME                 TYPE                                  DATA   AGE
# clerk-secrets        Opaque                                4      24h
# neon-credentials     Opaque                                1      24h
# ai-api-keys          Opaque                                4      24h
# todo-tls             kubernetes.io/tls                     2      24h
```

---

## CI/CD Pipeline

Images are built and pushed via GitHub Actions:

```yaml
# .github/workflows/build-and-deploy.yml
on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build and push images
        run: |
          docker buildx build --platform linux/amd64,linux/arm64 \
            -t ap-mumbai-1.ocir.io/ax12345/todo-chatbot-backend:${{ github.sha }} \
            --push ./backend
      - name: Deploy to OKE
        run: |
          helm upgrade --install todo-chatbot ./todo-chatbot \
            --namespace todo-app \
            -f ./todo-chatbot/values-prod.yaml
```

---

## Monitoring

| Tool | URL | Purpose |
|------|-----|---------|
| Dapr Dashboard | http://dapr-dashboard.dapr-system.svc.cluster.local:8080 | Service mesh overview |
| Grafana | https://grafana.todo-app.cloud | Metrics dashboards |
| Prometheus | Internal | Metrics collection |

---

## Rollback Procedure

```bash
# Rollback to previous deployment
kubectl rollout undo deployment/todo-chatbot-backend -n todo-app
kubectl rollout undo deployment/todo-chatbot-frontend -n todo-app

# Or rollback to specific revision
kubectl rollout undo deployment/todo-chatbot-backend -n todo-app --to-revision=2
```

---

## Last Updated

- **Date**: 2026-02-18
- **Deployed By**: DevOps Team
- **Version**: v1.0.0

# Data Model: Kubernetes Deployment (Local & Oracle OKE)

**Feature**: 014-k8s-deployment
**Date**: 2026-02-18
**Type**: Infrastructure Resources (Kubernetes Manifests)

## Overview

This feature defines infrastructure resources rather than traditional data entities. The "data model" describes Kubernetes resources, their relationships, and configuration schemas.

## Resource Hierarchy

```
Namespace: todo-app
├── Deployments
│   ├── todo-chatbot-backend (with Dapr sidecar)
│   └── todo-chatbot-frontend (with Dapr sidecar)
├── Services
│   ├── todo-chatbot-backend (ClusterIP)
│   └── todo-chatbot-frontend (ClusterIP)
├── Ingress
│   └── todo-chatbot-ingress (TLS enabled)
├── Dapr Components
│   ├── pubsub (Kafka)
│   └── statestore (Redis/PostgreSQL)
├── Secrets
│   ├── neon-credentials
│   ├── clerk-secrets
│   └── ai-api-keys
└── Kafka Resources (prod only)
    ├── Kafka (cluster)
    └── KafkaTopic (task-events, reminders)
```

## Resource Definitions

### 1. Container Images

**Entity**: ContainerImage

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| name | string | Image name in registry | `ghcr.io/owner/todo-chatbot-backend` |
| tag | string | Image version | `sha-abc123`, `latest` |
| platforms | array | Supported architectures | `["linux/amd64", "linux/arm64"]` |
| registry | enum | Container registry | `ghcr` |

**Naming Convention**:
```
ghcr.io/{owner}/{app}-{service}:{tag}
```

**Examples**:
- `ghcr.io/madoo/todo-chatbot-backend:sha-abc123`
- `ghcr.io/madoo/todo-chatbot-frontend:latest`

---

### 2. Kubernetes Deployment

**Entity**: Deployment

| Attribute | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| name | string | Deployment name | `todo-chatbot-{backend\|frontend}` |
| replicas | integer | Pod count | `1` (free tier) |
| image | ContainerImage | Container reference | - |
| resources | object | CPU/Memory limits | See Resource Budget |
| dapr.enabled | boolean | Dapr sidecar injection | `true` |
| dapr.appId | string | Dapr application ID | `todo-backend`, `todo-frontend` |
| dapr.appPort | integer | Application port | `8000`, `3000` |

**Dapr Annotations**:
```yaml
annotations:
  dapr.io/enabled: "true"
  dapr.io/app-id: "{{ .Values.dapr.appId }}"
  dapr.io/app-port: "{{ .Values.container.port }}"
  dapr.io/app-protocol: "http"
  dapr.io/config: "appconfig"
  dapr.io/log-level: "info"
```

---

### 3. Dapr Component

**Entity**: DaprComponent

| Attribute | Type | Description | Valid Values |
|-----------|------|-------------|--------------|
| name | string | Component name | `pubsub`, `statestore` |
| type | string | Component type | `pubsub.kafka`, `state.redis`, `state.postgresql` |
| version | string | API version | `v1` |
| metadata | array | Component configuration | Type-specific |

**Pub/Sub Component (Kafka)**:
```yaml
metadata:
  - name: brokers
    value: "kafka-bootstrap:9092"
  - name: authRequired
    value: "false"
  - name: consumerGroup
    value: "todo-app"
  - name: schemaRegistryURL
    value: "http://schema-registry:8081"  # Optional
```

**State Store Component (Redis)**:
```yaml
metadata:
  - name: redisHost
    value: "redis:6379"
  - name: redisPassword
    secretKeyRef:
      name: redis-secrets
      key: password
```

**State Store Component (PostgreSQL)**:
```yaml
metadata:
  - name: connectionString
    secretKeyRef:
      name: neon-credentials
      key: connection-string
```

---

### 4. Kafka Cluster (Strimzi)

**Entity**: Kafka (CRD)

| Attribute | Type | Description | Free Tier Value |
|-----------|------|-------------|-----------------|
| name | string | Cluster name | `todo-kafka` |
| kafka.replicas | integer | Broker count | `1` |
| kafka.storage.type | enum | Storage type | `ephemeral` |
| kafka.resources.limits.cpu | string | CPU limit | `1` |
| kafka.resources.limits.memory | string | Memory limit | `2Gi` |
| zookeeper.replicas | integer | ZK count | `1` |

**Kafka Custom Resource**:
```yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: Kafka
metadata:
  name: todo-kafka
spec:
  kafka:
    replicas: 1
    version: 3.6.0
    listeners:
      - name: plain
        port: 9092
        type: internal
        tls: false
    storage:
      type: ephemeral
    resources:
      limits:
        cpu: "1"
        memory: 2Gi
      requests:
        cpu: "500m"
        memory: 1Gi
  zookeeper:
    replicas: 1
    storage:
      type: ephemeral
```

---

### 5. Kafka Topic

**Entity**: KafkaTopic (CRD)

| Attribute | Type | Description | Value |
|-----------|------|-------------|-------|
| name | string | Topic name | `task-events`, `reminders`, `task-updates` |
| partitions | integer | Partition count | `1` |
| replicationFactor | integer | Replicas | `1` |

**Topics Required**:

| Topic | Purpose | Consumer |
|-------|---------|----------|
| `task-events` | Task CRUD events | Audit, Analytics |
| `reminders` | Reminder triggers | Notification Service |
| `task-updates` | Task state changes | Real-time UI updates |

---

### 6. Ingress Resource

**Entity**: Ingress

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| name | string | Ingress name | `todo-chatbot-ingress` |
| className | string | Ingress class | `nginx` |
| tls.hosts | array | TLS-enabled hosts | `["todo.example.com"]` |
| tls.secretName | string | TLS secret | `todo-tls` |
| rules | array | Routing rules | See below |

**Ingress Configuration**:
```yaml
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - todo.example.com
      secretName: todo-tls
  rules:
    - host: todo.example.com
      http:
        paths:
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: todo-chatbot-backend
                port:
                  number: 8000
          - path: /
            pathType: Prefix
            backend:
              service:
                name: todo-chatbot-frontend
                port:
                  number: 3000
```

---

### 7. Secrets

**Entity**: Secret

| Secret Name | Keys | Purpose |
|-------------|------|---------|
| `neon-credentials` | `connection-string` | PostgreSQL connection |
| `clerk-secrets` | `secret-key`, `issuer` | Clerk authentication |
| `ai-api-keys` | `gemini-key`, `z-ai-key` | AI provider keys |
| `todo-tls` | `tls.crt`, `tls.key` | TLS certificate |

**Secret Schema**:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: neon-credentials
  namespace: todo-app
type: Opaque
stringData:
  connection-string: "postgresql://..."
```

---

## Resource Budget

### Oracle Free Tier Allocation

| Resource | Capacity | Allocated | Headroom |
|----------|----------|-----------|----------|
| CPU (cores) | 4 | ~1.1 | ~2.9 |
| Memory (GB) | 24 | ~2.5 | ~21.5 |

### Per-Component Budget

| Component | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-----------|-------------|-----------|----------------|--------------|
| Backend | 100m | 500m | 256Mi | 512Mi |
| Frontend | 100m | 500m | 256Mi | 512Mi |
| Dapr Sidecar | 50m | 200m | 64Mi | 128Mi |
| Kafka (Strimzi) | 500m | 1000m | 1Gi | 2Gi |
| Zookeeper | 100m | 200m | 256Mi | 512Mi |
| NGINX Ingress | 50m | 200m | 64Mi | 128Mi |
| Cert-Manager | 25m | 100m | 32Mi | 64Mi |

---

## State Transitions

### Deployment Rollout

```
[New Image] -> [Build] -> [Push to GHCR] -> [Helm Upgrade]
                                                    |
                                                    v
[Pending] -> [ContainerCreating] -> [Running] -> [Ready]
                                                    |
                                                    v
[Service Updated] -> [Ingress Routes Traffic] -> [Live]
```

### Kafka Cluster Lifecycle

```
[Strimzi Operator Installed]
            |
            v
[Kafka CR Created] -> [Zookeeper Pod] -> [Kafka Pod] -> [Ready]
                                                            |
                                                            v
[KafkaTopic CRs Created] -> [Topics Ready] -> [Apps Connect]
```

---

## Validation Rules

### Container Images
- MUST support both `linux/amd64` and `linux/arm64`
- MUST be pushed to GHCR before deployment
- MUST include health check endpoint

### Deployments
- MUST have resource limits defined
- MUST have liveness and readiness probes
- MUST have Dapr annotations if using Dapr

### Dapr Components
- MUST reference valid Kafka/Redis/PostgreSQL endpoints
- MUST use secretKeyRef for sensitive values (not plaintext)
- MUST be applied before application deployment

### Secrets
- MUST be created before deployment
- MUST be in same namespace as consumers
- MUST NOT be committed to git (use sealed-secrets or external-secrets in production)

---

## Environment Differences

| Aspect | Local (Minikube) | Production (OKE) |
|--------|------------------|------------------|
| Kafka | Redpanda (single pod) | Strimzi (operator) |
| State Store | Redis (in-cluster) | Neon PostgreSQL (external) |
| Ingress | NodePort / minikube tunnel | NGINX + TLS |
| Image Registry | Local / minikube cache | GHCR |
| Secrets | Plain K8s secrets | External Secrets (recommended) |
| Storage | Ephemeral | Ephemeral (free tier) |

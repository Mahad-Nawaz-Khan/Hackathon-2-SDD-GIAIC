# Implementation Plan: Kubernetes Deployment (Local & Oracle OKE)

**Branch**: `014-k8s-deployment` | **Date**: 2026-02-18 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/014-k8s-deployment/spec.md`

## Summary

Deploy the Event-Driven Todo Chatbot to Kubernetes clusters - locally via Minikube for development/validation, and to Oracle OKE for production. The implementation follows a Dapr-first architecture where all infrastructure concerns (Pub/Sub, State, Jobs, Secrets) are abstracted through Dapr sidecars. Key deliverables include: multi-arch Docker images (amd64/arm64 for Oracle ARM Free Tier), Dapr-integrated Helm charts, local Redpanda/Strimzi Kafka, and automated CI/CD via GitHub Actions.

## Technical Context

**Language/Version**: Python 3.11 (Backend), Node.js 20 (Frontend)
**Primary Dependencies**: FastAPI, Next.js 16, Dapr SDK, SQLModel, OpenAI Agents SDK
**Storage**: PostgreSQL (Neon - external managed service), Redis (local state), Kafka (Redpanda/Strimzi)
**Testing**: pytest (Backend), Jest/Vitest (Frontend), Helm test hooks
**Target Platform**: Kubernetes (Minikube local, Oracle OKE cloud)
**Project Type**: web (Frontend + Backend microservices)
**Performance Goals**: <200ms p95 API latency, 10-minute CI/CD pipeline, zero-downtime rolling updates
**Constraints**: Oracle Free Tier (4 OCPUs ARM, 24GB RAM), Dapr-first architecture, no app code changes
**Scale/Scope**: Single-tenant MVP, 1-10 concurrent users initially

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| **Event-Driven First** | PASS | Using Kafka via Dapr for all events (task-created, reminder-triggered) |
| **Infrastructure Abstraction** | PASS | All infrastructure through Dapr (pubsub, statestore, jobs, secrets) |
| **Cloud-Native Architecture** | PASS | Containerized services, Helm-managed, Dapr sidecars |
| **Automation & IaC** | PASS | GitHub Actions for build/deploy, no manual kubectl |

**No violations detected.** Architecture fully aligns with constitution.

## Project Structure

### Documentation (this feature)

```text
specs/014-k8s-deployment/
├── spec.md              # Feature specification (completed)
├── plan.md              # This file
├── research.md          # Phase 0 research findings
├── data-model.md        # Infrastructure data model
├── quickstart.md        # Deployment guide
├── contracts/           # API contracts (inherited from existing)
└── tasks.md             # Implementation tasks
```

### Source Code (repository root)

```text
# Existing structure (no modifications to app code per constraints)
backend/
├── src/
│   ├── models/
│   ├── services/
│   ├── api/
│   └── main.py
├── tests/
├── Dockerfile           # EXISTS - needs multi-arch updates
└── requirements.txt

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
├── tests/
├── Dockerfile           # EXISTS - needs multi-arch updates
├── next.config.ts
└── package.json

# NEW: Infrastructure directory
infra/
├── local/
│   ├── redpanda/
│   │   └── values.yaml          # Helm values for Redpanda (local Kafka)
│   ├── redis/
│   │   └── values.yaml          # Helm values for Redis (local state)
│   └── dapr/
│       └── components/
│           ├── pubsub.yaml      # Kafka pub/sub component
│           └── statestore.yaml  # Redis state store component
├── prod/
│   ├── strimzi/
│   │   ├── kafka-cluster.yaml   # Kafka CR for Strimzi (1 replica, ephemeral)
│   │   └── kafka-topic.yaml     # Topics: task-events, reminders
│   ├── cert-manager/
│   │   ├── issuer.yaml          # Let's Encrypt ClusterIssuer
│   │   └── certificate.yaml     # TLS certificate definition
│   └── dapr/
│       └── components/
│           ├── pubsub.yaml      # Strimzi Kafka pub/sub
│           └── statestore.yaml  # Neon PostgreSQL state store
└── nginx-ingress/
    └── values.yaml              # NGINX ingress controller config

# EXISTING: Helm chart (needs Dapr annotations)
todo-chatbot/
├── Chart.yaml
├── values.yaml
├── values-local.yaml            # NEW: Local environment overrides
├── values-prod.yaml             # NEW: Production environment overrides
└── templates/
    ├── deployment.yaml          # NEEDS: Dapr annotations
    ├── service.yaml
    ├── ingress.yaml             # NEEDS: TLS configuration
    └── ...

# NEW: GitHub Actions
.github/
└── workflows/
    ├── build.yaml               # Multi-arch build + GHCR push
    └── deploy.yaml              # Helm deployment to OKE
```

**Structure Decision**: Augmenting existing `todo-chatbot/` Helm chart with Dapr annotations and environment-specific values. Creating new `infra/` directory for infrastructure components (Kafka, Redis, Dapr, Ingress) following separation of concerns principle.

## Phase 0: Research Findings

### Decision 1: Multi-Arch Docker Builds

**Decision**: Use `docker buildx` for multi-platform images (`linux/amd64`, `linux/arm64`)

**Rationale**: Oracle Cloud Free Tier uses Ampere ARM processors. Images MUST be arm64-compatible to run on OKE. Developers on x86 machines (Intel/AMD) need amd64 images for local Minikube testing.

**Alternatives Considered**:
- Single arch images: REJECTED - Would require separate image builds or emulation
- QEMU emulation: REJECTED - Unacceptable performance penalty (10x slower)
- Cloud-native builds (Cloud Build, CodeBuild): REJECTED - Adds external dependency, GitHub Actions is already available

**Implementation**:
```yaml
# .github/workflows/build.yaml
- name: Set up Docker Buildx
  uses: docker/setup-buildx-action@v3

- name: Build and push
  uses: docker/build-push-action@v5
  with:
    platforms: linux/amd64,linux/arm64
    push: true
    tags: ghcr.io/${{ github.repository }}/${{ matrix.service }}:${{ github.sha }}
```

### Decision 2: Component Separation (local vs prod)

**Decision**: Maintain separate `components/local/` and `components/prod/` directories for Dapr component YAMLs

**Rationale**:
1. **Security**: Prevents accidental leakage of production hostnames/secrets into local development
2. **Clarity**: Explicit environment mapping reduces configuration errors
3. **Simplicity**: Helm templating adds complexity; separate files are easier to understand and debug

**Alternatives Considered**:
- Single parameterized component file: REJECTED - Risk of production secrets in git history, harder to audit
- Environment variables in components: REJECTED - Dapr components don't natively support env var interpolation in all fields

### Decision 3: Strimzi vs Managed Kafka

**Decision**: Use Self-Hosted Strimzi Kafka Operator on Oracle OKE

**Rationale**:
| Factor | Strimzi (Self-Hosted) | Confluent/Managed Kafka |
|--------|----------------------|------------------------|
| Cost | Free (within OKE limits) | $0.50/hour minimum = $365/month |
| Control | Full control over config | Limited to provider options |
| Free Tier Fit | Fits within 4 OCPU/24GB | Exceeds free tier budget |
| Operations | Higher ops burden | Managed for you |

**For Oracle Free Tier, Strimzi is the ONLY viable option.**

**Configuration for Free Tier**:
```yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: Kafka
metadata:
  name: todo-kafka
spec:
  kafka:
    replicas: 1  # Single replica (no HA)
    resources:
      limits:
        cpu: "1"
        memory: 2Gi
      requests:
        cpu: "500m"
        memory: 1Gi
    storage:
      type: ephemeral  # No persistent volumes (free tier)
  zookeeper:
    replicas: 1
    storage:
      type: ephemeral
```

### Decision 4: Local Kafka - Redpanda vs Strimzi

**Decision**: Use Redpanda for local Minikube (simpler setup), Strimzi for production (Kubernetes-native)

**Rationale**: Redpanda offers single-binary deployment ideal for local development, while Strimzi provides production-grade Kafka operator for cloud.

### Decision 5: State Store Strategy

**Decision**:
- **Local**: Redis (ephemeral, fast, simple)
- **Production**: PostgreSQL via Neon (existing managed service, already configured)

**Rationale**: Using the existing Neon PostgreSQL connection for state store aligns with current architecture. Redis for local keeps resource footprint minimal.

## Phase 1: Design & Contracts

### Data Model

Infrastructure resources are defined as Kubernetes manifests:

```yaml
# Key entities managed by this feature
ContainerImage:
  name: string           # ghcr.io/owner/todo-chatbot-{backend|frontend}
  platforms: [amd64, arm64]
  registry: GHCR

DaprComponent:
  name: string           # pubsub, statestore
  type: string           # pubsub.kafka, state.redis, state.postgresql
  version: string        # v1
  metadata:
    - name: string
      value: string|secretKeyRef

KafkaCluster:
  name: string           # todo-kafka
  replicas: integer      # 1 (free tier)
  storage: ephemeral

KafkaTopic:
  name: string           # task-events, reminders, task-updates
  partitions: integer    # 1
  replicationFactor: 1   # No HA in free tier
```

### Contracts

#### Dapr Component Contract: Pub/Sub (Local)

```yaml
# infra/local/dapr/components/pubsub.yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: pubsub
spec:
  type: pubsub.kafka
  version: v1
  metadata:
    - name: brokers
      value: "redpanda:9092"
    - name: authRequired
      value: "false"
    - name: schemaRegistryURL
      value: "http://redpanda:8081"
```

#### Dapr Component Contract: State Store (Local)

```yaml
# infra/local/dapr/components/statestore.yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: statestore
spec:
  type: state.redis
  version: v1
  metadata:
    - name: redisHost
      value: "redis:6379"
```

#### Dapr Component Contract: Pub/Sub (Production)

```yaml
# infra/prod/dapr/components/pubsub.yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: pubsub
spec:
  type: pubsub.kafka
  version: v1
  metadata:
    - name: brokers
      value: "todo-kafka-kafka-bootstrap:9092"
    - name: authRequired
      value: "false"
    - name: schemaRegistryURL
      value: ""  # Optional in production
```

#### Dapr Component Contract: State Store (Production)

```yaml
# infra/prod/dapr/components/statestore.yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: statestore
spec:
  type: state.postgresql
  version: v1
  metadata:
    - name: connectionString
      secretKeyRef:
        name: neon-credentials
        key: connection-string
```

### Deployment Template with Dapr Annotations

```yaml
# Updated deployment.yaml with Dapr sidecar
apiVersion: apps/v1
kind: Deployment
metadata:
  name: todo-chatbot-backend
spec:
  template:
    metadata:
      annotations:
        dapr.io/enabled: "true"
        dapr.io/app-id: "todo-backend"
        dapr.io/app-port: "8000"
        dapr.io/app-protocol: "http"
        dapr.io/config: "appconfig"
        dapr.io/log-level: "info"
    spec:
      containers:
        - name: backend
          # ... existing config
```

### Resource Budget (Oracle Free Tier)

| Component | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-----------|-------------|-----------|----------------|--------------|
| Backend | 100m | 500m | 256Mi | 512Mi |
| Frontend | 100m | 500m | 256Mi | 512Mi |
| Dapr Sidecar (x2) | 50m each | 200m each | 64Mi each | 128Mi each |
| Strimzi Kafka | 500m | 1 | 1Gi | 2Gi |
| Strimzi Zookeeper | 100m | 200m | 256Mi | 512Mi |
| NGINX Ingress | 50m | 200m | 64Mi | 128Mi |
| Cert-Manager | 25m | 100m | 32Mi | 64Mi |
| **Total** | ~1.1 cores | ~3.2 cores | ~2.5Gi | ~5Gi |

**Free Tier Capacity**: 4 OCPUs ARM, 24GB RAM
**Headroom**: ~2.9 cores, ~19GB RAM available

### CI/CD Pipeline Design

```yaml
# .github/workflows/build.yaml
name: Build Multi-Arch Images

on:
  push:
    branches: [main]
    paths:
      - 'backend/**'
      - 'frontend/**'

jobs:
  build:
    strategy:
      matrix:
        service: [backend, frontend]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: ./${{ matrix.service }}
          platforms: linux/amd64,linux/arm64
          push: true
          tags: |
            ghcr.io/${{ github.repository }}/${{ matrix.service }}:${{ github.sha }}
            ghcr.io/${{ github.repository }}/${{ matrix.service }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

```yaml
# .github/workflows/deploy.yaml
name: Deploy to OKE

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install kubectl
        uses: azure/setup-kubectl@v3

      - name: Install Helm
        uses: azure/setup-helm@v3

      - name: Configure kubeconfig
        run: |
          echo "${{ secrets.OKE_KUBECONFIG }}" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig

      - name: Deploy with Helm
        run: |
          helm upgrade --install todo-chatbot ./todo-chatbot \
            -f ./todo-chatbot/values-prod.yaml \
            --set backend.image.tag=${{ github.sha }} \
            --set frontend.image.tag=${{ github.sha }} \
            --namespace todo-app --create-namespace
```

## Architecture Decision Records

The following decisions are architecturally significant and should be documented:

1. **ADR-001: Multi-Arch Docker Builds** - Required for Oracle ARM Free Tier compatibility
2. **ADR-002: Component Directory Separation** - Security boundary between local and prod
3. **ADR-003: Strimzi Kafka on OKE** - Cost-effective Kafka for Free Tier

Run `/sp.adr <title>` to document these decisions.

## Quickstart Commands

```bash
# Local Development (Minikube)
minikube start --cpus=4 --memory=8192
dapr init -k

# Deploy infrastructure
helm repo add redpanda https://charts.redpanda.com
helm install redpanda redpanda/redpanda -f infra/local/redpanda/values.yaml
helm install redis bitnami/redis -f infra/local/redis/values.yaml

# Deploy Dapr components
kubectl apply -f infra/local/dapr/components/

# Deploy application
helm install todo-chatbot ./todo-chatbot -f todo-chatbot/values-local.yaml

# Production Deployment (OKE)
helm repo add strimzi https://strimzi.io/charts/
helm install strimzi strimzi/strimzi-kafka-operator
kubectl apply -f infra/prod/strimzi/
kubectl apply -f infra/prod/dapr/components/
helm install todo-chatbot ./todo-chatbot -f todo-chatbot/values-prod.yaml
```

## Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| Strimzi exceeds Free Tier limits | Critical | Resource quotas enforced; single replica with ephemeral storage |
| Multi-arch build failures | High | CI pipeline tests both platforms; fallback to emulation |
| Dapr sidecar memory leak | Medium | Resource limits; liveness probes; pod restart policies |
| TLS certificate provisioning fails | Medium | Use HTTP-01 challenge; pre-provision staging cert |
| GitHub Actions timeout | Low | Optimize build caching; parallel job execution |

## Follow-ups

1. Document ADRs for the three key architectural decisions
2. Create detailed `tasks.md` with implementation steps
3. Set up Oracle OKE cluster and configure kubeconfig secrets
4. Configure DNS for production domain pointing to OKE ingress

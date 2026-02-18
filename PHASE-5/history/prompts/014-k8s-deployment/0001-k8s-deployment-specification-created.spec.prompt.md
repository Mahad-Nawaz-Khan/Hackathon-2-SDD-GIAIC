---
id: 0001
title: K8s deployment specification created
stage: spec
date: 2026-02-17
surface: agent
model: claude-opus-4-6
feature: 014-k8s-deployment
branch: 014-k8s-deployment
user: madoo
command: /sp.specify
labels: ["specification","kubernetes","deployment"]
links:
  spec: specs/014-k8s-deployment/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/014-k8s-deployment/spec.md
  - specs/014-k8s-deployment/checklists/requirements.md
tests:
  - Specification quality validation (PASSED)
---

## Prompt

Phase V: Local & Cloud Deployment (Minikube -> Oracle OKE)

## Intent
Deploy the completed Event-Driven Todo App (Frontend + Backend + Dapr) to a local Kubernetes cluster (Minikube) for validation, and then to a production-grade Oracle Kubernetes Engine (OKE) cluster using CI/CD.

## Scope & Features

### 1. Containerization (Docker)
- **Artifacts**: Create optimized `Dockerfile`s for Frontend (Next.js standalone) and Backend (FastAPI).
- **Multi-Arch**: Configure builds for both `amd64` (Local) and `arm64` (Oracle Ampere Free Tier).
- **Registry**: Push images to GitHub Container Registry (GHCR).

### 2. Part B: Local Deployment (Minikube)
- **Infrastructure**:
  - Enable Dapr on Minikube (`dapr init -k`).
  - Deploy **Redpanda** (single node) or **Strimzi** for local Kafka.
  - Deploy **Redis** for local State Store.
- **Dapr Components**:
  - Create `components/local/*.yaml` configured for local internal DNS (e.g., `kafka.default.svc.cluster.local`).
- **Manifests**: Create a Helm Chart `todo-app` that deploys Frontend and Backend with Dapr sidecar annotations.

### 3. Part C: Cloud Deployment (Oracle OKE)
- **Infrastructure**:
  - Setup OKE Cluster (Free Tier compatible).
  - Install **Nginx Ingress Controller** and **Cert-Manager** (Let's Encrypt).
  - Deploy **Strimzi Kafka Operator** with a `Kafka` Custom Resource optimized for free tier (1 replica, ephemeral storage).
- **Dapr Components**:
  - Create `components/prod/*.yaml` configured for the Cloud Kafka/Postgres services.
- **CI/CD (GitHub Actions)**:
  - Workflow `deploy.yaml`:
    1. Lint/Test.
    2. Build & Push Docker Images (Multi-arch).
    3. Update OKE Cluster (via `kubectl apply` or Helm Upgrade).

## Constraints
- **Oracle Free Tier**: strictly adhere to resource limits (4 OCPUs ARM, 24GB RAM).
- **Dapr-First**: All service-to-service and event-bus communication MUST go through Dapr sidecars.
- **No Azure/GCP**: Exclude AKS/GKE configurations.
- **No App Logic**: Do not modify Python/TypeScript code; only add Dockerfiles, YAML manifests, and CI/CD configs.

## Success Criteria (SMART)
- ✅ **Local**: `kubectl get pods` on Minikube shows healthy pods; Dapr dashboard shows apps connected to Redis/Redpanda.
- ✅ **Cloud**: Public URL (HTTPS) on Oracle Cloud loads the app.
- ✅ **Event Flow**: Creating a recurring task in the Cloud UI triggers a Kafka event (verified in logs) via Strimzi.
- ✅ **CI/CD**: A push to `main` automatically updates the running version on OKE within 10 minutes.

## Non-Goals
- implementing the application logic (Advanced/Intermediate features are assumed complete).
- Setting up complex observability stacks beyond basic Dapr logging.

## Response snapshot

Created comprehensive specification for Kubernetes deployment with 4 prioritized user stories:

**User Stories**:
1. **P1 - Containerize Applications**: Multi-arch Dockerfiles (amd64/arm64) for Frontend/Backend, push to GHCR
2. **P2 - Local Minikube Deployment**: Deploy full stack with Dapr, Kafka (Redpanda/Strimzi), Redis
3. **P3 - Oracle OKE Cloud Deployment**: Nginx ingress, Cert-Manager TLS, Strimzi Kafka within free tier limits
4. **P4 - Automated CI/CD**: GitHub Actions pipeline for build-test-deploy under 10 minutes

**Requirements**: 22 functional requirements covering Dockerfiles, Kubernetes manifests, Dapr components (local/prod), Strimzi Kafka, ingress/Cert-Manager, and GitHub Actions workflow.

**Success Criteria**: 8 measurable outcomes covering build times, deployment health checks, event flow verification, and CI/CD performance.

## Outcome

- Impact: Clear roadmap for containerizing and deploying the Event-Driven Todo App to both local (Minikube) and production (Oracle OKE) Kubernetes environments
- Tests: Specification quality validation PASSED
- Files: 2 files created (spec.md, requirements checklist)
- Next prompts: `/sp.plan` to create architecture and implementation plan
- Reflection: Specification balances comprehensive deployment needs with Oracle Free Tier constraints; Dapr-first architecture enforced throughout

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results: Specification quality validation PASSED
- Prompt variant: N/A
- Next experiment: Architecture planning should focus on separating local vs prod configurations while maintaining Dapr abstraction

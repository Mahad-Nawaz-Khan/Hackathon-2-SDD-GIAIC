---
id: 0002
title: K8s deployment plan created
stage: plan
date: 2026-02-18
surface: agent
model: claude-opus-4-6
feature: 014-k8s-deployment
branch: 014-k8s-deployment
user: madoo
command: /sp.plan
labels: ["architecture","kubernetes","dapr","oci"]
links:
  spec: specs/014-k8s-deployment/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/014-k8s-deployment/plan.md
  - specs/014-k8s-deployment/research.md
  - specs/014-k8s-deployment/data-model.md
  - specs/014-k8s-deployment/quickstart.md
tests:
  - Constitution check PASSED (all 4 principles aligned)
---

## Prompt

Architecture:
- **Pattern**: Cloud-Native Event-Driven Microservices on Kubernetes.
- **Runtime**: Dapr (Distributed Application Runtime) for abstracting Kafka/State interactions.
- **Infrastructure**:
  - **Local**: Minikube + Redpanda (Kafka) + Redis (State) + Dapr.
  - **Cloud**: Oracle OKE + Strimzi (Kafka) + Neon (External State) + Dapr.

Phases:
1. **Container Strategy**:
   - Set up `docker buildx` to generate multi-arch images (`linux/amd64`, `linux/arm64`). This is non-negotiable for Oracle's ARM-based Free Tier.
   - Configure GitHub Container Registry (GHCR) access.

2. **Local Environment (Minikube)**:
   - Create `infra/local` folder.
   - Define Helm values for Redpanda and Redis.
   - Define Dapr Component manifests (`pubsub.yaml`, `statestore.yaml`) pointing to local services.
   - Create the `todo-app` Helm Chart to deploy Frontend+Backend with Dapr sidecar annotations.

3. **Cloud Environment (Oracle OKE)**:
   - Create `infra/prod` folder.
   - Define Strimzi Kafka Cluster manifest (optimized for low resources: 1 replica, ephemeral).
   - Define Production Dapr Components pointing to Strimzi (Kafka) and Neon (Postgres).
   - Set up NGINX Ingress Controller for public access.

4. **CI/CD Pipeline**:
   - Design GitHub Actions workflow:
     - `build.yaml`: Builds and pushes multi-arch images on push to main.
     - `deploy.yaml`: Deploys the Helm chart to OKE using the new image tag.

Design Decisions (ADRs) to Document:
- **Multi-Arch Builds**: Why we MUST use `docker buildx` (Oracle ARM instances).
- **Component Separation**: Why we maintain separate `components/local` and `components/prod` directories instead of one parameterized file (avoids leaking local secrets/hostnames into prod).
- **Strimzi vs Managed**: Why we use Self-Hosted Strimzi on OKE (Cost: Free) vs Managed Kafka (Cost: $$).

Dependencies:
- `helm` for package management.
- `dapr-cli` for initializing the cluster.
- `kubectl` for applying manifests.
- `docker` with `buildx` plugin.

## Response snapshot

Created comprehensive implementation plan for Kubernetes deployment with 4 phases:

**Technical Context**:
- Python 3.11 Backend, Node.js 20 Frontend
- Target: Kubernetes (Minikube local, Oracle OKE cloud)
- Constraints: Oracle Free Tier (4 OCPUs ARM, 24GB RAM)

**Constitution Check**: All 4 principles PASS
- Event-Driven First: Kafka via Dapr
- Infrastructure Abstraction: All infra through Dapr
- Cloud-Native Architecture: Containerized, Helm-managed, Dapr sidecars
- Automation & IaC: GitHub Actions for build/deploy

**Phase 0 Research**:
1. Multi-Arch Docker Builds: docker buildx for amd64/arm64
2. Dapr Resources: 50m/200m CPU, 64Mi/128Mi memory
3. Strimzi Kafka: Fits free tier with 1 replica, ephemeral storage
4. Local Kafka: Redpanda for simplicity
5. Component Separation: Security boundary between local/prod
6. NGINX Ingress: With Cert-Manager for TLS
7. CI/CD Auth: kubeconfig as GitHub secret
8. Health Checks: Both probes + graceful shutdown

**Deliverables Created**:
- plan.md: Full implementation plan
- research.md: Phase 0 findings
- data-model.md: Infrastructure resource definitions
- quickstart.md: Step-by-step deployment guide

**ADR Suggestions**:
- ADR-001: Multi-Arch Docker Builds
- ADR-002: Component Directory Separation
- ADR-003: Strimzi Kafka on OKE

## Outcome

- Impact: Complete architecture plan for local and cloud Kubernetes deployment with Dapr-first design
- Tests: Constitution check PASSED (all 4 principles aligned)
- Files: 4 files created (plan.md, research.md, data-model.md, quickstart.md)
- Next prompts: `/sp.tasks` to create implementation tasks, `/sp.adr` to document architectural decisions
- Reflection: Plan balances Oracle Free Tier constraints with production-ready architecture; resource budget calculated with headroom

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results: Constitution check PASSED
- Prompt variant: N/A
- Next experiment: Task generation should focus on incremental deliverables (P1-P4 from spec)

---
id: 001
title: constitution-initial
stage: constitution
date: 2026-02-16
surface: agent
model: claude-opus-4-6
feature: none
branch: master
user: madoo
command: /sp.constitution
labels: ["constitution","setup"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
  - N/A (constitution setup)
---

## Prompt

Project: Phase V Advanced Cloud-Native Todo Chatbot (Event-Driven & Distributed)

Core Principles:
- **Event-Driven First**: Decouple services using asynchronous events (Kafka) for side effects (reminders, audit logs, recurring tasks).
- **Infrastructure Abstraction**: Use Dapr for all infrastructure plumbing. Application code must NOT couple directly to Kafka, Redis, or Secret APIs.
- **Cloud-Native**: All components must be containerized, managed via Helm, and deployable to any Kubernetes cluster (Minikube/AKS/GKE).
- **Automation**: Zero manual deployment. Infrastructure changes must be defined in Code (Helm/YAML) and deployed via CI/CD.

Key Standards:
- **Architecture**: Distributed Microservices pattern using Sidecar architecture (Dapr).
- **Tech Stack**:
  - Frontend: Next.js + ChatKit + Dapr Sidecar.
  - Backend: FastAPI + AI Agents SDK + Dapr Sidecar.
  - Runtime: Dapr (Pub/Sub, State, Secrets, Jobs/Bindings).
  - Event Bus: Kafka (Redpanda/Strimzi).
  - DB: Neon (PostgreSQL) via Dapr State Store & Direct SQLModel where complex queries are needed.
- **Dapr Implementation**:
  - Pub/Sub: Use `pubsub.kafka` component for `task-events`, `reminders`, `task-updates`.
  - State: Use `state.postgresql` for conversation history.
  - Jobs: Use Dapr Jobs API for precise reminder scheduling (replacing Cron polling).
  - Secrets: Use `secretstores.kubernetes` for API keys.
- **Data Integrity**: All Kafka events must adhere to strict schemas (Task Event, Reminder Event) defined in the Hackathon spec.

Constraints:
- **No Direct Drivers**: Do not use `kafka-python` or direct `boto3` style clients. Communication must go through Dapr HTTP/gRPC APIs.
- **Dual Compatibility**: Configuration must support switching between Local (Minikube/Redpanda) and Cloud (AKS/GKE/Confluent) via Helm values, without code changes.
- **CI/CD**: GitHub Actions must be used for testing, building Docker images, and deploying to K8s.

Success Criteria:
- **Feature Completeness**: Recurring tasks and Reminders function end-to-end via Kafka events.
- **Deployment**: `kubectl get pods` shows all services (Frontend, Backend, Dapr Sidecars, Kafka, Redis/Zipkin) Running and Healthy on both Minikube and Cloud.
- **Observability**: Distributed tracing (Zipkin/Dapr Dashboard) clearly shows the flow of a request from Chat -> Backend -> Kafka -> Consumer.
- **Resilience**: Services auto-recover/retry connections via Dapr resilience policies.

## Response snapshot

Created constitution v1.0.0 for Phase V Advanced Cloud-Native Todo Chatbot with 4 core principles:

1. **Event-Driven First**: Kafka for side effects with strict schema requirements
2. **Infrastructure Abstraction**: Dapr for all infrastructure (no direct Kafka/Redis clients)
3. **Cloud-Native Architecture**: Containerized with Helm, deployable to any K8s cluster
4. **Automation & IaC**: Zero manual deployment via GitHub Actions

The constitution includes detailed technology standards, Dapr component specifications, development constraints (no direct drivers, dual environment support), and success criteria covering deployment, observability, and resilience.

Template validation completed - all existing templates (plan, spec, tasks) remain compatible with the new constitution.

## Outcome

- Impact: Project constitution established with 4 core principles and comprehensive governance framework
- Tests: N/A (constitution setup)
- Files: `.specify/memory/constitution.md` created with Sync Impact Report
- Next prompts: Consider `/sp.plan` for feature architecture planning using constitution guidance
- Reflection: Constitution successfully captures event-driven microservices architecture with Dapr abstraction as primary constraint

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results: N/A
- Prompt variant: N/A
- Next experiment: Validate constitution alignment with first feature specification

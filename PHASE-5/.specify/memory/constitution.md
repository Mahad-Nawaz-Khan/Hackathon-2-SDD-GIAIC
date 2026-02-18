<!--
=============================================================================
SYNC IMPACT REPORT
=============================================================================
Version change: initial -> 1.0.0

Constitution Created: Phase V Advanced Cloud-Native Todo Chatbot

Principles Added:
- Event-Driven First
- Infrastructure Abstraction
- Cloud-Native Architecture
- Automation & IaC

Sections Added:
- Core Principles (4 principles)
- Technology Standards
- Architecture Constraints
- Development Constraints
- Success Criteria
- Governance

Templates Status:
- plan-template.md: Compatible (no changes needed)
- spec-template.md: Compatible (no changes needed)
- tasks-template.md: Compatible (no changes needed)

Follow-up TODOs: None
=============================================================================
-->

# Phase V Todo Chatbot Constitution

## Core Principles

### Event-Driven First

Decouple all services using asynchronous events for side effects including reminders, audit logs, and recurring tasks. All cross-service communication MUST use event-driven patterns rather than synchronous calls where feasible.

**Rationale**: Event-driven architecture enables loose coupling between services, allowing independent deployment and scaling. Critical for distributed systems reliability.

**Non-Negotiable Rules**:
- Use Kafka event bus for all side effects (reminders, audit, recurring tasks)
- Events MUST adhere to strict schemas (Task Event, Reminder Event)
- Producers MUST NOT depend on consumer implementation
- Consumers MUST be idempotent

### Infrastructure Abstraction

Use Dapr for all infrastructure plumbing. Application code MUST NOT couple directly to Kafka, Redis, Secret APIs, or other infrastructure concerns.

**Rationale**: Dapr provides a consistent abstraction layer enabling portability across cloud providers and reducing vendor lock-in.

**Non-Negotiable Rules**:
- All Pub/Sub via `pubsub.kafka` component (Dapr HTTP/gRPC APIs)
- All state via `state.postgresql` component (conversation history)
- All scheduled jobs via Dapr Jobs API (reminder scheduling)
- All secrets via `secretstores.kubernetes`
- NO direct `kafka-python`, `boto3`, `redis-py` style clients in application code

### Cloud-Native Architecture

All components MUST be containerized, managed via Helm, and deployable to any Kubernetes cluster (Minikube/AKS/GKE).

**Rationale**: Kubernetes provides consistent runtime across environments, enabling development-to-production parity.

**Non-Negotiable Rules**:
- Every service containerized with explicit Dockerfile
- Helm charts for all deployments (local and cloud)
- Dapr sidecar injection for all microservices
- Configuration via Helm values (no hardcoded environments)
- Support for both Minikube/Redpanda (local) and AKS/GKE/Confluent (cloud)

### Automation & Infrastructure as Code

Zero manual deployment. All infrastructure changes MUST be defined in Code (Helm/YAML) and deployed via CI/CD.

**Rationale**: Automation reduces human error, ensures repeatability, and enables rapid iteration.

**Non-Negotiable Rules**:
- All infrastructure changes committed to Git
- GitHub Actions for CI/CD pipeline
- Automated testing, building, and deployment
- No manual `kubectl apply` or `helm install` workflows

## Technology Standards

### Architecture Pattern

Distributed Microservices with Sidecar architecture (Dapr). Each service runs with a Dapr sidecar handling infrastructure concerns.

### Tech Stack

**Frontend**:
- Next.js (App Router)
- ChatKit for conversational interface
- Dapr Sidecar for backend communication

**Backend**:
- FastAPI for REST APIs
- OpenAI Agents SDK for AI processing
- Dapr Sidecar for infrastructure abstraction

**Runtime & Infrastructure**:
- Dapr for Pub/Sub, State, Secrets, Jobs/Bindings
- Kafka (Redpanda/Strimzi) for event bus
- Neon PostgreSQL (via Dapr State Store)
- Direct SQLModel for complex queries

### Dapr Component Configuration

| Component | Dapr Implementation | Purpose |
|-----------|---------------------|---------|
| Pub/Sub | `pubsub.kafka` | `task-events`, `reminders`, `task-updates` topics |
| State | `state.postgresql` | Conversation history persistence |
| Jobs | Dapr Jobs API | Precise reminder scheduling (replaces Cron polling) |
| Secrets | `secretstores.kubernetes` | API keys and sensitive configuration |

### Data Integrity

All Kafka events MUST adhere to strict schemas defined in the Hackathon spec:
- Task Event Schema
- Reminder Event Schema

Schema validation MUST occur at event publication boundaries.

## Development Constraints

### No Direct Infrastructure Drivers

Application code MUST communicate through Dapr HTTP/gRPC APIs only. Forbidden:
- `kafka-python` or similar direct Kafka clients
- Direct Redis clients (`redis-py`)
- Direct cloud SDK calls for infrastructure (`boto3`, `google-cloud-storage`)
- Direct secret management API calls

### Dual Environment Compatibility

Configuration MUST support switching between Local and Cloud via Helm values only:
- **Local**: Minikube + Redpanda
- **Cloud**: AKS/GKE + Confluent

NO code changes required for environment switching.

### CI/CD Requirements

GitHub Actions MUST handle:
1. Running tests (unit, integration)
2. Building Docker images
3. Deploying to Kubernetes (Helm)

## Success Criteria

### Feature Completeness
- Recurring tasks function end-to-end via Kafka events
- Reminders trigger correctly via Dapr Jobs API and Kafka events
- All event schemas validated and documented

### Deployment
- `kubectl get pods` shows all services Running and Healthy on Minikube
- Same deployment works on AKS/GKE without code changes
- Services include: Frontend, Backend, Dapr Sidecars, Kafka, Zipkin

### Observability
- Distributed tracing (Zipkin/Dapr Dashboard) shows request flow
- Trace path visible: Chat -> Backend -> Kafka -> Consumer
- All services emit structured logs with correlation IDs

### Resilience
- Services auto-recover from connection failures
- Dapr resilience policies configured for retries and timeouts
- Graceful degradation when dependencies unavailable

## Governance

### Amendment Process

1. All constitutional changes require documentation
2. Changes must be proposed via ADR (Architecture Decision Record)
3. ADR must include: rationale, tradeoffs, migration plan
4. Constitution version follows semantic versioning (MAJOR.MINOR.PATCH)

### Versioning Policy

- **MAJOR**: Backward incompatible governance/principle removals or redefinitions
- **MINOR**: New principle or section added or materially expanded guidance
- **PATCH**: Clarifications, wording, typo fixes, non-semantic refinements

### Compliance Review

All PRs MUST:
- Verify compliance with current constitution
- Reference relevant constitution principles in description
- Update ADRs for constitutional changes

### Runtime Guidance

For development-specific guidance not covered in this constitution, reference:
- README.md for project setup
- `/specs/<feature>/plan.md` for architecture decisions
- `/specs/<feature>/tasks.md` for implementation guidance

---

**Version**: 1.0.0 | **Ratified**: 2026-02-16 | **Last Amended**: 2026-02-16

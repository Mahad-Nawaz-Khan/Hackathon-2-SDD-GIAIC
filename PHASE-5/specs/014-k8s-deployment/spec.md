# Feature Specification: Kubernetes Deployment (Local & Oracle OKE)

**Feature Branch**: `014-k8s-deployment`
**Created**: 2026-02-17
**Status**: Draft
**Input**: User description: "Phase V: Local & Cloud Deployment (Minikube -> Oracle OKE)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Containerize Applications (Priority: P1)

As a DevOps engineer, I need to package the Frontend and Backend applications as container images that can run on any Kubernetes cluster, so that the application can be deployed consistently across environments.

**Why this priority**: Containerization is foundational - without images, nothing can be deployed to Kubernetes.

**Independent Test**: Can be fully tested by building Docker images locally and running them with docker-compose, delivering portable container artifacts.

**Acceptance Scenarios**:

1. **Given** a clean build environment, **When** I run the Docker build for the Backend, **Then** a multi-arch image (amd64/arm64) is created and can be pushed to GHCR
2. **Given** a clean build environment, **When** I run the Docker build for the Frontend, **Then** a multi-arch image (amd64/arm64) is created and can be pushed to GHCR
3. **Given** both images are built, **When** I run containers locally, **Then** both applications start without errors and serve the UI

---

### User Story 2 - Local Kubernetes Deployment (Priority: P2)

As a developer, I need to deploy the entire application stack to a local Minikube cluster with Dapr, Kafka, and Redis, so that I can validate the event-driven architecture and Dapr integration before deploying to production.

**Why this priority**: Local validation prevents production issues and allows debugging without cloud costs.

**Independent Test**: Can be fully tested by deploying to Minikube and verifying all pods are healthy and Dapr dashboard shows connectivity.

**Acceptance Scenarios**:

1. **Given** Minikube is running with Dapr enabled, **When** I deploy Kafka (Redpanda/Strimzi) and Redis, **Then** all infrastructure pods are Running and Ready
2. **Given** infrastructure is deployed, **When** I apply Dapr component configurations, **Then** Dapr dashboard shows components are connected to local services
3. **Given** Dapr components are configured, **When** I deploy the Backend and Frontend using Helm, **Then** all application pods start successfully with Dapr sidecars
4. **Given** all pods are running, **When** I create a task through the UI, **Then** the task appears in the database and is visible in the UI
5. **Given** a task is created with a reminder, **When** the reminder time is reached, **Then** a Dapr Jobs callback is received and logged

---

### User Story 3 - Oracle OKE Cloud Deployment (Priority: P3)

As a DevOps engineer, I need to deploy the application to Oracle Kubernetes Engine with TLS-enabled ingress, production-grade Kafka, and PostgreSQL, so that users can access the application over HTTPS with reliable infrastructure.

**Why this priority**: Production deployment provides real user access and validates cloud infrastructure compatibility.

**Independent Test**: Can be fully tested by deploying to OKE cluster and accessing the application via public HTTPS URL.

**Acceptance Scenarios**:

1. **Given** an OKE cluster is provisioned, **When** I install Nginx Ingress Controller and Cert-Manager, **Then** both controllers are running and can provision TLS certificates
2. **Given** ingress is configured, **When** I deploy Strimzi Kafka Operator with 1 replica and ephemeral storage, **Then** Kafka cluster is healthy and within free tier resource limits
3. **Given** Kafka is running, **When** I configure Dapr components for cloud PostgreSQL and Kafka, **Then** Dapr can connect to both services
4. **Given** Dapr components are configured, **When** I deploy the Backend and Frontend using Helm, **Then** all pods start successfully and are accessible via ingress
5. **Given** the application is deployed, **When** I access the public URL, **Then** the application loads over HTTPS with valid TLS certificate
6. **Given** a task with recurrence is created, **When** the task is completed, **Then** a Kafka event is published (verified in pod logs)

---

### User Story 4 - Automated CI/CD Pipeline (Priority: P4)

As a developer, I need an automated pipeline that builds, tests, and deploys the application to Oracle OKE on every push to the main branch, so that deployments are consistent and manual intervention is minimized.

**Why this priority**: Automation reduces deployment errors and ensures rapid delivery of updates.

**Independent Test**: Can be fully tested by pushing a code change and verifying the pipeline builds and deploys within 10 minutes.

**Acceptance Scenarios**:

1. **Given** the GitHub Actions workflow is configured, **When** I push a commit to the main branch, **Then** the workflow triggers automatically
2. **Given** the workflow is running, **When** lint and test steps complete, **Then** all tests pass and code quality checks succeed
3. **Given** tests pass, **When** the build step executes, **Then** multi-arch Docker images are built and pushed to GHCR
4. **Given** images are pushed, **When** the deploy step runs, **Then** kubectl apply or Helm upgrade updates the OKE deployment
5. **Given** deployment completes, **When** I check the application URL, **Then** the new version is live within 10 minutes of push

---

### Edge Cases

- What happens when Minikube runs out of resources during local deployment?
- How does the system handle Oracle OKE free tier resource exhaustion (4 OCPUs, 24GB RAM)?
- What happens when Dapr sidecar cannot connect to Kafka during startup?
- How does the system handle multi-arch image build failures for arm64?
- What happens when Cert-Manager cannot provision a TLS certificate (rate limits, DNS issues)?
- How does the system handle GitHub Actions workflow timeout during deployment?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide Dockerfiles for Frontend (Next.js) and Backend (FastAPI) that build multi-arch images (amd64/arm64)
- **FR-002**: System MUST push container images to GitHub Container Registry (GHCR) with appropriate tags
- **FR-003**: System MUST provide Kubernetes manifests or Helm chart for deploying the application to any Kubernetes cluster
- **FR-004**: System MUST configure Dapr sidecar annotations on Frontend and Backend pods
- **FR-005**: System MUST provide Dapr component configurations for local deployment (Redis state store, Kafka/Redpanda pub/sub)
- **FR-006**: System MUST provide Dapr component configurations for Oracle OKE (PostgreSQL state store, Strimzi Kafka pub/sub)
- **FR-007**: System MUST include Redpanda or Strimzi Kafka deployment manifests for local Minikube testing
- **FR-008**: System MUST include Redis deployment manifest for local Minikube state store
- **FR-009**: System MUST include Strimzi Kafka Operator deployment manifests for Oracle OKE with 1 replica and ephemeral storage
- **FR-010**: System MUST configure Strimzi Kafka to use resources within Oracle Free Tier limits (4 OCPUs ARM, 24GB RAM)
- **FR-011**: System MUST provide Nginx Ingress Controller deployment manifest for Oracle OKE
- **FR-012**: System MUST provide Cert-Manager deployment manifest with Let's Encrypt issuer for Oracle OKE
- **FR-013**: System MUST configure ingress resources to use TLS termination and route traffic to Frontend
- **FR-014**: System MUST provide GitHub Actions workflow that triggers on push to main branch
- **FR-015**: System MUST configure workflow to run linting and testing before building images
- **FR-016**: System MUST configure workflow to build and push multi-arch Docker images using buildx
- **FR-017**: System MUST configure workflow to deploy to Oracle OKE using kubectl apply or Helm upgrade
- **FR-018**: System MUST ensure all service-to-service communication goes through Dapr sidecars (no direct Kafka/Redis/Postgres connections)
- **FR-019**: System MUST configure Dapr components to use internal Kubernetes DNS for local (kafka.default.svc.cluster.local) and cloud service names
- **FR-020**: System MUST include resource limits on all deployments to prevent Oracle Free Tier overage
- **FR-021**: System MUST provide health check endpoints that Kubernetes can use for readiness/liveness probes
- **FR-022**: System MUST handle graceful shutdown of pods during rolling updates

### Key Entities

- **Container Image**: Frontend and Backend application packages with multi-arch support, stored in GHCR
- **Kubernetes Deployment**: Pod specifications with Dapr sidecar annotations, resource limits, and health probes
- **Helm Chart**: Package of Kubernetes manifests for todo-app with configurable values
- **Dapr Component**: Configuration for pub/sub (Kafka) and state store (Redis/PostgreSQL) bindings
- **GitHub Actions Workflow**: CI/CD pipeline automation for build, test, and deploy

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Developers can build multi-arch Docker images and push to GHCR in under 10 minutes
- **SC-002**: Local Minikube deployment shows all pods Running and Ready within 5 minutes of deployment
- **SC-003**: Dapr dashboard shows all applications connected to Redis/Redpanda in local environment
- **SC-004**: Oracle OKE deployment serves the application over public HTTPS with valid TLS certificate
- **SC-005**: Creating a recurring task in the Cloud UI triggers a Kafka event visible in Strimzi logs
- **SC-006**: GitHub Actions workflow completes full build-test-deploy cycle within 10 minutes of push
- **SC-007**: All pods run within Oracle Free Tier resource limits (4 OCPUs ARM, 24GB RAM)
- **SC-008**: Rolling updates to Frontend or Backend result in zero downtime for users

## Constraints

### Oracle Free Tier Limits

- **4 OCPUs ARM** (Ampere processors)
- **24GB RAM** total across all nodes
- **Storage**: Ephemeral storage only (no persistent block storage)

### Dapr-First Architecture

- All service-to-service communication MUST go through Dapr sidecars
- No direct connections to Kafka, Redis, or PostgreSQL from application code
- Dapr handles service discovery, pub/sub, state management, and job scheduling

### Platform Scope

- **Include**: Minikube (local), Oracle Kubernetes Engine (cloud)
- **Exclude**: Azure AKS, Google Cloud GKE, AWS EKS
- **Registry**: GitHub Container Registry (GHCR) only

### Application Scope

- No modifications to Python/TypeScript application logic
- Only add deployment artifacts: Dockerfiles, Kubernetes manifests, CI/CD configs

## Dependencies

### Prerequisites

- Existing Event-Driven Todo App codebase with Dapr integration
- Docker installed locally for image building
- Minikube installed locally for Kubernetes testing
- Oracle Cloud account with OKE free tier access
- GitHub repository with Actions enabled

### External Services

- GitHub Container Registry (GHCR) for image storage
- Docker Hub for base images (if needed)
- Strimzi Kafka Operator (from GitHub or Quay)
- Nginx Ingress Controller (from Kubernetes community)
- Cert-Manager (from Jetstack)
- Redpanda (for local Kafka) or Strimzi

## Assumptions

- Application code already has Dapr integration implemented
- PostgreSQL database will use managed service (Neon or similar) for cloud deployment
- Local development can use SQLite or PostgreSQL in Docker container
- User has Oracle Cloud account set up with appropriate quotas
- GitHub repository is properly configured for Actions and OIDC
- TLS certificate domain is pre-configured with DNS pointing to Oracle OKE ingress

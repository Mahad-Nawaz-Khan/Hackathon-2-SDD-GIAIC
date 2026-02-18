# ADR-0001: Multi-Arch Docker Builds for Oracle ARM Free Tier

- **Status:** Accepted
- **Date:** 2026-02-18
- **Feature:** 014-k8s-deployment
- **Context:** The production deployment targets Oracle Cloud Free Tier, which exclusively uses Ampere ARM (aarch64) processors. Developer machines are predominantly x86_64 (Intel/AMD). Docker images must run natively on both architectures to avoid emulation overhead and ensure development-production parity.

## Decision

Use `docker buildx` with multi-platform manifest lists to produce images for both `linux/amd64` and `linux/arm64` from a single build command.

- **Build tool**: Docker Buildx (BuildKit-based)
- **Platforms**: `linux/amd64,linux/arm64`
- **Registry**: GitHub Container Registry (GHCR)
- **CI integration**: `docker/setup-buildx-action@v3` + `docker/build-push-action@v5` in GitHub Actions
- **Local convenience**: `docker-bake.hcl` for parallel multi-service builds
- **Base images**: `python:3.11-slim` (backend) and `node:20-alpine` (frontend) -- both provide official multi-arch manifests

## Consequences

### Positive

- **Native performance on OKE**: ARM images run without QEMU emulation on Oracle Ampere instances, eliminating 5-10x performance penalty
- **Development parity**: Same image manifests work on x86 developer machines and ARM production nodes
- **Single CI pipeline**: One build step produces both architectures with shared cache layers
- **Future-proof**: Supports migration to any cloud provider (AWS Graviton, Azure Arm) without image changes
- **Cost savings**: Oracle ARM Free Tier provides 4 OCPUs + 24GB RAM at zero cost

### Negative

- **Longer CI builds**: Multi-arch builds take ~2x longer than single-arch (QEMU cross-compilation for non-native arch)
- **Cache complexity**: GitHub Actions cache (`type=gha`) must be scoped per-service to avoid conflicts
- **Buildx dependency**: Requires Docker Buildx plugin (included in Docker Desktop, needs setup on CI runners)
- **Debugging difficulty**: Issues may manifest on only one architecture, requiring platform-specific testing

## Alternatives Considered

### Alternative A: Single-arch (amd64 only) + QEMU emulation on OKE

- Run x86 images on ARM via QEMU user-mode emulation
- **Rejected**: 5-10x performance penalty makes application unusable; memory overhead exceeds free tier budget

### Alternative B: Separate build pipelines per architecture

- Maintain two Dockerfiles or two CI workflows (one for amd64, one for arm64)
- **Rejected**: Doubles maintenance burden; risk of configuration drift between architectures; Docker Buildx solves this natively

### Alternative C: Cloud-native build services (Oracle Cloud Build, AWS CodeBuild)

- Use cloud provider's build service with native ARM runners
- **Rejected**: Adds external dependency outside GitHub ecosystem; Oracle doesn't offer free-tier build service; GitHub Actions already available

### Alternative D: Build only arm64 images

- Target only the production architecture
- **Rejected**: Developers on x86 machines can't test locally; breaks development workflow

## References

- Feature Spec: [specs/014-k8s-deployment/spec.md](../../specs/014-k8s-deployment/spec.md)
- Implementation Plan: [specs/014-k8s-deployment/plan.md](../../specs/014-k8s-deployment/plan.md)
- Related ADRs: ADR-0002 (Component Separation), ADR-0003 (Strimzi Kafka)
- Evaluator Evidence: [PHR-0002](../prompts/014-k8s-deployment/0002-k8s-deployment-plan-created.plan.prompt.md)

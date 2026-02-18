---
id: 0005
title: K8s ADRs documented three decisions
stage: misc
date: 2026-02-18
surface: agent
model: claude-opus-4-6
feature: 014-k8s-deployment
branch: 014-k8s-deployment
user: madoo
command: /sp.adr
labels: ["adr","architecture","kubernetes","decisions"]
links:
  spec: specs/014-k8s-deployment/spec.md
  ticket: null
  adr: history/adr/0001-multi-arch-docker-builds-for-oracle-arm-free-tier.md
  pr: null
files:
  - history/adr/0001-multi-arch-docker-builds-for-oracle-arm-free-tier.md
  - history/adr/0002-dapr-component-directory-separation.md
  - history/adr/0003-self-hosted-strimzi-kafka-on-oracle-oke.md
tests:
  - Significance test PASSED for all 3 ADRs (impact + alternatives + scope)
---

## Prompt

/sp.adr multi-arch-docker-builds

## Response snapshot

Created 3 Architecture Decision Records for the K8s deployment feature:

1. **ADR-0001: Multi-Arch Docker Builds for Oracle ARM Free Tier**
   - Decision: Use docker buildx for linux/amd64,linux/arm64
   - 4 alternatives evaluated and rejected
   - Key tradeoff: 2x longer CI builds vs native ARM performance

2. **ADR-0002: Dapr Component Directory Separation**
   - Decision: Separate infra/local/ and infra/prod/ directories
   - 3 alternatives evaluated (Helm templating, Kustomize, env vars)
   - Key tradeoff: File duplication vs security boundary

3. **ADR-0003: Self-Hosted Strimzi Kafka on Oracle OKE**
   - Decision: Strimzi with 1 replica, ephemeral storage
   - 4 alternatives evaluated (Confluent, MSK, Redis Streams, no broker)
   - Key tradeoff: Operational burden vs zero cost within free tier

All ADRs pass significance test: cross-cutting impact, multiple alternatives with tradeoffs, likely to be revisited.

## Outcome

- Impact: 3 architectural decisions permanently documented with rationale and alternatives
- Tests: All 3 ADRs pass significance checklist (impact, alternatives, scope)
- Files: 3 ADR files created in history/adr/
- Next prompts: Review ADRs with team; proceed with implementation testing
- Reflection: Clustered decisions appropriately (not over-granular); each ADR covers an independent concern

## Evaluation notes (flywheel)

- Failure modes observed: create-adr.sh failed on Windows (used agent-native fallback)
- Graders run and results: Significance test PASSED for all 3
- Prompt variant: N/A
- Next experiment: N/A

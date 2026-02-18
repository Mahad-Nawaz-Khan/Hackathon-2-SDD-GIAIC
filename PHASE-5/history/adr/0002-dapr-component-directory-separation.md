# ADR-0002: Dapr Component Directory Separation (Local vs Production)

- **Status:** Accepted
- **Date:** 2026-02-18
- **Feature:** 014-k8s-deployment
- **Context:** Dapr components (pub/sub, state store) require different connection strings and credentials for local development (Redpanda + Redis) versus production (Strimzi Kafka + Neon PostgreSQL). A strategy is needed for managing these environment-specific configurations without risking credential leakage or misconfiguration.

## Decision

Maintain physically separate directories for Dapr component manifests per environment, rather than using a single parameterized template.

- **Local components**: `infra/local/dapr/components/` (Redpanda broker, Redis state store)
- **Production components**: `infra/prod/dapr/components/` (Strimzi Kafka broker, Neon PostgreSQL state store)
- **Application**: No change to application code -- Dapr abstracts the infrastructure
- **Deployment**: `kubectl apply -f infra/{local|prod}/dapr/components/` selects the environment

## Consequences

### Positive

- **Security boundary**: Production hostnames, credentials, and secretKeyRefs are physically isolated from local configs; no risk of local dev accidentally connecting to production Kafka/PostgreSQL
- **Auditability**: Git history clearly shows what changed per environment; diffs are isolated
- **Simplicity**: No Helm templating complexity for Dapr components; plain YAML files are easy to read, debug, and validate
- **Dapr compatibility**: Works with Dapr's native component loading without requiring custom value injection

### Negative

- **Duplication**: Component structure is repeated across directories (pubsub.yaml, statestore.yaml exist in both)
- **Drift risk**: Changes to component schema must be applied in both directories manually
- **More files**: Two sets of files vs one parameterized template

## Alternatives Considered

### Alternative A: Single Helm-templated component files

- Use `{{ .Values.kafka.brokers }}` in a single pubsub.yaml template
- Deploy via `helm template` with environment-specific values
- **Rejected**: Dapr components don't support Helm templating natively; requires a pre-processing step; production secrets could leak into values files committed to git

### Alternative B: Kustomize overlays

- Base components with environment patches via Kustomize
- **Rejected**: Adds Kustomize as a dependency; team unfamiliar with overlay semantics; added complexity for only 2-4 files

### Alternative C: Environment variables in Dapr components

- Use `${KAFKA_BROKERS}` in component metadata
- **Rejected**: Dapr components don't support environment variable interpolation in all metadata fields; inconsistent behavior across component types

## References

- Feature Spec: [specs/014-k8s-deployment/spec.md](../../specs/014-k8s-deployment/spec.md)
- Implementation Plan: [specs/014-k8s-deployment/plan.md](../../specs/014-k8s-deployment/plan.md)
- Related ADRs: ADR-0001 (Multi-Arch Builds), ADR-0003 (Strimzi Kafka)
- Evaluator Evidence: [PHR-0002](../prompts/014-k8s-deployment/0002-k8s-deployment-plan-created.plan.prompt.md)

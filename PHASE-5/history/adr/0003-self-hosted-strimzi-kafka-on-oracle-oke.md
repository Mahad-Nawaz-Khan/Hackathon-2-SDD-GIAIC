# ADR-0003: Self-Hosted Strimzi Kafka on Oracle OKE

- **Status:** Accepted
- **Date:** 2026-02-18
- **Feature:** 014-k8s-deployment
- **Context:** The event-driven architecture requires a Kafka-compatible message broker in production. Oracle Cloud Free Tier provides 4 ARM OCPUs and 24GB RAM. A Kafka solution must fit within these resource constraints while supporting the Dapr pub/sub component interface. The choice is between self-hosted Kafka operators and managed Kafka services.

## Decision

Use the Strimzi Kafka Operator to deploy a self-hosted, single-replica Kafka cluster on Oracle OKE, with ephemeral storage optimized for free tier constraints.

- **Operator**: Strimzi Kafka Operator (Helm chart)
- **Kafka version**: 3.6.0
- **Replicas**: 1 broker, 1 ZooKeeper (no HA -- acceptable for MVP)
- **Storage**: Ephemeral (no persistent volumes, data lost on restart)
- **Resources**: Kafka 500m-1CPU / 1-2Gi RAM; ZK 100-200m / 256-512Mi RAM
- **Topics**: task-events, reminders, task-updates (1 partition each, replication factor 1)
- **Local equivalent**: Redpanda (Kafka API-compatible, simpler for development)

## Consequences

### Positive

- **Zero cost**: Runs entirely within Oracle Free Tier budget (total cluster ~1.1 cores / 2.5Gi from 4 cores / 24Gi available)
- **Kubernetes-native**: Operator pattern with CRDs for declarative Kafka management (KafkaCluster, KafkaTopic)
- **Full control**: Complete control over broker configuration, retention, partitioning
- **Dapr compatible**: Standard Kafka protocol works with `pubsub.kafka` Dapr component without changes
- **Production-grade operator**: Strimzi handles rolling upgrades, monitoring, and certificate management

### Negative

- **No high availability**: Single replica means service disruption on pod restart or node failure
- **Data loss risk**: Ephemeral storage means all messages are lost on Kafka restart (acceptable for MVP; consumers must be idempotent)
- **Operational burden**: Team responsible for Kafka upgrades, monitoring, and troubleshooting (no managed service SLA)
- **Resource-constrained**: Cannot scale beyond free tier limits without cost
- **Startup time**: Strimzi cluster takes 5-10 minutes to become ready (ZooKeeper + Kafka + Operator reconciliation)

## Alternatives Considered

### Alternative A: Confluent Cloud (Managed Kafka)

- Fully managed Kafka service with 99.95% SLA
- **Cost**: Starting at ~$0.50/hour = ~$365/month minimum
- **Rejected**: Exceeds budget constraints; the project requires zero infrastructure cost for MVP

### Alternative B: Amazon MSK / Azure Event Hubs

- Cloud-provider managed Kafka services
- **Cost**: $0.10-0.25/hour per broker = $75-180/month
- **Rejected**: Not available on Oracle Cloud; would require multi-cloud networking; exceeds budget

### Alternative C: Redis Streams as Kafka replacement

- Use Redis Streams (already deployed for state store) as pub/sub
- **Rejected**: Dapr's `pubsub.redis` has different semantics than `pubsub.kafka`; would require application changes; Redis Streams lacks Kafka's consumer group semantics needed for event replay

### Alternative D: No message broker (direct HTTP calls)

- Replace event-driven communication with synchronous REST calls
- **Rejected**: Violates the constitution's "Event-Driven First" principle; creates tight coupling between services; no event replay capability

## References

- Feature Spec: [specs/014-k8s-deployment/spec.md](../../specs/014-k8s-deployment/spec.md)
- Implementation Plan: [specs/014-k8s-deployment/plan.md](../../specs/014-k8s-deployment/plan.md)
- Related ADRs: ADR-0001 (Multi-Arch Builds), ADR-0002 (Component Separation)
- Evaluator Evidence: [PHR-0002](../prompts/014-k8s-deployment/0002-k8s-deployment-plan-created.plan.prompt.md)

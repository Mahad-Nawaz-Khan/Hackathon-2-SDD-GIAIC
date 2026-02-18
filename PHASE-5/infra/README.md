# Infrastructure Directory

This directory contains all Kubernetes infrastructure manifests organized by environment.

## Structure

```
infra/
├── namespace.yaml              # Shared namespace definition
├── README.md                   # This file
├── local/                      # Minikube local development
│   ├── redpanda/
│   │   └── values.yaml         # Redpanda (Kafka) Helm values
│   ├── redis/
│   │   └── values.yaml         # Redis Helm values
│   └── dapr/
│       └── components/
│           ├── pubsub.yaml     # Kafka pub/sub (Redpanda)
│           └── statestore.yaml # Redis state store
├── prod/                       # Oracle OKE production
│   ├── strimzi/
│   │   ├── kafka-cluster.yaml  # Strimzi Kafka cluster (1 replica)
│   │   └── kafka-topic.yaml    # Kafka topics
│   ├── cert-manager/
│   │   ├── issuer.yaml         # Let's Encrypt ClusterIssuer
│   │   └── certificate.yaml    # TLS certificate
│   ├── dapr/
│   │   └── components/
│   │       ├── pubsub.yaml     # Kafka pub/sub (Strimzi)
│   │       └── statestore.yaml # PostgreSQL state store (Neon)
│   └── secrets-template.yaml   # Template for K8s secrets (DO NOT COMMIT REAL VALUES)
└── nginx-ingress/
    └── values.yaml             # NGINX Ingress Controller config
```

## Environments

### Local (Minikube)

- **Kafka**: Redpanda (single pod, Kafka-compatible)
- **State Store**: Redis (ephemeral)
- **Ingress**: Minikube ingress addon
- **TLS**: Disabled

### Production (Oracle OKE)

- **Kafka**: Strimzi Operator (1 replica, ephemeral, Free Tier optimized)
- **State Store**: Neon PostgreSQL (external managed)
- **Ingress**: NGINX Ingress Controller
- **TLS**: Cert-Manager with Let's Encrypt

## GHCR Image Naming

```
ghcr.io/{github-owner}/todo-chatbot-backend:{tag}
ghcr.io/{github-owner}/todo-chatbot-frontend:{tag}
```

Tags:
- `latest` - Latest main branch build
- `{sha}` - Git commit SHA
- `{branch}` - Branch name

## Resource Budget (Oracle Free Tier: 4 OCPU ARM, 24GB RAM)

| Component | CPU Req | CPU Limit | Mem Req | Mem Limit |
|-----------|---------|-----------|---------|-----------|
| Backend + Dapr | 150m | 700m | 320Mi | 640Mi |
| Frontend + Dapr | 150m | 700m | 320Mi | 640Mi |
| Strimzi Kafka | 500m | 1000m | 1Gi | 2Gi |
| Zookeeper | 100m | 200m | 256Mi | 512Mi |
| NGINX Ingress | 50m | 200m | 64Mi | 128Mi |
| Cert-Manager | 25m | 100m | 32Mi | 64Mi |
| **Total** | **~1.1** | **~2.9** | **~2Gi** | **~4Gi** |

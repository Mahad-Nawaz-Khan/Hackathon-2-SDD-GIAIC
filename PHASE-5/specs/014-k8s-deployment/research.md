# Research: Kubernetes Deployment (Local & Oracle OKE)

**Feature**: 014-k8s-deployment
**Date**: 2026-02-18
**Status**: Complete

## Research Topics

### 1. Multi-Architecture Docker Builds

**Question**: How to build Docker images that run on both x86 (local development) and ARM (Oracle Free Tier)?

**Findings**:

- **Docker Buildx** is the standard solution for multi-platform builds
- Supports cross-compilation without emulation for most languages
- Requires QEMU for languages without native cross-compilation (minimal impact)
- GitHub Actions has first-class support via `docker/setup-buildx-action`

**Decision**: Use `docker buildx` with `linux/amd64,linux/arm64` platforms

**Implementation Pattern**:
```bash
# Local testing
docker buildx build --platform linux/amd64,linux/arm64 -t myimage:latest .

# GitHub Actions
- uses: docker/setup-buildx-action@v3
- uses: docker/build-push-action@v5
  with:
    platforms: linux/amd64,linux/arm64
```

**Risks Mitigated**:
- No runtime emulation overhead (images are native to each platform)
- Build cache shared across platforms via GitHub Actions cache

---

### 2. Dapr Sidecar Resource Requirements

**Question**: What are the memory/CPU requirements for Dapr sidecars?

**Findings**:

From Dapr documentation and production benchmarks:
- **Minimal config**: 50m CPU, 64Mi memory per sidecar
- **Recommended**: 100m CPU, 128Mi memory per sidecar
- **With actors**: Up to 250m CPU, 256Mi memory

**Decision**: Use 50m/200m CPU and 64Mi/128Mi memory (request/limit) for sidecars

**Calculation for Free Tier**:
- Backend sidecar: 50m CPU, 64Mi memory
- Frontend sidecar: 50m CPU, 64Mi memory
- **Total sidecar overhead**: 100m CPU, 128Mi memory

---

### 3. Strimzi Kafka on Kubernetes

**Question**: Can Strimzi run within Oracle Free Tier constraints?

**Findings**:

| Component | Min Recommended | Our Config | Fits Free Tier? |
|-----------|----------------|------------|-----------------|
| Kafka broker | 1 core, 2Gi | 1 core, 2Gi | Yes |
| Zookeeper | 0.5 core, 1Gi | 200m, 512Mi | Yes |
| Operator | 200m, 256Mi | Shared namespace | Yes |

**Critical Configuration for Free Tier**:
```yaml
spec:
  kafka:
    replicas: 1              # Single broker (no HA)
    storage:
      type: ephemeral        # No PV needed
    resources:
      limits:
        cpu: "1"
        memory: 2Gi
  zookeeper:
    replicas: 1
    storage:
      type: ephemeral
```

**Decision**: Use Strimzi with 1 replica, ephemeral storage

**Caveats**:
- No persistence means data loss on pod restart (acceptable for MVP)
- No HA means service degradation if broker fails

---

### 4. Redpanda vs Strimzi for Local Development

**Question**: Which is better for local Minikube development?

**Comparison**:

| Factor | Redpanda | Strimzi |
|--------|----------|---------|
| Setup complexity | Single Helm chart | Operator + CRD |
| Resource usage | Lower (no Zookeeper) | Higher (needs Zookeeper) |
| Kafka API compatible | Yes (100%) | Yes (native) |
| Local dev focus | Yes | Production focus |

**Decision**: Use Redpanda for local, Strimzi for production

**Rationale**:
- Redpanda is simpler for local development (no Zookeeper)
- Strimzi is more production-ready with operator pattern
- Both are Kafka API compatible, so Dapr configuration is identical

---

### 5. Dapr Component Configuration

**Question**: How to structure Dapr components for local vs production?

**Findings**:

Dapr components are namespace-scoped and loaded from `/components` directory. Options:

1. **Separate directories**: `components/local/*.yaml`, `components/prod/*.yaml`
2. **Helm templating**: Single template with values injection
3. **Kustomize overlays**: Base components with patches

**Decision**: Use separate directories with explicit kubectl apply

**Rationale**:
- Simplicity: No Helm templating complexity
- Security: No risk of prod secrets leaking to local configs
- Debuggability: Easy to inspect which config is applied
- Git history: Clear separation of environment changes

---

### 6. NGINX Ingress Controller for Oracle OKE

**Question**: How to set up ingress with TLS on OKE?

**Findings**:

Oracle OKE supports standard Kubernetes ingress. NGINX Ingress Controller is the most common choice.

**TLS Options**:
1. **Cert-Manager with Let's Encrypt**: Automated, free certificates
2. **Oracle Certificate Service**: Managed but costs extra
3. **Self-signed**: Not suitable for production

**Decision**: NGINX Ingress + Cert-Manager + Let's Encrypt

**Implementation**:
```yaml
# ClusterIssuer for Let's Encrypt
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

---

### 7. GitHub Actions for Kubernetes Deployment

**Question**: How to securely deploy to OKE from GitHub Actions?

**Findings**:

**Authentication Options**:
1. **Service Account Token**: Create K8s SA with limited permissions
2. **OIDC with Workload Identity**: Most secure, complex setup
3. **kubeconfig file**: Simple, stored as GitHub secret

**Decision**: Use kubeconfig stored as GitHub secret (base64 encoded)

**Security Considerations**:
- Use short-lived tokens when possible
- Limit service account permissions to specific namespace
- Rotate credentials regularly

**Workflow Structure**:
```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Configure kubectl
        run: echo "${{ secrets.KUBECONFIG }}" | base64 -d > ~/.kube/config

      - name: Deploy
        run: helm upgrade --install todo-chatbot ./todo-chatbot -f values-prod.yaml
```

---

### 8. Health Checks and Graceful Shutdown

**Question**: How to ensure zero-downtime deployments?

**Findings**:

**Kubernetes Probes**:
- **livenessProbe**: Restart pod if unhealthy
- **readinessProbe**: Remove from service if not ready
- **startupProbe**: Allow longer startup time

**Graceful Shutdown**:
- Kubernetes sends SIGTERM to pods
- Apps should handle SIGTERM and finish in-flight requests
- Default grace period is 30 seconds

**Decision**: Implement both probes and graceful shutdown

**Implementation**:
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10

# In deployment spec
lifecycle:
  preStop:
    exec:
      command: ["/bin/sh", "-c", "sleep 10"]  # Allow connections to drain
```

---

## Summary of Decisions

| Topic | Decision | Primary Rationale |
|-------|----------|-------------------|
| Multi-arch builds | Docker buildx | Native performance, GitHub Actions support |
| Dapr resources | 50m/200m CPU, 64Mi/128Mi | Minimal overhead, fits free tier |
| Kafka (local) | Redpanda | Simpler setup, no Zookeeper |
| Kafka (prod) | Strimzi | Kubernetes-native, operator pattern |
| Dapr components | Separate directories | Security, clarity, simplicity |
| Ingress/TLS | NGINX + Cert-Manager + Let's Encrypt | Free, automated, standard |
| CI/CD auth | kubeconfig as secret | Simplicity for MVP |
| Health checks | Both probes + graceful shutdown | Zero-downtime deployments |

## References

- [Dapr Documentation](https://docs.dapr.io/)
- [Strimzi Kafka Operator](https://strimzi.io/)
- [Redpanda Documentation](https://docs.redpanda.com/)
- [Docker Buildx](https://docs.docker.com/build/buildx/)
- [Cert-Manager](https://cert-manager.io/)
- [Oracle OKE Documentation](https://docs.oracle.com/en-us/iaas/Content/ContEng/home.htm)

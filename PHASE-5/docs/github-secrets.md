# GitHub Secrets Configuration

The CI/CD pipeline requires the following secrets configured in your GitHub repository.

## Required Secrets

| Secret | Description | How to Get |
|--------|-------------|------------|
| `OKE_KUBECONFIG` | Base64-encoded kubeconfig for Oracle OKE | `cat ~/.kube/config \| base64 -w 0` |

## Automatic Secrets

| Secret | Description |
|--------|-------------|
| `GITHUB_TOKEN` | Automatically provided by GitHub Actions for GHCR access |

## Setting Up Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** > **Secrets and variables** > **Actions**
3. Click **New repository secret**
4. Add each secret listed above

### OKE Kubeconfig

```bash
# Generate kubeconfig for OKE
oci ce cluster create-kubeconfig \
  --cluster-id <cluster-ocid> \
  --file oke-kubeconfig \
  --region <region>

# Encode for GitHub secret
cat oke-kubeconfig | base64 -w 0
```

### GHCR Authentication

GitHub Actions automatically authenticates with GHCR using `GITHUB_TOKEN`. No additional setup needed.

Ensure your repository **Settings** > **Actions** > **General** has:
- **Workflow permissions**: Read and write permissions
- **Allow GitHub Actions to create and approve pull requests**: Enabled (optional)

### Application Secrets (on Kubernetes)

Application secrets (database URLs, API keys) are NOT stored in GitHub.
They are created directly on the Kubernetes cluster:

```bash
# Create secrets on OKE (one-time setup)
kubectl create secret generic neon-credentials \
  --from-literal=connection-string='postgresql://...' -n todo-app

kubectl create secret generic clerk-secrets \
  --from-literal=secret-key='sk_...' \
  --from-literal=issuer='https://...' \
  --from-literal=jwks-url='https://.../.well-known/jwks.json' \
  --from-literal=publishable-key='pk_...' -n todo-app

kubectl create secret generic ai-api-keys \
  --from-literal=gemini-key='AIza...' \
  --from-literal=gemini-model='gemini-2.5-flash-lite' \
  --from-literal=z-ai-key='...' \
  --from-literal=z-ai-model='glm-4.5-air' -n todo-app
```

See `infra/prod/secrets-template.yaml` for the full template.

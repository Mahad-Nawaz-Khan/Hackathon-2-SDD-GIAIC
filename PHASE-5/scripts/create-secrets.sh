#!/bin/bash
# Create Kubernetes secrets for Oracle OKE deployment
# IMPORTANT: Update the values below before running!

set -e

NAMESPACE="todo-app"

echo "=== Creating secrets in namespace: ${NAMESPACE} ==="

# =============================================================================
# CLERK SECRETS - Update with your Clerk credentials
# =============================================================================
# Get these from: https://dashboard.clerk.com
CLERK_SECRET_KEY="${CLERK_SECRET_KEY:-placeholder-clerk-secret-key}"
CLERK_PUBLISHABLE_KEY="${CLERK_PUBLISHABLE_KEY:-placeholder-clerk-publishable-key}"
CLERK_ISSUER="${CLERK_ISSUER:-https://civil-corgi-51.clerk.accounts.dev}"
CLERK_JWKS_URL="${CLERK_JWKS_URL:-https://civil-corgi-51.clerk.accounts.dev/.well-known/jwks.json}"

kubectl create secret generic clerk-secrets \
    --namespace=${NAMESPACE} \
    --from-literal=secret-key="${CLERK_SECRET_KEY}" \
    --from-literal=publishable-key="${CLERK_PUBLISHABLE_KEY}" \
    --from-literal=issuer="${CLERK_ISSUER}" \
    --from-literal=jwks-url="${CLERK_JWKS_URL}" \
    --dry-run=client -o yaml | kubectl apply -f -

# =============================================================================
# NEON DATABASE - Update with your Neon PostgreSQL credentials
# =============================================================================
# Get these from: https://console.neon.tech
NEON_CONNECTION_STRING="${NEON_CONNECTION_STRING:-postgresql://neondb_owner:placeholder-password@ep-round-field-a1sjfms9-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require}"

kubectl create secret generic neon-credentials \
    --namespace=${NAMESPACE} \
    --from-literal=connection-string="${NEON_CONNECTION_STRING}" \
    --dry-run=client -o yaml | kubectl apply -f -

# =============================================================================
# AI API KEYS - Update with your AI provider credentials
# =============================================================================
GEMINI_API_KEY="${GEMINI_API_KEY:-placeholder-gemini-api-key}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash-lite}"
Z_AI_API_KEY="${Z_AI_API_KEY:-placeholder-z-ai-api-key}"
Z_AI_MODEL="${Z_AI_MODEL:-glm-4.7-flash}"

kubectl create secret generic ai-api-keys \
    --namespace=${NAMESPACE} \
    --from-literal=gemini-key="${GEMINI_API_KEY}" \
    --from-literal=gemini-model="${GEMINI_MODEL}" \
    --from-literal=z-ai-key="${Z_AI_API_KEY}" \
    --from-literal=z-ai-model="${Z_AI_MODEL}" \
    --dry-run=client -o yaml | kubectl apply -f -

echo ""
echo "=== Secrets created successfully ==="
echo ""
echo "Verify:"
echo "  kubectl get secrets -n ${NAMESPACE}"

# Build and push images to Oracle Container Image Registry (OCIR)
# Usage: .\scripts\build-push-oracle.ps1 -Region "ap-mumbai-1" -TenancyNamespace "your-namespace"

param(
    [string]$Region = "ap-mumbai-1",
    [Parameter(Mandatory=$true)]
    [string]$TenancyNamespace
)

$ErrorActionPreference = "Stop"

$Registry = "${Region}.ocir.io"
$RepoPrefix = "${Registry}/${TenancyNamespace}"

Write-Host "=== Building images for Oracle OKE ===" -ForegroundColor Cyan
Write-Host "Registry: $Registry"
Write-Host "Repository prefix: $RepoPrefix"

# Ensure buildx is available
docker buildx version

# Create/build with buildx for multi-arch
Write-Host ""
Write-Host "=== Building backend ===" -ForegroundColor Yellow
docker buildx build --platform linux/amd64,linux/arm64 `
    -t "${RepoPrefix}/todo-chatbot-backend:latest" `
    -t "${RepoPrefix}/todo-chatbot-backend:v1.0.0" `
    --push `
    ".\backend"

Write-Host ""
Write-Host "=== Building frontend ===" -ForegroundColor Yellow
$clerkKey = $env:NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
$apiUrl = "https://api.todo-app.cloud"

docker buildx build --platform linux/amd64,linux/arm64 `
    -t "${RepoPrefix}/todo-chatbot-frontend:latest" `
    -t "${RepoPrefix}/todo-chatbot-frontend:v1.0.0" `
    --build-arg "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=$clerkKey" `
    --build-arg "NEXT_PUBLIC_API_URL=$apiUrl" `
    --push `
    ".\frontend"

Write-Host ""
Write-Host "=== Images pushed successfully ===" -ForegroundColor Green
Write-Host "Backend: ${RepoPrefix}/todo-chatbot-backend:latest"
Write-Host "Frontend: ${RepoPrefix}/todo-chatbot-frontend:latest"
Write-Host ""
Write-Host "Update values-prod.yaml with these image URLs!" -ForegroundColor Cyan

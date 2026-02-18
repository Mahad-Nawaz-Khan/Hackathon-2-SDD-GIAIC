# docker-bake.hcl - Multi-arch build configuration
# Usage: docker buildx bake --push

variable "REGISTRY" {
  default = "ghcr.io/OWNER"  # CHANGE: Set your GitHub org/user
}

variable "TAG" {
  default = "latest"
}

group "default" {
  targets = ["backend", "frontend"]
}

target "backend" {
  context    = "./backend"
  dockerfile = "Dockerfile"
  platforms  = ["linux/amd64", "linux/arm64"]
  tags       = ["${REGISTRY}/todo-chatbot-backend:${TAG}"]
  cache-from = ["type=gha"]
  cache-to   = ["type=gha,mode=max"]
}

target "frontend" {
  context    = "./frontend"
  dockerfile = "Dockerfile"
  platforms  = ["linux/amd64", "linux/arm64"]
  tags       = ["${REGISTRY}/todo-chatbot-frontend:${TAG}"]
  cache-from = ["type=gha"]
  cache-to   = ["type=gha,mode=max"]
}

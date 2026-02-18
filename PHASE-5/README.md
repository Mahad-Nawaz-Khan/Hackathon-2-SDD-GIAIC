# AI-Powered Task Management Chatbot

A cloud-native AI-powered task management application with event-driven architecture, deployed on Kubernetes with Dapr.

## Features

- **AI-Powered Chat Interface**: Natural language task management via conversational AI
- **Event-Driven Architecture**: Kafka-based messaging for reminders, audit logs, and recurring tasks
- **Dapr Integration**: Abstracted infrastructure for pub/sub, state management, and secrets
- **Cloud-Native**: Kubernetes deployment with Helm, supporting local (Minikube) and cloud (Oracle OKE)
- User authentication and authorization with Clerk (JWT-based)
- Create, read, update, and delete tasks via natural language
- Task prioritization (HIGH, MEDIUM, LOW), due dates, tagging
- Rate limiting on all API endpoints
- Responsive dark-themed design

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Next.js 14, React 19, Tailwind CSS v4 |
| **Backend** | FastAPI, SQLModel, Python 3.13 |
| **AI** | Gemini 2.5 Flash Lite, Zhipu AI GLM-4.7 |
| **Authentication** | Clerk (JWT-based) |
| **Database** | PostgreSQL (Neon Serverless) |
| **Messaging** | Kafka (Redpanda local / Strimzi prod) |
| **State Cache** | Redis |
| **Service Mesh** | Dapr 1.16 |
| **Container Runtime** | Docker (multi-arch: amd64/arm64) |
| **Orchestration** | Kubernetes (Minikube / Oracle OKE) |
| **Package Manager** | Helm 3 |
| **CI/CD** | GitHub Actions |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              KUBERNETES CLUSTER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                    │
│  │  Frontend   │     │   Backend   │     │   Dapr      │                    │
│  │  (Next.js)  │────▶│  (FastAPI)  │────▶│  Sidecar    │                    │
│  │  :3000      │     │  :8000      │     │  :3500      │                    │
│  └─────────────┘     └──────┬──────┘     └──────┬──────┘                    │
│                             │                    │                           │
│                             ▼                    ▼                           │
│                    ┌─────────────┐     ┌─────────────────┐                  │
│                    │ PostgreSQL  │     │  Kafka/Redis    │                  │
│                    │   (Neon)    │     │  (via Dapr)     │                  │
│                    └─────────────┘     └─────────────────┘                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        NGINX Ingress                                 │   │
│  │    todo.todo-app.cloud → Frontend                                   │   │
│  │    api.todo-app.cloud → Backend                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Docker Desktop (with Kubernetes or Minikube)
- kubectl configured
- Helm 3
- Minikube (for local)

### Local Development (Minikube)

```bash
# 1. Start Minikube
minikube start --memory=8192 --cpus=4

# 2. Enable addons
minikube addons enable ingress
minikube addons enable metrics-server

# 3. Install Dapr
helm repo add dapr https://dapr.github.io/helm-charts
helm install dapr dapr/dapr --namespace dapr-system --create-namespace

# 4. Install Redis
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install redis bitnami/redis --namespace todo-app --create-namespace --set auth.enabled=false

# 5. Install Redpanda (Kafka)
helm repo add redpanda https://charts.redpanda.com
helm install redpanda redpanda/redpanda --namespace todo-app \
  --values infra/local/redpanda/values.yaml

# 6. Build and load images
docker build -t hackathon-backend:latest ./backend
docker build -t hackathon-frontend:latest \
  --build-arg NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=$CLERK_KEY \
  --build-arg NEXT_PUBLIC_API_URL=http://localhost:8000 \
  ./frontend
minikube image load hackathon-backend:latest
minikube image load hackathon-frontend:latest

# 7. Deploy with Helm
helm upgrade --install todo-chatbot ./todo-chatbot \
  --namespace todo-app \
  --values ./todo-chatbot/values-local.yaml \
  --set image.backend.repository=hackathon-backend \
  --set image.frontend.repository=hackathon-frontend

# 8. Port-forward to access
kubectl port-forward -n todo-app svc/todo-chatbot-frontend 3000:80
kubectl port-forward -n todo-app svc/todo-chatbot-backend 8000:8000
```

### Production (Oracle OKE)

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for full production deployment guide.

```bash
# Deploy to OKE
./scripts/deploy-to-oke.sh
```

## Project Structure

```
├── backend/                 # FastAPI backend service
│   ├── src/
│   │   ├── main.py         # Application entry point
│   │   ├── routes/         # API routes
│   │   ├── services/       # Business logic
│   │   └── models/         # SQLModel models
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/               # Next.js frontend
│   ├── src/
│   │   ├── app/           # App Router pages
│   │   └── components/    # React components
│   ├── Dockerfile
│   └── package.json
├── todo-chatbot/          # Helm chart
│   ├── Chart.yaml
│   ├── values.yaml
│   ├── values-local.yaml
│   ├── values-prod.yaml
│   └── templates/
├── infra/                 # Infrastructure manifests
│   ├── local/            # Minikube configs
│   │   ├── redpanda/
│   │   ├── redis/
│   │   └── dapr/
│   └── prod/             # OKE configs
│       ├── strimzi/
│       ├── cert-manager/
│       ├── dapr/
│       └── secrets-template.yaml
├── scripts/              # Deployment scripts
│   ├── build-push-oracle.ps1
│   ├── deploy-to-oke.sh
│   └── create-secrets.sh
├── .github/workflows/    # CI/CD pipelines
│   ├── ci.yml
│   └── build-deploy-prod.yml
└── docs/                 # Documentation
    └── DEPLOYMENT.md
```

## API Endpoints

### Tasks
- `GET /api/v1/tasks` - List tasks (with filters)
- `POST /api/v1/tasks` - Create task
- `GET /api/v1/tasks/{id}` - Get task
- `PUT /api/v1/tasks/{id}` - Update task
- `DELETE /api/v1/tasks/{id}` - Delete task
- `PATCH /api/v1/tasks/{id}/toggle-completion` - Toggle completion

### Tags
- `GET /api/v1/tags` - List tags
- `POST /api/v1/tags` - Create tag
- `PUT /api/v1/tags/{id}` - Update tag
- `DELETE /api/v1/tags/{id}` - Delete tag

### Chat (AI-Powered)
- `POST /api/v1/chat/message` - Send message
- `POST /api/v1/chat/message/stream` - Stream response (SSE)
- `GET /api/v1/chat/history` - Chat history

### Health
- `GET /health` - Health check

## Environment Variables

### Backend

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `CLERK_SECRET_KEY` | Clerk backend secret |
| `CLERK_ISSUER` | Clerk issuer URL |
| `CLERK_JWKS_URL` | Clerk JWKS URL |
| `GEMINI_API_KEY` | Gemini AI API key |
| `GEMINI_MODEL` | Gemini model name |
| `Z_AI_API_KEY` | Zhipu AI API key |
| `Z_AI_MODEL` | Zhipu AI model name |
| `CORS_ORIGINS` | Allowed CORS origins |
| `DAPR_HTTP_PORT` | Dapr HTTP port (default: 3500) |

### Frontend

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Clerk frontend key |
| `NEXT_PUBLIC_API_URL` | Backend API URL |

## Dapr Components

| Component | Type | Purpose |
|-----------|------|---------|
| `pubsub` | pubsub.kafka | Task events, reminders |
| `statestore` | state.redis | Task state cache |

## Event Topics

| Topic | Purpose |
|-------|---------|
| `task-events` | Task CRUD events |
| `reminders` | Scheduled reminders |
| `task-updates` | Task state changes |

## URLs

| Environment | Frontend | Backend |
|-------------|----------|---------|
| Local | http://localhost:3000 | http://localhost:8000 |
| Production | https://todo.todo-app.cloud | https://api.todo-app.cloud |

## Security

- JWT authentication via Clerk
- TLS via Let's Encrypt (production)
- mTLS between Dapr sidecars
- Rate limiting on all endpoints
- Secrets stored in Kubernetes Secrets
- Network policies (production)

## License

MIT

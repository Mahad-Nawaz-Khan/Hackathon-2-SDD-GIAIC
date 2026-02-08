---
id: 0001
title: Minikube Deployment and Networking Troubleshooting
stage: red
date: 2025-02-08
surface: agent
model: claude-opus-4-6
feature: 012-project-deliverables-nextjs
branch: master
user: unknown
command: /sp.phr
labels: ["kubernetes", "minikube", "networking", "cors", "loadbalancer", "troubleshooting"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - frontend/src/services/chatService.ts
  - frontend/next.config.ts
tests:
  - none

## Prompt

Deploy the frontend and backend Docker images to minikube. The frontend is failing to connect to the backend with various errors: DNS resolution failures (ERR_NAME_NOT_RESOLVED), connection refused, and CORS errors. The ports change each time minikube services are accessed. Fix all networking issues so the frontend can successfully call the backend API.

## Response snapshot

Attempted multiple approaches: (1) Using internal Kubernetes service names - failed because Next.js rewrites don't work for client-side fetch, (2) Using minikube NodePort directly (192.168.49.2:30800) - timed out due to Windows Docker driver networking, (3) Using relative URLs with Next.js rewrites - still called backend directly from browser. Final solution: Changed services to LoadBalancer type with minikube tunnel, exposing backend on 127.0.0.1:8000 and frontend on 127.0.0.1:3000. Also set CORS_ORIGINS="*" on backend to allow all origins.

## Outcome

- ✅ Impact: Successfully deployed with LoadBalancer services accessible via localhost
- 🧪 Tests: Verified backend health endpoint accessible at http://127.0.0.1:8000/health
- 📁 Files: frontend/src/services/chatService.ts (modified for conditional URL), frontend/next.config.ts
- 🔁 Next prompts: Document deployment commands for future use
- 🧠 Reflection: Next.js rewrites do NOT work for client-side fetch - browser calls backend directly, requiring external accessible URL

## Evaluation notes (flywheel)

- Failure modes observed: Multiple networking issues - CORS, DNS resolution, connection timeouts, minikube Docker driver limitations. Had to rebuild frontend 6 times (v1-v6) trying different URL approaches.
- Graders run and results (PASS/FAIL): PASS - Final solution works with LoadBalancer + tunnel
- Prompt variant (if applicable): none
- Next experiment (smallest change to try): Create a local.env file with configurable API_URL to avoid rebuilds when changing deployment targets

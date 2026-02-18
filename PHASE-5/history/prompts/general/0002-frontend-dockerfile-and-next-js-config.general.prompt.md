---
id: 0002
title: Frontend Dockerfile and Next.js Config
stage: green
date: 2025-02-08
surface: agent
model: claude-opus-4-6
feature: none
branch: master
user: unknown
command: /sp.phr
labels: ["docker", "frontend", "nextjs", "standalone", "clerk"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - frontend/Dockerfile
  - frontend/next.config.ts
tests:
  - none

## Prompt

Create an optimized Dockerfile for the Next.js 16 frontend application. The build was failing due to missing Clerk publishable key. Configure build arguments to pass Clerk keys during build time and enable standalone output for production deployment.

## Response snapshot

Created multi-stage Dockerfile for Next.js 16 with deps, builder, and runner stages. Added build arguments for NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY and NEXT_PUBLIC_API_URL. Updated next.config.ts to enable output: 'standalone' for Docker standalone output. Multiple rebuilds were required to get the correct Clerk key (civil-corgi-51 vs pleased-mammoth-33).

## Outcome

- ✅ Impact: Frontend Docker image built successfully (295MB) with standalone output enabled
- 🧪 Tests: none
- 📁 Files: frontend/Dockerfile, frontend/next.config.ts
- 🔁 Next prompts: Build and deploy Docker images
- 🧠 Reflection: NEXT_PUBLIC_* env vars must be baked in at build time - they cannot be changed at runtime

## Evaluation notes (flywheel)

- Failure modes observed: Build failed multiple times due to incorrect Clerk publishable key. Had to rebuild with correct key.
- Graders run and results (PASS/FAIL): PASS - After multiple iterations, correct key was identified and build succeeded
- Prompt variant (if applicable): none
- Next experiment (smallest change to try): Validate Clerk keys in pre-build script to avoid failed builds

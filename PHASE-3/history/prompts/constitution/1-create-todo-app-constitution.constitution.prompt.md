---
id: 1
title: "Create TODO App Constitution with Rate Limiting"
stage: constitution
date_iso: "2026-01-08"
surface: "agent"
model: "claude-opus-4-5-20251101"
feature: "none"
branch: "master"
user: "user"
command: "/sp.constitution"
labels: ["constitution", "todo-app", "auth", "security", "rate-limiting"]
spec: null
ticket: null
adr: null
pr: null
files:
  - ".specify/memory/constitution.md"
tests: []
outcome: "Successfully created a comprehensive constitution for the TODO application with Clerk authentication and rate limiting"
evaluation: "Constitution includes all required principles for auth, data isolation, tech stack adherence, and rate limiting implementation"
---

# Create TODO App Constitution

## User Prompt
```
TODO Application (Full-Stack Web) – Constitution (With Auth and Rate Limiting)

- Purpose
  - Build a secure, multi-user, full-stack TODO application with strict data isolation.
  - Ensure each user owns and accesses only their own tasks.
  - Implement rate limiting on all API endpoints to prevent abuse.

- Platform & Architecture
  - Frontend: Next.js (App Router)
  - Backend: FastAPI
  - Database: Neon Serverless PostgreSQL
  - ORM: SQLModel
  - Authentication & Identity: Clerk (official SDKs only)
  - Rate Limiting: slowapi for preventing API abuse

- Authentication & Identity (Mandatory)
  - User authentication must be handled exclusively by Clerk.
  - Login, signup, and account management pages must NOT be custom-built.
  - Only Clerk-provided components, routes, and hosted pages may be used.
  - Clerk documentation must be followed as the source of truth.
  - Clerk user ID is the authoritative identity for all backend operations.

- User Isolation Rules (Non-Negotiable)
  - Every task must belong to exactly one authenticated user.
  - No user may access, modify, or infer another user's data.
  - Backend must enforce user ownership at the database query level.
  - Frontend must never pass user IDs manually.

- Core Functional Features
  - Full CRUD for tasks.
  - Completion toggling.
  - Priorities (HIGH / MEDIUM / LOW).
  - Tags with many-to-many relationships to tasks.
  - Search, filter, and sorting.
  - Recurring tasks (DAILY / WEEKLY / MONTHLY).
  - Due dates and reminders.

- Backend Authority
  - Backend validates Clerk-issued JWTs.
  - Backend derives user identity exclusively from verified tokens.
  - Business logic and access control reside in backend only.

- Data Integrity & Rate Limiting
  - Tasks are always scoped to a user record in the database.
  - Clerk user ID must map to a single internal user record.
  - Cross-user data leakage is forbidden.
  - All API endpoints must implement rate limiting (GET: 100/min, POST: 20/min, PUT/PATCH: 30/min, DELETE: 30/min).

- Engineering Principles
  - Clerk for auth, Neon for data, FastAPI for logic, Next.js for UI.
  - Separation of concerns across all layers.
  - Production-first mindset, even at MVP stage.
  - Rate limiting implementation on all API endpoints.
```

## Assistant Response
Created a comprehensive constitution for the TODO application that includes:

1. Core Principles:
   - Authentication & Identity (Mandatory)
   - User Data Isolation (Non-Negotiable)
   - Technology Stack Adherence
   - Backend Authority (Mandatory)
   - Data Integrity & Security
   - Rate Limiting & Performance

2. Additional Constraints:
   - Platform & Architecture requirements
   - Core Functional Features list
   - Security Requirements including rate limiting

3. Development Workflow:
   - Code Review Process with rate limiting verification
   - Quality Gates with performance benchmarks
   - Deployment Policy with rate limiting configurations

4. Governance section with amendment procedures

The constitution has been saved to `.specify/memory/constitution.md` with version 1.1.0, ratified on 2026-01-08.
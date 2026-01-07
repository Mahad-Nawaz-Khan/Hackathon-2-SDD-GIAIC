---
id: 1
title: "Create TODO App Constitution"
stage: constitution
date_iso: "2026-01-06"
surface: "agent"
model: "claude-opus-4-5-20251101"
feature: "none"
branch: "master"
user: "user"
command: "/sp.constitution"
labels: ["constitution", "todo-app", "auth", "security"]
spec: null
ticket: null
adr: null
pr: null
files:
  - ".specify/memory/constitution.md"
tests: []
outcome: "Successfully created a comprehensive constitution for the TODO application with Clerk authentication"
evaluation: "Constitution includes all required principles for auth, data isolation, and tech stack adherence"
---

# Create TODO App Constitution

## User Prompt
```
TODO Application (Full-Stack Web) – Constitution (With Auth)

- Purpose
  - Build a secure, multi-user, full-stack TODO application with strict data isolation.
  - Ensure each user owns and accesses only their own tasks.

- Platform & Architecture
  - Frontend: Next.js (App Router)
  - Backend: FastAPI
  - Database: Neon Serverless PostgreSQL
  - ORM: SQLModel
  - Authentication & Identity: Clerk (official SDKs only)

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
  - Priorities (high / medium / low).
  - Tags / categories.
  - Search, filter, and sorting.
  - Recurring tasks.
  - Due dates and reminders.

- Backend Authority
  - Backend validates Clerk-issued JWTs.
  - Backend derives user identity exclusively from verified tokens.
  - Business logic and access control reside in backend only.

- Data Integrity
  - Tasks are always scoped to a user record in the database.
  - Clerk user ID must map to a single internal user record.
  - Cross-user data leakage is forbidden.

- Engineering Principles
  - Clerk for auth, Neon for data, FastAPI for logic, Next.js for UI.
  - Separation of concerns across all layers.
  - Production-first mindset, even at MVP stage.
```

## Assistant Response
Created a comprehensive constitution for the TODO application that includes:

1. Core Principles:
   - Authentication & Identity (Mandatory)
   - User Data Isolation (Non-Negotiable)
   - Technology Stack Adherence
   - Backend Authority (Mandatory)
   - Data Integrity & Security
   - Test-First (NON-NEGOTIABLE)

2. Additional Constraints:
   - Platform & Architecture requirements
   - Core Functional Features list
   - Security Requirements

3. Development Workflow:
   - Code Review Process
   - Quality Gates
   - Deployment Policy

4. Governance section with amendment procedures

The constitution has been saved to `.specify/memory/constitution.md` with version 1.0.0, ratified on 2026-01-06.
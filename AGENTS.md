# AGENTS.md

## Purpose

This project uses **Spec-Driven Development (SDD)** — a workflow where **no agent is allowed to write code until the specification is complete and approved**.  
All AI agents (Claude, Copilot, Gemini, local LLMs, etc.) must follow the **Spec-Kit lifecycle**:

> **Specify → Plan → Tasks → Implement**

This prevents "vibe coding", ensures alignment across agents, and guarantees that every implementation step maps back to an explicit requirement.

---

## The Mental Model: Who Does What?

| Component | Role | Responsibility |
| :--- | :--- | :--- |
| **AGENTS.md** | **The Brain** | Cross-agent source of truth. Defines how agents behave, what tools to use, and coding standards. |
| **Spec-Kit** | **The Architect** | Manages spec artifacts (`.specify`, `.plan`, `.tasks`). Ensures technical rigor before coding starts. |
| **AI Agents** | **The Executor** | Reads the project memory and executes tasks via tools and verified edits. |

---

## How Agents Must Work

Every agent in this project MUST obey these non-negotiable rules:

1. **Never generate code without a referenced Task ID.**
2. **Never modify architecture without updating `plan.md` / `speckit.plan`.**
3. **Never propose features without updating `spec.md` / `speckit.specify` (WHAT).**
4. **Never change fundamental approaches without updating `constitution.md` / `speckit.constitution` (Principles).**
5. **Every code file should map directly back to a Task and Spec section.**
6. **Strict Environment Security**: NEVER inspect, read, or edit any `.env` file (ONLY `.env.example` may be examined or referenced).

If an agent cannot find the required spec, it must **stop and request it**, not improvise.

---

## Spec-Kit Workflow (Source of Truth)

### 1. Constitution (WHY — Principles & Constraints)
**File**: `speckit.constitution` / `.specify/memory/constitution.md`  
Defines the project's non-negotiables: architectural values, security rules, tech stack constraints, performance expectations, and patterns allowed.

### 2. Specify (WHAT — Requirements, Journeys & Acceptance Criteria)
**File**: `specs/<feature>/spec.md`  
Contains:
- User journeys and user stories
- Functional and non-functional requirements
- Acceptance criteria and edge cases
- Domain rules and data models

### 3. Plan (HOW — Architecture, Components, Interfaces)
**File**: `specs/<feature>/plan.md`  
Contains:
- Component breakdown and service boundaries
- API schemas and contracts
- Database models and migrations
- Security and deployment topology

### 4. Tasks (EXECUTION — Ordered Checklist)
**File**: `specs/<feature>/tasks.md`  
Contains:
- Granular, testable steps
- Dependencies between tasks
- Explicit verification criteria for each task

### 5. Implement (CODE — Controlled Generation)
Code is generated strictly conforming to tasks and verified with automated test suites:
- Python: `pytest`
- TypeScript / Next.js: `npm run build` and `npx tsc --noEmit`

---

## Monorepo Phase Organization

| Phase | Directory | Description | Stack |
| :--- | :--- | :--- | :--- |
| **Phase I** | `PHASE-1/` | In-Memory Python Console Todo App | Python 3.10+, pytest |
| **Phase II** | `PHASE-2/` | Full-Stack Web App | Next.js, FastAPI, SQLModel, Neon, Clerk |
| **Phase III** | `PHASE-3/` | AI-Powered Todo Chatbot | FastAPI, OpenAI Agents SDK, MCP, Next.js |
| **Phase IV** | `PHASE-4/` | Local Kubernetes Deployment | Helm, Docker, Minikube, FastAPI, Next.js |
| **Phase V** | `PHASE-5/` | Advanced Cloud Deployment | Dapr (Pub/Sub, State), Kafka, Redis, K8s, FastAPI, Next.js |

---

## Developer–Agent Alignment

Humans and agents collaborate, but the **spec is the single source of truth**.  
Before modifying code in any phase:
1. Verify the phase-level spec under `PHASE-X/specs/`.
2. Ensure test coverage exists or is written first.
3. Validate that all existing unit tests and type checks pass cleanly without regressions.

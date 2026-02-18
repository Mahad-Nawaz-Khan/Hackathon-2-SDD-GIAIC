---
id: 002
title: advanced-todo-plan
stage: plan
date: 2026-02-16
surface: agent
model: claude-opus-4-6
feature: 013-advanced-todo-features
branch: 013-advanced-todo-features
user: madoo
command: /sp.plan
labels: ["plan","event-driven","dapr"]
links:
  spec: specs/013-advanced-todo-features/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/013-advanced-todo-features/plan.md
  - specs/013-advanced-todo-features/data-model.md
  - specs/013-advanced-todo-features/contracts/api-contracts.md
tests:
  - N/A (planning phase)
---

## Prompt

Architecture:
- **Pattern**: Event-Driven Microservices with Dapr Sidecar.
- **Data Flow**:
  - Direct SQLModel/Postgres for CRUD, Search, and Filtering (Intermediate features).
  - Dapr Pub/Sub (Kafka) for asynchronous logic (Recurring Tasks).
  - Dapr Jobs API for scheduled triggers (Reminders).

Phases:
1. **Data Layer Evolution**:
   - Update `Task` SQLModel to include `priority` (enum), `tags` (JSON/Array), `due_date` (datetime), `reminder_time` (datetime), and `recurrence_rule` (string).
   - Generate and apply Alembic migrations.

2. **Backend Logic (Intermediate)**:
   - Refactor `GET /api/tasks` to support dynamic filtering (SQL `WHERE` clauses) and sorting (SQL `ORDER BY`).
   - Update Create/Update endpoints to handle new fields.

3. **Backend Logic (Advanced - Dapr)**:
   - Implement Dapr Subscriber endpoint (`/dapr/subscribe`) to listen to `task-events` topic.
   - Implement `POST /api/events/task-completed` handler: When a recurring task is finished, calculate next due date and create new task.
   - Implement `POST /api/jobs/trigger` handler: Receives Dapr Job callbacks and processes reminders.
   - Update Task Creation logic to register Dapr Jobs when `reminder_time` is set.

4. **Frontend Implementation**:
   - **State**: Update TypeScript interfaces to match new backend schema.
   - **UI Components**:
     - Add `PriorityBadge` and `TagList` components.
     - Add `SearchFilterSidebar` component.
     - Upgrade `TaskForm` with DatePicker and Recurrence dropdown.
   - **Integration**: wire up new filters to the API query parameters.

Design Decisions (ADRs) to Document:
- **Search Strategy**: Use SQL-based filtering (Postgres) instead of Dapr State Store query API (too limited for complex sorting/filtering).
- **Recurrence Engine**: Use "Just-in-Time" creation (create next task only when current is completed) via Pub/Sub, rather than a cron job creating all future tasks at once.
- **Reminders**: Use Dapr Jobs API over internal Python schedulers (APScheduler) to ensure statelessness and horizontal scalability.

Dependencies:
- `dapr-python-sdk` or `httpx` for Dapr communication.
- `sqlmodel` for database schema updates.
- `shadcn/ui` (or existing UI lib) for DatePicker components.

## Response snapshot

Created comprehensive implementation plan for Advanced Todo Features with:

**Architecture**: Event-driven microservices with Dapr sidecar. Direct SQLModel for CRUD/filtering (Intermediate) and Dapr Pub/Sub/Jobs for async features (Advanced).

**Key Design Decisions**:
1. SQL-based filtering: PostgreSQL for complex queries (not Dapr State Store)
2. Just-in-time recurrence: Create next task instance only when current completes
3. Dapr Jobs API: Stateless reminder scheduling (not Python schedulers)

**Phase Structure**:
- Phase 1: Data Layer (add `reminder_time` field, migration)
- Phase 2: Event-Driven Backend (Dapr service, subscriber endpoints)
- Phase 3: Frontend Components (PriorityBadge, SearchFilterSidebar, TaskForm updates)

**Documents Created**:
- plan.md: Full implementation plan with architecture diagrams
- data-model.md: Entity relationships, event schemas, migration script
- contracts/api-contracts.md: API contracts for all endpoints

**Constitution Compliance**: All event-driven features use Dapr HTTP APIs (no direct Kafka clients). Infrastructure deployment out of scope per spec.

## Outcome

- Impact: Complete technical design for Intermediate + Advanced features with Dapr event-driven architecture
- Tests: N/A (planning phase)
- Files: plan.md, data-model.md, contracts/api-contracts.md created
- Next prompts: `/sp.tasks` for implementation tasks or `/sp.adr` to document design decisions
- Reflection: Existing codebase already has priority/due_date/recurrence_rule fields. Plan focuses on reminder_time addition and event-driven implementation

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results: N/A (planning phase)
- Prompt variant: N/A
- Next experiment: Validate tasks.md against plan estimates once generated

📋 Architectural decision detected: Event-driven recurring tasks with Dapr Pub/Sub
   Document reasoning and tradeoffs? Run `/sp.adr recurring-tasks-event-driven`

📋 Architectural decision detected: Dapr Jobs API for stateless reminder scheduling
   Document reasoning and tradeoffs? Run `/sp.adr dapr-jobs-reminders`

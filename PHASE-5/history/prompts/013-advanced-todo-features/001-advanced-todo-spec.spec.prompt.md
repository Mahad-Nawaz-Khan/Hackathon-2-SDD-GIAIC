---
id: 001
title: advanced-todo-spec
stage: spec
date: 2026-02-16
surface: agent
model: claude-opus-4-6
feature: 013-advanced-todo-features
branch: 013-advanced-todo-features
user: madoo
command: /sp.specify
labels: ["spec","event-driven","todos"]
links:
  spec: specs/013-advanced-todo-features/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/013-advanced-todo-features/spec.md
  - specs/013-advanced-todo-features/checklists/requirements.md
tests:
  - N/A (specification phase)
---

## Prompt

Phase V: Intermediate & Advanced Feature Implementation

## Intent
Upgrade the existing Todo application to support "Intermediate" organizational features and "Advanced" intelligent, event-driven capabilities. This specification focuses on the Application Layer (Frontend/Backend) and Data Layer changes required before deployment.

## Scope & Features

### 1. Intermediate Level (Organization & Usability)
- **Priorities & Tags**:
  - Add `priority` field (High, Medium, Low) and `tags` (Array of strings, e.g., "Work", "Home") to Task model.
  - UI: Color-coded badges for priorities/tags.
- **Search & Filter**:
  - Backend: Update `GET /api/tasks` to accept query params: `search` (keyword), `priority`, `status`, `tags`.
  - Frontend: Add search bar and filter sidebar/dropdowns.
- **Sorting**:
  - Backend: Support sorting by `due_date`, `priority`, `created_at` (asc/desc).
  - Frontend: Sort controls in the task list header.

### 2. Advanced Level (Event-Driven & Intelligent)
- **Recurring Tasks**:
  - Add `recurrence_rule` (e.g., "DAILY", "WEEKLY", or CRON string) to Task model.
  - **Architecture**: When a recurring task is completed, publish a `task.completed` event to Dapr Pub/Sub (Topic: `task-events`). A backend subscriber must listen to this and create the *next* instance of the task.
- **Due Dates & Reminders**:
  - Add `due_date` and `reminder_time` to Task model.
  - UI: Date/Time picker in Create/Update forms.
  - **Architecture**: When a task with `reminder_time` is created/updated, use **Dapr Jobs API** to schedule a precise callback. When triggered, publish to Dapr Pub/Sub (Topic: `reminders`) for the Notification Service (stub for now).

## Technical Implementation (Dapr-First)
- **Backend (FastAPI)**:
  - DO NOT use `kafka-python`. Use `dapr-python-sdk` or `httpx` to call Dapr sidecar (`http://localhost:3500/v1.0/publish/...`).
  - Create Dapr subscription endpoints (e.g., `/dapr/subscribe`) to listen for `task-events`.
- **Frontend (Next.js)**:
  - Update interfaces/types to match new schema.
  - Ensure UI components handle the new complex state (e.g., recurrence settings).

## Success Criteria (SMART)
- ✅ **Schema**: Database schema updated to include priority, tags, due_date, recurrence.
- ✅ **Search**: Searching "meeting" returns only relevant tasks within <200ms.
- ✅ **Events**: Marking a "Daily Standup" task as complete triggers a `task.completed` event (visible in logs) and auto-creates tomorrow's task via the subscriber.
- ✅ **Jobs**: Setting a reminder schedules a Dapr Job (verified via Dapr dashboard or logs).
- ✅ **UI**: Users can sort by Priority High->Low and see visual indicators for overdue tasks.

## Non-Goals
- Infrastructure deployment (Minikube/K8s/Cloud) - Spec specifically for App Logic.
- CI/CD pipelines.
- External Email/Push Notification integration (Logic only, use log outputs for verification).

## Response snapshot

Created specification for "Advanced Todo Features" with 4 prioritized user stories:

**P1 - Task Organization & Filtering**: Priorities (High/Medium/Low), tags, search, filter, and sort capabilities.

**P2 - Due Date Management**: Date picker, overdue indicators, visual styling for time-sensitive tasks.

**P3 - Recurring Tasks**: Event-driven recurring task creation using message bus events. When completed, triggers automatic next instance creation.

**P4 - Smart Reminders**: Scheduled callbacks via Jobs API that publish reminder events for notifications.

**Specification Quality**: All checklist items passed. No [NEEDS CLARIFICATION] markers. Requirements are testable and measurable. Success criteria are technology-agnostic and user-focused.

**Branch**: `013-advanced-todo-features`
**Spec File**: `specs/013-advanced-todo-features/spec.md`

## Outcome

- Impact: Complete specification with 45 functional requirements across 4 priority levels
- Tests: N/A (specification phase)
- Files: spec.md created, requirements.md checklist created
- Next prompts: `/sp.plan` for architecture design or `/sp.clarify` for refinement
- Reflection: Successfully abstracted event-driven architecture requirements into business terms (publish events, schedule callbacks) without leaking implementation details

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results: Checklist validation PASSED (all 12 items)
- Prompt variant: N/A
- Next experiment: Validate plan.md constitution compliance when generated

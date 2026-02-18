---
id: 004
title: advanced-todo-implementation
stage: red
date: 2026-02-16
surface: agent
model: claude-opus-4-6
feature: 013-advanced-todo-features
branch: 013-advanced-todo-features
user: madoo
command: /sp.implement
labels: ["implementation","mvp"]
links:
  spec: specs/013-advanced-todo-features/spec.md
  plan: specs/013-advanced-todo-features/plan.md
  ticket: null
  adr: null
  pr: null
files:
  - backend/requirements.txt
  - backend/alembic/versions/001_add_reminder_time_to_tasks.py
  - backend/src/models/task.py
  - backend/src/schemas/task.py
  - backend/src/services/dapr_service.py
  - backend/src/api/dapr_router.py
  - backend/src/api/jobs_router.py
  - backend/src/main.py
  - backend/src/services/task_service.py
  - backend/src/api/task_router.py
  - frontend/src/types/task.ts
  - frontend/src/components/PriorityBadge.tsx
  - frontend/src/components/SearchFilterSidebar.tsx
  - frontend/src/components/SortControls.tsx
  - frontend/src/hooks/useTasks.ts
tests:
  - N/A (implementation in progress)
---

## Prompt

(N/A - invoked via command)

## Response snapshot

Started implementation of Advanced Todo Features (MVP: User Story 1 - Task Organization & Filtering).

**COMPLETED PHASES**:
- Phase 1 (Setup): Added python-dateutil dependency, created Alembic migration for reminder_time
- Phase 2 (Foundational): Updated Task model, schemas, created DaprService, registered dapr_router and jobs_router
- Phase 3 (User Story 1 - IN PROGRESS): Added tag filtering to task_service, created frontend types and components

**FILES MODIFIED**:
- backend/requirements.txt: Added python-dateutil>=2.8.0
- backend/src/models/task.py: Added reminder_time to TaskBase
- backend/src/schemas/task.py: Added reminder_time to TaskResponse, TaskCreateRequest, TaskUpdateRequest
- backend/src/services/task_service.py: Added tags parameter with GROUP BY HAVING logic, case import for priority sorting
- backend/src/api/task_router.py: Added tags query parameter to GET /tasks, added reminder_time to _task_to_response
- backend/src/main.py: Registered dapr_router and jobs_router

**FILES CREATED**:
- backend/alembic/versions/001_add_reminder_time_to_tasks.py: Alembic migration for reminder_time column
- backend/src/services/dapr_service.py: Dapr HTTP API wrapper (publish_event, schedule_job, delete_job, state management)
- backend/src/api/dapr_router.py: Dapr subscription endpoints (/dapr/subscribe, /api/events/task-completed, /api/events/reminder-triggered)
- backend/src/api/jobs_router.py: Dapr Jobs callback endpoint (/api/jobs/trigger)
- frontend/src/types/task.ts: TypeScript types (Task, Tag, Priority, RecurrenceRule, TaskFilters, PRIORITY_COLORS, TASK_STATUS helpers)
- frontend/src/components/PriorityBadge.tsx: Color-coded priority badge component
- frontend/src/components/SearchFilterSidebar.tsx: Search and filter sidebar component
- frontend/src/components/SortControls.tsx: Sort controls component
- frontend/src/hooks/useTasks.ts: Task management hooks (useTasks, useCreateTask, useUpdateTask, useToggleTaskCompletion, useDeleteTask)

**REMAINING WORK FOR MVP**:
- T020-T022: Frontend integration (TaskList component updates, page integration)
- Phase 4-7: Additional user stories (Due Dates, Recurring Tasks, Reminders, Polish)

## Outcome

- Impact: Core infrastructure for advanced todo features implemented. Backend supports tag filtering with AND semantics, frontend components created for search/filter/sort UI.
- Tests: N/A (implementation in progress)
- Files: 17 files created/modified
- Next prompts: Continue frontend integration (T020-T022), then proceed to Phase 4 (Due Dates) or run full implementation again
- Reflection: Dapr-first architecture enforced - all event/Job communication goes through DaprService wrapper. Tag filtering uses proper GROUP BY HAVING for AND semantics.

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results: N/A (implementation in progress)
- Prompt variant: N/A
- Next experiment: Complete frontend integration to enable end-to-end testing of tag filtering feature

# Tasks: Advanced Todo Features (Intermediate & Advanced)

**Input**: Design documents from `/specs/013-advanced-todo-features/`
**Prerequisites**: plan.md, spec.md, data-model.md, contracts/api-contracts.md

**Tests**: Tests are OPTIONAL for this feature. The specification focuses on application layer functionality with log-based verification for event-driven features. Test tasks are not included unless explicitly requested during implementation.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Web application**: `backend/src/`, `frontend/src/`
- All paths are relative to repository root: `F:\MADOO\Governor House\Hackathon 2\PHASE-5/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Database migration and dependency setup

- [X] T001 Add python-dateutil dependency to backend/requirements.txt for recurrence calculation
- [X] T002 Create Alembic migration for reminder_time column in backend/alembic/versions/XXX_add_reminder_time_to_tasks.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T003 Add reminder_time field to TaskBase class in backend/src/models/task.py
- [X] T004 [P] Add reminder_time field to TaskResponse in backend/src/schemas/task.py
- [X] T005 [P] Add reminder_time field to TaskCreateRequest in backend/src/schemas/task.py
- [X] T006 [P] Add reminder_time field to TaskUpdateRequest in backend/src/schemas/task.py
- [X] T007 Create DaprService wrapper class in backend/src/services/dapr_service.py with publish_event, schedule_job, delete_job methods
- [X] T008 Register dapr_router in backend/src/main.py to include Dapr subscription endpoints

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Task Organization & Filtering (Priority: P1) 🎯 MVP

**Goal**: Enable users to organize tasks by priority/tags, search, filter, and sort

**Independent Test**: Create tasks with different priorities/tags, then search and filter to verify only matching tasks appear. Test sorting by priority. Verify combined filters (priority + tag + search) work correctly.

### Implementation for User Story 1

- [X] T009 [P] [US1] Add tags parameter to get_tasks method signature in backend/src/services/task_service.py
- [X] T010 [P] [US1] Add tag filtering logic with GROUP BY HAVING to get_tasks method in backend/src/services/task_service.py
- [X] T011 [US1] Add tags query parameter to GET /tasks endpoint in backend/src/api/task_router.py
- [X] T012 [P] [US1] Add reminder_time to _task_to_response helper in backend/src/api/task_router.py
- [X] T013-T015 [P] [US1] Create frontend Task types in frontend/src/types/task.ts
- [X] T016 [P] [US1] Create PriorityBadge component in frontend/src/components/PriorityBadge.tsx
- [X] T017 [P] [US1] Create SearchFilterSidebar component in frontend/src/components/SearchFilterSidebar.tsx
- [X] T018 [P] [US1] Create SortControls component in frontend/src/components/SortControls.tsx
- [X] T019 [US1] Create useTasks hook in frontend/src/hooks/useTasks.ts
- [ ] T020 [US1] Update TaskList component in frontend/src/components/TaskList.tsx to display PriorityBadge and TagList badges
- [ ] T021 [US1] Add SearchFilterSidebar and SortControls to task page in frontend/src/app/page.tsx or appropriate layout
- [ ] T022 [US1] Update task list API call to pass filter parameters from frontend

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Due Date Management (Priority: P2)

**Goal**: Enable users to set due dates and see visual indicators for overdue tasks

**Independent Test**: Create tasks with past/today/future due dates, verify overdue tasks show red indicator, due today shows yellow indicator, sorting by due date works correctly

### Implementation for User Story 2

- [ ] T023 [P] [US2] Add due_date input field to TaskForm component in frontend/src/components/TaskForm.tsx using HTML datetime-local input
- [ ] T024 [US2] Add overdue/due today calculation logic to TaskList component in frontend/src/components/TaskList.tsx
- [ ] T025 [US2] Add overdue visual styling (red/bold) to task items in frontend/src/components/TaskList.tsx
- [ ] T026 [US2] Add due today visual styling (yellow/orange) to task items in frontend/src/components/TaskList.tsx
- [ ] T027 [US2] Handle tasks without due dates in sort (appear at end) in frontend/src/components/TaskList.tsx or useTasks hook

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Recurring Tasks (Priority: P3)

**Goal**: Enable users to create recurring tasks that auto-regenerate when completed via event-driven architecture

**Independent Test**: Create "Daily Standup" recurring task, mark complete, verify new instance for tomorrow is created automatically. Check logs for task.completed event.

### Implementation for User Story 3

- [ ] T028 [P] [US3] Create RecurrenceService in backend/src/services/recurrence_service.py with create_next_instance and _calculate_next_due_date methods
- [ ] T029 [P] [US3] Add recurrence rule options to TaskForm component in frontend/src/components/TaskForm.tsx (DAILY, WEEKLY, MONTHLY dropdown)
- [ ] T030 [P] [US3] Create dapr_router with task-completed and reminder-triggered endpoints in backend/src/api/dapr_router.py
- [ ] T031 [P] [US3] Add dapr/subscribe endpoint to dapr_router in backend/src/api/dapr_router.py returning subscription configurations
- [ ] T032 [P] [US3] Add _calculate_next_due_date method to RecurrenceService in backend/src/services/recurrence_service.py handling DAILY, WEEKLY, MONTHLY with edge cases
- [ ] T033 [US3] Modify toggle_task_completion in backend/src/services/task_service.py to publish task.completed event via DaprService when recurring task is completed
- [ ] T034 [US3] Import recurrence_rule enum to frontend Task type in frontend/src/types/task.ts
- [ ] T035 [US3] Add recurrence_rule field to TaskCreateRequest in frontend/src/components/TaskForm.tsx

**Checkpoint**: At this point, User Stories 1, 2, AND 3 should all work independently

---

## Phase 6: User Story 4 - Smart Reminders (Priority: P4)

**Goal**: Enable users to set reminders that trigger via Dapr Jobs API and publish notification events

**Independent Test**: Create task with reminder 5 minutes in future, wait for trigger, verify reminder event is published and logged

### Implementation for User Story 4

- [ ] T036 [P] [US4] Create jobs_router with trigger endpoint in backend/src/api/jobs_router.py
- [ ] T037 [P] [US4] Add schedule_reminder method to DaprService in backend/src/services/dapr_service.py that calls Dapr Jobs API
- [ ] T038 [P] [US4] Add cancel_reminder method to DaprService in backend/src/services/dapr_service.py that deletes Dapr job
- [ ] T039 [P] [US4] Add reminder_time input field to TaskForm component in frontend/src/components/TaskForm.tsx using HTML datetime-local input
- [ ] T040 [US4] Modify create_task in backend/src/services/task_service.py to schedule Dapr job when reminder_time is set
- [ ] T041 [US4] Modify update_task in backend/src/services/task_service.py to reschedule/cancel Dapr job when reminder_time changes
- [ ] T042 [US4] Modify delete_task in backend/src/services/task_service.py to cancel pending Dapr job when task is deleted
- [ ] T043 [US4] Modify toggle_task_completion in backend/src/services/task_service.py to cancel pending reminder when task completed before reminder time
- [ ] T044 [US4] Register jobs_router in backend/src/main.py to include job trigger endpoint

**Checkpoint**: All user stories should now be independently functional

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T045 [P] Run database migration (alembic upgrade head) and verify reminder_time column exists
- [ ] T046 [P] Test search performance with 100+ tasks to verify <200ms target
- [ ] T047 [P] Add structured logging for all Dapr event publishing in backend/src/services/dapr_service.py
- [ ] T048 [P] Add structured logging for all Dapr job scheduling in backend/src/services/dapr_service.py
- [ ] T049 Verify event schema validation for task.completed events (check all required fields present)
- [ ] T050 Verify event schema validation for reminder.triggered events (check all required fields present)
- [ ] T051 Test combined filters (priority + tag + search + sort) return correct results
- [ ] T052 Test recurring task completion creates next instance with consistent properties
- [ ] T053 Test reminder rescheduling cancels old job and schedules new one
- [ ] T054 Test task deletion cancels pending reminder job
- [ ] T055 [P] Add validation: reminder_time must be before due_date when both set in backend/src/services/task_service.py
- [ ] T056 [P] Add validation: reminder_time must be in the future when creating task in backend/src/services/task_service.py
- [ ] T057 Update README.md with new API parameters (tags filter, reminder_time field)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion (T001-T002) - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion (T003-T008)
  - User stories can proceed in parallel after foundation is ready
  - Or sequentially in priority order: P1 → P2 → P3 → P4
- **Polish (Phase 7)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Builds on US1 for UI consistency but independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Independent of US1/US2, uses existing Task model
- **User Story 4 (P4)**: Can start after Foundational (Phase 2) - Builds on due dates from US2 but independently testable

### Within Each User Story

**User Story 1**:
- Backend changes (T009-T012) can run in parallel
- Frontend types (T013-T015) can run in parallel
- Frontend components (T016-T018) can run in parallel
- Integration (T019-T022) depends on previous components

**User Story 2**:
- All tasks can run in parallel (different UI components)

**User Story 3**:
- Backend services (T028) must come before event integration (T033)
- Frontend components (T029, T034-T035) can run in parallel
- Dapr endpoints (T030-T032) can run in parallel

**User Story 4**:
- Dapr service methods (T037-T038) must come before task service integration (T040-T043)
- UI component (T039) can run in parallel
- Jobs router (T036) must be registered (T044) after creation

### Parallel Opportunities

**After Foundational Phase Completes**:
```bash
# All four user stories can be worked on in parallel by different developers:
Developer A: T009-T022 (User Story 1 - Task Organization)
Developer B: T023-T027 (User Story 2 - Due Dates)
Developer C: T028-T035 (User Story 3 - Recurring Tasks)
Developer D: T036-T044 (User Story 4 - Reminders)
```

**Within User Story 1**:
```bash
# Backend parallel:
Task T009: Add tags parameter to task_service
Task T012: Update _task_to_response helper

# Frontend types parallel:
Task T013: Create Priority type
Task T014: Add reminder_time to Task interface
Task T015: Create TaskFilters interface

# Frontend components parallel:
Task T016: Create PriorityBadge
Task T017: Create SearchFilterSidebar
Task T018: Create SortControls
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T002)
2. Complete Phase 2: Foundational (T003-T008) - CRITICAL
3. Complete Phase 3: User Story 1 (T009-T022)
4. **STOP and VALIDATE**: Test search, filter, sort independently
5. Deploy/demo MVP

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add US1 (Task Organization) → Test independently → Deploy/Demo (MVP!)
3. Add US2 (Due Dates) → Test independently → Deploy/Demo
4. Add US3 (Recurring Tasks) → Test independently → Deploy/Demo
5. Add US4 (Reminders) → Test independently → Deploy/Demo
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With 4 developers after Foundational phase:
- Developer A: User Story 1 (Task Organization)
- Developer B: User Story 2 (Due Dates)
- Developer C: User Story 3 (Recurring Tasks)
- Developer D: User Story 4 (Reminders)

Stories complete independently and integrate seamlessly.

---

## Summary

| Metric | Count |
|--------|-------|
| **Total Tasks** | 57 |
| **Setup Tasks** | 2 |
| **Foundational Tasks** | 6 |
| **User Story 1 Tasks** | 14 |
| **User Story 2 Tasks** | 5 |
| **User Story 3 Tasks** | 8 |
| **User Story 4 Tasks** | 9 |
| **Polish Tasks** | 13 |
| **Parallel Opportunities** | 30+ tasks marked [P] |

**MVP Scope**: Tasks T001-T022 (Setup + Foundational + User Story 1) = 22 tasks for initial delivery

**Format Validation**: All tasks follow the required checklist format with checkbox, Task ID, optional [P] marker, optional [Story] label, and file paths.

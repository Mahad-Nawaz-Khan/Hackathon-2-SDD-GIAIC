---
description: "Task list for Console TODO Application implementation"
---

# Tasks: Console TODO Application

**Input**: Design documents from `/specs/001-console-todo-app/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create project directory structure per implementation plan
- [x] T002 [P] Create todo_app/ directory structure with models/, core/, ui/ subdirectories
- [x] T003 [P] Create main.py entry point file

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 [P] Create Task class/data structure in todo_app/models/task.py
- [x] T005 [P] Implement JSON file persistence functions (load and save) in todo_app/core/persistence.py
- [x] T006 [P] Create main menu loop structure with user input handling in todo_app/ui/menu.py
- [x] T007 Create base task management functions (add_task, view_tasks, update_task, delete_task, toggle_task_completion) in todo_app/core/tasks.py
- [x] T008 [P] Create input validation functions in todo_app/core/validation.py
- [x] T009 Create constants and configuration in todo_app/config.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Add and View Tasks (Priority: P1) 🎯 MVP

**Goal**: Allow users to add new tasks and view all existing tasks with completion status

**Independent Test**: User can add a task with a title and see it in the task list with a checkbox indicating completion status

### Tests for User Story 1 (OPTIONAL - only if tests requested) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T010 [P] [US1] Unit test for add_task function in tests/unit/test_tasks.py
- [x] T011 [P] [US1] Unit test for view_tasks function in tests/unit/test_tasks.py

### Implementation for User Story 1

- [x] T012 [P] [US1] Implement add_task function in todo_app/core/tasks.py (depends on T004)
- [x] T013 [P] [US1] Implement view_tasks function in todo_app/core/tasks.py (depends on T004)
- [x] T014 [US1] Integrate add_task functionality into main menu in todo_app/main.py (depends on T012)
- [x] T015 [US1] Integrate view_tasks functionality into main menu in todo_app/main.py (depends on T013)
- [x] T016 [US1] Add input validation for task titles in todo_app/core/validation.py (depends on T008)
- [x] T017 [US1] Add error handling for empty task titles in todo_app/core/tasks.py (depends on T012)
- [x] T018 [US1] Add unique ID generation mechanism in todo_app/core/tasks.py (depends on T004)

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Update and Delete Tasks (Priority: P2)

**Goal**: Allow users to modify or remove existing tasks when their requirements change

**Independent Test**: User can select a task by ID and modify its title/description or remove it entirely from the list

### Tests for User Story 2 (OPTIONAL - only if tests requested) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T019 [P] [US2] Unit test for update_task function in tests/unit/test_tasks.py
- [x] T020 [P] [US2] Unit test for delete_task function in tests/unit/test_tasks.py

### Implementation for User Story 2

- [x] T021 [P] [US2] Implement update_task function in todo_app/core/tasks.py (depends on T004, T007)
- [x] T022 [P] [US2] Implement delete_task function in todo_app/core/tasks.py (depends on T004, T007)
- [x] T023 [US2] Integrate update_task functionality into main menu in todo_app/main.py (depends on T021)
- [x] T024 [US2] Integrate delete_task functionality into main menu in todo_app/main.py (depends on T022)
- [x] T025 [US2] Add validation for task ID existence in todo_app/core/validation.py (depends on T008)
- [x] T026 [US2] Add error handling for invalid task IDs in todo_app/core/tasks.py (depends on T021, T022)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Mark Tasks as Complete (Priority: P3)

**Goal**: Allow users to track which tasks have been completed to maintain an accurate view of their outstanding work

**Independent Test**: User can select a task by ID and toggle its completion status, with the change immediately visible in the task list

### Tests for User Story 3 (OPTIONAL - only if tests requested) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [x] T027 [P] [US3] Unit test for toggle_task_completion function in tests/unit/test_tasks.py

### Implementation for User Story 3

- [x] T028 [P] [US3] Implement toggle_task_completion function in todo_app/core/tasks.py (depends on T004, T007)
- [x] T029 [US3] Integrate toggle_task_completion functionality into main menu in todo_app/main.py (depends on T028)
- [x] T030 [US3] Add visual indicators for completion status in view_tasks display in todo_app/core/tasks.py (depends on T013)

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Error Handling & Validation (Cross-cutting concerns)

**Goal**: Implement comprehensive error handling for all user inputs and edge cases

- [x] T031 [P] Implement validation for numeric input in todo_app/core/validation.py
- [x] T032 [P] Implement validation for menu selection in todo_app/ui/menu.py
- [x] T033 Add error handling for empty task list in view_tasks function in todo_app/core/tasks.py
- [x] T034 Add re-prompt functionality after invalid input in todo_app/ui/menu.py
- [x] T035 Add meaningful error messages throughout the application

---

## Phase 7: Persistence Implementation

**Goal**: Implement data persistence using JSON files

- [x] T036 [P] Integrate load_tasks on startup in main.py
- [x] T037 [P] Integrate save_tasks after each mutation (add, update, delete, toggle) in todo_app/main.py
- [x] T038 Add error handling for missing or corrupt JSON files in todo_app/core/persistence.py
- [x] T039 Test persistence functionality across application restarts

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T040 [P] Documentation updates in README.md
- [x] T041 Code cleanup and refactoring
- [x] T042 [P] Additional unit tests (if requested) in tests/unit/
- [x] T043 Final integration testing
- [x] T044 Run quickstart.md validation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Cross-cutting phases (6+)**: Depend on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
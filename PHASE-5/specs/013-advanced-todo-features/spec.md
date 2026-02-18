# Feature Specification: Advanced Todo Features (Intermediate & Advanced)

**Feature Branch**: `013-advanced-todo-features`
**Created**: 2026-02-16
**Status**: Draft
**Input**: User description: "Phase V: Intermediate & Advanced Feature Implementation"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Task Organization & Filtering (Priority: P1)

As a busy professional managing multiple projects, I want to organize my tasks by priority and category, and quickly find specific tasks through search and filtering, so that I can focus on the most important work without scrolling through endless task lists.

**Why this priority**: Task organization is foundational - without it, users struggle to manage even moderate task volumes. This delivers immediate value by making existing tasks more manageable.

**Independent Test**: Can be fully tested by creating tasks with different priorities/tags, then searching and filtering to verify only matching tasks appear. Delivers task management value without requiring event-driven features.

**Acceptance Scenarios**:

1. **Given** I have 20 tasks with various priorities, **When** I filter by "High" priority, **Then** only High priority tasks are displayed
2. **Given** I have tasks tagged "Work" and "Personal", **When** I select the "Work" tag filter, **Then** only Work-related tasks are shown
3. **Given** I have many tasks, **When** I search for "meeting", **Then** only tasks containing "meeting" in the title or description appear within 200ms
4. **Given** I have a list of tasks, **When** I sort by priority, **Then** tasks are ordered High → Medium → Low
5. **Given** I have a list of tasks, **When** I apply multiple filters (priority + tag + search), **Then** only tasks matching ALL criteria are displayed

---

### User Story 2 - Due Date Management (Priority: P2)

As a user with time-sensitive tasks, I want to set due dates and see visual indicators for overdue tasks, so that I can prioritize time-critical work and avoid missing deadlines.

**Why this priority**: Due dates are essential for time management but don't require complex event-driven architecture. Builds on P1 by adding temporal organization.

**Independent Test**: Can be tested by creating tasks with various due dates (past, today, future), then verifying overdue tasks are visually highlighted and sorting by due date works correctly.

**Acceptance Scenarios**:

1. **Given** I have a task due yesterday, **When** I view my task list, **Then** the task shows a clear "overdue" visual indicator (red/bold)
2. **Given** I have multiple tasks with different due dates, **When** I sort by due date, **Then** tasks are ordered from most urgent to least urgent
3. **Given** I am creating a new task, **When** I select a due date from the date picker, **Then** the date is saved and displayed in the task list
4. **Given** I have a task due today, **When** I view my task list, **Then** the task shows a "due today" visual indicator (yellow/orange)
5. **Given** I have a task with no due date, **When** I sort by due date, **Then** these tasks appear at the end

---

### User Story 3 - Recurring Tasks (Priority: P3)

As a user with repetitive responsibilities, I want to create recurring tasks (e.g., "Daily Standup", "Weekly Report") that automatically regenerate when completed, so that I don't have to manually recreate the same task every day/week.

**Why this priority**: Recurring tasks provide significant value for users with repeating responsibilities, but require event-driven architecture making them more complex to implement.

**Independent Test**: Can be tested by creating a "Daily Standup" recurring task, marking it complete, and verifying a new instance for tomorrow is automatically created with the same recurrence rule.

**Acceptance Scenarios**:

1. **Given** I create a task with "DAILY" recurrence, **When** I complete today's instance, **Then** a new task for tomorrow is automatically created with the same recurrence rule
2. **Given** I create a task with "WEEKLY" recurrence on Monday, **When** I complete the Monday instance, **Then** a new task for next Monday is automatically created
3. **Given** I have a recurring task, **When** I delete a single instance, **Then** the recurrence pattern continues for future instances
4. **Given** I have a recurring task, **When** I edit the recurrence rule, **Then** future instances follow the new rule
5. **Given** I complete a recurring task, **When** the completion event is published, **Then** the event is visible in system logs

---

### User Story 4 - Smart Reminders (Priority: P4)

As a user who forgets tasks, I want to set reminders that notify me before a task is due, so that I can complete tasks on time without constantly checking my task list.

**Why this priority**: Reminders are valuable but require the most complex event-driven architecture (Dapr Jobs API + Pub/Sub). Builds on due dates from P2.

**Independent Test**: Can be tested by creating a task with a reminder 5 minutes in the future, waiting for the trigger, and verifying a reminder event is published and logged.

**Acceptance Scenarios**:

1. **Given** I create a task with a reminder for 2:00 PM today, **When** the time reaches 2:00 PM, **Then** a reminder event is published to the notification system
2. **Given** I edit a task's reminder time, **When** the new reminder time is reached, **Then** the reminder triggers at the new time (not the old time)
3. **Given** I complete a task before its reminder, **When** the reminder time passes, **Then** no reminder event is published
4. **Given** I delete a task with a pending reminder, **When** the reminder time passes, **Then** no reminder event is published
5. **Given** a reminder is triggered, **When** the notification system receives the event, **Then** the event is logged for verification

---

### Edge Cases

- What happens when a user sets a recurrence rule that would create an infinite loop?
- How does the system handle timezones for recurring tasks and reminders?
- What happens when a recurring task's next occurrence would be in the past (e.g., system downtime)?
- How does the system handle deleted tags in existing task filters?
- What happens when search contains special characters or SQL injection patterns?
- What happens when a user sets a reminder in the past?
- What happens when the event bus is down and a recurring task is completed?
- What happens when recurrence calculation falls on a non-existent date (e.g., February 30)?

## Requirements *(mandatory)*

### Functional Requirements

#### Task Organization (P1)

- **FR-001**: System MUST allow users to assign a priority level (High, Medium, Low) to each task
- **FR-002**: System MUST allow users to assign multiple tags to each task (tags are free-form text strings)
- **FR-003**: System MUST provide color-coded visual indicators for priority levels (e.g., High=red, Medium=yellow, Low=green)
- **FR-004**: System MUST display tags as visual badges on tasks in the task list
- **FR-005**: System MUST support searching tasks by keyword across task title and description
- **FR-006**: System MUST support filtering tasks by priority level
- **FR-007**: System MUST support filtering tasks by completion status (active/completed)
- **FR-008**: System MUST support filtering tasks by one or more tags
- **FR-009**: System MUST support sorting tasks by due date (ascending/descending)
- **FR-010**: System MUST support sorting tasks by priority level
- **FR-011**: System MUST support sorting tasks by creation date (ascending/descending)
- **FR-012**: System MUST allow combining multiple filters simultaneously (e.g., High priority + Work tag + search term)
- **FR-013**: System MUST return filtered search results within 200ms for up to 1,000 tasks

#### Due Date Management (P2)

- **FR-014**: System MUST allow users to set an optional due date for each task
- **FR-015**: System MUST provide a date picker interface for selecting due dates
- **FR-016**: System MUST display overdue tasks with a distinct visual indicator (red/bold styling)
- **FR-017**: System MUST display tasks due today with a distinct visual indicator (yellow/orange styling)
- **FR-018**: System MUST allow users to modify or remove due dates from existing tasks
- **FR-019**: System MUST handle timezone-aware date storage and display
- **FR-020**: When sorting by due date, tasks without due dates MUST appear at the end of the list

#### Recurring Tasks (P3)

- **FR-021**: System MUST allow users to specify a recurrence rule when creating or editing a task
- **FR-022**: System MUST support standard recurrence patterns: DAILY, WEEKLY, MONTHLY
- **FR-023**: System MUST support custom recurrence rules via CRON-like syntax
- **FR-024**: When a recurring task is marked complete, System MUST publish a `task.completed` event containing the task ID and recurrence rule
- **FR-025**: System MUST subscribe to `task.completed` events and automatically create the next occurrence based on the recurrence rule
- **FR-026**: The newly created recurring task instance MUST inherit the title, description, priority, tags, and recurrence rule from the completed task
- **FR-027**: System MUST calculate the next occurrence date/time based on the recurrence rule
- **FR-028**: System MUST allow users to edit or remove the recurrence rule from existing tasks
- **FR-029**: System MUST allow users to delete individual instances of recurring tasks without affecting the recurrence pattern
- **FR-030**: System MUST log all recurring task creation events for debugging

#### Smart Reminders (P4)

- **FR-031**: System MUST allow users to set an optional reminder time for each task
- **FR-032**: System MUST provide a date/time picker interface for selecting reminder times
- **FR-033**: When a task with a reminder is created or updated, System MUST schedule a precise callback using the Jobs API
- **FR-034**: When a reminder triggers, System MUST publish a `reminder.triggered` event containing the task ID and reminder details
- **FR-035**: System MUST allow users to modify or remove reminder times from existing tasks
- **FR-036**: System MUST cancel scheduled reminders when a task is deleted
- **FR-037**: System MUST cancel scheduled reminders when a task is completed before the reminder time
- **FR-038**: System MUST reschedule reminders when a task's reminder time is modified
- **FR-039**: System MUST log all reminder scheduling and triggering events for debugging
- **FR-040**: System MUST handle reminder scheduling even if the exact trigger time is missed (e.g., system downtime)

#### Event-Driven Architecture (Cross-cutting)

- **FR-041**: System MUST publish events to the message bus without direct infrastructure client dependencies
- **FR-042**: System MUST subscribe to and process `task.completed` events for recurring task generation
- **FR-043**: System MUST subscribe to and process `reminder.triggered` events for notifications
- **FR-044**: System MUST ensure event messages adhere to defined schemas (task event, reminder event)
- **FR-045**: System MUST ensure idempotent event processing (duplicate events don't cause duplicate task creation)

### Key Entities

- **Task**: Represents a single todo item with attributes: title, description, completion status, priority (enum: High/Medium/Low), tags (array of strings), due date (optional, datetime), recurrence rule (optional, string), reminder time (optional, datetime), created at timestamp, updated at timestamp
- **TaskEvent**: Represents a domain event published when a task is completed; contains: event type, task ID, original task data (title, description, priority, tags, recurrence rule), timestamp
- **ReminderEvent**: Represents a notification event published when a reminder triggers; contains: event type, task ID, task title, reminder time, timestamp
- **Tag**: Represents a categorical label for organizing tasks; contains: name, color (for UI display), user ownership
- **RecurrenceRule**: Represents how a task repeats; contains: frequency (DAILY/WEEKLY/MONTHLY/custom), interval (every N periods), custom CRON expression (optional), end date (optional)

## Assumptions

1. **User Authentication**: Tasks are scoped to authenticated users; each user only sees their own tasks
2. **Timezone Handling**: All datetimes are stored in UTC and displayed in the user's local timezone
3. **Event Bus Reliability**: The message bus delivers events at-least-once; consumers handle duplicates idempotently
4. **Reminder Notification**: For this phase, reminder events are logged to verify functionality; external notifications (email/push) are out of scope
5. **Recurrence Calculation**: Standard recurrence patterns (DAILY, WEEKLY, MONTHLY) follow common calendar conventions
6. **Search Performance**: The 200ms search performance target assumes up to 1,000 tasks per user
7. **Tag Management**: Tags are user-specific and created implicitly when assigned to a task
8. **Database Schema**: The existing database can be migrated to add new columns without data loss

## Non-Goals (Explicitly Out of Scope)

- Infrastructure deployment (Minikube, Kubernetes, Cloud providers)
- CI/CD pipeline configuration
- External notification delivery (email, SMS, push notifications)
- Recurrence rule UI builder (users input predefined patterns or CRON strings)
- Task sharing between users
- Task dependencies or subtasks
- Recurrence end dates or limits (beyond manual removal)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Database schema includes new fields (priority, tags, due_date, recurrence_rule, reminder_time) with proper data types and indexes
- **SC-002**: Keyword search returns matching results within 200ms for a user with up to 1,000 tasks
- **SC-003**: Completing a "Daily Standup" recurring task triggers a `task.completed` event visible in logs and automatically creates tomorrow's task instance
- **SC-004**: Setting a reminder schedules a background job verified via system logs or administrative dashboard
- **SC-005**: Users can sort tasks by Priority High→Low and see visual indicators for overdue (red) and due today (yellow) tasks
- **SC-006**: Combining multiple filters (priority + tag + search) returns only tasks matching ALL criteria
- **SC-007**: Recurring task instances maintain consistent title, priority, and tags across all occurrences
- **SC-008**: Modifying a task's reminder time cancels the old reminder and schedules a new one
- **SC-009**: Deleting a task cancels any pending reminders without triggering events
- **SC-010**: All event messages contain required fields (task ID, timestamp, event type) matching defined schemas

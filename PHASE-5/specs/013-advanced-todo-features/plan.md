# Implementation Plan: Advanced Todo Features (Intermediate & Advanced)

**Branch**: `013-advanced-todo-features` | **Date**: 2026-02-16 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/013-advanced-todo-features/spec.md`

## Summary

This plan implements advanced todo features in two phases: **Intermediate** (task organization, search/filter/sort) and **Advanced** (event-driven recurring tasks and scheduled reminders via Dapr). The existing codebase already has foundational Task model fields (priority, due_date, recurrence_rule) and tag relationships. This plan focuses on completing the UI components, extending filtering with tag support, adding the missing `reminder_time` field, and implementing event-driven architecture using Dapr for recurring tasks and reminders.

**Key Architectural Decisions**:
1. **SQL-based filtering**: Use PostgreSQL for complex search/filter/sort (not Dapr State Store query API)
2. **Event-driven recurrence**: Publish `task.completed` events via Dapr Pub/Sub; subscriber creates next task instance
3. **Dapr Jobs API**: Schedule reminder callbacks via Dapr Jobs (not Python schedulers) for statelessness

## Technical Context

**Language/Version**: Python 3.11+ (backend), TypeScript/Next.js 15+ (frontend)
**Primary Dependencies**: FastAPI, SQLModel, PostgreSQL, Next.js, React 19, Dapr Python SDK, httpx
**Storage**: PostgreSQL (Neon) via SQLModel for CRUD; Dapr State Store for conversation history (existing)
**Testing**: pytest (backend), existing test structure in `backend/tests/`
**Target Platform**: Containerized services with Dapr sidecars for local development
**Project Type**: web application (backend + frontend)
**Performance Goals**: <200ms p95 for search/filter with up to 1,000 tasks per user
**Constraints**:
  - NO direct Kafka clients (`kafka-python` forbidden)
  - Use Dapr HTTP APIs for Pub/Sub and Jobs
  - Existing Task model already has `priority`, `due_date`, `recurrence_rule` fields
**Scale/Scope**: Single-user task management; horizontally scalable via stateless design

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| **Event-Driven First** | ✅ PASS | Recurring tasks and reminders use Dapr Pub/Sub; schemas defined |
| **Infrastructure Abstraction** | ✅ PASS | All event/Job communication via Dapr HTTP; no direct Kafka clients |
| **Cloud-Native Architecture** | ⚠️ NOT IN SCOPE | App-layer only; containerization/deployment out of scope per spec |
| **Automation & IaC** | ⚠️ NOT IN SCOPE | CI/CD and Helm charts explicitly excluded |

**Compliance**: This feature is Application Layer only (per spec Non-Goals). Infrastructure deployment and CI/CD are intentionally out of scope. The code follows Dapr-first patterns for event-driven features.

## Project Structure

### Documentation (this feature)

```text
specs/013-advanced-todo-features/
├── plan.md              # This file
├── spec.md              # Feature specification (already created)
├── data-model.md        # Phase 1 output (to be created)
├── contracts/           # Phase 1 output (to be created)
│   └── api-contracts.md # Updated API contracts with new fields/endpoints
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

**Structure Decision**: Web application (backend + frontend) - Option 2 from template

```text
backend/
├── src/
│   ├── models/
│   │   ├── task.py              # MODIFIED: Add reminder_time field
│   │   └── ...
│   ├── schemas/
│   │   └── task.py              # MODIFIED: Add reminder_time to DTOs
│   ├── services/
│   │   ├── task_service.py      # MODIFIED: Add tag filtering, event publishing
│   │   └── dapr_service.py      # NEW: Dapr client wrapper (Pub/Sub, Jobs)
│   ├── api/
│   │   ├── task_router.py       # MODIFIED: Add tags filter, Dapr subscriber endpoint
│   │   └── dapr_router.py       # NEW: Dapr event subscription endpoints
│   └── middleware/
│       └── ...
├── tests/
│   ├── test_task_service.py     # MODIFIED: Add tests for filtering, events
│   ├── test_dapr_service.py     # NEW: Test Dapr integration
│   └── test_recurrence_e2e.py   # NEW: End-to-end recurrence test
└── requirements.txt             # MODIFIED: Add dapr-sdk, dateutil

frontend/
├── src/
│   ├── components/
│   │   ├── TaskList.tsx         # MODIFIED: Add priority badges, tag badges
│   │   ├── TaskForm.tsx         # MODIFIED: Add due date picker, recurrence, reminder
│   │   ├── SearchFilterSidebar.tsx # NEW: Search, filter by priority/tags
│   │   ├── SortControls.tsx     # NEW: Sort dropdowns
│   │   └── PriorityBadge.tsx    # NEW: Color-coded priority indicator
│   ├── types/
│   │   └── task.ts              # MODIFIED: Add reminder_time field
│   ├── hooks/
│   │   └── useTasks.ts          # MODIFIED: Add filter/sort parameters
│   └── lib/
│       └── api.ts               # MODIFIED: Update API calls with new params
```

## Architecture Design

### Data Flow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW DIAGRAM                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     HTTP      ┌─────────────┐     SQLModel    ┌────────────┐
│   Frontend   │ ─────────────▶ │   Backend   │ ──────────────▶ │ PostgreSQL │
│  (Next.js)   │ ◀───────────── │   (FastAPI) │ ◀────────────── │   (Neon)   │
└──────────────┘                └─────────────┘                └────────────┘
                                      │
                                      │ Dapr HTTP (localhost:3500)
                                      ▼
                            ┌─────────────────────┐
                            │    Dapr Sidecar     │
                            │─────────────────────│
                            │ Pub/Sub (Kafka)     │
                            │ Jobs API            │
                            │ State Store         │
                            └─────────────────────┘

EVENT FLOWS:
───────────────────────────────────────────────────────────────────────────────

1. RECURRING TASK COMPLETION:
   Frontend ──POST /tasks/{id}/toggle-completion──▶ Backend
                                                      │
                                                      ▼
                                       Publish event to Dapr Pub/Sub
                                       Topic: task-events
                                       Payload: {event_type: "task.completed",
                                                 task_id, recurrence_rule, ...}
                                                      │
                                                      ▼
                                       Dapr forwards to subscriber endpoint
                                       POST /api/events/task-completed
                                                      │
                                                      ▼
                                       Calculate next occurrence date
                                                      │
                                                      ▼
                                       Create new task instance via SQLModel

2. REMINDER SCHEDULING:
   Frontend ─────POST /tasks (with reminder_time)────▶ Backend
                                                          │
                                                          ▼
                                       Schedule Dapr Job via HTTP
                                       POST /v1.0-alpha1/jobs
                                       Payload: {schedule: reminder_time,
                                                 http: {uri: /api/jobs/trigger}}
                                                          │
                                                          ▼
                                       [Wait until reminder_time]
                                                          │
                                                          ▼
                                       Dapr invokes callback endpoint
                                       POST /api/jobs/trigger
                                       Payload: {task_id, reminder_time}
                                                          │
                                                          ▼
                                       Publish reminder event to Dapr Pub/Sub
                                       Topic: reminders
                                       Payload: {event_type: "reminder.triggered",
                                                 task_id, title, ...}
                                                          │
                                                          ▼
                                       Notification Service (stub)
                                       Log event for verification
```

### Component Interactions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPONENT INTERACTIONS                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ INTERMEDIATE FEATURES (Direct Database)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────┐   GET /tasks?search=X&priority=HIGH&tags=1,2   ┌──────────┐   │
│  │ Frontend│ ─────────────────────────────────────────────────▶│ Backend  │   │
│  └─────────┘                                                    └──────────┘   │
│       ▲                                                              │        │
│       │                                                              ▼        │
│       │                                                    ┌───────────────┐  │
│       │                                                    │ SQLModel Query │  │
│       │                                                    │ + PostgreSQL   │  │
│       │                                                    └───────────────┘  │
│       │                                                              │        │
│       └──────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ ADVANCED FEATURES (Event-Driven via Dapr)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────┐ toggle complete   ┌──────────┐ publish event   ┌───────────┐  │
│  │ Frontend│ ──────────────────▶│ Backend  │ ────────────────▶│  Dapr    │  │
│  └─────────┘                   └──────────┘                  │  Pub/Sub │  │
│         ▲                           ▲                         └───────────┘  │
│         │                           │                                │        │
│         │                    ┌──────┴─────────┐                      │        │
│         │                    │ GET /tasks      │                      ▼        │
│         │                    │ (new instance)  │               ┌───────────┐  │
│         │                    └─────────────────┘               │   Kafka  │  │
│         │                                                         └───────────┘  │
│         │                                                              │        │
│         └──────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  REMINDERS FLOW:                                                             │
│                                                                              │
│  ┌─────────┐ create task   ┌──────────┐ schedule job    ┌───────────┐      │
│  │ Frontend│ ──────────────▶│ Backend  │ ────────────────▶│ Dapr Jobs │      │
│  └─────────┘ (reminder)     └──────────┘                  └───────────┘      │
│                                                                  │            │
│                                  [wait until scheduled time]    │            │
│                                                                  │            │
│                                                                  ▼            │
│                         ┌──────────┐ callback   ┌───────────┐                │
│                         │ Backend  │ ◀──────────│ Dapr Jobs │                │
│                         └──────────┘            └───────────┘                │
│                               │                                                  │
│                               │ publish reminder                                 │
│                               ▼                                                  │
│                         ┌───────────┐                                          │
│                         │ Dapr Pub/ │                                          │
│                         │   Sub     │                                          │
│                         └───────────┘                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Model Changes

### Task Model Addition

**Existing Fields** (already in database):
- `priority`: ENUM("HIGH", "MEDIUM", "LOW") - DEFAULT "MEDIUM"
- `due_date`: datetime - NULLABLE
- `recurrence_rule`: ENUM("DAILY", "WEEKLY", "MONTHLY") - NULLABLE
- `tags`: Many-to-many relationship via TaskTagLink

**New Field to Add**:
```python
reminder_time: Optional[datetime] = Field(
    default=None,
    sa_column=Column(DateTime, nullable=True),
    index=True  # Index for querying upcoming reminders
)
```

### Event Schemas

**Task Completed Event** (published to `task-events` topic):
```typescript
{
  "event_type": "task.completed",
  "event_id": "uuid-v4",
  "timestamp": "ISO-8601 datetime",
  "task_id": integer,
  "user_id": integer,
  "task_data": {
    "title": string,
    "description": string | null,
    "priority": "HIGH" | "MEDIUM" | "LOW",
    "tags": [string],
    "recurrence_rule": "DAILY" | "WEEKLY" | "MONTHLY" | null,
    "due_date": "ISO-8601 datetime" | null
  }
}
```

**Reminder Triggered Event** (published to `reminders` topic):
```typescript
{
  "event_type": "reminder.triggered",
  "event_id": "uuid-v4",
  "timestamp": "ISO-8601 datetime",
  "task_id": integer,
  "user_id": integer,
  "task_title": string,
  "reminder_time": "ISO-8601 datetime",
  "due_date": "ISO-8601 datetime" | null
}
```

## Phase 1: Data Layer & Backend Core

### 1.1 Database Migration

**File**: `backend/alembic/versions/XXX_add_reminder_time_to_task.py`

```python
def upgrade():
    op.add_column('task', sa.Column('reminder_time', sa.DateTime(), nullable=True))
    op.create_index('ix_task_reminder_time', 'task', ['reminder_time'])
    op.create_index('ix_task_user_reminder_time', 'task', ['user_id', 'reminder_time'])

def downgrade():
    op.drop_index('ix_task_user_reminder_time', table_name='task')
    op.drop_index('ix_task_reminder_time', table_name='task')
    op.drop_column('task', 'reminder_time')
```

### 1.2 Update Models

**File**: `backend/src/models/task.py`

Add `reminder_time` field to `TaskBase` and `Task` classes.

**File**: `backend/src/schemas/task.py`

Add `reminder_time` to `TaskResponse`, `TaskCreateRequest`, and `TaskUpdateRequest`.

### 1.3 Create Dapr Service

**File**: `backend/src/services/dapr_service.py` (NEW)

```python
import httpx
import logging
from typing import Dict, Any, Optional
from datetime import datetime

class DaprService:
    """Wrapper for Dapr HTTP API - enforces infrastructure abstraction"""

    DAPR_HTTP_PORT = 3500
    DAPR_BASE_URL = f"http://localhost:{DAPR_HTTP_PORT}/v1.0"

    async def publish_event(
        self,
        pubsub_name: str,
        topic: str,
        data: Dict[str, Any]
    ) -> bool:
        """Publish event via Dapr Pub/Sub"""
        url = f"{self.DAPR_BASE_URL}/publish/{pubsub_name}/{topic}"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=data, timeout=5.0)
                response.raise_for_status()
                logging.info(f"Published event to {pubsub_name}/{topic}: {data.get('event_type')}")
                return True
            except Exception as e:
                logging.error(f"Failed to publish event: {e}")
                return False

    async def schedule_job(
        self,
        job_name: str,
        schedule_time: datetime,
        payload: Dict[str, Any]
    ) -> bool:
        """Schedule a job via Dapr Jobs API"""
        url = f"{self.DAPR_BASE_URL}/jobs"
        job_data = {
            "name": job_name,
            "schedule": schedule_time.isoformat(),
            "data": payload,
            "http": {
                "uri": f"http://localhost:8000/api/jobs/trigger"
            }
        }
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=job_data, timeout=5.0)
                response.raise_for_status()
                logging.info(f"Scheduled Dapr job: {job_name} at {schedule_time}")
                return True
            except Exception as e:
                logging.error(f"Failed to schedule job: {e}")
                return False

    async def delete_job(self, job_name: str) -> bool:
        """Delete a scheduled job via Dapr Jobs API"""
        url = f"{self.DAPR_BASE_URL}/jobs/{job_name}"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.delete(url, timeout=5.0)
                # 404 is acceptable - job may not exist
                if response.status_code == 404:
                    return True
                response.raise_for_status()
                logging.info(f"Deleted Dapr job: {job_name}")
                return True
            except Exception as e:
                logging.error(f"Failed to delete job: {e}")
                return False
```

### 1.4 Update Task Service

**File**: `backend/src/services/task_service.py`

**Changes**:
1. Add tag filtering to `get_tasks()` method
2. Modify `create_task()` to schedule Dapr job when `reminder_time` is set
3. Modify `update_task()` to reschedule/cancel Dapr job when `reminder_time` changes
4. Modify `delete_task()` to cancel Dapr job for reminders
5. Replace direct recurrence handling in `toggle_task_completion()` with event publishing

**Tag Filtering Logic**:
```python
# Add to get_tasks() method
tags: Optional[List[int]] = None,

# In query building:
if tags:
    query = query.join(Task.tags).where(Tag.id.in_(tags))
```

**Reminder Scheduling**:
```python
async def create_task(self, ...):
    # ... existing code ...
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)

    # Schedule reminder if set
    if task_data.reminder_time:
        await dapr_service.schedule_reminder(task, db_session)

    return task
```

## Phase 2: Event-Driven Features (Advanced)

### 2.1 Dapr Subscription Endpoints

**File**: `backend/src/api/dapr_router.py` (NEW)

```python
from fastapi import APIRouter, Request, HTTPException
from ..services.recurrence_service import recurrence_service
from ..services.reminder_service import reminder_service
import logging

router = APIRouter(prefix="/api/events", tags=["events"])

@router.post("/task-completed")
async def handle_task_completed(request: Request):
    """Dapr subscriber endpoint for task.completed events"""
    try:
        event_data = await request.json()
        logging.info(f"Received task.completed event: {event_data}")

        # Extract task data
        task_id = event_data.get("task_id")
        recurrence_rule = event_data.get("task_data", {}).get("recurrence_rule")

        if not recurrence_rule:
            return {"status": "ignored", "reason": "not a recurring task"}

        # Create next instance
        new_task = await recurrence_service.create_next_instance(
            task_id=task_id,
            event_data=event_data
        )

        return {
            "status": "processed",
            "new_task_id": new_task.id if new_task else None
        }

    except Exception as e:
        logging.error(f"Error processing task.completed event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reminder-triggered")
async def handle_reminder_triggered(request: Request):
    """Dapr subscriber endpoint for reminder events"""
    try:
        event_data = await request.json()
        logging.info(f"Received reminder.triggered event: {event_data}")

        # Process reminder (currently just log)
        await reminder_service.process_reminder(event_data)

        return {"status": "processed"}

    except Exception as e:
        logging.error(f"Error processing reminder event: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### 2.2 Dapr Subscription Configuration

**File**: `backend/config/dapr/subscriptions.yaml` (NEW) OR use programmatic registration

**Dapr expects subscription registration at `/dapr/subscribe`**:

**File**: `backend/src/api/dapr_router.py` (add endpoint)

```python
@router.get("/dapr/subscribe")
async def dapr_subscribe():
    """Register Dapr subscriptions"""
    return [
        {
            "pubsubname": "pubsub.kafka",
            "topic": "task-events",
            "route": "/api/events/task-completed"
        },
        {
            "pubsubname": "pubsub.kafka",
            "topic": "reminders",
            "route": "/api/events/reminder-triggered"
        }
    ]
```

### 2.3 Job Trigger Endpoint

**File**: `backend/src/api/jobs_router.py` (NEW)

```python
from fastapi import APIRouter, Request
from ..services.dapr_service import dapr_service
import logging

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

@router.post("/trigger")
async def handle_job_trigger(request: Request):
    """Dapr Jobs callback endpoint for reminder triggers"""
    try:
        payload = await request.json()
        logging.info(f"Job triggered: {payload}")

        task_id = payload.get("task_id")
        user_id = payload.get("user_id")

        # Publish reminder event to Kafka via Dapr
        await dapr_service.publish_event(
            pubsub_name="pubsub.kafka",
            topic="reminders",
            data={
                "event_type": "reminder.triggered",
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "task_id": task_id,
                "user_id": user_id,
                "task_title": payload.get("task_title"),
                "reminder_time": payload.get("reminder_time"),
                "due_date": payload.get("due_date")
            }
        )

        return {"status": "processed"}

    except Exception as e:
        logging.error(f"Error processing job trigger: {e}")
        return {"status": "error", "message": str(e)}
```

## Phase 3: Frontend Implementation

### 3.1 Update TypeScript Types

**File**: `frontend/src/types/task.ts`

```typescript
export type Priority = "HIGH" | "MEDIUM" | "LOW";
export type RecurrenceRule = "DAILY" | "WEEKLY" | "MONTHLY";

export interface Task {
  id: number;
  title: string;
  description?: string;
  completed: boolean;
  priority?: Priority;
  due_date?: string;  // ISO-8601
  recurrence_rule?: RecurrenceRule;
  reminder_time?: string;  // NEW: ISO-8601
  created_at: string;
  updated_at: string;
  tags: Tag[];
}

export interface TaskFilters {
  completed?: boolean;
  priority?: Priority;
  tags?: number[];
  search?: string;
  sort_by?: "created_at" | "updated_at" | "due_date" | "priority";
  order?: "asc" | "desc";
}
```

### 3.2 New Components

**PriorityBadge Component** (`frontend/src/components/PriorityBadge.tsx`):
```typescript
interface Props {
  priority?: Priority;
}

const colors = {
  HIGH: "bg-red-100 text-red-800",
  MEDIUM: "bg-yellow-100 text-yellow-800",
  LOW: "bg-green-100 text-green-800"
};
```

**SearchFilterSidebar Component** (`frontend/src/components/SearchFilterSidebar.tsx`):
- Search input field
- Priority dropdown filter
- Tag multi-select
- Clear filters button

**SortControls Component** (`frontend/src/components/SortControls.tsx`):
- Sort by dropdown (created_at, due_date, priority)
- Order toggle (asc/desc)

### 3.3 Update TaskForm

**File**: `frontend/src/components/TaskForm.tsx`

Add fields:
- Due date picker (use HTML `<input type="datetime-local">`)
- Priority dropdown
- Tag selector (multi-select)
- Recurrence dropdown
- Reminder time picker

### 3.4 Update useTasks Hook

**File**: `frontend/src/hooks/useTasks.ts`

Accept filter parameters and pass to API:

```typescript
export function useTasks(filters: TaskFilters) {
  const params = new URLSearchParams();
  if (filters.completed !== undefined) params.set('completed', String(filters.completed));
  if (filters.priority) params.set('priority', filters.priority);
  if (filters.tags?.length) params.set('tags', filters.tags.join(','));
  if (filters.search) params.set('search', filters.search);
  if (filters.sort_by) params.set('sort_by', filters.sort_by);
  if (filters.order) params.set('order', filters.order);

  // ... fetch with params
}
```

## API Contracts

### Updated Endpoints

**GET /api/v1/tasks** (Modified)
- **New Query Param**: `tags` (comma-separated tag IDs)
- **Response**: Now includes `reminder_time` field

**POST /api/v1/tasks** (Modified)
- **New Request Field**: `reminder_time` (ISO-8601 datetime)
- **Side Effect**: Schedules Dapr job if `reminder_time` is provided

**PUT /api/v1/tasks/{id}** (Modified)
- **New Request Field**: `reminder_time`
- **Side Effect**: Reschedules/cancels Dapr job if `reminder_time` changes

**DELETE /api/v1/tasks/{id}** (Modified)
- **Side Effect**: Cancels pending Dapr job for reminder

**PATCH /api/v1/tasks/{id}/toggle-completion** (Modified)
- **Side Effect**: Publishes `task.completed` event if task has `recurrence_rule`

### New Endpoints

**GET /dapr/subscribe**
- **Purpose**: Dapr subscription registration
- **Response**: Array of subscription configurations

**POST /api/events/task-completed**
- **Purpose**: Handle `task.completed` events from Dapr
- **Internal**: Called by Dapr sidecar

**POST /api/events/reminder-triggered**
- **Purpose**: Handle reminder events from Dapr
- **Internal**: Called by Dapr sidecar

**POST /api/jobs/trigger**
- **Purpose**: Dapr Jobs callback for reminder triggers
- **Internal**: Called by Dapr sidecar

## Dependencies

### Backend Additions

**File**: `backend/requirements.txt`

```
# ============================================================================
# Dapr Integration
# ============================================================================
dapr-sdk>=1.14.0  # OR use httpx (already included) for HTTP API calls
python-dateutil>=2.8.0  # For recurrence calculation

# ============================================================================
# Testing
# ============================================================================
pytest-asyncio>=0.21.0  # For async event testing
```

**Note**: The plan recommends using `httpx` (already in requirements.txt) for Dapr HTTP API calls rather than the full Dapr SDK, reducing dependency overhead.

### Frontend Additions

**File**: `frontend/package.json`

```json
{
  "dependencies": {
    "react-datepicker": "^7.5.0",
    "@types/react-datepicker": "^7.5.0"
  }
}
```

Or use native HTML5 date inputs for simplicity.

## Testing Strategy

### Unit Tests

1. **Task Service Tests** (`backend/tests/test_task_service.py`)
   - Tag filtering logic
   - Reminder job scheduling (mocked Dapr)
   - Event publishing (mocked Dapr)

2. **Dapr Service Tests** (`backend/tests/test_dapr_service.py`)
   - Mock httpx calls to Dapr sidecar
   - Verify correct payload formats

3. **Recurrence Calculation Tests** (`backend/tests/test_recurrence.py`)
   - Next occurrence date calculation for DAILY/WEEKLY/MONTHLY
   - Edge cases: leap year, month-end

### Integration Tests

1. **End-to-End Recurring Task** (`backend/tests/test_recurrence_e2e.py`)
   - Create recurring task
   - Toggle completion
   - Verify event is published (mock Dapr)
   - Verify next task is created

2. **Reminder Scheduling** (`backend/tests/test_reminder_e2e.py`)
   - Create task with reminder
   - Verify Dapr job is scheduled (mock Dapr)
   - Simulate job trigger
   - Verify reminder event is published

## Complexity Tracking

> No constitutional violations requiring justification.

All architecture decisions align with the constitution:
- **Event-Driven First**: Recurring tasks and reminders use Dapr Pub/Sub
- **Infrastructure Abstraction**: All event/Job communication via Dapr HTTP APIs
- **Cloud-Native**: Design supports containerization with Dapr sidecar (deployment out of scope per spec)

## Success Metrics

| Success Criterion | Verification Method |
|-------------------|---------------------|
| SC-001: Schema includes new fields | Run `alembic upgrade head` and verify columns |
| SC-002: Search <200ms | Benchmark with 1,000 tasks, measure response time |
| SC-003: Recurring tasks create instances | Complete recurring task, verify event logged and new task created |
| SC-004: Reminders schedule Dapr jobs | Create task with reminder, verify job scheduled via logs |
| SC-005: UI sorting and indicators | Manual testing of sort controls and visual badges |
| SC-006: Combined filters work | Apply priority + tag + search, verify correct results |
| SC-007: Recurring instances consistent | Create "Daily Standup", complete 3 times, verify all instances identical |
| SC-008: Reminder rescheduling | Change reminder time, verify old job cancelled, new job scheduled |
| SC-009: Task deletion cancels reminders | Delete task with pending reminder, verify job cancelled |
| SC-010: Event schema validation | Capture event payloads, verify all required fields present |

## Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| Dapr sidecar not available during development | HIGH | Implement graceful degradation; log errors without failing requests |
| Event duplication (at-least-once delivery) | MEDIUM | Implement idempotency keys in task creation |
| Timezone handling complexity | MEDIUM | Store all datetimes in UTC; convert to user timezone in frontend |
| Recurrence calculation edge cases | LOW | Use `python-dateutil` for robust date math; extensive unit tests |

## Follow-up Considerations

1. **Future Enhancements** (out of scope):
   - Custom CRON syntax for recurrence
   - Recurrence end dates
   - Email/Push notification delivery
   - Task dependencies/subtasks

2. **Infrastructure** (explicitly out of scope):
   - Docker containerization
   - Helm chart development
   - Kubernetes deployment manifests
   - CI/CD pipeline configuration

3. **Next Phase Recommendations**:
   - Consider moving to `dapr-python-sdk` if gRPC benefits are needed
   - Implement dead-letter queue for failed events
   - Add circuit breakers for Dapr API calls
   - Consider event sourcing for full audit trail

# API Contracts: Advanced Todo Features

**Feature**: 013-Advanced Todo Features
**Date**: 2026-02-16
**Related**: [spec.md](./spec.md), [plan.md](./plan.md), [data-model.md](./data-model.md)

## Overview

This document defines the API contracts for the advanced todo features, including modified existing endpoints and new event-driven endpoints.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

All endpoints require Bearer token authentication via Clerk JWT.

```
Authorization: Bearer <clerk_jwt_token>
```

## Response Format

Success responses use appropriate HTTP status codes:

- `200 OK` - Successful GET
- `201 Created` - Successful POST
- `204 No Content` - Successful DELETE
- `400 Bad Request` - Validation error
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

Error response format:

```typescript
interface ErrorResponse {
  detail: string;
}
```

---

## Modified Endpoints

### GET /tasks

Get all tasks for the authenticated user with optional filtering and sorting.

**Request**:

```http
GET /api/v1/tasks?completed=false&priority=HIGH&tags=1,2&search=meeting&sort_by=due_date&order=asc&limit=20&offset=0
```

**Query Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| completed | boolean | No | Filter by completion status |
| priority | string | No | Filter by priority: HIGH, MEDIUM, LOW |
| tags | string | No | Comma-separated tag IDs for filtering |
| search | string | No | Keyword search in title and description |
| sort_by | string | No | Sort field: created_at, updated_at, due_date, priority |
| order | string | No | Sort direction: asc, desc (default: desc) |
| limit | integer | No | Results per page (default: 10, max: 100) |
| offset | integer | No | Results to skip (default: 0) |

**Response**: `200 OK`

```typescript
interface TaskResponse {
  id: number;
  title: string;
  description: string | null;
  completed: boolean;
  priority: "HIGH" | "MEDIUM" | "LOW" | null;
  due_date: string | null;        // ISO-8601
  recurrence_rule: "DAILY" | "WEEKLY" | "MONTHLY" | null;
  reminder_time: string | null;   // ISO-8601 [NEW]
  created_at: string;             // ISO-8601
  updated_at: string;             // ISO-8601
  tags: TagResponse[];
}

interface TagResponse {
  id: number;
  name: string;
  color: string | null;
  priority: number;
  user_id: number;
  created_at: string;
}

type GetTasksResponse = TaskResponse[];
```

**Example**:

```json
[
  {
    "id": 1,
    "title": "Complete project proposal",
    "description": "Finish the Q1 proposal document",
    "completed": false,
    "priority": "HIGH",
    "due_date": "2026-02-20T17:00:00Z",
    "recurrence_rule": null,
    "reminder_time": "2026-02-20T09:00:00Z",
    "created_at": "2026-02-16T10:00:00Z",
    "updated_at": "2026-02-16T10:00:00Z",
    "tags": [
      {
        "id": 1,
        "name": "Work",
        "color": "#3B82F6",
        "priority": 1,
        "user_id": 1,
        "created_at": "2026-02-01T10:00:00Z"
      }
    ]
  }
]
```

---

### POST /tasks

Create a new task. If `reminder_time` is provided, a Dapr job will be scheduled.

**Request**:

```http
POST /api/v1/tasks
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body**:

```typescript
interface TaskCreateRequest {
  title: string;
  description?: string;
  priority?: "HIGH" | "MEDIUM" | "LOW";
  due_date?: string;          // ISO-8601
  recurrence_rule?: "DAILY" | "WEEKLY" | "MONTHLY";
  reminder_time?: string;     // ISO-8601 [NEW]
  tag_ids?: number[];
}
```

**Validation**:

- `title`: Required, 1-255 characters
- `reminder_time`: Must be after current time if provided
- `reminder_time`: Must be before `due_date` if both are provided

**Response**: `201 Created`

Returns `TaskResponse` (see above).

**Example**:

```json
{
  "title": "Daily Standup",
  "description": "Team sync meeting",
  "priority": "MEDIUM",
  "due_date": "2026-02-17T09:00:00Z",
  "recurrence_rule": "DAILY",
  "reminder_time": "2026-02-17T08:45:00Z",
  "tag_ids": [1, 3]
}
```

**Side Effects**:
- If `reminder_time` is provided, schedules a Dapr job named `reminder-{task_id}`
- Job will invoke `/api/jobs/trigger` at the specified time

---

### PUT /tasks/{id}

Update an existing task. If `reminder_time` changes, the Dapr job is rescheduled.

**Request**:

```http
PUT /api/v1/tasks/123
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body**:

```typescript
interface TaskUpdateRequest {
  title?: string;
  description?: string;
  completed?: boolean;
  priority?: "HIGH" | "MEDIUM" | "LOW";
  due_date?: string;
  recurrence_rule?: "DAILY" | "WEEKLY" | "MONTHLY";
  reminder_time?: string;     // [NEW]
  tag_ids?: number[];
}
```

**Response**: `200 OK`

Returns `TaskResponse`.

**Side Effects**:
- If `reminder_time` is added or changed: deletes old job (if exists) and schedules new one
- If `reminder_time` is set to null: cancels existing job

---

### DELETE /tasks/{id}

Delete a task and cancel any pending reminder job.

**Request**:

```http
DELETE /api/v1/tasks/123
Authorization: Bearer <token>
```

**Response**: `204 No Content`

**Side Effects**:
- Cancels Dapr job `reminder-{task_id}` if it exists
- Deletes task from database

---

### PATCH /tasks/{id}/toggle-completion

Toggle task completion status. If completing a recurring task, publishes an event.

**Request**:

```http
PATCH /api/v1/tasks/123/toggle-completion
Authorization: Bearer <token>
```

**Response**: `200 OK`

Returns `TaskResponse`.

**Side Effects**:
- If task has `recurrence_rule` and is being marked complete:
  1. Publishes `task.completed` event to Dapr Pub/Sub topic `task-events`
  2. Event subscriber creates next task instance
- If task has `reminder_time` and is being marked complete:
  - Cancels the pending reminder job

---

## New Endpoints

### GET /dapr/subscribe

Dapr subscription registration endpoint. Called by Dapr sidecar on startup.

**Request**:

```http
GET /api/v1/dapr/subscribe
```

**Response**: `200 OK`

```typescript
interface DaprSubscription {
  pubsubname: string;
  topic: string;
  route: string;
}

type DaprSubscribeResponse = DaprSubscription[];
```

**Example**:

```json
[
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

---

### POST /events/task-completed

Internal endpoint called by Dapr when a `task.completed` event is received.

**Request**:

```http
POST /api/v1/events/task-completed
X-Dapr-Subscription-Id: task-events-subscription
Content-Type: application/json
```

**Request Body**:

```typescript
interface TaskCompletedEvent {
  event_type: "task.completed";
  event_id: string;
  timestamp: string;
  task_id: number;
  user_id: number;
  task_data: {
    title: string;
    description: string | null;
    priority: "HIGH" | "MEDIUM" | "LOW";
    tags: string[];
    recurrence_rule: "DAILY" | "WEEKLY" | "MONTHLY" | null;
    due_date: string | null;
  };
}
```

**Response**: `200 OK`

```typescript
interface EventProcessResponse {
  status: "processed" | "ignored";
  new_task_id: number | null;
  reason?: string;
}
```

**Behavior**:
- If task has no `recurrence_rule`: returns `{status: "ignored"}`
- If task has `recurrence_rule`: calculates next date, creates new task, returns `{status: "processed", new_task_id: N}`

---

### POST /events/reminder-triggered

Internal endpoint called by Dapr when a `reminder.triggered` event is received.

**Request**:

```http
POST /api/v1/events/reminder-triggered
X-Dapr-Subscription-Id: reminders-subscription
Content-Type: application/json
```

**Request Body**:

```typescript
interface ReminderTriggeredEvent {
  event_type: "reminder.triggered";
  event_id: string;
  timestamp: string;
  task_id: number;
  user_id: number;
  task_title: string;
  reminder_time: string;
  due_date: string | null;
}
```

**Response**: `200 OK`

```typescript
interface EventProcessResponse {
  status: "processed";
}
```

**Behavior**:
- Logs the reminder event
- In future phases: delivers notification (email, push, etc.)

---

### POST /jobs/trigger

Internal endpoint called by Dapr Jobs API when a scheduled reminder triggers.

**Request**:

```http
POST /api/v1/jobs/trigger
Content-Type: application/json
```

**Request Body** (from Dapr Job data):

```typescript
interface JobTriggerPayload {
  task_id: number;
  user_id: number;
  task_title: string;
  reminder_time: string;
  due_date: string | null;
}
```

**Response**: `200 OK`

```typescript
interface JobTriggerResponse {
  status: "processed" | "error";
  message?: string;
}
```

**Behavior**:
- Publishes `reminder.triggered` event to Dapr Pub/Sub topic `reminders`
- Event is consumed by notification service (currently just logs)

---

## Dapr HTTP API Calls

The backend makes the following calls to Dapr sidecar at `http://localhost:3500/v1.0`:

### Publish Event

```http
POST /v1.0/publish/pubsub.kafka/task-events
Content-Type: application/json

{
  "event_type": "task.completed",
  "event_id": "uuid-v4",
  "timestamp": "2026-02-16T10:00:00Z",
  "task_id": 123,
  "user_id": 1,
  "task_data": {...}
}
```

### Schedule Job

```http
POST /v1.0-alpha1/jobs
Content-Type: application/json

{
  "name": "reminder-123",
  "schedule": "2026-02-17T08:45:00Z",
  "data": {
    "task_id": 123,
    "user_id": 1,
    "task_title": "Daily Standup",
    "reminder_time": "2026-02-17T08:45:00Z",
    "due_date": "2026-02-17T09:00:00Z"
  },
  "http": {
    "uri": "http://localhost:8000/api/v1/jobs/trigger"
  }
}
```

### Delete Job

```http
DELETE /v1.0-alpha1/jobs/reminder-123
```

---

## Rate Limiting

| Endpoint | Rate Limit |
|----------|------------|
| GET /tasks | 100/minute |
| POST /tasks | 20/minute |
| GET /tasks/{id} | 50/minute |
| PUT /tasks/{id} | 30/minute |
| DELETE /tasks/{id} | 30/minute |
| PATCH /tasks/{id}/toggle-completion | 40/minute |

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Task title is required"
}
```

### 404 Not Found

```json
{
  "detail": "Task not found or access denied"
}
```

### 429 Too Many Requests

```json
{
  "detail": "Rate limit exceeded"
}
```

Headers:
```
Retry-After: 60
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1705420800
```

### 500 Internal Server Error

```json
{
  "detail": "Failed to create task"
}
```

---

## Webhook Headers

Dapr adds the following headers to webhook calls:

| Header | Description |
|--------|-------------|
| X-Dapr-Subscription-Id | Subscription identifier |
| X-Dapr-Trace-Parent | Distributed tracing ID |
| Content-Type | application/json or application/cloudevents+json |

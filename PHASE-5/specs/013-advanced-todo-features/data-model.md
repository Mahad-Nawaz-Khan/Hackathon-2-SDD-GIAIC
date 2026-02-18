# Data Model: Advanced Todo Features

**Feature**: 013-Advanced Todo Features
**Date**: 2026-02-16
**Related**: [spec.md](./spec.md), [plan.md](./plan.md)

## Overview

This document describes the data model changes required for the advanced todo features. The existing Task model already has `priority`, `due_date`, and `recurrence_rule` fields. This feature adds `reminder_time` and extends the event schemas for recurring tasks and reminders.

## Entity Relationship Diagram

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│    User     │────────▶│     Task     │◀────────│     Tag     │
│             │   1:N   │              │   M:N   │             │
│ - id        │         │ - id         │         │ - id        │
│ - email     │         │ - title      │         │ - name      │
│ - name      │         │ - description│         │ - color     │
│             │         │ - completed  │         │ - user_id   │
│             │         │ - priority   │         └─────────────┘
│             │         │ - due_date   │                 │
│             │         │ - recurrence │◀────────────────┘
│             │         │ - reminder   │  (via TaskTagLink)
│             │         │ - user_id    │
│             │         │ - created_at │
│             │         │ - updated_at │
│             │         └──────────────┘
└─────────────┘                 │
                                 │ 1:N (via events)
                                 ▼
                      ┌──────────────────┐
                      │   TaskEvent      │
                      │  (Domain Event)  │
                      │ - event_type     │
                      │ - event_id       │
                      │ - timestamp      │
                      │ - task_id        │
                      │ - task_data      │
                      └──────────────────┘
```

## Database Schema

### Task Table (Modifications)

**Existing Columns** (already implemented):
| Column | Type | Constraints | Default | Description |
|--------|------|-------------|---------|-------------|
| id | INTEGER | PRIMARY KEY | AUTO | Unique identifier |
| title | VARCHAR(255) | NOT NULL | - | Task title |
| description | TEXT | NULLABLE | NULL | Detailed description |
| completed | BOOLEAN | NOT NULL | FALSE | Completion status |
| priority | ENUM | NULLABLE | 'MEDIUM' | Priority level |
| due_date | TIMESTAMP | NULLABLE | NULL | When task is due |
| recurrence_rule | ENUM | NULLABLE | NULL | Recurrence pattern |
| user_id | INTEGER | FOREIGN KEY | - | Owner user |
| created_at | TIMESTAMP | NOT NULL | NOW() | Creation time |
| updated_at | TIMESTAMP | NOT NULL | NOW() | Last update |

**New Column** (to be added):
| Column | Type | Constraints | Default | Description |
|--------|------|-------------|---------|-------------|
| reminder_time | TIMESTAMP | NULLABLE | NULL | When to send reminder |

**Indexes**:
- `ix_task_user_completed` on (user_id, completed)
- `ix_task_user_priority` on (user_id, priority)
- `ix_task_user_due_date` on (user_id, due_date)
- `ix_task_user_created_at` on (user_id, created_at)
- `ix_task_completed_priority` on (completed, priority)
- `ix_task_user_updated_at` on (user_id, updated_at)
- `ix_task_user_reminder_time` on (user_id, reminder_time) - **NEW**

### Tag Table (Existing)

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY | Unique identifier |
| name | VARCHAR(50) | NOT NULL | Tag name |
| color | VARCHAR(7) | NULLABLE | Hex color code |
| priority | INTEGER | DEFAULT 0 | Display order |
| user_id | INTEGER | FOREIGN KEY | Owner user |
| created_at | TIMESTAMP | NOT NULL | Creation time |

### TaskTagLink Table (Existing - Many-to-Many)

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| task_id | INTEGER | FOREIGN KEY | Task reference |
| tag_id | INTEGER | FOREIGN KEY | Tag reference |

**Primary Key**: (task_id, tag_id)

## Domain Events

### Task Completed Event

Published to `task-events` topic via Dapr Pub/Sub when a recurring task is marked complete.

```typescript
interface TaskCompletedEvent {
  event_type: "task.completed";
  event_id: string;        // UUID v4
  timestamp: string;       // ISO-8601 datetime (UTC)
  task_id: number;
  user_id: number;
  task_data: {
    title: string;
    description: string | null;
    priority: "HIGH" | "MEDIUM" | "LOW";
    tags: string[];        // Array of tag names
    recurrence_rule: "DAILY" | "WEEKLY" | "MONTHLY" | null;
    due_date: string | null;  // ISO-8601
  };
}
```

**Purpose**: Triggers creation of the next recurring task instance.

**Idempotency Key**: `event_id` - consumers should track processed events to avoid duplicates.

### Reminder Triggered Event

Published to `reminders` topic via Dapr Pub/Sub when a scheduled reminder triggers.

```typescript
interface ReminderTriggeredEvent {
  event_type: "reminder.triggered";
  event_id: string;        // UUID v4
  timestamp: string;       // ISO-8601 datetime (UTC)
  task_id: number;
  user_id: number;
  task_title: string;
  reminder_time: string;   // ISO-8601
  due_date: string | null; // ISO-8601
}
```

**Purpose**: Triggers notification delivery (currently logs for verification).

**Idempotency Key**: `event_id` - ensures no duplicate notifications.

## Enumerations

### Priority Enum

```sql
CREATE TYPE priorityenum AS ENUM ('HIGH', 'MEDIUM', 'LOW');
```

### Recurrence Rule Enum

```sql
CREATE TYPE recurrenceruleenum AS ENUM ('DAILY', 'WEEKLY', 'MONTHLY');
```

## Migration Script

### Alembic Migration: Add reminder_time

```python
# backend/alembic/versions/XXX_add_reminder_time_to_tasks.py

from alembic import op
import sqlalchemy as sa
from datetime import datetime

revision = '001_add_reminder_time'
down_revision = '000_base'
branch_labels = None
depends_on = None


def upgrade():
    # Add reminder_time column
    op.add_column(
        'task',
        sa.Column('reminder_time', sa.DateTime(), nullable=True)
    )

    # Create indexes for reminder querying
    op.create_index(
        'ix_task_reminder_time',
        'task',
        ['reminder_time']
    )

    op.create_index(
        'ix_task_user_reminder_time',
        'task',
        ['user_id', 'reminder_time']
    )


def downgrade():
    # Drop indexes
    op.drop_index('ix_task_user_reminder_time', table_name='task')
    op.drop_index('ix_task_reminder_time', table_name='task')

    # Remove column
    op.drop_column('task', 'reminder_time')
```

## Data Validation Rules

### Task Validation

| Field | Validation | Error Message |
|-------|------------|---------------|
| title | Required, 1-255 chars | "Task title is required" |
| priority | Must be HIGH/MEDIUM/LOW if provided | "Invalid priority value" |
| due_date | Must be future if provided | "Due date must be in the future" |
| reminder_time | Must be before due_date if both set | "Reminder must be before due date" |
| recurrence_rule | Must be DAILY/WEEKLY/MONTHLY if provided | "Invalid recurrence rule" |

### Tag Validation

| Field | Validation | Error Message |
|-------|------------|---------------|
| name | Required, 1-50 chars | "Tag name is required" |
| color | Valid hex color if provided | "Invalid color format" |

## Recurrence Calculation Rules

### Next Occurrence Logic

| Rule | Calculation | Example |
|------|-------------|---------|
| DAILY | current_due_date + 1 day | Jan 1 → Jan 2 |
| WEEKLY | current_due_date + 7 days | Monday → Next Monday |
| MONTHLY | current_due_date + ~30 days | Jan 15 → Feb 15 |
| Custom | CRON expression parsing | "0 9 * * 1" (Every Monday 9am) |

**Edge Cases**:
- **Month-end**: Jan 31 + 1 month → Feb 28 (or 29 in leap year)
- **Leap year**: Feb 29 + 1 year → Feb 28 (non-leap year)
- **Daylight saving**: Preserve local time; adjust UTC offset accordingly

**Note**: For Phase 5, only DAILY, WEEKLY, MONTHLY enums are supported. Custom CRON is reserved for future enhancement.

## Data Access Patterns

### Tag Filtering Query

```sql
SELECT t.*
FROM task t
LEFT JOIN tasktaglink ttl ON t.id = ttl.task_id
LEFT JOIN tag tg ON ttl.tag_id = tg.id
WHERE t.user_id = :user_id
  AND tg.id IN (:tag_ids)
GROUP BY t.id
HAVING COUNT(DISTINCT tg.id) = :tag_count
```

**Rationale**: This ensures tasks match ALL specified tags (AND logic), not just ANY tag (OR logic).

### Search Query

```sql
SELECT t.*
FROM task t
WHERE t.user_id = :user_id
  AND (
    t.title ILIKE '%' || :search || '%'
    OR t.description ILIKE '%' || :search || '%'
  )
```

**Rationale**: ILIKE provides case-insensitive search in PostgreSQL.

### Upcoming Reminders Query

```sql
SELECT t.*
FROM task t
WHERE t.user_id = :user_id
  AND t.reminder_time BETWEEN NOW() AND NOW() + INTERVAL '1 hour'
  AND t.completed = FALSE
ORDER BY t.reminder_time ASC
```

**Rationale**: Efficiently retrieves reminders that should trigger soon.

## Data Retention

- **Completed Tasks**: Retain indefinitely (no auto-deletion)
- **Event Logs**: Retain for 30 days (for debugging)
- **Expired Reminders**: Clean up after task completion or deletion

## Performance Considerations

1. **Index Strategy**: Compound indexes on (user_id, field) optimize common filter patterns
2. **Tag Filtering**: The GROUP BY HAVING pattern ensures correct AND semantics but may be slow for users with many tags; consider denormalization if needed
3. **Search Performance**: ILIKE with leading wildcard (`%term%`) prevents index usage; consider full-text search (pg_trgm) if search performance degrades
4. **Partitioning**: For large-scale deployment, consider partitioning tasks by user_id

# Data Model: Console TODO Application

**Feature**: 001-console-todo-app
**Date**: 2025-12-30
**Input**: Feature specification and implementation plan

## Core Entities

### Task
The primary entity representing a single task in the system.

**Attributes:**
- `id` (integer, required, immutable): Unique identifier for the task, auto-generated
- `title` (string, required): The task title, non-empty
- `description` (string, optional): Optional detailed description of the task
- `completed` (boolean, required): Completion status of the task

**Structure (Dictionary/JSON representation):**
```python
{
    "id": 1,
    "title": "Sample task",
    "description": "Optional description",
    "completed": False
}
```

## Data Relationships

- Tasks are stored in a simple list/array structure
- Each task ID must be unique within the collection
- Task IDs are immutable and never reused after deletion

## Data Lifecycle

### Creation
- New tasks are created with:
  - Auto-generated unique ID (incremental)
  - Provided title (validated as non-empty)
  - Optional description (may be empty string)
  - `completed` set to `False`

### Update
- Task titles and descriptions can be modified
- Task ID and completion status remain unchanged unless explicitly toggled
- All attributes must maintain their data types

### Deletion
- Tasks are removed from the collection
- IDs are not reused (preserves data integrity)
- No cascade effects (no related entities)

### Persistence
- Tasks are serialized to JSON format
- Stored in a single JSON file
- Complete task list is saved/loaded as a single array

## Validation Rules

1. **ID Uniqueness**: All task IDs must be unique within the collection
2. **Title Non-Empty**: Task titles must not be empty strings
3. **Type Consistency**: Each attribute must maintain its defined data type
4. **ID Immutability**: Task IDs cannot be changed after creation
5. **Boolean Status**: The completed field must always be a boolean value

## JSON Schema

```json
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {"type": "integer"},
      "title": {"type": "string", "minLength": 1},
      "description": {"type": "string"},
      "completed": {"type": "boolean"}
    },
    "required": ["id", "title", "completed"]
  }
}
```
# Data Model: TODO Application (Full-Stack Web) with Authentication

## Entities

### User
**Description**: Represents an authenticated user in the system, linked to a Clerk user ID

**Fields**:
- `id` (Integer): Internal primary key
- `clerk_user_id` (String, unique): The unique identifier from Clerk, immutable
- `created_at` (DateTime): Timestamp when the user record was created

**Validation Rules**:
- `clerk_user_id` must be unique across all users
- `clerk_user_id` cannot be modified after creation
- `created_at` is set automatically on creation

### Task
**Description**: Represents a personal task belonging to a single user

**Fields**:
- `id` (Integer): Primary key
- `user_id` (Integer, foreign key): References User.id for ownership
- `title` (String): Task title/description
- `description` (String, optional): Detailed task description
- `completed` (Boolean): Whether the task is completed
- `priority` (String/Enum): Priority level (HIGH, MEDIUM, LOW)
- `due_date` (DateTime, optional): When the task is due
- `recurrence_rule` (String, optional): Recurrence pattern (DAILY, WEEKLY, MONTHLY)
- `created_at` (DateTime): When the task was created
- `updated_at` (DateTime): When the task was last updated

**Validation Rules**:
- `user_id` must reference a valid User record
- `title` is required and must not be empty
- `priority` must be one of the defined values (HIGH, MEDIUM, LOW)
- `due_date` must be in the future if provided
- `recurrence_rule` must be one of the defined values if provided

**State Transitions**:
- `completed` can transition from False to True (completed) or True to False (uncompleted)

### Tag
**Description**: Represents a structured object with additional metadata that can be associated with tasks

**Fields**:
- `id` (Integer): Primary key
- `name` (String): Tag name
- `color` (String): Color code for UI display
- `priority` (Integer): Display priority for the tag
- `user_id` (Integer, foreign key): References User.id for ownership
- `created_at` (DateTime): When the tag was created

**Validation Rules**:
- `name` is required and must be unique per user
- `color` must be a valid color code format
- `user_id` must reference a valid User record
- `priority` must be a non-negative integer

### TaskTag (Join Table)
**Description**: Junction table to connect tasks and tags (many-to-many relationship)

**Fields**:
- `task_id` (Integer, foreign key): References Task.id
- `tag_id` (Integer, foreign key): References Tag.id

**Validation Rules**:
- Both `task_id` and `tag_id` must reference valid records
- The combination of `task_id` and `tag_id` must be unique

## Relationships

### User → Task
- One-to-many relationship
- A user can have many tasks
- Tasks are always scoped to their owner user

### User → Tag
- One-to-many relationship
- A user can have many tags
- Tags are always scoped to their owner user

### Task ↔ Tag
- Many-to-many relationship through TaskTag
- A task can have multiple tags
- A tag can be applied to multiple tasks
- Both relationships are constrained by user ownership
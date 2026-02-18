# API Contracts: TODO Application (Full-Stack Web) with Authentication

## Authentication Endpoints

### GET /api/v1/auth/me
**Description**: Get current authenticated user information

**Headers**:
- `Authorization: Bearer {token}`

**Response**:
```json
{
  "id": 1,
  "clerk_user_id": "user_abc123",
  "created_at": "2023-01-01T00:00:00Z"
}
```

**Status Codes**:
- 200: Success
- 401: Unauthorized (invalid/expired token)

## Task Endpoints

### GET /api/v1/tasks
**Description**: Get all tasks for the authenticated user

**Query Parameters**:
- `completed` (optional): Filter by completion status
- `priority` (optional): Filter by priority level (HIGH, MEDIUM, LOW)
- `due_date_from` (optional): Filter tasks with due date after this date
- `due_date_to` (optional): Filter tasks with due date before this date
- `search` (optional): Search in title and description
- `sort_by` (optional): Sort by (created_at, updated_at, due_date, priority)
- `order` (optional): Sort order (asc, desc)
- `limit` (optional): Number of results to return
- `offset` (optional): Number of results to skip

**Headers**:
- `Authorization: Bearer {token}`

**Response**:
```json
{
  "tasks": [
    {
      "id": 1,
      "title": "Sample Task",
      "description": "Task description",
      "completed": false,
      "priority": "HIGH",
      "due_date": "2023-12-31T23:59:59Z",
      "recurrence_rule": "WEEKLY",
      "created_at": "2023-01-01T00:00:00Z",
      "updated_at": "2023-01-01T00:00:00Z",
      "tags": [
        {
          "id": 1,
          "name": "work",
          "color": "#FF0000",
          "priority": 1
        }
      ]
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

**Status Codes**:
- 200: Success
- 401: Unauthorized

### POST /api/v1/tasks
**Description**: Create a new task for the authenticated user

**Headers**:
- `Authorization: Bearer {token}`
- `Content-Type: application/json`

**Request Body**:
```json
{
  "title": "New Task",
  "description": "Task description",
  "priority": "MEDIUM",
  "due_date": "2023-12-31T23:59:59Z",
  "recurrence_rule": "DAILY",
  "tags": [1, 2]  // Array of tag IDs to associate with the task
}
```

**Response**:
```json
{
  "id": 1,
  "title": "New Task",
  "description": "Task description",
  "completed": false,
  "priority": "MEDIUM",
  "due_date": "2023-12-31T23:59:59Z",
  "recurrence_rule": "DAILY",
  "created_at": "2023-01-01T00:00:00Z",
  "updated_at": "2023-01-01T00:00:00Z",
  "tags": [
    {
      "id": 1,
      "name": "work",
      "color": "#FF0000",
      "priority": 1
    }
  ]
}
```

**Status Codes**:
- 201: Created
- 400: Invalid request body
- 401: Unauthorized

### GET /api/v1/tasks/{id}
**Description**: Get a specific task by ID

**Headers**:
- `Authorization: Bearer {token}`

**Response**:
```json
{
  "id": 1,
  "title": "Sample Task",
  "description": "Task description",
  "completed": false,
  "priority": "HIGH",
  "due_date": "2023-12-31T23:59:59Z",
  "recurrence_rule": "WEEKLY",
  "created_at": "2023-01-01T00:00:00Z",
  "updated_at": "2023-01-01T00:00:00Z",
  "tags": [
    {
      "id": 1,
      "name": "work",
      "color": "#FF0000",
      "priority": 1
    }
  ]
}
```

**Status Codes**:
- 200: Success
- 401: Unauthorized
- 404: Task not found (or not owned by user)

### PUT /api/v1/tasks/{id}
**Description**: Update a specific task

**Headers**:
- `Authorization: Bearer {token}`
- `Content-Type: application/json`

**Request Body**:
```json
{
  "title": "Updated Task",
  "description": "Updated description",
  "completed": true,
  "priority": "LOW",
  "due_date": "2023-12-31T23:59:59Z",
  "recurrence_rule": "MONTHLY",
  "tags": [1, 3]  // Updated array of tag IDs
}
```

**Response**:
```json
{
  "id": 1,
  "title": "Updated Task",
  "description": "Updated description",
  "completed": true,
  "priority": "LOW",
  "due_date": "2023-12-31T23:59:59Z",
  "recurrence_rule": "MONTHLY",
  "created_at": "2023-01-01T00:00:00Z",
  "updated_at": "2023-01-02T00:00:00Z",
  "tags": [
    {
      "id": 1,
      "name": "work",
      "color": "#FF0000",
      "priority": 1
    },
    {
      "id": 3,
      "name": "personal",
      "color": "#00FF00",
      "priority": 2
    }
  ]
}
```

**Status Codes**:
- 200: Success
- 400: Invalid request body
- 401: Unauthorized
- 404: Task not found (or not owned by user)

### DELETE /api/v1/tasks/{id}
**Description**: Delete a specific task

**Headers**:
- `Authorization: Bearer {token}`

**Status Codes**:
- 204: Success (No content)
- 401: Unauthorized
- 404: Task not found (or not owned by user)

### PATCH /api/v1/tasks/{id}/toggle-completion
**Description**: Toggle the completion status of a task

**Headers**:
- `Authorization: Bearer {token}`
- `Content-Type: application/json`

**Request Body**:
```json
{
  "completed": true
}
```

**Response**:
```json
{
  "id": 1,
  "title": "Sample Task",
  "completed": true,
  "updated_at": "2023-01-02T00:00:00Z"
}
```

**Status Codes**:
- 200: Success
- 400: Invalid request body
- 401: Unauthorized
- 404: Task not found (or not owned by user)

## Tag Endpoints

### GET /api/v1/tags
**Description**: Get all tags for the authenticated user

**Headers**:
- `Authorization: Bearer {token}`

**Response**:
```json
{
  "tags": [
    {
      "id": 1,
      "name": "work",
      "color": "#FF0000",
      "priority": 1,
      "created_at": "2023-01-01T00:00:00Z"
    }
  ]
}
```

**Status Codes**:
- 200: Success
- 401: Unauthorized

### POST /api/v1/tags
**Description**: Create a new tag for the authenticated user

**Headers**:
- `Authorization: Bearer {token}`
- `Content-Type: application/json`

**Request Body**:
```json
{
  "name": "personal",
  "color": "#00FF00",
  "priority": 2
}
```

**Response**:
```json
{
  "id": 2,
  "name": "personal",
  "color": "#00FF00",
  "priority": 2,
  "created_at": "2023-01-01T00:00:00Z"
}
```

**Status Codes**:
- 201: Created
- 400: Invalid request body
- 401: Unauthorized

### PUT /api/v1/tags/{id}
**Description**: Update a specific tag

**Headers**:
- `Authorization: Bearer {token}`
- `Content-Type: application/json`

**Request Body**:
```json
{
  "name": "updated tag",
  "color": "#0000FF",
  "priority": 3
}
```

**Response**:
```json
{
  "id": 1,
  "name": "updated tag",
  "color": "#0000FF",
  "priority": 3,
  "created_at": "2023-01-01T00:00:00Z",
  "updated_at": "2023-01-02T00:00:00Z"
}
```

**Status Codes**:
- 200: Success
- 400: Invalid request body
- 401: Unauthorized
- 404: Tag not found (or not owned by user)

### DELETE /api/v1/tags/{id}
**Description**: Delete a specific tag

**Headers**:
- `Authorization: Bearer {token}`

**Status Codes**:
- 204: Success (No content)
- 401: Unauthorized
- 404: Tag not found (or not owned by user)
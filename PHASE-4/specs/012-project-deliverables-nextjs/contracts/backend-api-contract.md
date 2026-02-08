# API Contract: AI Task Management Chatbot Backend

## Overview
This document specifies the API contracts for the backend services of the AI Task Management Chatbot.

## Authentication
All API endpoints (except `/auth/login` and `/auth/register`) require JWT authentication in the form of a Bearer token in the Authorization header:
```
Authorization: Bearer <jwt_token>
```

## API Base URL
```
http://localhost:8000/api/v1
```

## Endpoints

### Authentication

#### POST /auth/login
Authenticate a user and return a JWT token.

**Request:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response (200 OK):**
```json
{
  "access_token": "string",
  "token_type": "bearer"
}
```

**Response (401 Unauthorized):**
```json
{
  "detail": "Incorrect username or password"
}
```

#### POST /auth/register
Register a new user.

**Request:**
```json
{
  "email": "user@example.com",
  "username": "string",
  "password": "string"
}
```

**Response (200 OK):**
```json
{
  "id": 1,
  "email": "user@example.com",
  "username": "string"
}
```

**Response (400 Bad Request):**
```json
{
  "detail": "Username or email already registered"
}
```

### Messages

#### POST /messages
Send a message to the AI chatbot and receive a response.

**Headers:**
```
Content-Type: application/json
Authorization: Bearer <jwt_token>
```

**Request:**
```json
{
  "content": "string"
}
```

**Response (200 OK):**
```json
{
  "response": "string"
}
```

**Response (400 Bad Request):**
```json
{
  "detail": "Invalid message format"
}
```

#### GET /messages/history
Retrieve conversation history for the authenticated user.

**Headers:**
```
Authorization: Bearer <jwt_token>
```

**Query Parameters:**
- `conversation_id` (optional): Specific conversation to retrieve
- `limit` (optional): Number of messages to return (default: 50)
- `offset` (optional): Number of messages to skip (default: 0)

**Response (200 OK):**
```json
[
  {
    "id": 1,
    "role": "user",
    "content": "string",
    "timestamp": "2023-01-01T00:00:00Z"
  },
  {
    "id": 2,
    "role": "assistant",
    "content": "string",
    "timestamp": "2023-01-01T00:00:00Z"
  }
]
```

### Tasks

#### GET /tasks
Retrieve tasks for the authenticated user.

**Headers:**
```
Authorization: Bearer <jwt_token>
```

**Query Parameters:**
- `status` (optional): Filter by status ('pending', 'completed', 'archived')
- `limit` (optional): Number of tasks to return (default: 50)
- `offset` (optional): Number of tasks to skip (default: 0)

**Response (200 OK):**
```json
[
  {
    "id": 1,
    "title": "string",
    "status": "pending",
    "created_at": "2023-01-01T00:00:00Z",
    "updated_at": "2023-01-01T00:00:00Z"
  }
]
```

#### POST /tasks
Create a new task.

**Headers:**
```
Content-Type: application/json
Authorization: Bearer <jwt_token>
```

**Request:**
```json
{
  "title": "string"
}
```

**Response (201 Created):**
```json
{
  "id": 1,
  "title": "string",
  "status": "pending",
  "created_at": "2023-01-01T00:00:00Z",
  "updated_at": "2023-01-01T00:00:00Z"
}
```

#### PUT /tasks/{task_id}
Update an existing task.

**Headers:**
```
Content-Type: application/json
Authorization: Bearer <jwt_token>
```

**Request:**
```json
{
  "title": "string",
  "status": "pending"
}
```

**Response (200 OK):**
```json
{
  "id": 1,
  "title": "string",
  "status": "pending",
  "created_at": "2023-01-01T00:00:00Z",
  "updated_at": "2023-01-01T00:00:00Z"
}
```

#### DELETE /tasks/{task_id}
Delete a task.

**Headers:**
```
Authorization: Bearer <jwt_token>
```

**Response (204 No Content):**
```
(Empty response body)
```

**Response (404 Not Found):**
```json
{
  "detail": "Task not found"
}
```

### Health Check

#### GET /health
Check the health status of the API.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2023-01-01T00:00:00Z"
}
```

## Error Responses
All error responses follow the same structure:
```json
{
  "detail": "Error message"
}
```

## Rate Limiting
All endpoints are subject to rate limiting (100 requests per minute per IP).
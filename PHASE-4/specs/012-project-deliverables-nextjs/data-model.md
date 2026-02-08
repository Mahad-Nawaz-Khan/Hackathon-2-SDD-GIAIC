# Data Model: AI Task Management Chatbot

## Overview
This document defines the data models for the AI Task Management Chatbot, including database schemas, relationships, and validation rules based on the feature specification.

## Entity: User
**Description:** Represents a system user with authentication details

**Fields:**
- `id` (Integer, Primary Key, Auto-increment)
- `email` (String, Unique, Not Null)
- `username` (String, Unique, Not Null)
- `hashed_password` (String, Not Null)
- `created_at` (DateTime, Not Null, Default: now())
- `updated_at` (DateTime, Not Null, Default: now())
- `is_active` (Boolean, Not Null, Default: True)

**Validation Rules:**
- Email must be a valid email format
- Username must be 3-30 characters, alphanumeric + underscores/hyphens
- Password must meet complexity requirements (handled during hashing)

## Entity: Conversation
**Description:** Represents a conversation thread between user and AI assistant

**Fields:**
- `id` (Integer, Primary Key, Auto-increment)
- `user_id` (Integer, Foreign Key to User.id, Not Null)
- `title` (String, Optional, Max 200 characters)
- `created_at` (DateTime, Not Null, Default: now())
- `updated_at` (DateTime, Not Null, Default: now())

**Relationships:**
- One-to-Many: User → Conversations
- One-to-Many: Conversation → Messages

## Entity: Message
**Description:** Represents a single message in a conversation

**Fields:**
- `id` (Integer, Primary Key, Auto-increment)
- `conversation_id` (Integer, Foreign Key to Conversation.id, Not Null)
- `role` (String, Not Null, Values: 'user' | 'assistant')
- `content` (Text, Not Null)
- `timestamp` (DateTime, Not Null, Default: now())
- `token_count` (Integer, Optional)

**Validation Rules:**
- Role must be either 'user' or 'assistant'
- Content must not be empty

## Entity: Task
**Description:** Represents a task managed by the AI assistant

**Fields:**
- `id` (Integer, Primary Key, Auto-increment)
- `user_id` (Integer, Foreign Key to User.id, Not Null)
- `title` (String, Not Null, Max 500 characters)
- `status` (String, Not Null, Values: 'pending' | 'completed' | 'archived')
- `created_at` (DateTime, Not Null, Default: now())
- `updated_at` (DateTime, Not Null, Default: now())
- `completed_at` (DateTime, Optional)

**Validation Rules:**
- Title must not be empty
- Status must be one of the allowed values

## Entity: AuditLog
**Description:** Tracks system actions for observability and debugging

**Fields:**
- `id` (Integer, Primary Key, Auto-increment)
- `user_id` (Integer, Foreign Key to User.id, Optional)
- `action` (String, Not Null)
- `resource_type` (String, Not Null)
- `resource_id` (Integer, Optional)
- `details` (JSON, Optional)
- `timestamp` (DateTime, Not Null, Default: now())
- `ip_address` (String, Optional)
- `user_agent` (String, Optional)

**Validation Rules:**
- Action and resource_type must not be empty

## Relationships Summary

```
User (1) ←→ (Many) Conversation
Conversation (1) ←→ (Many) Message
User (1) ←→ (Many) Task
User (1) ←→ (Many) AuditLog (Optional)
```

## State Transitions

### Task Status Transitions
- `pending` → `completed` (when task is marked complete)
- `pending` → `archived` (when task is deleted)
- `completed` → `pending` (when task is reopened)
- `completed` → `archived` (when completed task is archived)
- `archived` → `pending` (when archived task is restored)

## Indexes

For Performance Optimization:
- User.email (Unique index)
- User.username (Unique index)
- Conversation.user_id (Index)
- Message.conversation_id (Index)
- Message.timestamp (Index)
- Task.user_id (Index)
- Task.status (Index)
- AuditLog.timestamp (Index)
- AuditLog.user_id (Index)

## Constraints

- Referential Integrity: Foreign key constraints to maintain data consistency
- Unique Constraints: Email and username uniqueness for users
- Check Constraints: Valid values for status fields
- Not Null Constraints: Required fields as specified above
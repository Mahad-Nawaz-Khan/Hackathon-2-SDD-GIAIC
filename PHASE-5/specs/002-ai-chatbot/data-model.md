# Data Model: AI Chatbot System

## Overview
This document defines the data structures needed for the AI Chatbot System that integrates with the existing TODO application.

## Core Entities

### ChatInteraction
Represents a conversation between a user and the AI chatbot.

- **id**: String (unique identifier)
- **userId**: String (references Clerk user ID via existing user table)
- **sessionId**: String (for grouping related messages in a conversation)
- **createdAt**: DateTime (timestamp of creation)
- **updatedAt**: DateTime (timestamp of last update)

### ChatMessage
Represents individual messages within a chat interaction.

- **id**: String (unique identifier)
- **chatInteractionId**: String (foreign key to ChatInteraction)
- **senderType**: Enum ['USER', 'AI'] (who sent the message)
- **content**: String (the actual message text)
- **intent**: String (detected user intent, e.g., CREATE_TASK, UPDATE_TASK, etc.)
- **processed**: Boolean (whether the message has been processed)
- **createdAt**: DateTime (timestamp of creation)

### Intent
Represents the recognized user intention from their natural language input.

- **type**: Enum ['CREATE_TASK', 'UPDATE_TASK', 'DELETE_TASK', 'SEARCH_TASKS', 'LIST_TASKS', 'READ_TASK']
- **parameters**: Object (extracted parameters like task title, due date, priority, etc.)
- **confidence**: Float (confidence score of intent recognition)

### OperationRequest
Structured request to be processed by the MCP server containing validated parameters.

- **id**: String (unique identifier)
- **chatMessageId**: String (foreign key to ChatMessage)
- **operationType**: Enum ['CREATE', 'READ', 'UPDATE', 'DELETE']
- **entityType**: String ('TASK', 'TAG', etc.)
- **parameters**: Object (validated parameters for the operation)
- **status**: Enum ['PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED']
- **response**: Object (result from MCP server)
- **createdAt**: DateTime (timestamp of creation)

## Relationships
- One ChatInteraction has many ChatMessages
- One ChatMessage has one Intent (detected from the message content)
- One ChatMessage generates zero or one OperationRequest (for data operations)

## Validation Rules
- All operations must validate against existing user authentication
- ChatMessage content must not exceed 5000 characters
- Intent confidence must be >0.7 for automated operations, otherwise requires user confirmation
- OperationRequest parameters must conform to existing TODO app validation rules

## State Transitions
- ChatMessage: [CREATED] → [INTENT_DETECTED] → [OPERATION_QUEUED/RESPONSE_ONLY] → [COMPLETED]
- OperationRequest: [PENDING] → [IN_PROGRESS] → [COMPLETED/FAILED]

## Integration with Existing Models
- Leverages existing User model through userId reference
- Leverages existing Task model through operations performed via MCP server
- Leverages existing Tag model through operations performed via MCP server
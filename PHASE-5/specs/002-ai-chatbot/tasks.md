# Tasks: AI Chatbot System

**Feature**: AI Chatbot System with OpenAI Agents SDK integration, MCP server for CRUD operations, and multi-model support (OpenAI and Gemini)
**Created**: 2026-02-06
**Status**: Ready for implementation
**MVP**: Complete US1 - Natural Language Task Management

## Dependencies

- User Story 2 depends on foundational infrastructure from User Story 1
- User Story 3 depends on MCP server implementation from foundational phase

## Parallel Execution Opportunities

- Contract definition can run parallel to implementation tasks
- UI component development can run parallel to backend service development
- Different models (OpenAI/Gemini) can be developed in parallel after core infrastructure

## Implementation Strategy

**MVP Scope**: Complete User Story 1 (Natural Language Task Management) with basic OpenAI integration. This delivers core value of natural language task management while keeping initial scope manageable.

**Incremental Delivery**:
- Phase 1-2: Core infrastructure and US1 implementation
- Phase 3: US2 Multi-Model support
- Phase 4: US3 MCP-mediated operations
- Phase 5: Polish and optimization

---

## Phase 1: Setup

**Goal**: Establish project infrastructure and dependencies per implementation plan

- [ ] T001 Set up backend directory structure for chatbot components: backend/src/models/chat_models.py, backend/src/services/chat_service.py, backend/src/services/ai_agents_service.py, backend/src/services/gemini_service.py, backend/src/services/mcp_adapter.py, backend/src/api/chat_router.py, backend/src/tools/
- [ ] T002 Set up frontend directory structure for chatbot components: frontend/src/components/ChatBot.tsx, frontend/src/services/chatService.ts
- [ ] T003 [P] Install required dependencies for OpenAI Agents SDK: openai, openai-agents, python-dotenv
- [ ] T004 [P] Install required dependencies for Google Generative AI: google-generativeai, vertexai
- [ ] T005 [P] Install MCP server framework dependencies
- [ ] T006 [P] Update existing requirements.txt and package.json with new dependencies
- [ ] T007 Set up environment variables for AI providers in backend/.env.template

---

## Phase 2: Foundational Infrastructure

**Goal**: Implement core infrastructure components needed by all user stories

- [ ] T010 Create ChatInteraction model in backend/src/models/chat_models.py following data-model.md specification
- [ ] T011 Create ChatMessage model in backend/src/models/chat_models.py following data-model.md specification
- [ ] T012 Create Intent model in backend/src/models/chat_models.py following data-model.md specification
- [ ] T013 Create OperationRequest model in backend/src/models/chat_models.py following data-model.md specification
- [ ] T014 [P] Implement database migrations for chatbot models
- [ ] T015 [P] Set up MCP server framework in backend/src/mcp_server/
- [ ] T016 Create MCP adapter service in backend/src/services/mcp_adapter.py
- [ ] T017 Implement basic chat service in backend/src/services/chat_service.py with skeleton methods
- [ ] T018 Implement base OpenAI agents service in backend/src/services/ai_agents_service.py
- [ ] T019 Implement base Gemini service in backend/src/services/gemini_service.py
- [ ] T020 Create chat API router in backend/src/api/chat_router.py with basic endpoints

---

## Phase 3: US1 - Natural Language Task Management

**Goal**: Enable users to interact with the TODO application using natural language to create, update, and manage tasks

**Independent Test**: User can type "Create a high priority task to buy groceries by Friday" and the system creates an appropriate task in the database.

- [ ] T030 [US1] Implement intent classification in ChatService to detect CREATE_TASK from user messages
- [ ] T031 [US1] Implement parameter extraction for task details (title, due date, priority) from user messages
- [ ] T032 [US1] Connect OpenAI Agents service to process task creation requests
- [ ] T033 [US1] Implement task creation tool for OpenAI Agents in backend/src/tools/task_crud_tools.py
- [ ] T034 [US1] Integrate with existing TODO app task creation via MCP adapter
- [ ] T035 [US1] Implement frontend chat interface in frontend/src/components/ChatBot.tsx
- [ ] T036 [US1] Implement chat API client in frontend/src/services/chatService.ts
- [ ] T037 [US1] Add chat UI to dashboard in frontend/src/pages/Dashboard.tsx
- [ ] T038 [US1] Test complete workflow: user input → intent detection → task creation → UI update

---

## Phase 4: US2 - Multi-Model Intelligence

**Goal**: Implement intelligent routing to appropriate AI models based on operation type

**Independent Test**: System uses Gemini for read-only queries and OpenAI for complex operations requiring MCP server integration

- [ ] T050 [US2] Implement model routing logic in backend/src/services/chat_service.py
- [ ] T051 [US2] Create query classifier to distinguish read vs. write operations
- [ ] T052 [US2] Implement read-only operation handler using Gemini model
- [ ] T053 [US2] Enhance OpenAI agents for complex operations requiring data changes
- [ ] T054 [US2] Add cost-optimization logic to prefer cheaper models when possible
- [ ] T055 [US2] Implement fallback mechanism between models
- [ ] T056 [US2] Update frontend to indicate which model is being used for each operation

---

## Phase 5: US3 - MCP Server Mediated Operations

**Goal**: Ensure all data operations go through MCP server with proper authentication and validation

**Independent Test**: All CRUD operations pass through MCP server which validates user permissions before executing

- [ ] T070 [US3] Complete MCP server implementation with all required endpoints
- [ ] T071 [US3] Implement authentication validation in MCP server for all operations
- [ ] T072 [US3] Create user data access tools in backend/src/tools/user_data_tools.py
- [ ] T073 [US3] Ensure all AI-driven operations route through MCP server
- [ ] T074 [US3] Implement proper error handling when MCP server denies operations
- [ ] T075 [US3] Add confirmation prompts for destructive operations
- [ ] T076 [US3] Test authorization enforcement: unauthorized users cannot access other users' data

---

## Phase 6: Polish & Cross-Cutting Concerns

**Goal**: Final touches and optimization of the entire system

- [ ] T090 Implement conversation context management and history
- [ ] T091 Add proper error handling and user feedback for AI processing failures
- [ ] T092 Optimize response times and implement caching where appropriate
- [ ] T093 Add logging and monitoring for AI interactions
- [ ] T094 Implement rate limiting for chatbot endpoints
- [ ] T095 Create comprehensive test suite for all chatbot functionality
- [ ] T096 Write documentation for chatbot features and API usage
- [ ] T097 Conduct end-to-end testing of all user stories
- [ ] T098 Performance testing and optimization
- [ ] T099 Final integration testing with existing TODO application
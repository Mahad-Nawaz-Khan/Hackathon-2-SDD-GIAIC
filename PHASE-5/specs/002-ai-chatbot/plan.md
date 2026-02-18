# Implementation Plan: AI Chatbot System

**Branch**: `002-ai-chatbot` | **Date**: 2026-02-06 | **Spec**: [link to spec](spec.md)
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of an AI chatbot system that enables natural language interactions with the TODO application. The system will utilize OpenAI Agents SDK for complex reasoning and orchestration, with MCP server mediating all data operations to ensure security and proper authorization. Simple read operations will leverage the free-tier Gemini model for cost efficiency.

## Technical Context

**Language/Version**: Python 3.11, TypeScript/JavaScript
**Primary Dependencies**: OpenAI Agents SDK, MCP Server Framework, FastAPI, Next.js
**Storage**: PostgreSQL (via existing TODO app infrastructure)
**Testing**: pytest, Jest
**Target Platform**: Web application environment (server-side AI integration)
**Project Type**: Web (integrated with existing TODO app)
**Performance Goals**: <5 second response time for 95% of AI interactions
**Constraints**: <200ms p95 for MCP-mediated operations, proper authentication validation
**Scale/Scope**: Single-tenant operation matching existing TODO app scope

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- ✅ Authentication & Identity: Chatbot will respect Clerk-based authentication and pass user context through MCP server
- ✅ User Data Isolation: All operations will go through MCP server which will enforce user ownership validation
- ✅ Technology Stack Adherence: Will integrate with existing FastAPI backend and Next.js frontend
- ✅ Backend Authority: MCP server will derive user identity from verified tokens and enforce authorization
- ✅ Data Integrity & Security: All CRUD operations will be mediated through MCP server with proper validation
- ✅ Rate Limiting & Performance: Existing rate limiting will apply to chatbot endpoints as well
- ✅ AI Chatbot, Multi-Model Setup, and MCP Integration Constitution: All AI operations will go through MCP server as single source of truth; OpenAI Agents SDK as primary reasoning layer; Gemini for free-tier operations; all models respect authorization rules

## Project Structure

### Documentation (this feature)

```text
specs/002-ai-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   │   └── chat_models.py           # Chat interaction and intent models
│   ├── services/
│   │   ├── chat_service.py          # Core chatbot business logic
│   │   ├── ai_agents_service.py     # OpenAI Agents SDK integration
│   │   ├── gemini_service.py        # Gemini API integration
│   │   └── mcp_adapter.py           # MCP server communication layer
│   ├── api/
│   │   └── chat_router.py           # Chat API endpoints
│   └── tools/
│       ├── task_crud_tools.py       # MCP tools for task operations
│       └── user_data_tools.py       # MCP tools for user data access
└── tests/

frontend/
├── src/
│   ├── components/
│   │   └── ChatBot.tsx              # Chatbot UI component
│   ├── services/
│   │   └── chatService.ts           # Frontend chat API client
│   └── pages/
│       └── Dashboard.tsx             # Integration with existing dashboard
└── tests/
```

**Structure Decision**: Following the web application pattern to integrate with the existing TODO app architecture, with backend services for AI processing and MCP integration, and frontend components for user interaction.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
|           |            |                                     |
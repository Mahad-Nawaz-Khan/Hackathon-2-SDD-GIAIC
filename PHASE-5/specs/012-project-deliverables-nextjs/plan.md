# Implementation Plan: AI Task Management Chatbot

**Branch**: `012-project-deliverables-nextjs` | **Date**: 2026-02-06 | **Spec**: [link to spec.md](spec.md)
**Input**: Feature specification from `/specs/012-project-deliverables-nextjs/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of an AI-powered task management chatbot with Next.js frontend, Python backend (FastAPI), OpenAI Agents SDK integration, and MCP server for safe CRUD operations. The system will use Neon Serverless PostgreSQL for data persistence, JWT-based authentication, and streaming responses via OpenAI's runStreamed option.

## Technical Context

**Language/Version**: Python 3.11, TypeScript 5.0, Next.js 14.x
**Primary Dependencies**: FastAPI, OpenAI Agents SDK, MCP Python SDK, Neon PostgreSQL driver, JWT libraries, Tailwind CSS
**Storage**: Neon Serverless PostgreSQL with SQLAlchemy ORM
**Testing**: pytest for backend, Jest/React Testing Library for frontend
**Target Platform**: Web application (Next.js SSR/Client components)
**Project Type**: Web application (frontend + backend + MCP server)
**Performance Goals**: <2 second latency for streaming responses, 99.9% uptime for authentication
**Constraints**: Free tier usage for Gemini API, JWT token-based auth, React native hooks for state management
**Scale/Scope**: Single user conversations initially, extensible to multi-user system

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Based on the project constitution, this implementation plan complies with the core principles:
- Library-first: Each component (frontend, backend, MCP server) will be modular
- CLI Interface: Backend will expose API endpoints following text-in/out protocols
- Test-First: All components will have comprehensive test coverage
- Integration Testing: Focus on API contract tests, MCP tool integration, and end-to-end flows
- Observability: Structured logging for debugging and monitoring

## Project Structure

### Documentation (this feature)

```text
specs/012-project-deliverables-nextjs/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
# Web application structure
backend/
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── config.py               # Configuration and environment variables
│   ├── database.py             # Database connection and session management
│   ├── auth.py                 # JWT authentication utilities
│   ├── models/                 # SQLAlchemy models
│   │   ├── user.py
│   │   ├── message.py
│   │   └── task.py
│   ├── schemas/                # Pydantic schemas
│   │   ├── user.py
│   │   ├── message.py
│   │   └── task.py
│   ├── api/                    # API routes
│   │   ├── deps.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── messages.py
│   │   │   └── tasks.py
│   ├── agents/                 # OpenAI Agents integration
│   │   ├── agent_orchestrator.py
│   │   ├── intent_classifier.py
│   │   └── gemini_router.py
│   ├── mcp_client.py           # MCP client abstraction
│   └── utils/                  # Utility functions
│       ├── logger.py
│       └── helpers.py
├── mcp_server/                 # MCP server implementation
│   ├── server.py
│   ├── tools/
│   │   ├── task_tools.py
│   │   └── validation.py
│   └── config.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── contract/
├── alembic/
│   └── versions/
├── requirements.txt
├── requirements-dev.txt
└── Dockerfile

frontend/
├── app/
│   ├── page.tsx               # Main chat interface
│   ├── layout.tsx
│   ├── globals.css
│   └── components/            # React components
│       ├── ChatInterface.tsx
│       ├── MessageList.tsx
│       ├── MessageBubble.tsx
│       ├── ChatInput.tsx
│       └── TaskActions.tsx
├── lib/
│   ├── api.ts                 # API client utilities
│   ├── auth.ts                # Authentication utilities
│   ├── types.ts               # TypeScript types
│   └── hooks/                 # Custom React hooks
│       ├── useMessages.ts
│       └── useTasks.ts
├── styles/
│   └── globals.css
├── public/
├── next.config.js
├── tailwind.config.js
├── tsconfig.json
├── package.json
└── Dockerfile

# Shared specifications
specs/
└── 012-project-deliverables-nextjs/
    ├── spec.md
    ├── plan.md
    ├── data-model.md
    ├── research.md
    ├── quickstart.md
    ├── contracts/
    └── tasks.md
```

**Structure Decision**: Web application structure with separate frontend and backend components. The backend contains both the main FastAPI application and the MCP server as separate modules. The frontend is a Next.js application with React components for the chat interface. This structure allows for clear separation of concerns while maintaining the ability to deploy components independently if needed.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| (None at this time) | (Not applicable) | (Not applicable) |

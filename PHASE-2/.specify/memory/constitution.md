<!--
SYNC IMPACT REPORT:
Version change: N/A -> 1.0.0
Modified principles: N/A (new constitution)
Added sections: Core Principles (6), Additional Constraints, Development Workflow, Governance
Removed sections: N/A
Templates requiring updates:
- ✅ .specify/templates/plan-template.md - updated to align with auth and data isolation principles
- ✅ .specify/templates/spec-template.md - updated to include auth requirements
- ✅ .specify/templates/tasks-template.md - updated to include security task types
- ⚠️  README.md - requires update to reflect new principles (pending manual update)
Follow-up TODOs: None
-->
# TODO Application (Full-Stack Web) Constitution

## Core Principles

### I. Authentication & Identity (Mandatory)
User authentication must be handled exclusively by Clerk. Login, signup, and account management pages must NOT be custom-built. Only Clerk-provided components, routes, and hosted pages may be used. Clerk documentation must be followed as the source of truth. Clerk user ID is the authoritative identity for all backend operations.

### II. User Data Isolation (Non-Negotiable)
Every task must belong to exactly one authenticated user. No user may access, modify, or infer another user's data. Backend must enforce user ownership at the database query level. Frontend must never pass user IDs manually. Cross-user data leakage is forbidden.

### III. Technology Stack Adherence
Frontend must use Next.js (App Router), backend must use FastAPI, and database must use Neon Serverless PostgreSQL with SQLModel ORM. All components must follow their respective framework best practices. Third-party libraries should align with the chosen stack and not introduce conflicting patterns.

### IV. Backend Authority (Mandatory)
Backend validates Clerk-issued JWTs. Backend derives user identity exclusively from verified tokens. Business logic and access control reside in backend only. Frontend should never make decisions about user permissions or data access - all authorization must be verified by the backend.

### V. Data Integrity & Security
Tasks are always scoped to a user record in the database. Clerk user ID must map to a single internal user record. All database queries must include user ownership validation. Input validation and sanitization must be performed on all user inputs to prevent injection attacks.

### VI. Rate Limiting & Performance
All API endpoints must implement rate limiting to prevent abuse. Endpoints should be optimized for performance with appropriate database indexing. Cache headers should be properly configured for different types of responses.

## Additional Constraints

### Platform & Architecture
- Frontend: Next.js (App Router)
- Backend: FastAPI
- Database: Neon Serverless PostgreSQL
- ORM: SQLModel
- Authentication & Identity: Clerk (official SDKs only)

### Core Functional Features
- Full CRUD for tasks
- Completion toggling
- Priorities (HIGH / MEDIUM / LOW)
- Tags / categories with many-to-many relationship to tasks
- Search, filter, and sorting capabilities
- Recurring tasks (DAILY / WEEKLY / MONTHLY)
- Due dates and reminders

### Security Requirements
- All API endpoints must validate authentication
- All data access must validate user ownership
- Passwords and sensitive data must never be logged
- API responses must not leak information about other users' data existence
- Rate limiting implemented on all endpoints to prevent abuse

## Development Workflow

### Code Review Process
- All PRs must verify compliance with user data isolation rules
- Authentication flow changes require special security review
- Database schema changes must maintain user data isolation
- New API endpoints must include proper authentication validation
- Rate limiting implementation must be verified for all new endpoints

### Quality Gates
- All tests must pass before merging
- Authentication and authorization logic must be covered by tests
- No hardcoded secrets or credentials allowed
- Security scanning must pass for all PRs
- Performance benchmarks must be maintained

### Deployment Policy
- Environment-specific configurations must be securely managed
- Database migrations must preserve data integrity
- Rollback procedures must maintain user data isolation
- Production deployments require authentication verification
- Rate limiting configurations must be appropriate for production traffic

## Governance

This constitution supersedes all other development practices and architectural decisions for this project. All code changes must align with the stated principles. Amendments to this constitution require explicit documentation, team approval, and a migration plan for existing code. All PRs/reviews must verify compliance with user isolation and authentication requirements. Complexity must be justified with clear security and architectural benefits.

**Version**: 1.1.0 | **Ratified**: 2026-01-08 | **Last Amended**: 2026-01-08

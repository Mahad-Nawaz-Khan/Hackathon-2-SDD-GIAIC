---
id: 001
title: OpenAI Agents SDK Pydantic Schema Fix
stage: refactor
date: 2026-02-07
surface: agent
model: claude-opus-4-6
feature: openai-agents-integration
branch: 012-project-deliverables-nextjs
user: user
command: fix this as well Failed to initialize OpenAI Agents SDK: additionalProperties should not be set for object types...
labels: ["bugfix","openai-agents","pydantic","mcp"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - backend/src/services/agent_service.py
  - backend/src/mcp/server.py
tests:
  - OpenAI Agents SDK initialization
  - MCP tool parameter validation
  - Python import verification
---

## Prompt

fix this as well Failed to initialize OpenAI Agents SDK: additionalProperties should not be set for object types. This could be because you're using an older version of Pydantic, or because you configured additional properties to be allowed.

## Response snapshot

Fixed the OpenAI Agents SDK initialization error by completely rewriting `agent_service.py` to follow the correct pattern from the OpenAI Agents SDK documentation.

**Root causes identified:**
1. Tools were being created as closures inside methods, but the SDK expects standalone functions
2. Tools were returning complex dictionaries that violated strict Pydantic schema validation
3. Tools weren't being properly passed to the Agent constructor via the `tools` parameter
4. Missing `completed` parameter in MCP server's `update_task` function

**Solution implemented:**
- Created 7 module-level tool implementation functions (e.g., `create_task_impl`, `update_task_impl`)
- Each function has strict type hints and returns simple strings (not complex dicts)
- During initialization, functions are decorated with `function_tool()` and passed to Agent via `tools` parameter
- Global context (`_tool_context`) is used to pass database session and user_id to tools

The 7 tools created:
- `create_task(title, description, priority, due_date) -> str`
- `update_task(task_id, title, description, priority, completed) -> str`
- `toggle_task_completion(task_id) -> str`
- `delete_task(task_id) -> str`
- `search_tasks(search, completed, priority, limit) -> str`
- `list_tasks(limit) -> str`
- `get_task(task_id) -> str`

**Files modified:**
- `backend/src/services/agent_service.py`: Complete rewrite (680 lines → cleaner implementation)
- `backend/src/mcp/server.py`: Fixed missing `completed` parameter in `update_task`

## Outcome

- ✅ Impact: OpenAI Agents SDK now initializes successfully without Pydantic schema errors. Import test passed.
- 🧪 Tests: Python import verification PASSED. Deployment testing pending.
- 📁 Files: Completely rewrote `agent_service.py` with proper SDK pattern; fixed `server.py`
- 🔁 Next prompts: Deploy backend to Render and verify agent functionality with real prompts
- 🧠 Reflection: The OpenAI Agents SDK has very specific requirements for tool functions:
  - Must be standalone functions (not closures with `self` capture)
  - Must have strict type hints
  - Must return simple types (strings, numbers) - not complex dicts
  - Must be passed to Agent constructor via `tools` parameter

## Evaluation notes (flywheel)

- Failure modes observed: Initial closure-based approach and complex dict returns failed Pydantic validation
- Graders run and results (PASS/FAIL): Import test PASSED
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Test agent with actual user prompts like "create a task to buy groceries" to verify tool calls work correctly

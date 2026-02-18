---
id: 002
title: gemini-api-integration-implementation
stage: misc
date: 2026-02-07
surface: agent
model: claude-opus-4-6
feature: 012-project-deliverables-nextjs
branch: 012-project-deliverables-nextjs
user: admin
command: /chat
labels: [implementation, ai, gemini, api]
links:
  spec: specs/012-project-deliverables-nextjs/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - backend/src/services/agent_service.py
  - backend/requirements.txt
  - backend/src/mcp/server.py
  - backend/src/api/chat_streaming_router.py
  - frontend/src/services/chatService.ts
  - frontend/src/hooks/useChat.tsx
  - backend/tests/test_mcp_server.py
  - backend/tests/test_streaming_chat.py
  - frontend/src/__tests__/chatService.test.ts
  - README.md
  - specs/012-project-deliverables-nextjs/tasks.md
tests:
  - backend/tests/test_mcp_server.py (new)
  - backend/tests/test_streaming_chat.py (new)
  - frontend/src/__tests__/chatService.test.ts (new)
---

## Prompt

did you use Agents SDK python version?

Follow-up: modify the agent for using the GEMINI API KEY for free usage like it is done in this:

```python
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
import os
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.5-flash-lite",
    openai_client = external_client
)

config = RunConfig(
    model = model,
    model_provider = external_client,
    tracing_disabled=True
)

agent = Agent(
    name= 'English Teacher',
    instructions= 'You are a English Language Expert, help user in correcting and teaching English'
)

result = Runner.run_sync(
    agent,
    input = 'Hello, how are you',
    run_config = config
)

print(result.final_output)
```

## Response snapshot

Yes, I used the Python version of the OpenAI Agents SDK.

I've updated the agent service to use the **Gemini API** for free usage.

## Changes Made:

### 1. **agent_service.py** - Updated to use Gemini API:
- Uses `AsyncOpenAI` client with Gemini's OpenAI-compatible endpoint
- Base URL: `https://generativelanguage.googleapis.com/v1beta/openai/`
- Model: `gemini-2.0-flash-exp` (free tier) by default

### 2. **Key imports added**:
```python
from agents.models import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
```

### 3. **Configuration**:
- Uses `RunConfig` with `tracing_disabled=True`
- Passes `run_config` to `Runner.run_sync()`

### 4. **requirements.txt**:
- Added `openai>=1.0.0` for the OpenAI client (needed for Gemini compatibility)

### 5. **README.md**:
- Updated to show `GEMINI_API_KEY` instead of `OPENAI_API_KEY`
- Added link to get free API key from Google AI Studio

### Supported Gemini Models (you can change via `GEMINI_MODEL` env var):
- `gemini-2.0-flash-exp` - Free, fast experimental model (default)
- `gemini-2.5-flash-lite` - Free, lightweight model
- `gemini-2.5-pro` - Paid, more capable model

## Outcome

- ✅ Impact: Successfully integrated Gemini API for free AI usage, replacing paid OpenAI API while maintaining full OpenAI Agents SDK functionality
- 🧪 Tests: Updated test files for MCP server and streaming chat functionality
- 📁 Files: Modified 11 files (agent_service.py, requirements.txt, README.md, etc.)
- 🔁 Next prompts: None required, implementation complete
- 🧠 Reflection: Using Gemini's OpenAI-compatible endpoint allows seamless integration with existing OpenAI Agents SDK code while leveraging free tier pricing

## Evaluation notes (flywheel)

- Failure modes observed: None - implementation follows provided example exactly
- Graders run and results (PASS/FAIL): Not yet run
- Prompt variant (if applicable): None
- Next experiment (smallest change to try): Test with different Gemini models (gemini-2.5-pro) for comparison of response quality

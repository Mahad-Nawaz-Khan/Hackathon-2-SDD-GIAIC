# Model Context Protocol (MCP) Server Skill

This skill equips you with a ready‑to‑use scaffold and best‑practice guide for interacting with **Model Context Protocol (MCP) servers**. It covers connection handling, resource querying, authentication, error management, and efficient data combination across multiple MCP sources.

---
## What the skill generates
When invoked, the skill creates a directory `mcp-client/` containing a minimal yet extensible Python package for MCP interactions:
```
mcp-client/
├─ mcp/
│  ├─ __init__.py               # package entry point
│  ├─ config.py                # MCP connection configuration (env vars, .env support)
│  ├─ client.py                # high‑level client wrapper built on httpx (async)
│  ├─ resources.py             # helper functions for files, databases, APIs
│  ├─ auth.py                  # token‑based, API‑key, or mutual‑TLS auth helpers
│  └─ errors.py                # custom MCPError hierarchy
├─ examples/
│  ├─ query_files.py           # demo: list files, read content
│  ├─ query_db.py              # demo: run a simple SELECT via MCP DB proxy
│  └─ combine_sources.py       # demo: merge data from files + API
├─ tests/
│  ├─ __init__.py
│  ├─ test_client.py           # unit tests for client behavior
│  └─ test_resources.py        # tests for resource helpers
├─ docs/
│  └─ usage.md                # quick start guide and best‑practice notes
├─ .env.example                # example environment variables
├─ pyproject.toml              # Poetry (or pip) config, dependencies
├─ requirements.txt            # fallback for pip
├─ .gitignore
└─ README.md
```

---
## Core Components
### 1. Configuration (`mcp/config.py`)
```python
from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env from project root if present
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

class MCPSettings:
    """Configuration for MCP server connection.

    Environment variables (prefix `MCP_`):
    - MCP_HOST (e.g. https://mcp.example.com)
    - MCP_PORT (default 443)
    - MCP_TIMEOUT (seconds, default 30)
    - MCP_AUTH_METHOD ("token", "apikey", "mtls")
    - MCP_TOKEN / MCP_API_KEY / MCP_CERT / MCP_KEY (depending on method)
    """
    HOST: str = os.getenv("MCP_HOST", "https://mcp.local")
    PORT: int = int(os.getenv("MCP_PORT", "443"))
    TIMEOUT: int = int(os.getenv("MCP_TIMEOUT", "30"))
    AUTH_METHOD: str = os.getenv("MCP_AUTH_METHOD", "token").lower()
    TOKEN: str | None = os.getenv("MCP_TOKEN")
    API_KEY: str | None = os.getenv("MCP_API_KEY")
    CERT: str | None = os.getenv("MCP_CERT")   # path to client cert PEM (mtls)
    KEY: str | None = os.getenv("MCP_KEY")     # path to client key PEM (mtls)
```

### 2. Authentication helpers (`mcp/auth.py`)
```python
from .config import MCPSettings
from httpx import Auth

class MCPBearerAuth(Auth):
    def auth_flow(self, request):
        if MCPSettings.TOKEN:
            request.headers["Authorization"] = f"Bearer {MCPSettings.TOKEN}"
        yield request

class MCPApiKeyAuth(Auth):
    def auth_flow(self, request):
        if MCPSettings.API_KEY:
            request.headers["X-API-Key"] = MCPSettings.API_KEY
        yield request

def get_auth() -> Auth | None:
    method = MCPSettings.AUTH_METHOD
    if method == "token":
        return MCPBearerAuth()
    if method == "apikey":
        return MCPApiKeyAuth()
    # mTLS handled by httpx client configuration directly
    return None
```

### 3. High‑level client (`mcp/client.py`)
```python
import httpx
from .config import MCPSettings
from .auth import get_auth
from .errors import MCPError, MCPConnectionError, MCPResponseError

class MCPClient:
    def __init__(self):
        base_url = f"{MCPSettings.HOST}:{MCPSettings.PORT}"
        self.auth = get_auth()
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=MCPSettings.TIMEOUT,
            auth=self.auth,
            verify=True,
            cert=(MCPSettings.CERT, MCPSettings.KEY) if MCPSettings.AUTH_METHOD == "mtls" else None,
        )

    async def request(self, method: str, path: str, **kwargs):
        try:
            response = await self.client.request(method, path, **kwargs)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise MCPConnectionError(str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            raise MCPResponseError(exc.response.status_code, exc.response.text) from exc
        except httpx.RequestError as exc:
            raise MCPError(str(exc)) from exc
        return response.json()

    async def close(self):
        await self.client.aclose()
```
### 4. Resource helpers (`mcp/resources.py`)
```python
from .client import MCPClient
from typing import List, Dict, Any

client = MCPClient()

async def list_files(directory: str) -> List[Dict[str, Any]]:
    """List files in a remote directory via MCP file service.

    Returns a list of dicts: {"name": str, "size": int, "modified": str}
    """
    return await client.request("GET", f"/files?dir={directory}")

async def read_file(path: str) -> str:
    return await client.request("GET", f"/files/content?path={path}")

async def query_database(sql: str) -> List[Dict[str, Any]]:
    """Execute a read‑only SQL query through MCP's DB proxy.
    The server must have the appropriate permissions; only SELECT is allowed.
    """
    payload = {"query": sql}
    return await client.request("POST", "/db/query", json=payload)

async def call_api(endpoint: str, method: str = "GET", json: dict | None = None) -> Any:
    """Generic wrapper to call an external API exposed via MCP.
    Useful for third‑party services that MCP proxies.
    """
    return await client.request(method, f"/api{endpoint}", json=json)

async def combine_sources(file_path: str, sql_query: str) -> List[Dict[str, Any]]:
    """Example pattern: fetch a JSON file, run a DB query, merge on a common key.
    Returns a list of merged dictionaries.
    """
    file_content = await read_file(file_path)
    # Assuming the file is JSON list of objects
    import json as _json
    file_data = _json.loads(file_content)
    db_rows = await query_database(sql_query)
    # Simple join on "id"
    db_by_id = {row["id"]: row for row in db_rows}
    merged = []
    for item in file_data:
        row = db_by_id.get(item.get("id"))
        merged.append({**item, **(row or {})})
    return merged
```
### 5. Error hierarchy (`mcp/errors.py`)
```python
class MCPError(Exception):
    """Base class for all MCP‑related errors."""
    pass

class MCPConnectionError(MCPError):
    """Raised when the client cannot reach the MCP server."""
    pass

class MCPResponseError(MCPError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(f"MCP returned {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail
```
---
## Example Usage (`examples/`)
### List files
```python
import asyncio
from mcp.resources import list_files

async def main():
    files = await list_files("/project/docs")
    for f in files:
        print(f"{f['name']} ({f['size']} bytes)")

if __name__ == "__main__":
    asyncio.run(main())
```
### Query a database
```python
import asyncio
from mcp.resources import query_database

async def main():
    rows = await query_database("SELECT id, name FROM users LIMIT 10;")
    for row in rows:
        print(row)

if __name__ == "__main__":
    asyncio.run(main())
```
### Combine file + DB data
```python
import asyncio
from mcp.resources import combine_sources

async def main():
    merged = await combine_sources("/data/users.json", "SELECT id, score FROM scores;")
    for rec in merged:
        print(rec)

if __name__ == "__main__":
    asyncio.run(main())
```
---
## Best‑Practice Checklist (generated by the skill)
- ✅ **Load configuration early** – use `.env` and `dotenv` to keep secrets out of code.
- ✅ **Prefer token or API‑key auth**; fall back to mTLS only when required.
- ✅ **Reuse a single `httpx.AsyncClient`** – avoids connection churn and enables pooling.
- ✅ **Wrap all network calls in `try/except`** and raise domain‑specific `MCPError` subclasses.
- ✅ **Validate responses** – ensure expected JSON shape before processing.
- ✅ **Implement timeouts** (`MCP_TIMEOUT`) to prevent hanging workflows.
- ✅ **Retry transient errors** – optionally use `httpx.Retry` or `tenacity`.
- ✅ **Log request/response metadata** (method, URL, status) at DEBUG level for audit trails.
- ✅ **Combine data lazily** – fetch and process streams when dealing with large files.
- ✅ **Close the client** (`await client.close()`) when the application shuts down.

---
## Dependencies (`pyproject.toml` excerpt)
```toml
[tool.poetry.dependencies]
python = ">=3.11"
httpx = "^0.27"
python-dotenv = "^1.0"
# optional for retries
tenacity = {version = "^9.0", optional = true}

[tool.poetry.extras]
retries = ["tenacity"]
```
---
## How to get started
1. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e .
   ```
2. **Copy `.env.example` to `.env`** and fill in your MCP host, auth method, and credentials.
3. Run any example script in `examples/` to verify the connection.
4. Integrate the `MCPClient` and resource helpers into your own workflows (data pipelines, LLM prompts, etc.).

---
## References
- MCP Protocol Specification (internal) – see the organization's documentation portal.
- `httpx` async client docs.
- `python-dotenv` usage guide.
- Security best practices for handling service tokens.

---
**Note:** The skill is intentionally generic; you can extend `resources.py` with additional MCP services (e.g., vector stores, message queues) by following the same pattern.

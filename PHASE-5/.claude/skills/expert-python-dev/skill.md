# Expert Python Development Skill

This skill provides a comprehensive scaffold and best‑practice guide for **professional Python 3.11+** projects across any domain. It covers everything from environment setup to production‑ready code structure, helping developers write clean, maintainable, and high‑performance Python applications.

---
## What the skill generates
When invoked, the skill creates a **template project** (`expert-python-app/`) with the following structure:
```
expert-python-app/
├─ .venv/                 # optional virtual environment (if using venv)
├─ pyproject.toml         # Poetry (or fallback to pip) configuration
├─ requirements.txt        # pinning for pip users
├─ src/
│  ├─ __init__.py        # makes src a package
│  ├─ main.py            # entry point, async example
│  ├─ config.py           # settings using pydantic BaseSettings + python-dotenv
│  ├─ logging.py          # structured logging setup (loguru or standard)
│  ├─ utils/
│  │  ├─ io.py           # file I/O helpers (streaming, pathlib, json, csv)
│  │  └─ data.py         # common data‑processing patterns (chunks, generators)
│  ├─ services/
│  │  └─ example_service.py  # async service example with type hints
│  └─ models/
│     ├─ __init__.py
│     ├─ schemas.py       # pydantic v2 models for validation & serialization
│     └─ entities.py     # domain entities using dataclasses (frozen)
├─ tests/
│  ├─ __init__.py
│  ├─ conftest.py        # pytest fixtures (tmp_path, async client)
│  ├─ unit/
│  │  └─ test_example.py # example unit test with pytest‑asyncio
│  └─ integration/
│     └─ test_integration.py # integration test example
├─ docs/
│  └─ index.md           # Sphinx or MkDocs starter
├─ .env.example          # example environment variables
├─ .gitignore
├─ .flake8                # optional flake8 config
├─ pyproject.toml         # Black, Ruff, MyPy configuration
├─ mypy.ini               # MyPy strict settings
└─ README.md
```

---
## Key Features Included
### 1. Modern Python 3.11+ language features
- **Pattern matching** (`match` statements) – example utilities.
- **Exception groups** and `except*` – robust multi‑exception handling.
- **Typed dicts, `typing.ParamSpec`, `typing.Concatenate`** – advanced typing for higher‑order functions.
- **Self‑type (`typing.Self`)** – for fluent APIs.

### 2. Virtual environment & package management
- **Poetry** (`pyproject.toml`) as the primary manager, with a fallback `requirements.txt` for pip users.
- Script `scripts/setup_venv.sh` (or Windows batch) to create a `.venv` and install dependencies.

### 3. Async/await patterns
- Boilerplate `async_main()` in `src/main.py` using `anyio`/`asyncio`.
- Example async service (`example_service.py`) that demonstrates concurrent I/O with `asyncio.gather`.

### 4. Type hints & static analysis
- **MyPy** strict mode (`strict = true`) with `pyproject.toml` section.
- **Ruff** for linting, configured for `F401`, `E722`, etc.
- **Black** code formatter.
- `py.typed` marker to indicate type‑checked package.

### 5. Error handling & logging
- Centralised `log_config()` in `src/logging.py` using **loguru** (or `structlog`).
- Custom `AppError` hierarchy with automatic traceback capture.
- Context‑aware exception handling with `except*` for asyncio tasks.

### 6. Testing with pytest
- `pytest-asyncio` for async tests.
- `pytest-cov` for coverage reporting.
- Example fixtures: `event_loop`, `tmp_path`, `client`.
- Test parametrization and property‑based testing with **hypothesis** (optional).

### 7. Code formatting & linting
- **Black** (line length 88) configured in `pyproject.toml`.
- **Ruff** for fast linting, includes `flake8`‑compatible rules.
- Pre‑commit hook template (`.pre-commit-config.yaml`).

### 8. File I/O & data processing utilities
- `src/utils/io.py` demonstrates safe file handling with `pathlib`, context managers, and streaming large files.
- CSV/JSON helpers with lazy parsing (`json.load` with `object_hook`, `csv.DictReader`).
- Example `chunks(iterable, size)` generator for batch processing.

### 9. Standard library patterns
- **Context manager** (`contextlib.ExitStack`), **cached_property**, **lru_cache** usage.
- **ThreadPoolExecutor** for CPU‑bound offloading, `asyncio.to_thread` for I/O.
- **Signal handling** (`signal.signal`) for graceful shutdown.

### 10. Dependency management & inversion of control
- Simple **service container** (`src/container.py`) using `dependency-injector` pattern (optional library).
- Example of injecting a repository into a service.

### 11. Debugging techniques
- `pdb.set_trace()` usage notes.
- `rich` console printing for pretty debugging.
- `py-spy` profiling guidance in `README.md`.

### 12. Performance optimization
- Profile with `cProfile` and `pyinstrument`.
- Use of **memoization** (`functools.cache`) and **type‑specific optimizations** (e.g., `array.array` for numeric data).
- Guidance for writing **C extensions** via `cffi`/`cython` (optional section).

### 13. Professional project structure
- Separation of concerns (`src/`, `tests/`, `docs/`).
- Configuration via **pydantic Settings** (`src/config.py`).
- Entry‑point script `run.sh` / `run.bat` for development and production.

---
## Usage Instructions
1. **Create the scaffold** – run the skill with a brief description (e.g., “Create a starter expert Python project”). The skill will generate the `expert-python-app/` directory as shown above.
2. **Initialize environment**:
   ```bash
   cd expert-python-app
   # Using Poetry
   poetry install
   # Or using pip + venv
   python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
   ```
3. **Run the app**:
   ```bash
   python -m src.main   # or `poetry run python -m src.main`
   ```
4. **Run tests**:
   ```bash
   pytest --cov=src
   ```
5. **Lint/format**:
   ```bash
   black src && ruff check src
   ```
6. **Build documentation** (MkDocs example):
   ```bash
   mkdocs serve
   ```

---
## Extensibility
- Add more domain‑specific modules under `src/services/`.
- Replace the async example with a web framework (FastAPI, Flask, aiohttp) by swapping `src/main.py`.
- Extend the CI configuration (GitHub Actions template provided in `.github/workflows/ci.yml`).

---
## References
- Python 3.11 release notes
- Poetry documentation
- MyPy strict mode guide
- AsyncIO best practices
- Black & Ruff style guides

---
**Note:** Adjust the `pyproject.toml` dependencies to suit your project (e.g., replace `loguru` with `structlog`). The skill is intentionally generic to serve any Python domain while embedding the listed best practices.

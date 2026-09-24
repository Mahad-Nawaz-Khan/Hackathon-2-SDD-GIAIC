# Phase 1: In-Memory Python Console Todo App

A modular, clean, and extensible Python command-line Todo application adhering to **Spec-Driven Development (SDD)**.

## Features

- **Core CRUD Operations**:
  - Add task with title, optional description, priority (`low`, `medium`, `high`), and optional due date.
  - View task list with formatted status indicators (`[x]` for complete, `[ ]` for pending).
  - Update task details (title, description, priority, due date).
  - Delete task by ID.
  - Mark task as complete or toggle status.
- **Persistence Layer**:
  - In-memory storage by default with optional file persistence (`tasks.json`).
  - Automatic serialization and deserialization.
- **Robust Validation**:
  - Non-empty title validation.
  - Priority level validation (`low`, `medium`, `high`).
  - Safe ID lookup with clean error handling.
- **Interactive CLI & Direct Command Modes**:
  - Command-line arguments for single-shot execution.
  - Interactive loop mode for continuous user interaction.

## Tech Stack

- **Language**: Python 3.10+
- **Architecture**: Modular separation of concerns:
  - `core/tasks.py`: Data models and in-memory business logic (`Task`, `TaskManager`).
  - `core/persistence.py`: JSON file serialization and loading.
  - `cli/commands.py`: Command parsers and formatters.
  - `main.py`: Application entrypoint.
- **Testing**: pytest

## Directory Structure

```
PHASE-1/
├── .claude/                   # Claude agent configuration
├── .specify/                  # Spec-Kit configuration & templates
├── specs/                     # Spec-Driven Development documentation
│   └── 001-console-todo-app/
│       ├── spec.md            # Feature requirements & user stories
│       ├── plan.md            # Technical architecture & design
│       ├── tasks.md           # Implementation task breakdown
│       ├── quickstart.md      # Usage guide
│       └── data-model.md      # Domain models
├── tests/
│   └── test_todo_app.py       # Comprehensive unit test suite (16 tests)
├── todo_app/
│   ├── __init__.py
│   ├── main.py                # Main entrypoint
│   ├── cli/
│   │   ├── __init__.py
│   │   └── commands.py
│   └── core/
│       ├── __init__.py
│       ├── tasks.py           # Task model and TaskManager
│       └── persistence.py     # JSON file persistence
├── requirements.txt           # Test dependencies
└── README.md                  # This file
```

## Setup & Usage

### 1. Requirements

Install test dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running the Application

Interactive mode:
```bash
python -m todo_app.main
# or directly:
python todo_app/main.py
```

Command-line usage:
```bash
# Add a task
python todo_app/main.py add "Buy groceries" --description "Milk, Bread, Eggs" --priority high

# List tasks
python todo_app/main.py list

# Mark task 1 as complete
python todo_app/main.py complete 1

# Update task 1
python todo_app/main.py update 1 --title "Buy groceries and drinks"

# Delete task 1
python todo_app/main.py delete 1
```

### 3. Running Unit Tests

Execute the unit test suite:
```bash
pytest tests/ -v
```

All 16 unit tests cover Task serialization, TaskManager CRUD operations, persistence loading/saving, and error edge cases.

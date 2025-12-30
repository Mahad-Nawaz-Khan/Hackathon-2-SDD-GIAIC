# Console TODO Application

A simple command-line based task management application written in Python.

## Features

- Add tasks with titles and optional descriptions
- View all tasks with completion status
- Update existing tasks
- Delete tasks
- Mark tasks as complete/incomplete
- Persistent storage using JSON files
- Menu-driven interface

## Requirements

- Python 3.x
- Standard library only (no external dependencies)

## Usage

1. Run the application:
   ```bash
   python todo_app/main.py
   ```

2. Use the menu to manage your tasks:
   - Option 1: Add a new task
   - Option 2: View all tasks
   - Option 3: Update an existing task
   - Option 4: Delete a task
   - Option 5: Toggle task completion status
   - Option 6: Exit the application

## Data Persistence

Tasks are automatically saved to `tasks.json` in the current directory. The file is loaded when the application starts.

## Project Structure

```
todo_app/
├── main.py                 # Main application entry point
├── config.py               # Configuration constants
├── models/
│   └── task.py            # Task data structure definition
├── core/
│   ├── tasks.py           # Core task operations
│   ├── persistence.py     # JSON load/save operations
│   └── validation.py      # Input validation functions
└── ui/
    └── menu.py            # Menu display and user interaction
```

## License

This project is created for educational purposes as part of a hackathon project.
# Quickstart Guide: Console TODO Application

**Feature**: 001-console-todo-app
**Date**: 2025-12-30

## Overview
A console-based TODO application that allows users to manage tasks through a menu-driven interface. The application runs continuously until the user explicitly exits.

## Prerequisites
- Python 3.x installed on your system
- Console/terminal access

## Running the Application

1. **Ensure Python is available**:
   ```bash
   python --version
   # or
   python3 --version
   ```

2. **Run the application**:
   ```bash
   python todo_app/main.py
   # or
   python3 todo_app/main.py
   ```

## Using the Application

### Main Menu Options
Once the application starts, you'll see a menu with the following options:
1. Add Task
2. View Tasks
3. Update Task
4. Delete Task
5. Toggle Task Completion
6. Exit

### Adding a Task
1. Select "Add Task" from the menu
2. Enter a non-empty title when prompted
3. Optionally enter a description
4. The task will be added with a unique ID and unchecked status

### Viewing Tasks
1. Select "View Tasks" from the menu
2. All tasks will be displayed in ascending order by ID
3. Each task shows its ID, title, and completion status ([ ] not completed, [✓] completed)

### Updating a Task
1. Select "Update Task" from the menu
2. Enter the ID of the task you want to update
3. Enter the new title (or press Enter to keep current)
4. Enter the new description (or press Enter to keep current)

### Deleting a Task
1. Select "Delete Task" from the menu
2. Enter the ID of the task you want to delete
3. The task will be permanently removed

### Toggling Task Completion
1. Select "Toggle Task Completion" from the menu
2. Enter the ID of the task you want to toggle
3. The completion status will be inverted

## Error Handling
- Invalid menu selections will show an error message and re-prompt
- Invalid task IDs will show an error message and re-prompt
- Empty task titles will be rejected
- The application will not crash on invalid input

## Persistence
- Tasks are automatically saved to a JSON file after each operation
- Tasks are loaded from the JSON file when the application starts
- The default filename is `tasks.json`

## Exiting the Application
- Select "Exit" from the main menu to cleanly exit the application
- All tasks will be saved before exiting
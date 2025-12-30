"""
Configuration module for the Console TODO Application
Contains constants and configuration settings.
"""


# Application settings
APP_NAME = "Console TODO Application"
APP_VERSION = "1.0.0"

# File settings
TASKS_FILE = "tasks.json"

# Menu options
MENU_OPTIONS = {
    1: "Add Task",
    2: "View Tasks",
    3: "Update Task",
    4: "Delete Task",
    5: "Toggle Task Completion",
    6: "Exit"
}

# Default values
DEFAULT_TASK_DESCRIPTION = ""
DEFAULT_TASK_COMPLETED = False

# Error messages
ERROR_INVALID_INPUT = "Invalid input. Please try again."
ERROR_TASK_NOT_FOUND = "Task not found."
ERROR_EMPTY_TITLE = "Task title cannot be empty."
ERROR_INVALID_MENU_CHOICE = "Invalid menu choice. Please select a number between 1 and 6."
ERROR_NO_TASKS = "No tasks found."

# Success messages
SUCCESS_TASK_ADDED = "Task added successfully."
SUCCESS_TASK_UPDATED = "Task updated successfully."
SUCCESS_TASK_DELETED = "Task deleted successfully."
SUCCESS_TASK_TOGGLED = "Task completion status updated."
SUCCESS_EXIT = "Thank you for using the Console TODO Application!"
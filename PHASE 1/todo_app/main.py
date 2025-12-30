#!/usr/bin/env python3
"""
Console TODO Application
A simple command-line based task management application.
"""
import sys
from core.tasks import add_task, view_tasks, update_task, delete_task, find_task_by_id, toggle_task_completion
from core.persistence import load_tasks, save_tasks
from ui.menu import (
    display_menu, get_user_choice, get_task_input, get_task_id,
    display_message, display_error, display_success, get_confirmation
)
from config import TASKS_FILE, SUCCESS_EXIT
from core.validation import validate_task_title, is_valid_menu_choice


def main():
    """Main entry point for the TODO application."""
    # Load existing tasks from file on startup
    tasks = load_tasks(TASKS_FILE)

    print("Welcome to the Console TODO Application!")
    if tasks:
        print(f"Loaded {len(tasks)} tasks from {TASKS_FILE}")

    while True:
        display_menu()
        choice = get_user_choice()

        if not is_valid_menu_choice(choice):
            display_error("Invalid menu choice. Please select a number between 1 and 6.")
            continue

        if choice == 1:
            # Add Task
            title, description = get_task_input()

            if not validate_task_title(title):
                display_error("Task title cannot be empty.")
                continue

            try:
                add_task(tasks, title, description)
                save_tasks(tasks, TASKS_FILE)  # Save after each operation
                display_success("Task added successfully.")
            except ValueError as e:
                display_error(str(e))

        elif choice == 2:
            # View Tasks
            if not tasks:
                display_message("No tasks found.")
            else:
                sorted_tasks = view_tasks(tasks)
                print("\nYour Tasks:")
                print("-" * 50)
                for task in sorted_tasks:
                    status = "✓" if task.completed else " "
                    print(f"[{status}] ID: {task.id} | {task.title}")
                    if task.description:
                        print(f"    Description: {task.description}")
                print("-" * 50)

        elif choice == 3:
            # Update Task
            task_id = get_task_id()
            if task_id <= 0:
                display_error("Invalid task ID. Please enter a valid number.")
                continue

            # Check if task exists
            task = find_task_by_id(tasks, task_id)
            if not task:
                display_error("Task not found.")
                continue

            # Get new values (or keep current if empty input)
            print(f"Current title: {task.title}")
            new_title = input("Enter new title (or press Enter to keep current): ").strip()
            if new_title == "":
                new_title = task.title

            print(f"Current description: {task.description}")
            new_description = input("Enter new description (or press Enter to keep current): ").strip()
            if new_description == "":
                new_description = task.description

            if not validate_task_title(new_title):
                display_error("Task title cannot be empty.")
                continue

            if update_task(tasks, task_id, new_title, new_description):
                save_tasks(tasks, TASKS_FILE)  # Save after each operation
                display_success("Task updated successfully.")
            else:
                display_error("Failed to update task.")

        elif choice == 4:
            # Delete Task
            task_id = get_task_id()
            if task_id <= 0:
                display_error("Invalid task ID. Please enter a valid number.")
                continue

            # Check if task exists
            task = find_task_by_id(tasks, task_id)
            if not task:
                display_error("Task not found.")
                continue

            if get_confirmation(f"Are you sure you want to delete task '{task.title}'?"):
                if delete_task(tasks, task_id):
                    save_tasks(tasks, TASKS_FILE)  # Save after each operation
                    display_success("Task deleted successfully.")
                else:
                    display_error("Failed to delete task.")
            else:
                display_message("Task deletion cancelled.")

        elif choice == 5:
            # Toggle Task Completion
            task_id = get_task_id()
            if task_id <= 0:
                display_error("Invalid task ID. Please enter a valid number.")
                continue

            # Check if task exists
            task = find_task_by_id(tasks, task_id)
            if not task:
                display_error("Task not found.")
                continue

            if toggle_task_completion(tasks, task_id):
                save_tasks(tasks, TASKS_FILE)  # Save after each operation
                status = "completed" if task.completed else "incomplete"
                display_success(f"Task marked as {status}.")
            else:
                display_error("Failed to update task completion status.")

        elif choice == 6:
            # Exit
            save_tasks(tasks, TASKS_FILE)  # Save before exiting
            print(f"\n{SUCCESS_EXIT}")
            break
        else:
            # For now, other options will show a message that they're not implemented yet
            # We'll implement them in the next phases
            display_message("This feature is not implemented yet. Please select 1, 2, 3, 4, 5, or 6.")


if __name__ == "__main__":
    main()
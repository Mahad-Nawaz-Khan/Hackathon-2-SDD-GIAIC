"""
Core task management functions for the Console TODO Application
Implements add, view, update, delete, and toggle completion operations.
"""
from typing import List, Optional
from models.task import Task


def add_task(tasks: List[Task], title: str, description: str = "") -> Task:
    """
    Add a new task to the task list.

    Args:
        tasks (List[Task]): The current list of tasks
        title (str): The title of the new task (must be non-empty)
        description (str): The optional description of the new task

    Returns:
        Task: The newly created task

    Raises:
        ValueError: If title is empty
    """
    if not title or not title.strip():
        raise ValueError("Task title cannot be empty")

    # Find the next available ID
    if tasks:
        next_id = max(task.id for task in tasks) + 1
    else:
        next_id = 1

    new_task = Task(next_id, title.strip(), description.strip())
    tasks.append(new_task)
    return new_task


def view_tasks(tasks: List[Task]) -> List[Task]:
    """
    Get all tasks sorted by ID.

    Args:
        tasks (List[Task]): The list of tasks to view

    Returns:
        List[Task]: The sorted list of tasks
    """
    return sorted(tasks, key=lambda task: task.id)


def update_task(tasks: List[Task], task_id: int, title: Optional[str] = None, description: Optional[str] = None) -> bool:
    """
    Update an existing task's title and/or description.

    Args:
        tasks (List[Task]): The current list of tasks
        task_id (int): The ID of the task to update
        title (Optional[str]): The new title (if provided)
        description (Optional[str]): The new description (if provided)

    Returns:
        bool: True if the task was found and updated, False otherwise
    """
    for task in tasks:
        if task.id == task_id:
            if title is not None:
                task.title = title.strip()
            if description is not None:
                task.description = description.strip()
            return True
    return False  # Task not found


def delete_task(tasks: List[Task], task_id: int) -> bool:
    """
    Delete a task by its ID.

    Args:
        tasks (List[Task]): The current list of tasks
        task_id (int): The ID of the task to delete

    Returns:
        bool: True if the task was found and deleted, False otherwise
    """
    for i, task in enumerate(tasks):
        if task.id == task_id:
            del tasks[i]
            return True
    return False  # Task not found


def toggle_task_completion(tasks: List[Task], task_id: int) -> bool:
    """
    Toggle the completion status of a task.

    Args:
        tasks (List[Task]): The current list of tasks
        task_id (int): The ID of the task to toggle

    Returns:
        bool: True if the task was found and toggled, False otherwise
    """
    for task in tasks:
        if task.id == task_id:
            task.completed = not task.completed
            return True
    return False  # Task not found


def find_task_by_id(tasks: List[Task], task_id: int) -> Optional[Task]:
    """
    Find a task by its ID.

    Args:
        tasks (List[Task]): The list of tasks to search
        task_id (int): The ID of the task to find

    Returns:
        Optional[Task]: The task if found, None otherwise
    """
    for task in tasks:
        if task.id == task_id:
            return task
    return None


def get_next_task_id(tasks: List[Task]) -> int:
    """
    Get the next available task ID.

    Args:
        tasks (List[Task]): The current list of tasks

    Returns:
        int: The next available task ID
    """
    if tasks:
        return max(task.id for task in tasks) + 1
    return 1
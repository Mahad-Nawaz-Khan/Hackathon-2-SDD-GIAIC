"""
Persistence module for the Console TODO Application
Handles loading and saving tasks to/from JSON files.
"""
import json
import os
from typing import List
from models.task import Task


def load_tasks(filename: str = "tasks.json") -> List[Task]:
    """
    Load tasks from a JSON file.

    Args:
        filename (str): Path to the JSON file to load tasks from

    Returns:
        List[Task]: List of Task objects loaded from the file
    """
    if not os.path.exists(filename):
        return []

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if not isinstance(data, list):
                return []
            return [Task.from_dict(task_data) for task_data in data]
    except json.JSONDecodeError:
        # Return empty list if file is corrupted
        print(f"Warning: Could not decode JSON from {filename}. Starting with empty task list.")
        return []
    except KeyError:
        # Return empty list if required keys are missing
        print(f"Warning: Invalid task data format in {filename}. Starting with empty task list.")
        return []
    except TypeError:
        # Return empty list if data types are incorrect
        print(f"Warning: Invalid data types in {filename}. Starting with empty task list.")
        return []
    except PermissionError:
        # Return empty list if file permissions prevent reading
        print(f"Warning: Permission denied reading {filename}. Starting with empty task list.")
        return []
    except Exception as e:
        # Return empty list for any other error
        print(f"Warning: Error reading {filename}: {str(e)}. Starting with empty task list.")
        return []


def save_tasks_safe(tasks: List[Task], filename: str = "tasks.json") -> bool:
    """
    Save tasks to a JSON file with error handling.

    Args:
        tasks (List[Task]): List of Task objects to save
        filename (str): Path to the JSON file to save tasks to

    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        # Convert tasks to dictionary format
        tasks_data = [task.to_dict() for task in tasks]

        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(tasks_data, file, indent=2, ensure_ascii=False)
        return True
    except PermissionError:
        print(f"Error: Permission denied writing to {filename}")
        return False
    except OSError as e:
        print(f"Error: Could not write to {filename}: {str(e)}")
        return False
    except Exception as e:
        print(f"Error: Failed to save tasks to {filename}: {str(e)}")
        return False


def save_tasks(tasks: List[Task], filename: str = "tasks.json") -> None:
    """
    Save tasks to a JSON file.

    Args:
        tasks (List[Task]): List of Task objects to save
        filename (str): Path to the JSON file to save tasks to
    """
    # Use the safe save function but maintain the original interface
    save_tasks_safe(tasks, filename)
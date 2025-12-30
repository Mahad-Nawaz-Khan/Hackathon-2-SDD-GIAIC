"""
Validation module for the Console TODO Application
Provides input validation functions for various user inputs.
"""


def validate_task_title(title: str) -> bool:
    """
    Validate that a task title is non-empty.

    Args:
        title (str): The task title to validate

    Returns:
        bool: True if the title is valid, False otherwise
    """
    return bool(title and title.strip())


def validate_task_id(task_id: int, tasks: list) -> bool:
    """
    Validate that a task ID exists in the task list.

    Args:
        task_id (int): The task ID to validate
        tasks (list): The list of existing tasks

    Returns:
        bool: True if the task ID is valid (exists in the list), False otherwise
    """
    return any(task.id == task_id for task in tasks)


def validate_numeric_input(input_str: str) -> bool:
    """
    Validate that a string can be converted to a number.

    Args:
        input_str (str): The string to validate

    Returns:
        bool: True if the string can be converted to an integer, False otherwise
    """
    try:
        int(input_str)
        return True
    except ValueError:
        return False


def is_valid_menu_choice(choice: int) -> bool:
    """
    Validate that a menu choice is within the valid range (1-6).

    Args:
        choice (int): The menu choice to validate

    Returns:
        bool: True if the choice is valid (1-6), False otherwise
    """
    return 1 <= choice <= 6


def validate_task_id_exists(task_id: int, tasks: list) -> bool:
    """
    Validate that a task ID exists in the task list.

    Args:
        task_id (int): The task ID to validate
        tasks (list): The list of existing tasks

    Returns:
        bool: True if the task ID exists in the list, False otherwise
    """
    return any(task.id == task_id for task in tasks)


def validate_positive_number(value: int) -> bool:
    """
    Validate that a number is positive.

    Args:
        value (int): The number to validate

    Returns:
        bool: True if the number is positive, False otherwise
    """
    return isinstance(value, int) and value > 0
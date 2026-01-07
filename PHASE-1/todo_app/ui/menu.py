"""
Menu module for the Console TODO Application
Handles the main menu interface and user input.
"""


def display_menu():
    """
    Display the main menu options to the user.
    """
    print("\n" + "="*40)
    print("         CONSOLE TODO APPLICATION")
    print("="*40)
    print("1. Add Task")
    print("2. View Tasks")
    print("3. Update Task")
    print("4. Delete Task")
    print("5. Toggle Task Completion")
    print("6. Exit")
    print("="*40)


def get_user_choice():
    """
    Prompt the user for their menu choice and validate the input.

    Returns:
        int: The user's menu choice (1-6), or -1 if invalid input
    """
    try:
        choice = input("Enter your choice (1-6): ").strip()
        return int(choice)
    except ValueError:
        return -1  # Invalid input


def get_task_input():
    """
    Prompt the user for task details.

    Returns:
        tuple: (title: str, description: str) entered by the user
    """
    title = input("Enter task title: ").strip()
    description = input("Enter task description (optional): ").strip()
    return title, description


def get_task_id():
    """
    Prompt the user for a task ID.

    Returns:
        int: The task ID entered by the user, or -1 if invalid input
    """
    try:
        task_id = input("Enter task ID: ").strip()
        return int(task_id)
    except ValueError:
        return -1  # Invalid input


def get_numeric_input(prompt: str) -> int:
    """
    Prompt the user for a numeric input.

    Args:
        prompt (str): The prompt to display to the user

    Returns:
        int: The numeric value entered by the user, or -1 if invalid input
    """
    try:
        value = input(prompt).strip()
        return int(value)
    except ValueError:
        return -1  # Invalid input


def get_confirmation(prompt: str) -> bool:
    """
    Ask the user for confirmation.

    Args:
        prompt (str): The confirmation message to display

    Returns:
        bool: True if user confirms, False otherwise
    """
    response = input(f"{prompt} (y/N): ").strip().lower()
    return response in ['y', 'yes']


def display_message(message: str):
    """
    Display a message to the user.

    Args:
        message (str): The message to display
    """
    print(f"\n{message}")


def display_success(message: str):
    """
    Display a success message to the user.

    Args:
        message (str): The success message to display
    """
    print(f"\n✓ {message}")


def display_error(message: str):
    """
    Display an error message to the user.

    Args:
        message (str): The error message to display
    """
    print(f"\n✗ {message}")
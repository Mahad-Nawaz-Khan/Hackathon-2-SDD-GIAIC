"""
Task model for the Console TODO Application
Defines the structure and basic operations for a task.
"""


class Task:
    """
    Represents a single task in the TODO application.

    Attributes:
        id (int): Unique identifier for the task
        title (str): The task title (non-empty)
        description (str): Optional detailed description of the task
        completed (bool): Completion status of the task
    """

    def __init__(self, task_id, title, description="", completed=False):
        """
        Initialize a new Task instance.

        Args:
            task_id (int): Unique identifier for the task
            title (str): The task title (non-empty)
            description (str): Optional detailed description of the task
            completed (bool): Completion status of the task (default: False)
        """
        if not isinstance(task_id, int):
            raise TypeError("Task ID must be an integer")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Task title must be a non-empty string")
        if not isinstance(completed, bool):
            raise TypeError("Task completion status must be a boolean")

        self.id = task_id
        self.title = title.strip()
        self.description = description.strip() if description else ""
        self.completed = completed

    def to_dict(self):
        """
        Convert the Task instance to a dictionary representation.

        Returns:
            dict: Dictionary representation of the task
        """
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "completed": self.completed
        }

    @classmethod
    def from_dict(cls, data):
        """
        Create a Task instance from a dictionary representation.

        Args:
            data (dict): Dictionary containing task data

        Returns:
            Task: Task instance created from the dictionary
        """
        return cls(
            task_id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            completed=data.get("completed", False)
        )

    def __repr__(self):
        """
        String representation of the Task instance.

        Returns:
            str: String representation of the task
        """
        status = "✓" if self.completed else " "
        return f"[{status}] Task {self.id}: {self.title}"

    def __eq__(self, other):
        """
        Check equality with another Task instance.

        Args:
            other (Task): Another Task instance to compare with

        Returns:
            bool: True if both tasks have the same attributes, False otherwise
        """
        if not isinstance(other, Task):
            return False
        return (self.id == other.id and
                self.title == other.title and
                self.description == other.description and
                self.completed == other.completed)
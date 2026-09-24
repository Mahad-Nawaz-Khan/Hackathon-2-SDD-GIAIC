"""
Unit Test Suite for Phase I Console TODO Application
Tests all 5 Basic Level Features:
1. Add Task
2. View Tasks
3. Update Task
4. Delete Task
5. Toggle Completion Status
Plus persistence and input validation.
"""

import os
import sys
import tempfile
import pytest

# Ensure todo_app is on sys.path
TODO_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "todo_app"))
if TODO_APP_DIR not in sys.path:
    sys.path.insert(0, TODO_APP_DIR)

from models.task import Task
from core.tasks import (
    add_task,
    view_tasks,
    update_task,
    delete_task,
    find_task_by_id,
    toggle_task_completion,
)
from core.persistence import load_tasks, save_tasks
from core.validation import validate_task_title, is_valid_menu_choice, validate_task_id


class TestTaskModel:
    """Tests for the Task data model."""

    def test_task_creation(self):
        task = Task(task_id=1, title="Buy groceries", description="Milk, eggs")
        assert task.id == 1
        assert task.title == "Buy groceries"
        assert task.description == "Milk, eggs"
        assert task.completed is False

    def test_task_to_dict_and_from_dict(self):
        task = Task(task_id=2, title="Learn Python", description="Read docs", completed=True)
        data = task.to_dict()
        assert data["id"] == 2
        assert data["title"] == "Learn Python"
        assert data["completed"] is True

        reconstructed = Task.from_dict(data)
        assert reconstructed.id == 2
        assert reconstructed.title == "Learn Python"
        assert reconstructed.completed is True


class TestTaskOperations:
    """Tests for core task operations (CRUD)."""

    def test_add_task_success(self):
        tasks = []
        task = add_task(tasks, "Task 1", "First description")
        assert len(tasks) == 1
        assert task.id == 1
        assert task.title == "Task 1"
        assert task.description == "First description"
        assert task.completed is False

    def test_add_task_empty_title_raises_error(self):
        tasks = []
        with pytest.raises(ValueError, match="cannot be empty"):
            add_task(tasks, "")
        with pytest.raises(ValueError, match="cannot be empty"):
            add_task(tasks, "   ")

    def test_add_multiple_tasks_increments_id(self):
        tasks = []
        t1 = add_task(tasks, "First")
        t2 = add_task(tasks, "Second")
        t3 = add_task(tasks, "Third")
        assert t1.id == 1
        assert t2.id == 2
        assert t3.id == 3
        assert len(tasks) == 3

    def test_view_tasks_sorted_by_id(self):
        t1 = Task(1, "A")
        t2 = Task(2, "B")
        t3 = Task(3, "C")
        tasks = [t3, t1, t2]
        sorted_tasks = view_tasks(tasks)
        assert [t.id for t in sorted_tasks] == [1, 2, 3]

    def test_update_task_title_and_description(self):
        tasks = [Task(1, "Old Title", "Old Desc")]
        success = update_task(tasks, 1, title="New Title", description="New Desc")
        assert success is True
        assert tasks[0].title == "New Title"
        assert tasks[0].description == "New Desc"

    def test_update_nonexistent_task_returns_false(self):
        tasks = [Task(1, "A")]
        assert update_task(tasks, 999, title="New") is False

    def test_delete_task_success(self):
        tasks = [Task(1, "A"), Task(2, "B")]
        success = delete_task(tasks, 1)
        assert success is True
        assert len(tasks) == 1
        assert tasks[0].id == 2

    def test_delete_nonexistent_task_returns_false(self):
        tasks = [Task(1, "A")]
        assert delete_task(tasks, 999) is False
        assert len(tasks) == 1

    def test_toggle_task_completion(self):
        tasks = [Task(1, "Task 1", completed=False)]
        # Toggle to completed
        success = toggle_task_completion(tasks, 1)
        assert success is True
        assert tasks[0].completed is True

        # Toggle back to incomplete
        success = toggle_task_completion(tasks, 1)
        assert success is True
        assert tasks[0].completed is False

    def test_find_task_by_id(self):
        tasks = [Task(1, "A"), Task(2, "B")]
        found = find_task_by_id(tasks, 2)
        assert found is not None
        assert found.title == "B"

        not_found = find_task_by_id(tasks, 999)
        assert not_found is None


class TestPersistence:
    """Tests for saving and loading tasks from JSON."""

    def test_save_and_load_tasks(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            original_tasks = [
                Task(1, "Task A", "Desc A", completed=False),
                Task(2, "Task B", "Desc B", completed=True),
            ]
            save_tasks(original_tasks, tmp_path)

            loaded_tasks = load_tasks(tmp_path)
            assert len(loaded_tasks) == 2
            assert loaded_tasks[0].id == 1
            assert loaded_tasks[0].title == "Task A"
            assert loaded_tasks[0].completed is False
            assert loaded_tasks[1].id == 2
            assert loaded_tasks[1].title == "Task B"
            assert loaded_tasks[1].completed is True
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_load_nonexistent_file_returns_empty(self):
        loaded = load_tasks("non_existent_file_xyz123.json")
        assert loaded == []


class TestValidation:
    """Tests for user input validation."""

    def test_validate_task_title(self):
        assert validate_task_title("Buy milk") is True
        assert validate_task_title("") is False
        assert validate_task_title("   ") is False

    def test_is_valid_menu_choice(self):
        assert is_valid_menu_choice(1) is True
        assert is_valid_menu_choice(6) is True
        assert is_valid_menu_choice(0) is False
        assert is_valid_menu_choice(7) is False

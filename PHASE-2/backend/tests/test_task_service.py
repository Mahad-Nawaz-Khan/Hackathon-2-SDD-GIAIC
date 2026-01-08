import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from src.models.task import Task
from src.models.tag import Tag
from src.models.user import User
from src.schemas.task import PriorityEnum, RecurrenceRuleEnum, TaskCreateRequest, TaskUpdateRequest
from src.services.task_service import TaskService


@pytest.fixture
def db_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(db_engine):
    with Session(db_engine) as session:
        yield session


@pytest.fixture
def user(db_session):
    user = User(clerk_user_id="test_user_123")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


class TestTaskService:
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.task_service = TaskService()
        self.mock_session = Mock(spec=Session)

    def test_create_task_success(self):
        """Test successful task creation."""
        # Arrange
        user_id = 1
        task_data = TaskCreateRequest(
            title="Test Task",
            description="Test Description",
            priority=PriorityEnum.MEDIUM
        )

        # Mock the database session behavior
        mock_task = Task(
            id=1,
            title="Test Task",
            description="Test Description",
            priority="MEDIUM",
            user_id=user_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        self.mock_session.add = Mock()
        self.mock_session.commit = Mock()
        self.mock_session.refresh = Mock(return_value=mock_task)

        # Act
        with patch.object(self.mock_session, 'add'), \
             patch.object(self.mock_session, 'commit'), \
             patch.object(self.mock_session, 'refresh', side_effect=lambda obj: setattr(obj, 'id', 1)):
            result = self.task_service.create_task(task_data, user_id, self.mock_session)

        # Assert
        assert result is not None
        assert result.title == "Test Task"
        assert result.description == "Test Description"

    def test_create_task_missing_title(self):
        """Test task creation with missing title."""
        # Arrange
        user_id = 1
        task_data = TaskCreateRequest(
            title="",  # Empty title
            description="Test Description",
            priority=PriorityEnum.MEDIUM
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Task title is required"):
            self.task_service.create_task(task_data, user_id, self.mock_session)

    def test_get_task_by_id_success(self):
        """Test getting a task by ID successfully."""
        # Arrange
        task_id = 1
        user_id = 1
        mock_task = Task(
            id=task_id,
            title="Test Task",
            user_id=user_id
        )

        # Mock the session execution
        mock_exec_result = Mock()
        mock_exec_result.first.return_value = mock_task
        self.mock_session.exec.return_value = mock_exec_result

        # Act
        result = self.task_service.get_task_by_id(task_id, user_id, self.mock_session)

        # Assert
        assert result is not None
        assert result.id == task_id
        assert result.title == "Test Task"

    def test_get_task_by_id_not_found(self):
        """Test getting a non-existent task."""
        # Arrange
        task_id = 999
        user_id = 1

        # Mock the session execution to return None
        mock_exec_result = Mock()
        mock_exec_result.first.return_value = None
        self.mock_session.exec.return_value = mock_exec_result

        # Act
        result = self.task_service.get_task_by_id(task_id, user_id, self.mock_session)

        # Assert
        assert result is None

    def test_update_task_success(self):
        """Test successful task update."""
        # Arrange
        task_id = 1
        user_id = 1
        existing_task = Task(
            id=task_id,
            title="Old Title",
            description="Old Description",
            user_id=user_id
        )

        update_data = TaskUpdateRequest(
            title="Updated Title",
            description="Updated Description"
        )

        # Mock the session execution for getting the task
        mock_exec_result = Mock()
        mock_exec_result.first.return_value = existing_task
        self.mock_session.exec.return_value = mock_exec_result

        # Mock the session for update operations
        self.mock_session.add = Mock()
        self.mock_session.commit = Mock()
        self.mock_session.refresh = Mock()

        # Act
        result = self.task_service.update_task(task_id, update_data, user_id, self.mock_session)

        # Assert
        assert result is not None
        assert result.title == "Updated Title"
        assert result.description == "Updated Description"

    def test_delete_task_success(self):
        """Test successful task deletion."""
        # Arrange
        task_id = 1
        user_id = 1
        mock_task = Task(
            id=task_id,
            title="Test Task",
            user_id=user_id
        )

        # Mock the session execution for getting the task
        mock_exec_result = Mock()
        mock_exec_result.first.return_value = mock_task
        self.mock_session.exec.return_value = mock_exec_result

        # Mock the session for delete operations
        self.mock_session.delete = Mock()
        self.mock_session.commit = Mock()

        # Act
        result = self.task_service.delete_task(task_id, user_id, self.mock_session)

        # Assert
        assert result is True
        self.mock_session.delete.assert_called_once_with(mock_task)
        self.mock_session.commit.assert_called_once()

    def test_delete_task_not_found(self):
        """Test deleting a non-existent task."""
        # Arrange
        task_id = 999
        user_id = 1

        # Mock the session execution to return None
        mock_exec_result = Mock()
        mock_exec_result.first.return_value = None
        self.mock_session.exec.return_value = mock_exec_result

        # Act
        result = self.task_service.delete_task(task_id, user_id, self.mock_session)

        # Assert
        assert result is False

    def test_toggle_task_completion_returns_completed_original_and_hides_future_instance(self, db_session, user):
        """Test toggle task completion returns completed original and hides future instance."""
        task_service = TaskService()
        now = datetime.utcnow()

        root = task_service.create_task(
            TaskCreateRequest(
                title="Recurring Task",
                priority=PriorityEnum.MEDIUM,
                due_date=now,
                recurrence_rule=RecurrenceRuleEnum.DAILY,
            ),
            user.id,
            db_session,
        )

        updated = task_service.toggle_task_completion(root.id, user.id, db_session)
        assert updated is not None
        assert updated.id == root.id
        assert updated.completed is True

        all_tasks = db_session.exec(select(Task).where(Task.user_id == user.id)).all()
        assert len(all_tasks) == 2
        child = next(t for t in all_tasks if t.id != root.id)
        assert child.parent_task_id == root.id
        assert child.completed is False
        assert child.due_date == root.due_date + timedelta(days=1)

        visible = task_service.get_tasks(user.id, db_session, limit=100, offset=0)
        visible_ids = {t.id for t in visible}
        assert root.id in visible_ids
        assert child.id not in visible_ids

    def test_get_tasks_includes_due_recurring_child_instance(self, db_session, user):
        """Test get tasks includes due recurring child instance."""
        task_service = TaskService()
        now = datetime.utcnow()

        root = task_service.create_task(
            TaskCreateRequest(
                title="Recurring Root",
                priority=PriorityEnum.MEDIUM,
                due_date=now,
                recurrence_rule=RecurrenceRuleEnum.DAILY,
            ),
            user.id,
            db_session,
        )

        child = Task(
            title=root.title,
            description=root.description,
            completed=False,
            priority=root.priority,
            due_date=now - timedelta(days=1),
            recurrence_rule=root.recurrence_rule,
            parent_task_id=root.id,
            user_id=user.id,
        )
        db_session.add(child)
        db_session.commit()
        db_session.refresh(child)

        visible = task_service.get_tasks(user.id, db_session, limit=100, offset=0)
        visible_ids = {t.id for t in visible}
        assert child.id in visible_ids

    def test_delete_task_deletes_all_incomplete_tasks_in_recurring_series(self, db_session, user):
        """Test delete task deletes all incomplete tasks in recurring series."""
        task_service = TaskService()
        now = datetime.utcnow()

        root = task_service.create_task(
            TaskCreateRequest(
                title="Recurring Delete Root",
                priority=PriorityEnum.MEDIUM,
                due_date=now,
                recurrence_rule=RecurrenceRuleEnum.DAILY,
            ),
            user.id,
            db_session,
        )
        task_service.toggle_task_completion(root.id, user.id, db_session)

        all_tasks = db_session.exec(select(Task).where(Task.user_id == user.id)).all()
        child1 = next(t for t in all_tasks if t.parent_task_id == root.id)

        child2 = Task(
            title=root.title,
            description=root.description,
            completed=False,
            priority=root.priority,
            due_date=child1.due_date + timedelta(days=1),
            recurrence_rule=root.recurrence_rule,
            parent_task_id=root.id,
            user_id=user.id,
        )
        db_session.add(child2)
        db_session.commit()
        db_session.refresh(child2)

        result = task_service.delete_task(child1.id, user.id, db_session)
        assert result is True

        remaining = db_session.exec(select(Task).where(Task.user_id == user.id)).all()
        remaining_ids = {t.id for t in remaining}
        assert root.id in remaining_ids
        assert child1.id not in remaining_ids
        assert child2.id not in remaining_ids


if __name__ == "__main__":
    pytest.main()
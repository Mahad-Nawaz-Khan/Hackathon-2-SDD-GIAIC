import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from sqlmodel import create_engine, Session, SQLModel
from sqlmodel.pool import StaticPool
from src.services.recurrence_service import RecurrenceService, recurrence_service
from src.models.task import Task
from src.models.tag import Tag
from src.models.task_tag import TaskTagLink


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return engine


@pytest.fixture
def db_session(in_memory_db):
    """Create a database session for testing."""
    SQLModel.metadata.create_all(in_memory_db)
    with Session(in_memory_db) as session:
        yield session


class TestRecurrenceServiceCalculation:
    """Unit tests for next due date calculation in RecurrenceService."""

    def test_daily_recurrence(self):
        service = RecurrenceService()
        base_date = datetime(2025, 1, 15, 10, 0, 0)
        next_date = service._calculate_next_due_date(base_date, "DAILY")
        assert next_date == datetime(2025, 1, 16, 10, 0, 0)

    def test_weekly_recurrence(self):
        service = RecurrenceService()
        base_date = datetime(2025, 1, 15, 10, 0, 0)
        next_date = service._calculate_next_due_date(base_date, "WEEKLY")
        assert next_date == datetime(2025, 1, 22, 10, 0, 0)

    def test_monthly_recurrence_normal(self):
        service = RecurrenceService()
        base_date = datetime(2025, 3, 15, 0, 0, 0)
        next_date = service._calculate_next_due_date(base_date, "MONTHLY")
        assert next_date == datetime(2025, 4, 15, 0, 0, 0)

    def test_monthly_recurrence_year_rollover(self):
        service = RecurrenceService()
        base_date = datetime(2025, 12, 10, 0, 0, 0)
        next_date = service._calculate_next_due_date(base_date, "MONTHLY")
        assert next_date == datetime(2026, 1, 10, 0, 0, 0)

    def test_monthly_recurrence_month_end_clamping(self):
        service = RecurrenceService()
        # Jan 31 -> Feb has 28 days in 2025
        base_date = datetime(2025, 1, 31, 0, 0, 0)
        next_date = service._calculate_next_due_date(base_date, "MONTHLY")
        assert next_date == datetime(2025, 2, 28, 0, 0, 0)

    def test_monthly_recurrence_leap_year_feb(self):
        service = RecurrenceService()
        # Jan 31 -> Feb has 29 days in 2024 (leap year)
        base_date = datetime(2024, 1, 31, 0, 0, 0)
        next_date = service._calculate_next_due_date(base_date, "MONTHLY")
        assert next_date == datetime(2024, 2, 29, 0, 0, 0)

    def test_none_recurrence_rule_returns_none(self):
        service = RecurrenceService()
        assert service._calculate_next_due_date(datetime.now(), None) is None

    def test_invalid_recurrence_rule_returns_none(self):
        service = RecurrenceService()
        assert service._calculate_next_due_date(datetime.now(), "YEARLY") is None

    def test_none_base_date_uses_current_time(self):
        service = RecurrenceService()
        before = datetime.now()
        next_date = service._calculate_next_due_date(None, "DAILY")
        after = datetime.now()
        assert next_date is not None
        assert next_date > before


@pytest.mark.asyncio
class TestRecurrenceServiceCreateNextInstance:
    """Unit tests for create_next_instance method."""

    async def test_create_next_instance_task_not_found(self, db_session):
        service = RecurrenceService()
        with patch("src.services.recurrence_service.get_session", return_value=iter([db_session])):
            result = await service.create_next_instance(9999, {"task_id": 9999})
            assert result is None

    async def test_create_next_instance_success(self, db_session):
        service = RecurrenceService()

        # Create original task
        original_task = Task(
            title="Daily Standup",
            description="Discuss blockers",
            priority="HIGH",
            due_date=datetime(2025, 6, 1, 9, 0, 0),
            recurrence_rule="DAILY",
            completed=True,
            user_id="user_test_123"
        )
        db_session.add(original_task)
        db_session.commit()
        db_session.refresh(original_task)

        # Add tag link
        tag = Tag(name="work", color="#FF0000", user_id="user_test_123")
        db_session.add(tag)
        db_session.commit()
        db_session.refresh(tag)

        tag_link = TaskTagLink(task_id=original_task.id, tag_id=tag.id)
        db_session.add(tag_link)
        db_session.commit()

        with patch("src.services.recurrence_service.get_session", return_value=iter([db_session])):
            new_task = await service.create_next_instance(original_task.id, {"task_id": original_task.id})

            assert new_task is not None
            assert new_task.id != original_task.id
            assert new_task.title == "Daily Standup"
            assert new_task.description == "Discuss blockers"
            assert new_task.priority == "HIGH"
            assert new_task.due_date == datetime(2025, 6, 2, 9, 0, 0)
            assert new_task.completed is False
            assert new_task.user_id == "user_test_123"
            assert new_task.recurrence_rule == "DAILY"

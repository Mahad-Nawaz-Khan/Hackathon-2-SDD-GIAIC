# ============================================================================
# Recurrence Service
# ============================================================================
# Service for handling recurring task logic:
# - Calculates next due date based on recurrence rule (DAILY, WEEKLY, MONTHLY)
# - Creates next task instance when recurring task is completed
#
# This service integrates with Dapr event system to process task.completed
# events and automatically create the next instance of recurring tasks.
# ============================================================================

from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from sqlmodel import Session, select
import logging

from ..models.task import Task, TaskBase
from ..models.user import User
from ..database import get_session

logger = logging.getLogger(__name__)


class RecurrenceService:
    """
    Service for managing recurring task logic.

    Handles the calculation of next due dates and creation of next
    task instances when recurring tasks are completed.
    """

    def __init__(self):
        """Initialize the recurrence service."""
        pass

    async def create_next_instance(
        self,
        task_id: int,
        event_data: Dict[str, Any]
    ) -> Optional[Task]:
        """
        Create the next instance of a recurring task.

        Called when a recurring task is completed. Calculates the next
        due date based on the recurrence rule and creates a new task
        with the same properties (except completed status).

        Args:
            task_id: The ID of the completed recurring task
            event_data: The event data from the task.completed event

        Returns:
            The newly created Task instance, or None if creation failed

        Example:
            >>> # When a daily task "Standup" due on 2025-01-01 is completed
            >>> new_task = await recurrence_service.create_next_instance(
            ...     task_id=123,
            ...     event_data={"task_id": 123, "user_id": 1, ...}
            ... )
            >>> # Returns new task "Standup" due on 2025-01-02
        """
        try:
            with next(get_session()) as session:
                # Get the original task
                statement = select(Task).where(Task.id == task_id)
                original_task = session.exec(statement).first()

                if not original_task:
                    logger.error(f"Original task {task_id} not found")
                    return None

                # Calculate next due date
                next_due_date = self._calculate_next_due_date(
                    original_task.due_date,
                    original_task.recurrence_rule
                )

                if not next_due_date:
                    logger.error(f"Failed to calculate next due date for task {task_id}")
                    return None

                # Create new task instance
                new_task = Task(
                    title=original_task.title,
                    description=original_task.description,
                    priority=original_task.priority,
                    due_date=next_due_date,
                    recurrence_rule=original_task.recurrence_rule,
                    reminder_time=original_task.reminder_time,
                    user_id=original_task.user_id,
                    completed=False  # New instance is not completed
                )

                session.add(new_task)
                session.commit()
                session.refresh(new_task)

                # Copy tags from original task
                from ..models.tag import TaskTagLink
                tag_links = session.exec(
                    select(TaskTagLink).where(TaskTagLink.task_id == task_id)
                ).all()

                for link in tag_links:
                    new_link = TaskTagLink(
                        task_id=new_task.id,
                        tag_id=link.tag_id
                    )
                    session.add(new_link)

                session.commit()

                logger.info(
                    f"Created next instance of recurring task {task_id}: {new_task.id}",
                    extra={
                        "original_task_id": task_id,
                        "new_task_id": new_task.id,
                        "next_due_date": next_due_date.isoformat(),
                        "recurrence_rule": original_task.recurrence_rule
                    }
                )

                return new_task

        except Exception as e:
            logger.error(f"Error creating next instance: {e}", exc_info=True)
            return None

    def _calculate_next_due_date(
        self,
        current_due_date: Optional[datetime],
        recurrence_rule: Optional[str]
    ) -> Optional[datetime]:
        """
        Calculate the next due date based on recurrence rule.

        Args:
            current_due_date: The current task's due date (or created_at if no due_date)
            recurrence_rule: One of DAILY, WEEKLY, MONTHLY

        Returns:
            The next due date as datetime, or None if calculation failed

        Rules:
            - DAILY: Add 1 day to current due date
            - WEEKLY: Add 7 days to current due date
            - MONTHLY: Add 1 month to current due date (handles month-end edge cases)

        Edge Cases:
            - If current_due_date is None, use current date
            - For monthly recurrence on day 31, next month with fewer days uses last day
            - For monthly recurrence on Feb 29, next non-leap year uses Feb 28
        """
        if not recurrence_rule:
            return None

        # Use current date if no due_date set
        if current_due_date is None:
            base_date = datetime.now()
        else:
            base_date = current_due_date

        try:
            if recurrence_rule == "DAILY":
                return base_date + timedelta(days=1)

            elif recurrence_rule == "WEEKLY":
                return base_date + timedelta(weeks=1)

            elif recurrence_rule == "MONTHLY":
                # Handle month addition with edge cases
                # For day 31 in a 30-day month, use the last day of the next month
                # For Feb 29 in non-leap year, use Feb 28

                year = base_date.year
                month = base_date.month
                day = base_date.day

                # Calculate next month
                if month == 12:
                    next_month = 1
                    next_year = year + 1
                else:
                    next_month = month + 1
                    next_year = year

                # Get the last day of the next month
                import calendar
                last_day_of_next_month = calendar.monthrange(next_year, next_month)[1]

                # Use the smaller of: original day or last day of next month
                next_day = min(day, last_day_of_next_month)

                return datetime(next_year, next_month, next_day)

            else:
                logger.error(f"Unknown recurrence rule: {recurrence_rule}")
                return None

        except Exception as e:
            logger.error(f"Error calculating next due date: {e}", exc_info=True)
            return None


# Singleton instance
recurrence_service = RecurrenceService()

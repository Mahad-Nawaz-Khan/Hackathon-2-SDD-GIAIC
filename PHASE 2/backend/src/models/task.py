from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from sqlalchemy import Index

from .task_tag import TaskTagLink

if TYPE_CHECKING:
    from .user import User
    from .tag import Tag


class TaskBase(SQLModel):
    title: str
    description: Optional[str] = None
    completed: bool = False
    priority: str = "MEDIUM"
    due_date: Optional[datetime] = None
    recurrence_rule: Optional[str] = None
    user_id: int = Field(foreign_key="user.id")


class Task(TaskBase, table=True):
    __table_args__ = (
        Index("idx_task_user_completed", "user_id", "completed"),
        Index("idx_task_user_priority", "user_id", "priority"),
        Index("idx_task_user_due_date", "user_id", "due_date"),
        Index("idx_task_user_created_at", "user_id", "created_at"),
        Index("idx_task_completed_priority", "completed", "priority"),
        Index("idx_task_user_updated_at", "user_id", "updated_at"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    description: Optional[str] = None
    completed: bool = False
    priority: str = "MEDIUM"
    due_date: Optional[datetime] = None
    recurrence_rule: Optional[str] = None
    user_id: int = Field(foreign_key="user.id", index=True)  # Index for user_id
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)  # Index for created_at
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)  # Index for updated_at

    # Relationships
    user: "User" = Relationship(back_populates="tasks")
    tags: list["Tag"] = Relationship(back_populates="tasks", link_model=TaskTagLink)
# ============================================================================
# Alembic Migration: Add reminder_time to Task table
# ============================================================================
# This migration adds the reminder_time column to the task table along with
# appropriate indexes for efficient querying of upcoming reminders.
# ============================================================================

from alembic import op
import sqlalchemy as sa


# Revision identifiers, used by Alembic.
revision = '001_add_reminder_time'
down_revision = None  # This is the first migration
branch_labels = None
depends_on = None


def upgrade():
    """
    Add reminder_time column and indexes to the task table.
    """
    # Add reminder_time column (nullable, existing tasks will have NULL)
    op.add_column(
        'task',
        sa.Column('reminder_time', sa.DateTime(), nullable=True)
    )

    # Create index on reminder_time for efficient querying
    op.create_index(
        'ix_task_reminder_time',
        'task',
        ['reminder_time']
    )

    # Create composite index on user_id and reminder_time for user-specific queries
    op.create_index(
        'ix_task_user_reminder_time',
        'task',
        ['user_id', 'reminder_time']
    )


def downgrade():
    """
    Remove reminder_time column and indexes from the task table.
    """
    # Drop indexes first
    op.drop_index('ix_task_user_reminder_time', table_name='task')
    op.drop_index('ix_task_reminder_time', table_name='task')

    # Remove column
    op.drop_column('task', 'reminder_time')

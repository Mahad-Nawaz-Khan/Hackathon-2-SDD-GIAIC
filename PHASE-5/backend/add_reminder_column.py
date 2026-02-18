"""
Quick script to add reminder_time column to task table.
Run this to fix the "column task.reminder_time does not exist" error.
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

# Get database URL
DATABASE_URL = os.getenv("DATABASE_URL")

# Normalize Postgres URLs to psycopg v3
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

print(f"Connecting to database...")

# Create engine
engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'task'
            AND column_name = 'reminder_time'
        """))

        if result.fetchone():
            print("OK: Column 'reminder_time' already exists in task table")
        else:
            print("Adding 'reminder_time' column to task table...")
            conn.execute(text("""
                ALTER TABLE task
                ADD COLUMN reminder_time TIMESTAMP
            """))
            conn.commit()
            print("OK: Successfully added 'reminder_time' column")

        # Create indexes for better performance
        print("Creating indexes...")

        # Check if index already exists
        result = conn.execute(text("""
            SELECT indexname
            FROM pg_indexes
            WHERE indexname = 'ix_task_reminder_time'
        """))

        if not result.fetchone():
            conn.execute(text("""
                CREATE INDEX ix_task_reminder_time ON task (reminder_time)
            """))
            conn.commit()
            print("OK: Created index on reminder_time")
        else:
            print("OK: Index on reminder_time already exists")

        # Check if composite index exists
        result = conn.execute(text("""
            SELECT indexname
            FROM pg_indexes
            WHERE indexname = 'ix_task_user_id_reminder_time'
        """))

        if not result.fetchone():
            conn.execute(text("""
                CREATE INDEX ix_task_user_id_reminder_time ON task (user_id, reminder_time)
            """))
            conn.commit()
            print("OK: Created composite index on (user_id, reminder_time)")
        else:
            print("OK: Composite index already exists")

    print("\nOK: Migration complete! The database is now up to date.")

except Exception as e:
    print(f"\nERROR: {e}")
    print("Please check your DATABASE_URL in .env file")
    raise

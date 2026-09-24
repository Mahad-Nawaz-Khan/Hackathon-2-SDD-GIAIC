"""
Quick script to add timezone column to user table.
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

print("Connecting to database...")

# Create engine
engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'user'
            AND column_name = 'timezone'
        """))

        if result.fetchone():
            print("OK: Column 'timezone' already exists in user table")
        else:
            print("Adding 'timezone' column to user table...")
            conn.execute(text("""
                ALTER TABLE "user"
                ADD COLUMN timezone VARCHAR(50) NOT NULL DEFAULT 'UTC'
            """))
            conn.commit()
            print("OK: Successfully added 'timezone' column to user table")

    print("\nOK: Migration complete! The user table now has timezone support.")

except Exception as e:
    print(f"\nERROR: {e}")
    print("Please check your DATABASE_URL in .env file")
    raise

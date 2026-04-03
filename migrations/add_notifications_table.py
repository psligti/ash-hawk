"""
Migration: Add Notifications Table

This migration creates the notifications table to store user notifications.
"""

from datetime import datetime
from datetime import timezone as tz


def upgrade(connection):
    """
    Apply the migration: Create the notifications table.

    Args:
        connection: Database connection object
    """
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id VARCHAR(255) PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            message TEXT NOT NULL,
            type VARCHAR(50) NOT NULL DEFAULT 'info',
            metadata JSON,
            read BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create indexes for better query performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_notifications_user_id
        ON notifications(user_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_notifications_user_read
        ON notifications(user_id, read)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_notifications_created_at
        ON notifications(created_at DESC)
    """)

    connection.commit()
    print("✓ Migration 'add_notifications_table' applied successfully")


def downgrade(connection):
    """
    Rollback the migration: Drop the notifications table.

    Args:
        connection: Database connection object
    """
    cursor = connection.cursor()

    cursor.execute("DROP INDEX IF EXISTS idx_notifications_user_id")
    cursor.execute("DROP INDEX IF EXISTS idx_notifications_user_read")
    cursor.execute("DROP INDEX IF EXISTS idx_notifications_created_at")
    cursor.execute("DROP TABLE IF EXISTS notifications")

    connection.commit()
    print("✓ Migration 'add_notifications_table' rolled back successfully")


def run_migration(connection, direction="up"):
    """
    Run the migration in the specified direction.

    Args:
        connection: Database connection object
        direction: 'up' to apply, 'down' to rollback
    """
    if direction == "up":
        upgrade(connection)
    elif direction == "down":
        downgrade(connection)
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'up' or 'down'")


if __name__ == "__main__":
    # Example usage (requires actual database connection)
    import sqlite3

    # Create test SQLite database for demonstration
    test_conn = sqlite3.connect(":memory:")

    # Run upgrade
    run_migration(test_conn, direction="up")

    # Verify table was created
    cursor = test_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='notifications'")
    result = cursor.fetchone()
    print(f"Table created: {result}")

    # Show table schema
    cursor.execute("PRAGMA table_info(notifications)")
    columns = cursor.fetchall()
    print("\nTable schema:")
    for col in columns:
        print(f"  - {col[1]}: {col[2]}")

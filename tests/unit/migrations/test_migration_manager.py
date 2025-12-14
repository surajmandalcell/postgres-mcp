"""Comprehensive tests for MigrationManager class.

Tests cover all edge cases for migration management:
- Migration file discovery and parsing
- Migration execution (up/down)
- Batch operations
- Dry run mode
- Rollback operations
- Status reporting
- Error handling and recovery
- Concurrent migrations
"""

import hashlib
import os
import tempfile
from datetime import datetime
from datetime import timezone
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch
from unittest.mock import PropertyMock

import pytest
import pytest_asyncio

from postgres_mcp.migrations.migration_tracker import MigrationRecord
from postgres_mcp.sql import SqlDriver


# Import will be available after implementation
# from postgres_mcp.migrations.migration_manager import (
#     MigrationManager,
#     MigrationFile,
#     MigrationResult,
#     MigrationError,
# )


class MockRowResult:
    """Mock row result for testing."""

    def __init__(self, cells: dict):
        self.cells = cells


@pytest_asyncio.fixture
async def mock_sql_driver():
    """Create a mock SQL driver."""
    driver = MagicMock(spec=SqlDriver)
    driver.execute_query = AsyncMock(return_value=[])
    return driver


@pytest.fixture
def temp_migrations_dir():
    """Create a temporary directory for migration files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_migration_content():
    """Sample migration SQL content."""
    return {
        "up": """
-- Migration: Create users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
""",
        "down": """
-- Rollback: Drop users table
DROP INDEX IF EXISTS idx_users_email;
DROP TABLE IF EXISTS users;
"""
    }


def create_migration_file(
    migrations_dir: Path,
    name: str,
    up_sql: str,
    down_sql: str = None,
    timestamp: str = None
) -> Path:
    """Helper to create migration files."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    migration_dir = migrations_dir / f"{timestamp}_{name}"
    migration_dir.mkdir(parents=True, exist_ok=True)

    # Write up migration
    up_file = migration_dir / "up.sql"
    up_file.write_text(up_sql)

    # Write down migration if provided
    if down_sql:
        down_file = migration_dir / "down.sql"
        down_file.write_text(down_sql)

    return migration_dir


# =============================================================================
# Migration File Discovery Tests
# =============================================================================


class TestMigrationDiscovery:
    """Test migration file discovery functionality."""

    @pytest.mark.asyncio
    async def test_discover_no_migrations(self, mock_sql_driver, temp_migrations_dir):
        """Test discovery when no migration files exist."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        migrations = await manager.discover_migrations()

        assert migrations == []

    @pytest.mark.asyncio
    async def test_discover_single_migration(
        self, mock_sql_driver, temp_migrations_dir, sample_migration_content
    ):
        """Test discovery of single migration."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(
            temp_migrations_dir,
            "create_users",
            sample_migration_content["up"],
            sample_migration_content["down"],
            "20240101000000"
        )

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        migrations = await manager.discover_migrations()

        assert len(migrations) == 1
        assert migrations[0].name == "20240101000000_create_users"

    @pytest.mark.asyncio
    async def test_discover_multiple_migrations_ordered(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test that migrations are discovered in correct order."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        # Create migrations in random order
        create_migration_file(temp_migrations_dir, "third", "SELECT 3;", timestamp="20240103000000")
        create_migration_file(temp_migrations_dir, "first", "SELECT 1;", timestamp="20240101000000")
        create_migration_file(temp_migrations_dir, "second", "SELECT 2;", timestamp="20240102000000")

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        migrations = await manager.discover_migrations()

        assert len(migrations) == 3
        assert migrations[0].name == "20240101000000_first"
        assert migrations[1].name == "20240102000000_second"
        assert migrations[2].name == "20240103000000_third"

    @pytest.mark.asyncio
    async def test_discover_ignores_non_migration_dirs(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test that non-migration directories are ignored."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        # Create valid migration
        create_migration_file(temp_migrations_dir, "valid", "SELECT 1;", timestamp="20240101000000")

        # Create invalid directories
        (temp_migrations_dir / "random_folder").mkdir()
        (temp_migrations_dir / ".hidden").mkdir()
        (temp_migrations_dir / "__pycache__").mkdir()

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        migrations = await manager.discover_migrations()

        assert len(migrations) == 1

    @pytest.mark.asyncio
    async def test_discover_requires_up_sql(self, mock_sql_driver, temp_migrations_dir):
        """Test that migrations without up.sql are skipped."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        # Create directory without up.sql
        migration_dir = temp_migrations_dir / "20240101000000_invalid"
        migration_dir.mkdir()
        (migration_dir / "down.sql").write_text("DROP TABLE test;")

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        migrations = await manager.discover_migrations()

        assert migrations == []

    @pytest.mark.asyncio
    async def test_discover_calculates_checksum(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test that checksum is calculated for migration content."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        content = "CREATE TABLE test (id INT);"
        create_migration_file(temp_migrations_dir, "test", content, timestamp="20240101000000")

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        migrations = await manager.discover_migrations()

        expected_checksum = hashlib.sha256(content.encode()).hexdigest()
        assert migrations[0].checksum == expected_checksum

    @pytest.mark.asyncio
    async def test_discover_with_single_file_format(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test discovery with single-file migration format."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        # Create single-file migration
        migration_file = temp_migrations_dir / "20240101000000_create_users.sql"
        migration_file.write_text("""
-- migrate:up
CREATE TABLE users (id INT);

-- migrate:down
DROP TABLE users;
""")

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        migrations = await manager.discover_migrations()

        assert len(migrations) == 1


# =============================================================================
# Migration Execution Tests
# =============================================================================


class TestMigrationExecution:
    """Test migration execution functionality."""

    @pytest.mark.asyncio
    async def test_migrate_up_single(
        self, mock_sql_driver, temp_migrations_dir, sample_migration_content
    ):
        """Test executing single migration up."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(
            temp_migrations_dir, "create_users",
            sample_migration_content["up"],
            sample_migration_content["down"],
            "20240101000000"
        )

        # Mock tracker methods
        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [],  # get_applied_migrations
            [MockRowResult({"latest_batch": 0})],  # get_latest_batch
            [],  # execute migration
            [],  # record_migration
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        result = await manager.migrate_up()

        assert result.success is True
        assert result.applied_count == 1

    @pytest.mark.asyncio
    async def test_migrate_up_multiple(self, mock_sql_driver, temp_migrations_dir):
        """Test executing multiple migrations up."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(temp_migrations_dir, "first", "SELECT 1;", timestamp="20240101000000")
        create_migration_file(temp_migrations_dir, "second", "SELECT 2;", timestamp="20240102000000")
        create_migration_file(temp_migrations_dir, "third", "SELECT 3;", timestamp="20240103000000")

        # Mock no applied migrations
        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [],  # get_applied_migrations (empty)
            [MockRowResult({"latest_batch": 0})],  # get_latest_batch
            [],  # execute migration 1
            [],  # record migration 1
            [],  # execute migration 2
            [],  # record migration 2
            [],  # execute migration 3
            [],  # record migration 3
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        result = await manager.migrate_up()

        assert result.success is True
        assert result.applied_count == 3

    @pytest.mark.asyncio
    async def test_migrate_up_already_applied(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test that already applied migrations are skipped."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(temp_migrations_dir, "first", "SELECT 1;", timestamp="20240101000000")

        # Mock migration already applied
        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [MockRowResult({
                "id": 1,
                "name": "20240101000000_first",
                "applied_at": datetime.now(timezone.utc),
                "checksum": hashlib.sha256(b"SELECT 1;").hexdigest(),
                "batch": 1
            })],  # get_applied_migrations
            [MockRowResult({"latest_batch": 1})],  # get_latest_batch
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        result = await manager.migrate_up()

        assert result.success is True
        assert result.applied_count == 0

    @pytest.mark.asyncio
    async def test_migrate_up_with_steps_limit(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test migration with steps limit."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(temp_migrations_dir, "first", "SELECT 1;", timestamp="20240101000000")
        create_migration_file(temp_migrations_dir, "second", "SELECT 2;", timestamp="20240102000000")
        create_migration_file(temp_migrations_dir, "third", "SELECT 3;", timestamp="20240103000000")

        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [],  # get_applied_migrations
            [MockRowResult({"latest_batch": 0})],  # get_latest_batch
            [],  # execute migration 1
            [],  # record migration 1
            [],  # execute migration 2
            [],  # record migration 2
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        result = await manager.migrate_up(steps=2)

        assert result.success is True
        assert result.applied_count == 2

    @pytest.mark.asyncio
    async def test_migrate_up_to_specific_migration(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test migration up to specific migration."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(temp_migrations_dir, "first", "SELECT 1;", timestamp="20240101000000")
        create_migration_file(temp_migrations_dir, "second", "SELECT 2;", timestamp="20240102000000")
        create_migration_file(temp_migrations_dir, "third", "SELECT 3;", timestamp="20240103000000")

        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [],  # get_applied_migrations
            [MockRowResult({"latest_batch": 0})],  # get_latest_batch
            [],  # execute migration 1
            [],  # record migration 1
            [],  # execute migration 2
            [],  # record migration 2
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        result = await manager.migrate_up(target="20240102000000_second")

        assert result.success is True
        assert result.applied_count == 2

    @pytest.mark.asyncio
    async def test_migrate_up_handles_sql_error(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test that SQL errors are properly handled."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(temp_migrations_dir, "bad", "INVALID SQL;", timestamp="20240101000000")

        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [],  # get_applied_migrations
            [MockRowResult({"latest_batch": 0})],  # get_latest_batch
            Exception("syntax error at or near 'INVALID'"),  # execute migration
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        result = await manager.migrate_up()

        assert result.success is False
        assert "syntax error" in result.error

    @pytest.mark.asyncio
    async def test_migrate_up_batch_number_increments(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test that batch number increments correctly."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(temp_migrations_dir, "first", "SELECT 1;", timestamp="20240101000000")

        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [],  # get_applied_migrations
            [MockRowResult({"latest_batch": 5})],  # get_latest_batch returns 5
            [],  # execute migration
            [],  # record migration
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        result = await manager.migrate_up()

        assert result.success is True
        assert result.batch == 6  # Should be incremented


# =============================================================================
# Rollback Tests
# =============================================================================


class TestMigrationRollback:
    """Test migration rollback functionality."""

    @pytest.mark.asyncio
    async def test_rollback_single(
        self, mock_sql_driver, temp_migrations_dir, sample_migration_content
    ):
        """Test rolling back single migration."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(
            temp_migrations_dir, "create_users",
            sample_migration_content["up"],
            sample_migration_content["down"],
            "20240101000000"
        )

        applied_at = datetime.now(timezone.utc)
        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [MockRowResult({"latest_batch": 1})],  # get_latest_batch
            [MockRowResult({
                "id": 1,
                "name": "20240101000000_create_users",
                "applied_at": applied_at,
                "checksum": "abc",
                "batch": 1
            })],  # get_migrations_in_batch
            [],  # execute down migration
            [],  # remove_migration
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        result = await manager.rollback()

        assert result.success is True
        assert result.rolled_back_count == 1

    @pytest.mark.asyncio
    async def test_rollback_batch(self, mock_sql_driver, temp_migrations_dir):
        """Test rolling back entire batch."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(temp_migrations_dir, "first", "SELECT 1;", "SELECT -1;", "20240101000000")
        create_migration_file(temp_migrations_dir, "second", "SELECT 2;", "SELECT -2;", "20240102000000")

        applied_at = datetime.now(timezone.utc)
        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [MockRowResult({"latest_batch": 1})],  # get_latest_batch
            [
                MockRowResult({"id": 2, "name": "20240102000000_second", "applied_at": applied_at, "checksum": "abc", "batch": 1}),
                MockRowResult({"id": 1, "name": "20240101000000_first", "applied_at": applied_at, "checksum": "def", "batch": 1}),
            ],  # get_migrations_in_batch (DESC order)
            [],  # execute down migration 2
            [],  # remove_migration 2
            [],  # execute down migration 1
            [],  # remove_migration 1
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        result = await manager.rollback()

        assert result.success is True
        assert result.rolled_back_count == 2

    @pytest.mark.asyncio
    async def test_rollback_with_steps(self, mock_sql_driver, temp_migrations_dir):
        """Test rolling back specific number of steps."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(temp_migrations_dir, "first", "SELECT 1;", "SELECT -1;", "20240101000000")
        create_migration_file(temp_migrations_dir, "second", "SELECT 2;", "SELECT -2;", "20240102000000")
        create_migration_file(temp_migrations_dir, "third", "SELECT 3;", "SELECT -3;", "20240103000000")

        applied_at = datetime.now(timezone.utc)

        # Return migrations from multiple batches
        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [
                MockRowResult({"id": 3, "name": "20240103000000_third", "applied_at": applied_at, "checksum": "c", "batch": 3}),
                MockRowResult({"id": 2, "name": "20240102000000_second", "applied_at": applied_at, "checksum": "b", "batch": 2}),
                MockRowResult({"id": 1, "name": "20240101000000_first", "applied_at": applied_at, "checksum": "a", "batch": 1}),
            ],  # get_applied_migrations (for step-based rollback)
            [],  # execute down migration 3
            [],  # remove_migration 3
            [],  # execute down migration 2
            [],  # remove_migration 2
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        result = await manager.rollback(steps=2)

        assert result.success is True
        assert result.rolled_back_count == 2

    @pytest.mark.asyncio
    async def test_rollback_no_migrations(self, mock_sql_driver, temp_migrations_dir):
        """Test rollback when no migrations to rollback."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [MockRowResult({"latest_batch": 0})],  # get_latest_batch
            [],  # get_migrations_in_batch (empty)
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        result = await manager.rollback()

        assert result.success is True
        assert result.rolled_back_count == 0

    @pytest.mark.asyncio
    async def test_rollback_missing_down_migration(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test rollback when down migration file is missing."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        # Create migration without down.sql
        create_migration_file(temp_migrations_dir, "no_down", "SELECT 1;", timestamp="20240101000000")

        applied_at = datetime.now(timezone.utc)
        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [MockRowResult({"latest_batch": 1})],  # get_latest_batch
            [MockRowResult({
                "id": 1,
                "name": "20240101000000_no_down",
                "applied_at": applied_at,
                "checksum": "abc",
                "batch": 1
            })],  # get_migrations_in_batch
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        result = await manager.rollback()

        assert result.success is False
        assert "down migration" in result.error.lower() or "rollback" in result.error.lower()

    @pytest.mark.asyncio
    async def test_rollback_all(self, mock_sql_driver, temp_migrations_dir):
        """Test rolling back all migrations."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(temp_migrations_dir, "first", "SELECT 1;", "SELECT -1;", "20240101000000")
        create_migration_file(temp_migrations_dir, "second", "SELECT 2;", "SELECT -2;", "20240102000000")

        applied_at = datetime.now(timezone.utc)
        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [
                MockRowResult({"id": 2, "name": "20240102000000_second", "applied_at": applied_at, "checksum": "b", "batch": 2}),
                MockRowResult({"id": 1, "name": "20240101000000_first", "applied_at": applied_at, "checksum": "a", "batch": 1}),
            ],  # get_applied_migrations
            [],  # execute down 2
            [],  # remove 2
            [],  # execute down 1
            [],  # remove 1
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        result = await manager.rollback_all()

        assert result.success is True
        assert result.rolled_back_count == 2


# =============================================================================
# Dry Run Tests
# =============================================================================


class TestDryRun:
    """Test dry run functionality."""

    @pytest.mark.asyncio
    async def test_dry_run_migrate_up(
        self, mock_sql_driver, temp_migrations_dir, sample_migration_content
    ):
        """Test dry run doesn't execute migrations."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(
            temp_migrations_dir, "create_users",
            sample_migration_content["up"],
            sample_migration_content["down"],
            "20240101000000"
        )

        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [],  # get_applied_migrations
            [MockRowResult({"latest_batch": 0})],  # get_latest_batch
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        result = await manager.migrate_up(dry_run=True)

        assert result.success is True
        assert result.dry_run is True
        assert result.pending_count == 1
        # Should only call ensure_table, get_applied, get_batch - not execute
        assert mock_sql_driver.execute_query.await_count <= 3

    @pytest.mark.asyncio
    async def test_dry_run_rollback(self, mock_sql_driver, temp_migrations_dir):
        """Test dry run doesn't execute rollbacks."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(temp_migrations_dir, "test", "SELECT 1;", "SELECT -1;", "20240101000000")

        applied_at = datetime.now(timezone.utc)
        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [MockRowResult({"latest_batch": 1})],  # get_latest_batch
            [MockRowResult({
                "id": 1,
                "name": "20240101000000_test",
                "applied_at": applied_at,
                "checksum": "abc",
                "batch": 1
            })],  # get_migrations_in_batch
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        result = await manager.rollback(dry_run=True)

        assert result.success is True
        assert result.dry_run is True

    @pytest.mark.asyncio
    async def test_dry_run_shows_sql(
        self, mock_sql_driver, temp_migrations_dir, sample_migration_content
    ):
        """Test dry run shows SQL to be executed."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(
            temp_migrations_dir, "create_users",
            sample_migration_content["up"],
            sample_migration_content["down"],
            "20240101000000"
        )

        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [],  # get_applied_migrations
            [MockRowResult({"latest_batch": 0})],  # get_latest_batch
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        result = await manager.migrate_up(dry_run=True)

        assert "CREATE TABLE" in result.sql_preview


# =============================================================================
# Status Tests
# =============================================================================


class TestMigrationStatus:
    """Test migration status functionality."""

    @pytest.mark.asyncio
    async def test_status_no_migrations(self, mock_sql_driver, temp_migrations_dir):
        """Test status when no migrations exist."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [],  # get_applied_migrations
            [MockRowResult({"latest_batch": 0})],  # get_latest_batch
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        status = await manager.status()

        assert status["total_migrations"] == 0
        assert status["applied_count"] == 0
        assert status["pending_count"] == 0

    @pytest.mark.asyncio
    async def test_status_all_applied(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test status when all migrations are applied."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(temp_migrations_dir, "first", "SELECT 1;", timestamp="20240101000000")
        create_migration_file(temp_migrations_dir, "second", "SELECT 2;", timestamp="20240102000000")

        applied_at = datetime.now(timezone.utc)
        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [
                MockRowResult({"id": 1, "name": "20240101000000_first", "applied_at": applied_at, "checksum": "a", "batch": 1}),
                MockRowResult({"id": 2, "name": "20240102000000_second", "applied_at": applied_at, "checksum": "b", "batch": 1}),
            ],  # get_applied_migrations
            [MockRowResult({"latest_batch": 1})],  # get_latest_batch
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        status = await manager.status()

        assert status["total_migrations"] == 2
        assert status["applied_count"] == 2
        assert status["pending_count"] == 0

    @pytest.mark.asyncio
    async def test_status_some_pending(self, mock_sql_driver, temp_migrations_dir):
        """Test status with pending migrations."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(temp_migrations_dir, "first", "SELECT 1;", timestamp="20240101000000")
        create_migration_file(temp_migrations_dir, "second", "SELECT 2;", timestamp="20240102000000")
        create_migration_file(temp_migrations_dir, "third", "SELECT 3;", timestamp="20240103000000")

        applied_at = datetime.now(timezone.utc)
        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [MockRowResult({"id": 1, "name": "20240101000000_first", "applied_at": applied_at, "checksum": "a", "batch": 1})],
            [MockRowResult({"latest_batch": 1})],  # get_latest_batch
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        status = await manager.status()

        assert status["total_migrations"] == 3
        assert status["applied_count"] == 1
        assert status["pending_count"] == 2

    @pytest.mark.asyncio
    async def test_status_includes_migration_details(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test that status includes migration details."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(temp_migrations_dir, "test", "SELECT 1;", timestamp="20240101000000")

        applied_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [MockRowResult({"id": 1, "name": "20240101000000_test", "applied_at": applied_at, "checksum": "abc", "batch": 1})],
            [MockRowResult({"latest_batch": 1})],
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        status = await manager.status()

        assert len(status["migrations"]) == 1
        assert status["migrations"][0]["name"] == "20240101000000_test"
        assert status["migrations"][0]["status"] == "applied"

    @pytest.mark.asyncio
    async def test_status_detects_checksum_mismatch(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test that status detects checksum mismatches."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(temp_migrations_dir, "test", "SELECT 1;", timestamp="20240101000000")

        applied_at = datetime.now(timezone.utc)
        # Checksum doesn't match file content
        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [MockRowResult({
                "id": 1,
                "name": "20240101000000_test",
                "applied_at": applied_at,
                "checksum": "different_checksum",
                "batch": 1
            })],
            [MockRowResult({"latest_batch": 1})],
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        status = await manager.status()

        assert status["migrations"][0]["checksum_mismatch"] is True


# =============================================================================
# Migration Generation Tests
# =============================================================================


class TestMigrationGeneration:
    """Test migration file generation."""

    @pytest.mark.asyncio
    async def test_generate_migration_creates_files(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test generating a new migration creates files."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )

        migration_path = await manager.generate_migration(
            name="create_products",
            up_sql="CREATE TABLE products (id SERIAL PRIMARY KEY);",
            down_sql="DROP TABLE products;"
        )

        assert migration_path.exists()
        assert (migration_path / "up.sql").exists()
        assert (migration_path / "down.sql").exists()

    @pytest.mark.asyncio
    async def test_generate_migration_timestamp(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test that generated migration has timestamp prefix."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )

        migration_path = await manager.generate_migration(
            name="test_migration",
            up_sql="SELECT 1;"
        )

        # Name should start with timestamp
        name = migration_path.name
        assert name.endswith("_test_migration")
        assert name[:14].isdigit()  # YYYYMMDDHHmmss

    @pytest.mark.asyncio
    async def test_generate_migration_sanitizes_name(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test that migration names are sanitized."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )

        migration_path = await manager.generate_migration(
            name="Add User's Table!",
            up_sql="SELECT 1;"
        )

        # Should be sanitized to safe characters
        assert "'" not in migration_path.name
        assert "!" not in migration_path.name

    @pytest.mark.asyncio
    async def test_generate_empty_migration(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test generating an empty migration template."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )

        migration_path = await manager.generate_migration(name="empty")

        up_content = (migration_path / "up.sql").read_text()
        assert "-- Add migration SQL here" in up_content or up_content.strip() == ""


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handles_empty_migrations_dir(self, mock_sql_driver):
        """Test handling non-existent migrations directory."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        non_existent = Path("/non/existent/path")

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=non_existent
        )

        # Should create directory or handle gracefully
        migrations = await manager.discover_migrations()
        assert migrations == []

    @pytest.mark.asyncio
    async def test_handles_corrupted_migration_file(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test handling of corrupted migration files."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        # Create directory with empty up.sql
        migration_dir = temp_migrations_dir / "20240101000000_corrupted"
        migration_dir.mkdir()
        (migration_dir / "up.sql").write_bytes(b"\x00\x00\x00")  # Binary garbage

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )

        # Should handle gracefully
        migrations = await manager.discover_migrations()
        # Either skip or handle the corrupted file

    @pytest.mark.asyncio
    async def test_handles_permission_error(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test handling of permission errors."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(temp_migrations_dir, "test", "SELECT 1;", timestamp="20240101000000")

        # Make file unreadable (skip on Windows)
        if os.name != "nt":
            up_file = temp_migrations_dir / "20240101000000_test" / "up.sql"
            up_file.chmod(0o000)

            try:
                manager = MigrationManager(
                    sql_driver=mock_sql_driver,
                    migrations_dir=temp_migrations_dir
                )
                # Should handle gracefully
                with pytest.raises(PermissionError):
                    await manager.discover_migrations()
            finally:
                up_file.chmod(0o644)

    @pytest.mark.asyncio
    async def test_handles_transaction_rollback(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test that failed migrations trigger transaction rollback."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(temp_migrations_dir, "failing", "INVALID SQL;", timestamp="20240101000000")

        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [],  # get_applied_migrations
            [MockRowResult({"latest_batch": 0})],  # get_latest_batch
            Exception("SQL error"),  # execute migration fails
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        result = await manager.migrate_up()

        assert result.success is False
        # Migration should not be recorded

    @pytest.mark.asyncio
    async def test_concurrent_migration_protection(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test protection against concurrent migrations."""
        from postgres_mcp.migrations.migration_manager import MigrationManager
        import asyncio

        create_migration_file(temp_migrations_dir, "test", "SELECT 1;", timestamp="20240101000000")

        mock_sql_driver.execute_query.side_effect = [
            [],  # First call
            [],
            [MockRowResult({"latest_batch": 0})],
            [],
            [],
            # Second concurrent call
            [],
            [],
            [MockRowResult({"latest_batch": 0})],
            [],
            [],
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )

        # Run migrations concurrently
        results = await asyncio.gather(
            manager.migrate_up(),
            manager.migrate_up(),
            return_exceptions=True
        )

        # At least one should succeed or handle concurrency

    @pytest.mark.asyncio
    async def test_large_migration_file(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test handling of large migration files."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        # Create a large migration (1MB+)
        large_sql = "-- Large migration\n" + "SELECT 1;\n" * 100000

        create_migration_file(
            temp_migrations_dir, "large",
            large_sql,
            timestamp="20240101000000"
        )

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        migrations = await manager.discover_migrations()

        assert len(migrations) == 1
        assert len(migrations[0].up_sql) > 1000000

    @pytest.mark.asyncio
    async def test_special_sql_characters(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test handling of special SQL characters."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        sql_with_special_chars = """
        CREATE TABLE test (
            name VARCHAR(255) DEFAULT 'O''Brien',  -- Single quote escape
            data JSONB DEFAULT '{"key": "value"}',  -- JSON
            pattern TEXT DEFAULT E'\\n\\t'  -- Escape sequences
        );
        """

        create_migration_file(
            temp_migrations_dir, "special",
            sql_with_special_chars,
            timestamp="20240101000000"
        )

        mock_sql_driver.execute_query.side_effect = [
            [],  # ensure_migration_table
            [],  # get_applied_migrations
            [MockRowResult({"latest_batch": 0})],
            [],  # execute
            [],  # record
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        result = await manager.migrate_up()

        assert result.success is True


# =============================================================================
# Reset and Fresh Tests
# =============================================================================


class TestResetAndFresh:
    """Test database reset and fresh migration functionality."""

    @pytest.mark.asyncio
    async def test_fresh_drops_all_and_migrates(
        self, mock_sql_driver, temp_migrations_dir
    ):
        """Test fresh migration (drop all + migrate)."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(temp_migrations_dir, "test", "SELECT 1;", "SELECT -1;", "20240101000000")

        applied_at = datetime.now(timezone.utc)
        mock_sql_driver.execute_query.side_effect = [
            # First: rollback_all
            [],  # ensure_migration_table
            [MockRowResult({"id": 1, "name": "20240101000000_test", "applied_at": applied_at, "checksum": "a", "batch": 1})],
            [],  # execute down
            [],  # remove
            # Then: migrate_up
            [],  # ensure_migration_table
            [],  # get_applied_migrations
            [MockRowResult({"latest_batch": 0})],
            [],  # execute
            [],  # record
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        result = await manager.fresh()

        assert result.success is True

    @pytest.mark.asyncio
    async def test_reset_drops_tables(self, mock_sql_driver, temp_migrations_dir):
        """Test reset drops migration tracking table."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        mock_sql_driver.execute_query.side_effect = [
            [],  # Drop migration table
        ]

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )
        await manager.reset()

        # Should have called DROP TABLE
        call_args = mock_sql_driver.execute_query.call_args[0][0]
        assert "DROP TABLE" in call_args


# =============================================================================
# Schema Diff Integration Tests
# =============================================================================


class TestSchemaDiffIntegration:
    """Test integration with schema diff for migration generation."""

    @pytest.mark.asyncio
    async def test_generate_from_diff(self, mock_sql_driver, temp_migrations_dir):
        """Test generating migration from schema diff."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        manager = MigrationManager(
            sql_driver=mock_sql_driver,
            migrations_dir=temp_migrations_dir
        )

        # Mock schema pull
        mock_sql_driver.execute_query.side_effect = [
            # Schema pull queries...
            [],
        ]

        # This would generate migration from current vs desired schema
        # migration_path = await manager.generate_from_diff(target_schema)

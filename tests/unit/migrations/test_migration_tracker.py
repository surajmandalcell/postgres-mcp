"""Comprehensive tests for MigrationTracker class.

Tests cover all edge cases for migration tracking functionality:
- Migration table creation and management
- Recording and removing migrations
- Batch operations
- Status queries
- Error handling
- Concurrent access scenarios
"""

from datetime import datetime
from datetime import timezone
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import pytest_asyncio

from postgres_mcp.migrations.migration_tracker import MigrationRecord
from postgres_mcp.migrations.migration_tracker import MigrationTracker
from postgres_mcp.migrations.migration_tracker import MIGRATION_TABLE_NAME
from postgres_mcp.sql import SqlDriver


class MockRowResult:
    """Mock row result for testing."""

    def __init__(self, cells: dict):
        self.cells = cells


@pytest_asyncio.fixture
async def mock_sql_driver():
    """Create a mock SQL driver for testing."""
    driver = MagicMock(spec=SqlDriver)
    driver.execute_query = AsyncMock(return_value=[])
    return driver


@pytest_asyncio.fixture
async def tracker(mock_sql_driver):
    """Create a MigrationTracker instance with mock driver."""
    return MigrationTracker(sql_driver=mock_sql_driver)


@pytest_asyncio.fixture
async def tracker_custom_schema(mock_sql_driver):
    """Create a MigrationTracker instance with custom schema."""
    return MigrationTracker(sql_driver=mock_sql_driver, schema="custom_schema")


# =============================================================================
# Migration Table Creation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_ensure_migration_table_creates_table(tracker, mock_sql_driver):
    """Test that ensure_migration_table creates the migration tracking table."""
    await tracker.ensure_migration_table()

    mock_sql_driver.execute_query.assert_awaited_once()
    call_args = mock_sql_driver.execute_query.call_args[0][0]

    assert "CREATE TABLE IF NOT EXISTS" in call_args
    assert MIGRATION_TABLE_NAME in call_args
    assert "id SERIAL PRIMARY KEY" in call_args
    assert "name VARCHAR(255) NOT NULL UNIQUE" in call_args
    assert "applied_at TIMESTAMP WITH TIME ZONE" in call_args
    assert "checksum VARCHAR(64) NOT NULL" in call_args
    assert "batch INTEGER NOT NULL" in call_args
    assert "executed_sql TEXT" in call_args


@pytest.mark.asyncio
async def test_ensure_migration_table_creates_indexes(tracker, mock_sql_driver):
    """Test that ensure_migration_table creates necessary indexes."""
    await tracker.ensure_migration_table()

    call_args = mock_sql_driver.execute_query.call_args[0][0]

    assert f"idx_{MIGRATION_TABLE_NAME}_name" in call_args
    assert f"idx_{MIGRATION_TABLE_NAME}_batch" in call_args


@pytest.mark.asyncio
async def test_ensure_migration_table_custom_schema(tracker_custom_schema, mock_sql_driver):
    """Test migration table creation in custom schema."""
    await tracker_custom_schema.ensure_migration_table()

    call_args = mock_sql_driver.execute_query.call_args[0][0]
    assert "custom_schema._postgres_mcp_migrations" in call_args


@pytest.mark.asyncio
async def test_ensure_migration_table_idempotent(tracker, mock_sql_driver):
    """Test that ensure_migration_table can be called multiple times safely."""
    await tracker.ensure_migration_table()
    await tracker.ensure_migration_table()

    assert mock_sql_driver.execute_query.await_count == 2


@pytest.mark.asyncio
async def test_ensure_migration_table_handles_error(tracker, mock_sql_driver):
    """Test error handling during table creation."""
    mock_sql_driver.execute_query.side_effect = Exception("Database error")

    with pytest.raises(Exception, match="Database error"):
        await tracker.ensure_migration_table()


# =============================================================================
# Get Applied Migrations Tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_applied_migrations_empty(tracker, mock_sql_driver):
    """Test getting applied migrations when none exist."""
    mock_sql_driver.execute_query.return_value = []

    result = await tracker.get_applied_migrations()

    assert result == []


@pytest.mark.asyncio
async def test_get_applied_migrations_none_returned(tracker, mock_sql_driver):
    """Test getting applied migrations when query returns None."""
    mock_sql_driver.execute_query.return_value = None

    result = await tracker.get_applied_migrations()

    assert result == []


@pytest.mark.asyncio
async def test_get_applied_migrations_single(tracker, mock_sql_driver):
    """Test getting a single applied migration."""
    applied_at = datetime.now(timezone.utc)
    mock_sql_driver.execute_query.return_value = [
        MockRowResult({
            "id": 1,
            "name": "0001_initial",
            "applied_at": applied_at,
            "checksum": "abc123",
            "batch": 1,
        })
    ]

    result = await tracker.get_applied_migrations()

    assert len(result) == 1
    assert result[0].id == 1
    assert result[0].name == "0001_initial"
    assert result[0].applied_at == applied_at
    assert result[0].checksum == "abc123"
    assert result[0].batch == 1


@pytest.mark.asyncio
async def test_get_applied_migrations_multiple(tracker, mock_sql_driver):
    """Test getting multiple applied migrations."""
    applied_at1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    applied_at2 = datetime(2024, 1, 2, tzinfo=timezone.utc)
    applied_at3 = datetime(2024, 1, 3, tzinfo=timezone.utc)

    mock_sql_driver.execute_query.return_value = [
        MockRowResult({"id": 1, "name": "0001_initial", "applied_at": applied_at1, "checksum": "abc1", "batch": 1}),
        MockRowResult({"id": 2, "name": "0002_add_users", "applied_at": applied_at2, "checksum": "abc2", "batch": 1}),
        MockRowResult({"id": 3, "name": "0003_add_orders", "applied_at": applied_at3, "checksum": "abc3", "batch": 2}),
    ]

    result = await tracker.get_applied_migrations()

    assert len(result) == 3
    assert result[0].name == "0001_initial"
    assert result[1].name == "0002_add_users"
    assert result[2].name == "0003_add_orders"


@pytest.mark.asyncio
async def test_get_applied_migrations_ordered_by_id(tracker, mock_sql_driver):
    """Test that migrations are ordered by id ASC."""
    await tracker.get_applied_migrations()

    call_args = mock_sql_driver.execute_query.call_args[0][0]
    assert "ORDER BY id ASC" in call_args


# =============================================================================
# Get Latest Batch Tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_latest_batch_no_migrations(tracker, mock_sql_driver):
    """Test getting latest batch when no migrations exist."""
    mock_sql_driver.execute_query.return_value = [
        MockRowResult({"latest_batch": 0})
    ]

    result = await tracker.get_latest_batch()

    assert result == 0


@pytest.mark.asyncio
async def test_get_latest_batch_empty_result(tracker, mock_sql_driver):
    """Test getting latest batch when query returns empty."""
    mock_sql_driver.execute_query.return_value = []

    result = await tracker.get_latest_batch()

    assert result == 0


@pytest.mark.asyncio
async def test_get_latest_batch_with_migrations(tracker, mock_sql_driver):
    """Test getting latest batch with existing migrations."""
    mock_sql_driver.execute_query.return_value = [
        MockRowResult({"latest_batch": 5})
    ]

    result = await tracker.get_latest_batch()

    assert result == 5


@pytest.mark.asyncio
async def test_get_latest_batch_uses_coalesce(tracker, mock_sql_driver):
    """Test that COALESCE is used to handle NULL."""
    await tracker.get_latest_batch()

    call_args = mock_sql_driver.execute_query.call_args[0][0]
    assert "COALESCE(MAX(batch), 0)" in call_args


# =============================================================================
# Record Migration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_record_migration_basic(tracker, mock_sql_driver):
    """Test recording a basic migration."""
    await tracker.record_migration(
        name="0001_initial",
        checksum="abc123",
        batch=1,
    )

    mock_sql_driver.execute_query.assert_awaited_once()
    call_args = mock_sql_driver.execute_query.call_args

    assert "INSERT INTO" in call_args[0][0]
    assert call_args[1]["params"] == ("0001_initial", "abc123", 1, None)


@pytest.mark.asyncio
async def test_record_migration_with_sql(tracker, mock_sql_driver):
    """Test recording a migration with executed SQL."""
    executed_sql = "CREATE TABLE users (id SERIAL PRIMARY KEY);"

    await tracker.record_migration(
        name="0001_initial",
        checksum="abc123",
        batch=1,
        executed_sql=executed_sql,
    )

    call_args = mock_sql_driver.execute_query.call_args
    assert call_args[1]["params"] == ("0001_initial", "abc123", 1, executed_sql)


@pytest.mark.asyncio
async def test_record_migration_with_special_characters(tracker, mock_sql_driver):
    """Test recording migration with special characters in name."""
    await tracker.record_migration(
        name="0001_add_user's_table",
        checksum="abc123",
        batch=1,
    )

    call_args = mock_sql_driver.execute_query.call_args
    assert call_args[1]["params"][0] == "0001_add_user's_table"


@pytest.mark.asyncio
async def test_record_migration_with_unicode(tracker, mock_sql_driver):
    """Test recording migration with unicode characters."""
    await tracker.record_migration(
        name="0001_добавить_таблицу",
        checksum="abc123",
        batch=1,
    )

    call_args = mock_sql_driver.execute_query.call_args
    assert call_args[1]["params"][0] == "0001_добавить_таблицу"


@pytest.mark.asyncio
async def test_record_migration_with_long_name(tracker, mock_sql_driver):
    """Test recording migration with maximum length name."""
    long_name = "0001_" + "a" * 250

    await tracker.record_migration(
        name=long_name,
        checksum="abc123",
        batch=1,
    )

    call_args = mock_sql_driver.execute_query.call_args
    assert call_args[1]["params"][0] == long_name


@pytest.mark.asyncio
async def test_record_migration_duplicate_error(tracker, mock_sql_driver):
    """Test that recording duplicate migration raises error."""
    mock_sql_driver.execute_query.side_effect = Exception("duplicate key value")

    with pytest.raises(Exception, match="duplicate key value"):
        await tracker.record_migration(
            name="0001_initial",
            checksum="abc123",
            batch=1,
        )


# =============================================================================
# Remove Migration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_remove_migration_basic(tracker, mock_sql_driver):
    """Test removing a migration record."""
    await tracker.remove_migration("0001_initial")

    mock_sql_driver.execute_query.assert_awaited_once()
    call_args = mock_sql_driver.execute_query.call_args

    assert "DELETE FROM" in call_args[0][0]
    assert call_args[1]["params"] == ("0001_initial",)


@pytest.mark.asyncio
async def test_remove_migration_nonexistent(tracker, mock_sql_driver):
    """Test removing a non-existent migration (should not error)."""
    mock_sql_driver.execute_query.return_value = []

    await tracker.remove_migration("nonexistent")

    mock_sql_driver.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_remove_migration_with_special_characters(tracker, mock_sql_driver):
    """Test removing migration with special characters."""
    await tracker.remove_migration("0001_user's_table")

    call_args = mock_sql_driver.execute_query.call_args
    assert call_args[1]["params"] == ("0001_user's_table",)


# =============================================================================
# Get Migrations in Batch Tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_migrations_in_batch_empty(tracker, mock_sql_driver):
    """Test getting migrations from empty batch."""
    mock_sql_driver.execute_query.return_value = []

    result = await tracker.get_migrations_in_batch(1)

    assert result == []


@pytest.mark.asyncio
async def test_get_migrations_in_batch_none_returned(tracker, mock_sql_driver):
    """Test getting migrations when query returns None."""
    mock_sql_driver.execute_query.return_value = None

    result = await tracker.get_migrations_in_batch(1)

    assert result == []


@pytest.mark.asyncio
async def test_get_migrations_in_batch_single(tracker, mock_sql_driver):
    """Test getting single migration from batch."""
    applied_at = datetime.now(timezone.utc)
    mock_sql_driver.execute_query.return_value = [
        MockRowResult({
            "id": 1,
            "name": "0001_initial",
            "applied_at": applied_at,
            "checksum": "abc123",
            "batch": 1,
        })
    ]

    result = await tracker.get_migrations_in_batch(1)

    assert len(result) == 1
    assert result[0].name == "0001_initial"
    assert result[0].batch == 1


@pytest.mark.asyncio
async def test_get_migrations_in_batch_multiple(tracker, mock_sql_driver):
    """Test getting multiple migrations from batch."""
    applied_at = datetime.now(timezone.utc)
    mock_sql_driver.execute_query.return_value = [
        MockRowResult({"id": 3, "name": "0003_third", "applied_at": applied_at, "checksum": "abc3", "batch": 2}),
        MockRowResult({"id": 2, "name": "0002_second", "applied_at": applied_at, "checksum": "abc2", "batch": 2}),
    ]

    result = await tracker.get_migrations_in_batch(2)

    assert len(result) == 2
    # Should be ordered DESC for rollback
    assert result[0].name == "0003_third"
    assert result[1].name == "0002_second"


@pytest.mark.asyncio
async def test_get_migrations_in_batch_ordered_desc(tracker, mock_sql_driver):
    """Test that batch migrations are ordered DESC for rollback."""
    await tracker.get_migrations_in_batch(1)

    call_args = mock_sql_driver.execute_query.call_args[0][0]
    assert "ORDER BY id DESC" in call_args


@pytest.mark.asyncio
async def test_get_migrations_in_batch_uses_parameter(tracker, mock_sql_driver):
    """Test that batch number is parameterized."""
    await tracker.get_migrations_in_batch(5)

    call_args = mock_sql_driver.execute_query.call_args
    assert call_args[1]["params"] == (5,)


# =============================================================================
# Is Migration Applied Tests
# =============================================================================


@pytest.mark.asyncio
async def test_is_migration_applied_true(tracker, mock_sql_driver):
    """Test checking applied migration returns True."""
    mock_sql_driver.execute_query.return_value = [MockRowResult({"1": 1})]

    result = await tracker.is_migration_applied("0001_initial")

    assert result is True


@pytest.mark.asyncio
async def test_is_migration_applied_false(tracker, mock_sql_driver):
    """Test checking unapplied migration returns False."""
    mock_sql_driver.execute_query.return_value = []

    result = await tracker.is_migration_applied("0001_initial")

    assert result is False


@pytest.mark.asyncio
async def test_is_migration_applied_none_result(tracker, mock_sql_driver):
    """Test checking migration with None result returns False."""
    mock_sql_driver.execute_query.return_value = None

    result = await tracker.is_migration_applied("0001_initial")

    assert result is False


@pytest.mark.asyncio
async def test_is_migration_applied_parameterized(tracker, mock_sql_driver):
    """Test that migration name is parameterized."""
    await tracker.is_migration_applied("0001_initial")

    call_args = mock_sql_driver.execute_query.call_args
    assert call_args[1]["params"] == ("0001_initial",)


# =============================================================================
# Get Migration Status Tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_migration_status_empty(tracker, mock_sql_driver):
    """Test getting status when no migrations exist."""
    mock_sql_driver.execute_query.side_effect = [
        [],  # get_applied_migrations
        [MockRowResult({"latest_batch": 0})],  # get_latest_batch
    ]

    result = await tracker.get_migration_status()

    assert result["total_applied"] == 0
    assert result["latest_batch"] == 0
    assert result["migrations"] == []


@pytest.mark.asyncio
async def test_get_migration_status_with_migrations(tracker, mock_sql_driver):
    """Test getting status with existing migrations."""
    applied_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    mock_sql_driver.execute_query.side_effect = [
        [
            MockRowResult({"id": 1, "name": "0001_initial", "applied_at": applied_at, "checksum": "abc1", "batch": 1}),
            MockRowResult({"id": 2, "name": "0002_add_users", "applied_at": applied_at, "checksum": "abc2", "batch": 2}),
        ],
        [MockRowResult({"latest_batch": 2})],
    ]

    result = await tracker.get_migration_status()

    assert result["total_applied"] == 2
    assert result["latest_batch"] == 2
    assert len(result["migrations"]) == 2
    assert result["migrations"][0]["name"] == "0001_initial"
    assert result["migrations"][1]["name"] == "0002_add_users"


@pytest.mark.asyncio
async def test_get_migration_status_includes_applied_at(tracker, mock_sql_driver):
    """Test that status includes applied_at as ISO format."""
    applied_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    mock_sql_driver.execute_query.side_effect = [
        [MockRowResult({"id": 1, "name": "0001_initial", "applied_at": applied_at, "checksum": "abc1", "batch": 1})],
        [MockRowResult({"latest_batch": 1})],
    ]

    result = await tracker.get_migration_status()

    assert result["migrations"][0]["applied_at"] == applied_at.isoformat()


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_tracker_handles_connection_error(tracker, mock_sql_driver):
    """Test that tracker handles connection errors gracefully."""
    mock_sql_driver.execute_query.side_effect = ConnectionError("Connection lost")

    with pytest.raises(ConnectionError, match="Connection lost"):
        await tracker.get_applied_migrations()


@pytest.mark.asyncio
async def test_tracker_handles_timeout_error(tracker, mock_sql_driver):
    """Test that tracker handles timeout errors."""
    mock_sql_driver.execute_query.side_effect = TimeoutError("Query timeout")

    with pytest.raises(TimeoutError, match="Query timeout"):
        await tracker.record_migration("test", "abc", 1)


@pytest.mark.asyncio
async def test_migration_record_dataclass():
    """Test MigrationRecord dataclass."""
    applied_at = datetime.now(timezone.utc)
    record = MigrationRecord(
        id=1,
        name="0001_initial",
        applied_at=applied_at,
        checksum="abc123",
        batch=1,
    )

    assert record.id == 1
    assert record.name == "0001_initial"
    assert record.applied_at == applied_at
    assert record.checksum == "abc123"
    assert record.batch == 1


@pytest.mark.asyncio
async def test_table_name_constant():
    """Test that table name constant is correct."""
    assert MIGRATION_TABLE_NAME == "_postgres_mcp_migrations"


@pytest.mark.asyncio
async def test_tracker_with_different_schemas(mock_sql_driver):
    """Test tracker works with different schema names."""
    schemas = ["public", "app", "migrations", "my_schema", "UPPERCASE"]

    for schema in schemas:
        tracker = MigrationTracker(sql_driver=mock_sql_driver, schema=schema)
        assert tracker.table_name == f"{schema}.{MIGRATION_TABLE_NAME}"


@pytest.mark.asyncio
async def test_tracker_concurrent_migrations(tracker, mock_sql_driver):
    """Test handling of concurrent migration recording."""
    import asyncio

    async def record_migration(name: str):
        await tracker.record_migration(name=name, checksum=f"hash_{name}", batch=1)

    # Simulate concurrent migration recording
    await asyncio.gather(
        record_migration("0001_first"),
        record_migration("0002_second"),
        record_migration("0003_third"),
    )

    assert mock_sql_driver.execute_query.await_count == 3


@pytest.mark.asyncio
async def test_empty_checksum_handling(tracker, mock_sql_driver):
    """Test handling of empty checksum."""
    await tracker.record_migration(name="0001_test", checksum="", batch=1)

    call_args = mock_sql_driver.execute_query.call_args
    assert call_args[1]["params"][1] == ""


@pytest.mark.asyncio
async def test_large_batch_number(tracker, mock_sql_driver):
    """Test handling of large batch numbers."""
    mock_sql_driver.execute_query.return_value = [
        MockRowResult({"latest_batch": 999999})
    ]

    result = await tracker.get_latest_batch()

    assert result == 999999


@pytest.mark.asyncio
async def test_very_long_sql_content(tracker, mock_sql_driver):
    """Test handling of very long SQL content."""
    long_sql = "CREATE TABLE test (" + ", ".join([f"col{i} TEXT" for i in range(1000)]) + ");"

    await tracker.record_migration(
        name="0001_test",
        checksum="abc123",
        batch=1,
        executed_sql=long_sql,
    )

    call_args = mock_sql_driver.execute_query.call_args
    assert call_args[1]["params"][3] == long_sql

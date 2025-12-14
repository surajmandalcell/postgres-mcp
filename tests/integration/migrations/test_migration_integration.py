"""Integration tests for migrations against real PostgreSQL database.

These tests run against actual PostgreSQL containers and verify:
- Full migration workflow (up, down, status)
- Schema introspection accuracy
- Migration file handling
- Error recovery
- Multi-version PostgreSQL compatibility
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio

from postgres_mcp.sql import DbConnPool
from postgres_mcp.sql import SqlDriver


def create_migration_file(migrations_dir: Path, name: str, up_sql: str, down_sql: str = None, timestamp: str = None) -> Path:
    """Helper to create migration files."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    migration_dir = migrations_dir / f"{timestamp}_{name}"
    migration_dir.mkdir(parents=True, exist_ok=True)

    (migration_dir / "up.sql").write_text(up_sql)
    if down_sql:
        (migration_dir / "down.sql").write_text(down_sql)

    return migration_dir


@pytest.fixture
def temp_migrations_dir():
    """Create a temporary directory for migration files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.usefixtures("test_postgres_connection_string")
class TestMigrationIntegration:
    """Integration tests for migration functionality."""

    @pytest_asyncio.fixture
    async def db_pool(self, test_postgres_connection_string):
        """Create a database connection pool."""
        conn_str, version = test_postgres_connection_string
        pool = DbConnPool()
        await pool.pool_connect(conn_str)
        yield pool
        await pool.close()

    @pytest_asyncio.fixture
    async def sql_driver(self, db_pool):
        """Create a SQL driver from the pool."""
        return SqlDriver(conn=db_pool)

    @pytest.mark.asyncio
    async def test_migration_tracker_table_creation(self, sql_driver):
        """Test that migration tracking table is created correctly."""
        from postgres_mcp.migrations.migration_tracker import MigrationTracker

        tracker = MigrationTracker(sql_driver=sql_driver)
        await tracker.ensure_migration_table()

        # Verify table exists
        result = await sql_driver.execute_query("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = '_postgres_mcp_migrations'
        """)

        assert len(result) == 1
        assert result[0].cells["table_name"] == "_postgres_mcp_migrations"

    @pytest.mark.asyncio
    async def test_migration_record_and_retrieve(self, sql_driver):
        """Test recording and retrieving migrations."""
        from postgres_mcp.migrations.migration_tracker import MigrationTracker

        tracker = MigrationTracker(sql_driver=sql_driver)
        await tracker.ensure_migration_table()

        # Record a migration
        await tracker.record_migration(
            name="20240101000000_test_migration",
            checksum="abc123def456",
            batch=1,
            executed_sql="CREATE TABLE test_table (id INT);"
        )

        # Retrieve migrations
        migrations = await tracker.get_applied_migrations()

        assert len(migrations) >= 1
        test_migration = [m for m in migrations if m.name == "20240101000000_test_migration"]
        assert len(test_migration) == 1
        assert test_migration[0].checksum == "abc123def456"
        assert test_migration[0].batch == 1

    @pytest.mark.asyncio
    async def test_migration_is_applied_check(self, sql_driver):
        """Test checking if migration is applied."""
        from postgres_mcp.migrations.migration_tracker import MigrationTracker

        tracker = MigrationTracker(sql_driver=sql_driver)
        await tracker.ensure_migration_table()

        # Record a migration
        await tracker.record_migration(
            name="20240101000001_check_test",
            checksum="check123",
            batch=1
        )

        # Check if applied
        is_applied = await tracker.is_migration_applied("20240101000001_check_test")
        is_not_applied = await tracker.is_migration_applied("nonexistent_migration")

        assert is_applied is True
        assert is_not_applied is False

    @pytest.mark.asyncio
    async def test_migration_removal(self, sql_driver):
        """Test removing a migration record."""
        from postgres_mcp.migrations.migration_tracker import MigrationTracker

        tracker = MigrationTracker(sql_driver=sql_driver)
        await tracker.ensure_migration_table()

        # Record and then remove
        await tracker.record_migration(
            name="20240101000002_to_remove",
            checksum="remove123",
            batch=1
        )

        assert await tracker.is_migration_applied("20240101000002_to_remove") is True

        await tracker.remove_migration("20240101000002_to_remove")

        assert await tracker.is_migration_applied("20240101000002_to_remove") is False

    @pytest.mark.asyncio
    async def test_batch_tracking(self, sql_driver):
        """Test batch number tracking."""
        from postgres_mcp.migrations.migration_tracker import MigrationTracker

        tracker = MigrationTracker(sql_driver=sql_driver)
        await tracker.ensure_migration_table()

        initial_batch = await tracker.get_latest_batch()

        # Record migrations in different batches
        await tracker.record_migration(name="batch_test_1", checksum="b1", batch=initial_batch + 1)
        await tracker.record_migration(name="batch_test_2", checksum="b2", batch=initial_batch + 1)
        await tracker.record_migration(name="batch_test_3", checksum="b3", batch=initial_batch + 2)

        latest_batch = await tracker.get_latest_batch()
        assert latest_batch == initial_batch + 2

        # Get migrations in batch
        batch_migrations = await tracker.get_migrations_in_batch(initial_batch + 1)
        assert len(batch_migrations) == 2

    @pytest.mark.asyncio
    async def test_schema_pull_tables(self, sql_driver):
        """Test pulling schema information for tables."""
        from postgres_mcp.migrations.schema_pull import SchemaPull

        # Create a test table
        await sql_driver.execute_query("""
            DROP TABLE IF EXISTS schema_pull_test CASCADE;
            CREATE TABLE schema_pull_test (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email TEXT UNIQUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                data JSONB
            );
            CREATE INDEX idx_schema_pull_test_name ON schema_pull_test(name);
            COMMENT ON TABLE schema_pull_test IS 'Test table for schema pulling';
            COMMENT ON COLUMN schema_pull_test.id IS 'Primary key';
        """)

        try:
            schema_pull = SchemaPull(sql_driver=sql_driver)
            tables = await schema_pull.pull_tables("public")

            # Find our test table
            test_table = [t for t in tables if t.name == "schema_pull_test"]
            assert len(test_table) == 1

            table = test_table[0]
            assert table.comment == "Test table for schema pulling"
            assert len(table.columns) == 5

            # Check column details
            id_col = [c for c in table.columns if c.name == "id"][0]
            assert id_col.is_nullable is False

            name_col = [c for c in table.columns if c.name == "name"][0]
            assert name_col.data_type == "character varying"
            assert name_col.character_maximum_length == 255

            # Check constraints
            pk = [c for c in table.constraints if c.constraint_type == "PRIMARY KEY"]
            assert len(pk) == 1
            assert pk[0].columns == ["id"]

            # Check indexes
            assert len(table.indexes) >= 2  # Primary key + custom index

        finally:
            await sql_driver.execute_query("DROP TABLE IF EXISTS schema_pull_test CASCADE;")

    @pytest.mark.asyncio
    async def test_schema_pull_views(self, sql_driver):
        """Test pulling schema information for views."""
        from postgres_mcp.migrations.schema_pull import SchemaPull

        # Create test table and view
        await sql_driver.execute_query("""
            DROP VIEW IF EXISTS test_view;
            DROP TABLE IF EXISTS view_source_table CASCADE;
            CREATE TABLE view_source_table (id INT, name TEXT, active BOOLEAN);
            CREATE VIEW test_view AS SELECT id, name FROM view_source_table WHERE active = true;
        """)

        try:
            schema_pull = SchemaPull(sql_driver=sql_driver)
            views = await schema_pull.pull_views("public")

            test_view = [v for v in views if v.name == "test_view"]
            assert len(test_view) == 1
            assert "SELECT" in test_view[0].definition
            assert test_view[0].is_materialized is False

        finally:
            await sql_driver.execute_query("""
                DROP VIEW IF EXISTS test_view;
                DROP TABLE IF EXISTS view_source_table CASCADE;
            """)

    @pytest.mark.asyncio
    async def test_schema_pull_sequences(self, sql_driver):
        """Test pulling schema information for sequences."""
        from postgres_mcp.migrations.schema_pull import SchemaPull

        # Create a test sequence
        await sql_driver.execute_query("""
            DROP SEQUENCE IF EXISTS test_sequence;
            CREATE SEQUENCE test_sequence
                START WITH 100
                INCREMENT BY 5
                MINVALUE 1
                MAXVALUE 1000
                NO CYCLE;
        """)

        try:
            schema_pull = SchemaPull(sql_driver=sql_driver)
            sequences = await schema_pull.pull_sequences("public")

            test_seq = [s for s in sequences if s.name == "test_sequence"]
            assert len(test_seq) == 1

            seq = test_seq[0]
            assert seq.start_value == 100
            assert seq.increment == 5
            assert seq.min_value == 1
            assert seq.max_value == 1000
            assert seq.cycle is False

        finally:
            await sql_driver.execute_query("DROP SEQUENCE IF EXISTS test_sequence;")

    @pytest.mark.asyncio
    async def test_schema_pull_enums(self, sql_driver):
        """Test pulling schema information for enum types."""
        from postgres_mcp.migrations.schema_pull import SchemaPull

        # Create a test enum
        await sql_driver.execute_query("""
            DROP TYPE IF EXISTS test_status_enum;
            CREATE TYPE test_status_enum AS ENUM ('pending', 'active', 'completed', 'cancelled');
        """)

        try:
            schema_pull = SchemaPull(sql_driver=sql_driver)
            enums = await schema_pull.pull_enums("public")

            test_enum = [e for e in enums if e.name == "test_status_enum"]
            assert len(test_enum) == 1
            assert test_enum[0].values == ["pending", "active", "completed", "cancelled"]

        finally:
            await sql_driver.execute_query("DROP TYPE IF EXISTS test_status_enum;")

    @pytest.mark.asyncio
    async def test_full_migration_workflow(self, sql_driver, temp_migrations_dir):
        """Test complete migration up and down workflow."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        # Create migration files
        create_migration_file(
            temp_migrations_dir,
            "create_users",
            """
            CREATE TABLE migration_test_users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) NOT NULL UNIQUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            DROP TABLE IF EXISTS migration_test_users;
            """,
            "20240101000000"
        )

        create_migration_file(
            temp_migrations_dir,
            "add_name_column",
            """
            ALTER TABLE migration_test_users ADD COLUMN name VARCHAR(255);
            """,
            """
            ALTER TABLE migration_test_users DROP COLUMN name;
            """,
            "20240101000001"
        )

        try:
            manager = MigrationManager(
                sql_driver=sql_driver,
                migrations_dir=temp_migrations_dir
            )

            # Run migrations up
            result = await manager.migrate_up()
            assert result.success is True
            assert result.applied_count == 2

            # Verify table exists with column
            table_check = await sql_driver.execute_query("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'migration_test_users'
                ORDER BY ordinal_position
            """)
            column_names = [r.cells["column_name"] for r in table_check]
            assert "id" in column_names
            assert "email" in column_names
            assert "name" in column_names

            # Check status
            status = await manager.status()
            assert status["applied_count"] == 2
            assert status["pending_count"] == 0

            # Rollback one step
            rollback_result = await manager.rollback(steps=1)
            assert rollback_result.success is True
            assert rollback_result.rolled_back_count == 1

            # Verify name column is gone
            table_check = await sql_driver.execute_query("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'migration_test_users'
            """)
            column_names = [r.cells["column_name"] for r in table_check]
            assert "name" not in column_names

            # Check status shows pending
            status = await manager.status()
            assert status["applied_count"] == 1
            assert status["pending_count"] == 1

        finally:
            # Cleanup
            await sql_driver.execute_query("DROP TABLE IF EXISTS migration_test_users CASCADE;")

    @pytest.mark.asyncio
    async def test_migration_dry_run(self, sql_driver, temp_migrations_dir):
        """Test dry run doesn't actually execute migrations."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(
            temp_migrations_dir,
            "should_not_run",
            "CREATE TABLE dry_run_test (id INT);",
            "DROP TABLE dry_run_test;",
            "20240101000000"
        )

        manager = MigrationManager(
            sql_driver=sql_driver,
            migrations_dir=temp_migrations_dir
        )

        # Run dry run
        result = await manager.migrate_up(dry_run=True)
        assert result.success is True
        assert result.dry_run is True

        # Verify table was NOT created
        table_check = await sql_driver.execute_query("""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = 'dry_run_test'
        """)
        assert len(table_check) == 0

    @pytest.mark.asyncio
    async def test_migration_error_handling(self, sql_driver, temp_migrations_dir):
        """Test that migration errors are handled properly."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(
            temp_migrations_dir,
            "bad_migration",
            "CREATE TABLE test (id INT); THIS IS INVALID SQL;",
            timestamp="20240101000000"
        )

        manager = MigrationManager(
            sql_driver=sql_driver,
            migrations_dir=temp_migrations_dir
        )

        result = await manager.migrate_up()
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_migration_with_foreign_keys(self, sql_driver, temp_migrations_dir):
        """Test migrations with foreign key constraints."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        create_migration_file(
            temp_migrations_dir,
            "create_users",
            "CREATE TABLE fk_test_users (id SERIAL PRIMARY KEY, name TEXT);",
            "DROP TABLE IF EXISTS fk_test_users CASCADE;",
            "20240101000000"
        )

        create_migration_file(
            temp_migrations_dir,
            "create_orders",
            """
            CREATE TABLE fk_test_orders (
                id SERIAL PRIMARY KEY,
                user_id INT REFERENCES fk_test_users(id) ON DELETE CASCADE,
                total NUMERIC(10,2)
            );
            """,
            "DROP TABLE IF EXISTS fk_test_orders CASCADE;",
            "20240101000001"
        )

        try:
            manager = MigrationManager(
                sql_driver=sql_driver,
                migrations_dir=temp_migrations_dir
            )

            result = await manager.migrate_up()
            assert result.success is True

            # Verify FK exists
            fk_check = await sql_driver.execute_query("""
                SELECT constraint_name FROM information_schema.table_constraints
                WHERE table_name = 'fk_test_orders' AND constraint_type = 'FOREIGN KEY'
            """)
            assert len(fk_check) == 1

            # Rollback should work due to CASCADE
            rollback_result = await manager.rollback_all()
            assert rollback_result.success is True

        finally:
            await sql_driver.execute_query("""
                DROP TABLE IF EXISTS fk_test_orders CASCADE;
                DROP TABLE IF EXISTS fk_test_users CASCADE;
            """)

    @pytest.mark.asyncio
    async def test_generate_create_table_sql_accuracy(self, sql_driver):
        """Test that generated CREATE TABLE SQL is accurate."""
        from postgres_mcp.migrations.schema_pull import SchemaPull

        # Create a complex test table
        await sql_driver.execute_query("""
            DROP TABLE IF EXISTS sql_gen_test CASCADE;
            CREATE TABLE sql_gen_test (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                price NUMERIC(10,2) DEFAULT 0.00,
                tags TEXT[],
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                CONSTRAINT price_positive CHECK (price >= 0)
            );
            CREATE UNIQUE INDEX idx_sql_gen_test_name ON sql_gen_test(name);
        """)

        try:
            schema_pull = SchemaPull(sql_driver=sql_driver)
            tables = await schema_pull.pull_tables("public")

            test_table = [t for t in tables if t.name == "sql_gen_test"][0]
            sql = schema_pull.generate_create_table_sql(test_table)

            # Verify SQL contains expected elements
            assert "CREATE TABLE" in sql
            assert "sql_gen_test" in sql
            assert "PRIMARY KEY" in sql
            assert "NOT NULL" in sql
            assert "CHECK" in sql or "price_positive" in sql

        finally:
            await sql_driver.execute_query("DROP TABLE IF EXISTS sql_gen_test CASCADE;")

    @pytest.mark.asyncio
    async def test_migration_status_report(self, sql_driver, temp_migrations_dir):
        """Test migration status reporting."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        # Create migrations
        create_migration_file(temp_migrations_dir, "first", "SELECT 1;", "SELECT -1;", "20240101000000")
        create_migration_file(temp_migrations_dir, "second", "SELECT 2;", "SELECT -2;", "20240101000001")
        create_migration_file(temp_migrations_dir, "third", "SELECT 3;", "SELECT -3;", "20240101000002")

        manager = MigrationManager(
            sql_driver=sql_driver,
            migrations_dir=temp_migrations_dir
        )

        # Initial status - all pending
        status = await manager.status()
        assert status["total_migrations"] == 3
        assert status["pending_count"] == 3
        assert status["applied_count"] == 0

        # Apply first two
        await manager.migrate_up(steps=2)

        # Updated status
        status = await manager.status()
        assert status["applied_count"] == 2
        assert status["pending_count"] == 1

        # Check migration details
        assert len(status["migrations"]) == 3
        applied = [m for m in status["migrations"] if m["status"] == "applied"]
        pending = [m for m in status["migrations"] if m["status"] == "pending"]
        assert len(applied) == 2
        assert len(pending) == 1

    @pytest.mark.asyncio
    async def test_migration_checksum_verification(self, sql_driver, temp_migrations_dir):
        """Test that checksum changes are detected."""
        from postgres_mcp.migrations.migration_manager import MigrationManager

        # Create and apply migration
        migration_path = create_migration_file(
            temp_migrations_dir,
            "checksum_test",
            "SELECT 'original';",
            timestamp="20240101000000"
        )

        manager = MigrationManager(
            sql_driver=sql_driver,
            migrations_dir=temp_migrations_dir
        )

        await manager.migrate_up()

        # Modify the migration file
        (migration_path / "up.sql").write_text("SELECT 'modified';")

        # Status should detect mismatch
        status = await manager.status()
        migration_status = status["migrations"][0]
        assert migration_status.get("checksum_mismatch") is True


@pytest.mark.usefixtures("test_postgres_connection_string")
class TestSchemaComparison:
    """Integration tests for schema comparison."""

    @pytest_asyncio.fixture
    async def db_pool(self, test_postgres_connection_string):
        """Create a database connection pool."""
        conn_str, version = test_postgres_connection_string
        pool = DbConnPool()
        await pool.pool_connect(conn_str)
        yield pool
        await pool.close()

    @pytest_asyncio.fixture
    async def sql_driver(self, db_pool):
        """Create a SQL driver from the pool."""
        return SqlDriver(conn=db_pool)

    @pytest.mark.asyncio
    async def test_schema_diff_detect_new_table(self, sql_driver):
        """Test detecting new table in schema diff."""
        from postgres_mcp.migrations.schema_diff import SchemaDiff
        from postgres_mcp.migrations.schema_pull import SchemaInfo, TableInfo, ColumnInfo

        # Current schema is empty
        source = SchemaInfo()

        # Target has a table
        target = SchemaInfo(tables=[
            TableInfo(
                schema="public",
                name="new_table",
                columns=[
                    ColumnInfo(
                        name="id", data_type="integer", is_nullable=False,
                        column_default=None, character_maximum_length=None,
                        numeric_precision=32, numeric_scale=0,
                        is_identity=True, identity_generation="BY DEFAULT"
                    )
                ]
            )
        ])

        differ = SchemaDiff()
        result = differ.diff(source, target)

        assert len(result.tables_to_create) == 1
        assert result.tables_to_create[0].name == "new_table"

    @pytest.mark.asyncio
    async def test_schema_diff_generate_sql(self, sql_driver):
        """Test generating migration SQL from schema diff."""
        from postgres_mcp.migrations.schema_diff import SchemaDiff
        from postgres_mcp.migrations.schema_pull import SchemaInfo, TableInfo, ColumnInfo

        source = SchemaInfo()
        target = SchemaInfo(tables=[
            TableInfo(
                schema="public",
                name="users",
                columns=[
                    ColumnInfo(
                        name="id", data_type="integer", is_nullable=False,
                        column_default=None, character_maximum_length=None,
                        numeric_precision=32, numeric_scale=0,
                        is_identity=False, identity_generation=None
                    ),
                    ColumnInfo(
                        name="email", data_type="text", is_nullable=False,
                        column_default=None, character_maximum_length=None,
                        numeric_precision=None, numeric_scale=None,
                        is_identity=False, identity_generation=None
                    )
                ]
            )
        ])

        differ = SchemaDiff()
        result = differ.diff(source, target)
        sql = differ.generate_migration_sql(result)

        assert "CREATE TABLE" in sql
        assert "users" in sql
        assert "email" in sql

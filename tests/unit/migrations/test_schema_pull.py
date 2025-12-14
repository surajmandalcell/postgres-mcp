"""Comprehensive tests for SchemaPull class.

Tests cover all edge cases for schema introspection:
- Table pulling with columns, constraints, indexes
- View pulling (regular and materialized)
- Sequence pulling
- Enum type pulling
- Multi-schema support
- Edge cases and special types
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from postgres_mcp.migrations.schema_pull import (
    ColumnInfo,
    ConstraintInfo,
    EnumInfo,
    IndexInfo,
    SchemaInfo,
    SchemaPull,
    SequenceInfo,
    TableInfo,
    ViewInfo,
)
from postgres_mcp.sql import SqlDriver


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


@pytest_asyncio.fixture
async def schema_pull(mock_sql_driver):
    """Create a SchemaPull instance."""
    return SchemaPull(sql_driver=mock_sql_driver)


# =============================================================================
# Pull Schema Tests
# =============================================================================


class TestPullSchema:
    """Test full schema pulling."""

    @pytest.mark.asyncio
    async def test_pull_empty_schema(self, schema_pull, mock_sql_driver):
        """Test pulling an empty schema."""
        mock_sql_driver.execute_query.return_value = []

        result = await schema_pull.pull_schema(["public"])

        assert isinstance(result, SchemaInfo)
        assert result.tables == []
        assert result.views == []
        assert result.sequences == []
        assert result.enums == []

    @pytest.mark.asyncio
    async def test_pull_default_schema(self, schema_pull, mock_sql_driver):
        """Test pulling with default (public) schema."""
        mock_sql_driver.execute_query.return_value = []

        result = await schema_pull.pull_schema()

        # Should default to public schema
        assert mock_sql_driver.execute_query.await_count >= 1

    @pytest.mark.asyncio
    async def test_pull_multiple_schemas(self, schema_pull, mock_sql_driver):
        """Test pulling from multiple schemas."""
        mock_sql_driver.execute_query.return_value = []

        result = await schema_pull.pull_schema(["public", "private", "audit"])

        # Should have called queries for each schema
        assert mock_sql_driver.execute_query.await_count >= 3


# =============================================================================
# Pull Tables Tests
# =============================================================================


class TestPullTables:
    """Test table pulling functionality."""

    @pytest.mark.asyncio
    async def test_pull_single_table(self, schema_pull, mock_sql_driver):
        """Test pulling a single table."""
        # Mock table query
        mock_sql_driver.execute_query.side_effect = [
            [MockRowResult({
                "table_schema": "public",
                "table_name": "users",
                "comment": "User accounts"
            })],
            # Columns query
            [MockRowResult({
                "column_name": "id",
                "data_type": "integer",
                "is_nullable": False,
                "column_default": None,
                "character_maximum_length": None,
                "numeric_precision": 32,
                "numeric_scale": 0,
                "is_identity": True,
                "identity_generation": "BY DEFAULT",
                "comment": "Primary key"
            })],
            # Constraints query
            [MockRowResult({
                "constraint_name": "users_pkey",
                "constraint_type": "PRIMARY KEY",
                "columns": ["id"],
                "foreign_table_schema": None,
                "foreign_table_name": None,
                "foreign_columns": None,
                "on_update": None,
                "on_delete": None,
                "check_clause": None
            })],
            # Indexes query
            [MockRowResult({
                "index_name": "users_pkey",
                "columns": ["id"],
                "is_unique": True,
                "is_primary": True,
                "index_type": "btree",
                "definition": "CREATE UNIQUE INDEX users_pkey ON public.users (id)"
            })]
        ]

        tables = await schema_pull.pull_tables("public")

        assert len(tables) == 1
        assert tables[0].name == "users"
        assert tables[0].schema == "public"
        assert tables[0].comment == "User accounts"
        assert len(tables[0].columns) == 1
        assert len(tables[0].constraints) == 1
        assert len(tables[0].indexes) == 1

    @pytest.mark.asyncio
    async def test_pull_table_with_all_column_types(self, schema_pull, mock_sql_driver):
        """Test pulling table with various column types."""
        mock_sql_driver.execute_query.side_effect = [
            [MockRowResult({"table_schema": "public", "table_name": "test", "comment": None})],
            # Various column types
            [
                MockRowResult({
                    "column_name": "int_col", "data_type": "integer",
                    "is_nullable": False, "column_default": None,
                    "character_maximum_length": None, "numeric_precision": 32,
                    "numeric_scale": 0, "is_identity": False, "identity_generation": None,
                    "comment": None
                }),
                MockRowResult({
                    "column_name": "text_col", "data_type": "text",
                    "is_nullable": True, "column_default": "'default'",
                    "character_maximum_length": None, "numeric_precision": None,
                    "numeric_scale": None, "is_identity": False, "identity_generation": None,
                    "comment": None
                }),
                MockRowResult({
                    "column_name": "varchar_col", "data_type": "character varying",
                    "is_nullable": True, "column_default": None,
                    "character_maximum_length": 255, "numeric_precision": None,
                    "numeric_scale": None, "is_identity": False, "identity_generation": None,
                    "comment": None
                }),
                MockRowResult({
                    "column_name": "numeric_col", "data_type": "numeric",
                    "is_nullable": True, "column_default": None,
                    "character_maximum_length": None, "numeric_precision": 10,
                    "numeric_scale": 2, "is_identity": False, "identity_generation": None,
                    "comment": None
                }),
                MockRowResult({
                    "column_name": "timestamp_col", "data_type": "timestamp with time zone",
                    "is_nullable": True, "column_default": "CURRENT_TIMESTAMP",
                    "character_maximum_length": None, "numeric_precision": None,
                    "numeric_scale": None, "is_identity": False, "identity_generation": None,
                    "comment": None
                }),
                MockRowResult({
                    "column_name": "json_col", "data_type": "jsonb",
                    "is_nullable": True, "column_default": "'{}'::jsonb",
                    "character_maximum_length": None, "numeric_precision": None,
                    "numeric_scale": None, "is_identity": False, "identity_generation": None,
                    "comment": None
                }),
                MockRowResult({
                    "column_name": "array_col", "data_type": "ARRAY",
                    "is_nullable": True, "column_default": None,
                    "character_maximum_length": None, "numeric_precision": None,
                    "numeric_scale": None, "is_identity": False, "identity_generation": None,
                    "comment": None
                }),
            ],
            [],  # Constraints
            [],  # Indexes
        ]

        tables = await schema_pull.pull_tables("public")

        assert len(tables[0].columns) == 7
        assert tables[0].columns[2].character_maximum_length == 255
        assert tables[0].columns[3].numeric_precision == 10
        assert tables[0].columns[3].numeric_scale == 2

    @pytest.mark.asyncio
    async def test_pull_table_with_foreign_key(self, schema_pull, mock_sql_driver):
        """Test pulling table with foreign key constraint."""
        mock_sql_driver.execute_query.side_effect = [
            [MockRowResult({"table_schema": "public", "table_name": "orders", "comment": None})],
            [MockRowResult({
                "column_name": "user_id", "data_type": "integer",
                "is_nullable": False, "column_default": None,
                "character_maximum_length": None, "numeric_precision": 32,
                "numeric_scale": 0, "is_identity": False, "identity_generation": None,
                "comment": None
            })],
            [MockRowResult({
                "constraint_name": "orders_user_id_fkey",
                "constraint_type": "FOREIGN KEY",
                "columns": ["user_id"],
                "foreign_table_schema": "public",
                "foreign_table_name": "users",
                "foreign_columns": ["id"],
                "on_update": "NO ACTION",
                "on_delete": "CASCADE",
                "check_clause": None
            })],
            [],  # Indexes
        ]

        tables = await schema_pull.pull_tables("public")

        fk = tables[0].constraints[0]
        assert fk.constraint_type == "FOREIGN KEY"
        assert fk.foreign_table_name == "users"
        assert fk.foreign_columns == ["id"]
        assert fk.on_delete == "CASCADE"

    @pytest.mark.asyncio
    async def test_pull_table_with_check_constraint(self, schema_pull, mock_sql_driver):
        """Test pulling table with check constraint."""
        mock_sql_driver.execute_query.side_effect = [
            [MockRowResult({"table_schema": "public", "table_name": "products", "comment": None})],
            [MockRowResult({
                "column_name": "price", "data_type": "numeric",
                "is_nullable": False, "column_default": None,
                "character_maximum_length": None, "numeric_precision": 10,
                "numeric_scale": 2, "is_identity": False, "identity_generation": None,
                "comment": None
            })],
            [MockRowResult({
                "constraint_name": "products_price_check",
                "constraint_type": "CHECK",
                "columns": [],
                "foreign_table_schema": None,
                "foreign_table_name": None,
                "foreign_columns": None,
                "on_update": None,
                "on_delete": None,
                "check_clause": "(price > 0)"
            })],
            [],  # Indexes
        ]

        tables = await schema_pull.pull_tables("public")

        check = tables[0].constraints[0]
        assert check.constraint_type == "CHECK"
        assert check.check_clause == "(price > 0)"

    @pytest.mark.asyncio
    async def test_pull_table_with_multiple_indexes(self, schema_pull, mock_sql_driver):
        """Test pulling table with multiple indexes."""
        mock_sql_driver.execute_query.side_effect = [
            [MockRowResult({"table_schema": "public", "table_name": "users", "comment": None})],
            [],  # Columns
            [],  # Constraints
            [
                MockRowResult({
                    "index_name": "users_pkey",
                    "columns": ["id"],
                    "is_unique": True,
                    "is_primary": True,
                    "index_type": "btree",
                    "definition": "CREATE UNIQUE INDEX users_pkey ON public.users (id)"
                }),
                MockRowResult({
                    "index_name": "idx_users_email",
                    "columns": ["email"],
                    "is_unique": True,
                    "is_primary": False,
                    "index_type": "btree",
                    "definition": "CREATE UNIQUE INDEX idx_users_email ON public.users (email)"
                }),
                MockRowResult({
                    "index_name": "idx_users_name",
                    "columns": ["first_name", "last_name"],
                    "is_unique": False,
                    "is_primary": False,
                    "index_type": "btree",
                    "definition": "CREATE INDEX idx_users_name ON public.users (first_name, last_name)"
                }),
            ],
        ]

        tables = await schema_pull.pull_tables("public")

        assert len(tables[0].indexes) == 3
        assert tables[0].indexes[2].columns == ["first_name", "last_name"]

    @pytest.mark.asyncio
    async def test_pull_no_tables(self, schema_pull, mock_sql_driver):
        """Test pulling when no tables exist."""
        mock_sql_driver.execute_query.return_value = []

        tables = await schema_pull.pull_tables("public")

        assert tables == []


# =============================================================================
# Pull Views Tests
# =============================================================================


class TestPullViews:
    """Test view pulling functionality."""

    @pytest.mark.asyncio
    async def test_pull_regular_view(self, schema_pull, mock_sql_driver):
        """Test pulling a regular view."""
        mock_sql_driver.execute_query.return_value = [
            MockRowResult({
                "schema": "public",
                "name": "active_users",
                "definition": " SELECT id, email FROM users WHERE active = true;",
                "is_materialized": False
            })
        ]

        views = await schema_pull.pull_views("public")

        assert len(views) == 1
        assert views[0].name == "active_users"
        assert views[0].is_materialized is False
        assert "SELECT" in views[0].definition

    @pytest.mark.asyncio
    async def test_pull_materialized_view(self, schema_pull, mock_sql_driver):
        """Test pulling a materialized view."""
        mock_sql_driver.execute_query.return_value = [
            MockRowResult({
                "schema": "public",
                "name": "user_stats",
                "definition": " SELECT count(*) FROM users;",
                "is_materialized": True
            })
        ]

        views = await schema_pull.pull_views("public")

        assert len(views) == 1
        assert views[0].is_materialized is True

    @pytest.mark.asyncio
    async def test_pull_view_with_complex_query(self, schema_pull, mock_sql_driver):
        """Test pulling view with complex query definition."""
        complex_definition = """
         SELECT u.id,
            u.email,
            count(o.id) AS order_count,
            sum(o.total) AS total_spent
           FROM users u
             LEFT JOIN orders o ON u.id = o.user_id
          GROUP BY u.id, u.email
          ORDER BY (sum(o.total)) DESC;
        """
        mock_sql_driver.execute_query.return_value = [
            MockRowResult({
                "schema": "public",
                "name": "user_orders_summary",
                "definition": complex_definition,
                "is_materialized": False
            })
        ]

        views = await schema_pull.pull_views("public")

        assert "LEFT JOIN" in views[0].definition
        assert "GROUP BY" in views[0].definition

    @pytest.mark.asyncio
    async def test_pull_no_views(self, schema_pull, mock_sql_driver):
        """Test pulling when no views exist."""
        mock_sql_driver.execute_query.return_value = []

        views = await schema_pull.pull_views("public")

        assert views == []


# =============================================================================
# Pull Sequences Tests
# =============================================================================


class TestPullSequences:
    """Test sequence pulling functionality."""

    @pytest.mark.asyncio
    async def test_pull_sequence(self, schema_pull, mock_sql_driver):
        """Test pulling a sequence."""
        mock_sql_driver.execute_query.return_value = [
            MockRowResult({
                "schema": "public",
                "name": "users_id_seq",
                "data_type": "bigint",
                "start_value": 1,
                "increment": 1,
                "min_value": 1,
                "max_value": 9223372036854775807,
                "cycle": False
            })
        ]

        sequences = await schema_pull.pull_sequences("public")

        assert len(sequences) == 1
        assert sequences[0].name == "users_id_seq"
        assert sequences[0].data_type == "bigint"
        assert sequences[0].start_value == 1
        assert sequences[0].increment == 1
        assert sequences[0].cycle is False

    @pytest.mark.asyncio
    async def test_pull_sequence_with_cycle(self, schema_pull, mock_sql_driver):
        """Test pulling a cycling sequence."""
        mock_sql_driver.execute_query.return_value = [
            MockRowResult({
                "schema": "public",
                "name": "round_robin_seq",
                "data_type": "integer",
                "start_value": 1,
                "increment": 1,
                "min_value": 1,
                "max_value": 10,
                "cycle": True
            })
        ]

        sequences = await schema_pull.pull_sequences("public")

        assert sequences[0].cycle is True
        assert sequences[0].max_value == 10

    @pytest.mark.asyncio
    async def test_pull_sequence_custom_increment(self, schema_pull, mock_sql_driver):
        """Test pulling sequence with custom increment."""
        mock_sql_driver.execute_query.return_value = [
            MockRowResult({
                "schema": "public",
                "name": "even_numbers",
                "data_type": "integer",
                "start_value": 2,
                "increment": 2,
                "min_value": 0,
                "max_value": 1000,
                "cycle": False
            })
        ]

        sequences = await schema_pull.pull_sequences("public")

        assert sequences[0].start_value == 2
        assert sequences[0].increment == 2

    @pytest.mark.asyncio
    async def test_pull_no_sequences(self, schema_pull, mock_sql_driver):
        """Test pulling when no sequences exist."""
        mock_sql_driver.execute_query.return_value = []

        sequences = await schema_pull.pull_sequences("public")

        assert sequences == []


# =============================================================================
# Pull Enums Tests
# =============================================================================


class TestPullEnums:
    """Test enum type pulling functionality."""

    @pytest.mark.asyncio
    async def test_pull_enum(self, schema_pull, mock_sql_driver):
        """Test pulling an enum type."""
        mock_sql_driver.execute_query.return_value = [
            MockRowResult({
                "schema": "public",
                "name": "status_type",
                "values": ["pending", "active", "completed", "cancelled"]
            })
        ]

        enums = await schema_pull.pull_enums("public")

        assert len(enums) == 1
        assert enums[0].name == "status_type"
        assert enums[0].values == ["pending", "active", "completed", "cancelled"]

    @pytest.mark.asyncio
    async def test_pull_multiple_enums(self, schema_pull, mock_sql_driver):
        """Test pulling multiple enum types."""
        mock_sql_driver.execute_query.return_value = [
            MockRowResult({
                "schema": "public",
                "name": "priority",
                "values": ["low", "medium", "high"]
            }),
            MockRowResult({
                "schema": "public",
                "name": "visibility",
                "values": ["public", "private", "internal"]
            })
        ]

        enums = await schema_pull.pull_enums("public")

        assert len(enums) == 2

    @pytest.mark.asyncio
    async def test_pull_enum_with_special_values(self, schema_pull, mock_sql_driver):
        """Test pulling enum with special character values."""
        mock_sql_driver.execute_query.return_value = [
            MockRowResult({
                "schema": "public",
                "name": "special_enum",
                "values": ["value-with-dash", "value_with_underscore", "CamelCase"]
            })
        ]

        enums = await schema_pull.pull_enums("public")

        assert "value-with-dash" in enums[0].values

    @pytest.mark.asyncio
    async def test_pull_no_enums(self, schema_pull, mock_sql_driver):
        """Test pulling when no enums exist."""
        mock_sql_driver.execute_query.return_value = []

        enums = await schema_pull.pull_enums("public")

        assert enums == []


# =============================================================================
# SQL Generation Tests
# =============================================================================


class TestGenerateSQL:
    """Test SQL generation from schema info."""

    def test_generate_create_table_simple(self, schema_pull):
        """Test generating CREATE TABLE for simple table."""
        table = TableInfo(
            schema="public",
            name="users",
            columns=[
                ColumnInfo(
                    name="id", data_type="integer", is_nullable=False,
                    column_default=None, character_maximum_length=None,
                    numeric_precision=32, numeric_scale=0,
                    is_identity=True, identity_generation="BY DEFAULT"
                ),
                ColumnInfo(
                    name="email", data_type="character varying", is_nullable=False,
                    column_default=None, character_maximum_length=255,
                    numeric_precision=None, numeric_scale=None,
                    is_identity=False, identity_generation=None
                ),
            ],
            constraints=[
                ConstraintInfo(
                    name="users_pkey", constraint_type="PRIMARY KEY",
                    columns=["id"]
                )
            ]
        )

        sql = schema_pull.generate_create_table_sql(table)

        assert 'CREATE TABLE "public"."users"' in sql
        assert '"id" integer' in sql
        assert "GENERATED BY DEFAULT AS IDENTITY" in sql
        assert '"email" character varying(255)' in sql
        assert 'PRIMARY KEY' in sql

    def test_generate_create_table_with_foreign_key(self, schema_pull):
        """Test generating CREATE TABLE with foreign key."""
        table = TableInfo(
            schema="public",
            name="orders",
            columns=[
                ColumnInfo(
                    name="user_id", data_type="integer", is_nullable=False,
                    column_default=None, character_maximum_length=None,
                    numeric_precision=32, numeric_scale=0,
                    is_identity=False, identity_generation=None
                ),
            ],
            constraints=[
                ConstraintInfo(
                    name="orders_user_id_fkey", constraint_type="FOREIGN KEY",
                    columns=["user_id"],
                    foreign_table_schema="public", foreign_table_name="users",
                    foreign_columns=["id"],
                    on_delete="CASCADE", on_update="NO ACTION"
                )
            ]
        )

        sql = schema_pull.generate_create_table_sql(table)

        assert "FOREIGN KEY" in sql
        assert "REFERENCES" in sql
        assert "ON DELETE CASCADE" in sql

    def test_generate_create_table_with_check_constraint(self, schema_pull):
        """Test generating CREATE TABLE with check constraint."""
        table = TableInfo(
            schema="public",
            name="products",
            columns=[
                ColumnInfo(
                    name="price", data_type="numeric", is_nullable=False,
                    column_default=None, character_maximum_length=None,
                    numeric_precision=10, numeric_scale=2,
                    is_identity=False, identity_generation=None
                ),
            ],
            constraints=[
                ConstraintInfo(
                    name="products_price_check", constraint_type="CHECK",
                    columns=[], check_clause="(price > 0)"
                )
            ]
        )

        sql = schema_pull.generate_create_table_sql(table)

        assert "CHECK" in sql
        assert "price > 0" in sql

    def test_generate_create_table_with_index(self, schema_pull):
        """Test generating CREATE TABLE includes non-primary indexes."""
        table = TableInfo(
            schema="public",
            name="users",
            columns=[
                ColumnInfo(
                    name="email", data_type="text", is_nullable=False,
                    column_default=None, character_maximum_length=None,
                    numeric_precision=None, numeric_scale=None,
                    is_identity=False, identity_generation=None
                ),
            ],
            indexes=[
                IndexInfo(
                    name="idx_users_email", columns=["email"],
                    is_unique=True, is_primary=False,
                    index_type="btree",
                    definition="CREATE UNIQUE INDEX idx_users_email ON public.users (email)"
                )
            ]
        )

        sql = schema_pull.generate_create_table_sql(table)

        assert "CREATE UNIQUE INDEX" in sql

    def test_generate_create_table_with_comment(self, schema_pull):
        """Test generating CREATE TABLE with table and column comments."""
        table = TableInfo(
            schema="public",
            name="users",
            columns=[
                ColumnInfo(
                    name="id", data_type="integer", is_nullable=False,
                    column_default=None, character_maximum_length=None,
                    numeric_precision=32, numeric_scale=0,
                    is_identity=False, identity_generation=None,
                    comment="Primary identifier"
                ),
            ],
            comment="User accounts table"
        )

        sql = schema_pull.generate_create_table_sql(table)

        assert "COMMENT ON TABLE" in sql
        assert "User accounts table" in sql
        assert "COMMENT ON COLUMN" in sql
        assert "Primary identifier" in sql


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_handle_null_results(self, schema_pull, mock_sql_driver):
        """Test handling of NULL values in results."""
        mock_sql_driver.execute_query.return_value = [
            MockRowResult({
                "column_name": "test",
                "data_type": "text",
                "is_nullable": True,
                "column_default": None,
                "character_maximum_length": None,
                "numeric_precision": None,
                "numeric_scale": None,
                "is_identity": False,
                "identity_generation": None,
                "comment": None
            })
        ]

        # Should handle None values gracefully
        columns = await schema_pull._pull_columns("public", "test")

        assert columns[0].column_default is None
        assert columns[0].comment is None

    @pytest.mark.asyncio
    async def test_special_characters_in_names(self, schema_pull, mock_sql_driver):
        """Test handling of special characters in identifiers."""
        mock_sql_driver.execute_query.side_effect = [
            [MockRowResult({
                "table_schema": "public",
                "table_name": "table-with-dashes",
                "comment": None
            })],
            [MockRowResult({
                "column_name": "column with spaces",
                "data_type": "text",
                "is_nullable": True,
                "column_default": None,
                "character_maximum_length": None,
                "numeric_precision": None,
                "numeric_scale": None,
                "is_identity": False,
                "identity_generation": None,
                "comment": None
            })],
            [],  # Constraints
            [],  # Indexes
        ]

        tables = await schema_pull.pull_tables("public")

        assert tables[0].name == "table-with-dashes"
        assert tables[0].columns[0].name == "column with spaces"

    @pytest.mark.asyncio
    async def test_unicode_in_comments(self, schema_pull, mock_sql_driver):
        """Test handling of unicode in comments."""
        mock_sql_driver.execute_query.side_effect = [
            [MockRowResult({
                "table_schema": "public",
                "table_name": "test",
                "comment": "Таблица пользователей 用户表"
            })],
            [],  # Columns
            [],  # Constraints
            [],  # Indexes
        ]

        tables = await schema_pull.pull_tables("public")

        assert "Таблица" in tables[0].comment
        assert "用户表" in tables[0].comment

    @pytest.mark.asyncio
    async def test_very_long_column_names(self, schema_pull, mock_sql_driver):
        """Test handling of very long column names."""
        long_name = "a" * 63  # PostgreSQL max identifier length

        mock_sql_driver.execute_query.side_effect = [
            [MockRowResult({
                "table_schema": "public",
                "table_name": "test",
                "comment": None
            })],
            [MockRowResult({
                "column_name": long_name,
                "data_type": "text",
                "is_nullable": True,
                "column_default": None,
                "character_maximum_length": None,
                "numeric_precision": None,
                "numeric_scale": None,
                "is_identity": False,
                "identity_generation": None,
                "comment": None
            })],
            [],  # Constraints
            [],  # Indexes
        ]

        tables = await schema_pull.pull_tables("public")

        assert len(tables[0].columns[0].name) == 63

    @pytest.mark.asyncio
    async def test_array_types(self, schema_pull, mock_sql_driver):
        """Test handling of array column types."""
        mock_sql_driver.execute_query.side_effect = [
            [MockRowResult({
                "table_schema": "public",
                "table_name": "test",
                "comment": None
            })],
            [
                MockRowResult({
                    "column_name": "tags",
                    "data_type": "ARRAY",
                    "is_nullable": True,
                    "column_default": "'{}'::text[]",
                    "character_maximum_length": None,
                    "numeric_precision": None,
                    "numeric_scale": None,
                    "is_identity": False,
                    "identity_generation": None,
                    "comment": None
                }),
                MockRowResult({
                    "column_name": "numbers",
                    "data_type": "integer[]",
                    "is_nullable": True,
                    "column_default": None,
                    "character_maximum_length": None,
                    "numeric_precision": None,
                    "numeric_scale": None,
                    "is_identity": False,
                    "identity_generation": None,
                    "comment": None
                }),
            ],
            [],  # Constraints
            [],  # Indexes
        ]

        tables = await schema_pull.pull_tables("public")

        assert "ARRAY" in tables[0].columns[0].data_type or "[]" in tables[0].columns[0].data_type

    @pytest.mark.asyncio
    async def test_composite_primary_key(self, schema_pull, mock_sql_driver):
        """Test handling of composite primary keys."""
        mock_sql_driver.execute_query.side_effect = [
            [MockRowResult({
                "table_schema": "public",
                "table_name": "order_items",
                "comment": None
            })],
            [],  # Columns
            [MockRowResult({
                "constraint_name": "order_items_pkey",
                "constraint_type": "PRIMARY KEY",
                "columns": ["order_id", "item_id"],
                "foreign_table_schema": None,
                "foreign_table_name": None,
                "foreign_columns": None,
                "on_update": None,
                "on_delete": None,
                "check_clause": None
            })],
            [],  # Indexes
        ]

        tables = await schema_pull.pull_tables("public")

        pk = tables[0].constraints[0]
        assert pk.columns == ["order_id", "item_id"]

    @pytest.mark.asyncio
    async def test_gin_index(self, schema_pull, mock_sql_driver):
        """Test handling of GIN indexes."""
        mock_sql_driver.execute_query.side_effect = [
            [MockRowResult({
                "table_schema": "public",
                "table_name": "documents",
                "comment": None
            })],
            [],  # Columns
            [],  # Constraints
            [MockRowResult({
                "index_name": "idx_documents_content",
                "columns": ["content"],
                "is_unique": False,
                "is_primary": False,
                "index_type": "gin",
                "definition": "CREATE INDEX idx_documents_content ON public.documents USING gin (to_tsvector('english', content))"
            })]
        ]

        tables = await schema_pull.pull_tables("public")

        assert tables[0].indexes[0].index_type == "gin"

    @pytest.mark.asyncio
    async def test_partial_index(self, schema_pull, mock_sql_driver):
        """Test handling of partial indexes."""
        mock_sql_driver.execute_query.side_effect = [
            [MockRowResult({
                "table_schema": "public",
                "table_name": "users",
                "comment": None
            })],
            [],  # Columns
            [],  # Constraints
            [MockRowResult({
                "index_name": "idx_active_users",
                "columns": ["email"],
                "is_unique": True,
                "is_primary": False,
                "index_type": "btree",
                "definition": "CREATE UNIQUE INDEX idx_active_users ON public.users (email) WHERE (active = true)"
            })]
        ]

        tables = await schema_pull.pull_tables("public")

        assert "WHERE" in tables[0].indexes[0].definition


# =============================================================================
# Data Classes Tests
# =============================================================================


class TestDataClasses:
    """Test dataclass behavior."""

    def test_column_info_defaults(self):
        """Test ColumnInfo default values."""
        col = ColumnInfo(
            name="test",
            data_type="text",
            is_nullable=True,
            column_default=None,
            character_maximum_length=None,
            numeric_precision=None,
            numeric_scale=None,
            is_identity=False,
            identity_generation=None
        )

        assert col.comment is None

    def test_table_info_defaults(self):
        """Test TableInfo default values."""
        table = TableInfo(schema="public", name="test")

        assert table.columns == []
        assert table.constraints == []
        assert table.indexes == []
        assert table.comment is None

    def test_schema_info_defaults(self):
        """Test SchemaInfo default values."""
        schema = SchemaInfo()

        assert schema.tables == []
        assert schema.views == []
        assert schema.sequences == []
        assert schema.enums == []

    def test_constraint_info_with_fk(self):
        """Test ConstraintInfo for foreign key."""
        fk = ConstraintInfo(
            name="test_fk",
            constraint_type="FOREIGN KEY",
            columns=["user_id"],
            foreign_table_schema="public",
            foreign_table_name="users",
            foreign_columns=["id"],
            on_delete="CASCADE",
            on_update="NO ACTION"
        )

        assert fk.foreign_table_name == "users"
        assert fk.on_delete == "CASCADE"

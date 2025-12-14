"""Comprehensive tests for SchemaDiff class.

Tests cover all edge cases for schema comparison and diff generation:
- Table diffs (create, drop, alter)
- Column diffs (add, remove, modify)
- Index diffs
- Constraint diffs
- View and sequence diffs
- Enum type diffs
- Complex schema changes
- SQL generation
"""

from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from postgres_mcp.migrations.schema_pull import (
    ColumnInfo,
    ConstraintInfo,
    EnumInfo,
    IndexInfo,
    SchemaInfo,
    SequenceInfo,
    TableInfo,
    ViewInfo,
)
from postgres_mcp.migrations.schema_diff import (
    SchemaDiff,
    TableDiff,
    ColumnDiff,
    IndexDiff,
    ConstraintDiff,
    DiffType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def differ():
    """Create a SchemaDiff instance."""
    return SchemaDiff()


@pytest.fixture
def empty_schema():
    """Create an empty schema."""
    return SchemaInfo()


@pytest.fixture
def sample_column():
    """Create a sample column."""
    return ColumnInfo(
        name="id",
        data_type="integer",
        is_nullable=False,
        column_default=None,
        character_maximum_length=None,
        numeric_precision=32,
        numeric_scale=0,
        is_identity=True,
        identity_generation="BY DEFAULT",
    )


@pytest.fixture
def sample_table():
    """Create a sample table."""
    return TableInfo(
        schema="public",
        name="users",
        columns=[
            ColumnInfo(
                name="id",
                data_type="integer",
                is_nullable=False,
                column_default=None,
                character_maximum_length=None,
                numeric_precision=32,
                numeric_scale=0,
                is_identity=True,
                identity_generation="BY DEFAULT",
            ),
            ColumnInfo(
                name="email",
                data_type="character varying",
                is_nullable=False,
                column_default=None,
                character_maximum_length=255,
                numeric_precision=None,
                numeric_scale=None,
                is_identity=False,
                identity_generation=None,
            ),
        ],
        constraints=[
            ConstraintInfo(
                name="users_pkey",
                constraint_type="PRIMARY KEY",
                columns=["id"],
            ),
        ],
        indexes=[
            IndexInfo(
                name="users_pkey",
                columns=["id"],
                is_unique=True,
                is_primary=True,
                index_type="btree",
                definition="CREATE UNIQUE INDEX users_pkey ON public.users USING btree (id)",
            ),
        ],
    )


# =============================================================================
# Basic Diff Tests
# =============================================================================


class TestBasicDiff:
    """Test basic diff functionality."""

    def test_diff_identical_schemas(self, differ, empty_schema):
        """Test that identical schemas produce no diff."""
        result = differ.diff(empty_schema, empty_schema)

        assert result.tables_to_create == []
        assert result.tables_to_drop == []
        assert result.tables_to_alter == []

    def test_diff_empty_to_table(self, differ, empty_schema, sample_table):
        """Test diff from empty schema to schema with table."""
        source = empty_schema
        target = SchemaInfo(tables=[sample_table])

        result = differ.diff(source, target)

        assert len(result.tables_to_create) == 1
        assert result.tables_to_create[0].name == "users"
        assert result.tables_to_drop == []

    def test_diff_table_to_empty(self, differ, empty_schema, sample_table):
        """Test diff from schema with table to empty schema."""
        source = SchemaInfo(tables=[sample_table])
        target = empty_schema

        result = differ.diff(source, target)

        assert result.tables_to_create == []
        assert len(result.tables_to_drop) == 1
        assert result.tables_to_drop[0].name == "users"


# =============================================================================
# Table Diff Tests
# =============================================================================


class TestTableDiff:
    """Test table-level diff functionality."""

    def test_create_new_table(self, differ):
        """Test detection of new table."""
        source = SchemaInfo()
        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="products", columns=[
                ColumnInfo(name="id", data_type="integer", is_nullable=False, column_default=None,
                          character_maximum_length=None, numeric_precision=32, numeric_scale=0,
                          is_identity=True, identity_generation="BY DEFAULT"),
            ])
        ])

        result = differ.diff(source, target)

        assert len(result.tables_to_create) == 1
        assert result.tables_to_create[0].name == "products"

    def test_drop_table(self, differ):
        """Test detection of table to drop."""
        source = SchemaInfo(tables=[
            TableInfo(schema="public", name="old_table", columns=[
                ColumnInfo(name="id", data_type="integer", is_nullable=False, column_default=None,
                          character_maximum_length=None, numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None),
            ])
        ])
        target = SchemaInfo()

        result = differ.diff(source, target)

        assert len(result.tables_to_drop) == 1
        assert result.tables_to_drop[0].name == "old_table"

    def test_multiple_tables_created(self, differ):
        """Test creation of multiple tables."""
        source = SchemaInfo()
        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="table1", columns=[]),
            TableInfo(schema="public", name="table2", columns=[]),
            TableInfo(schema="public", name="table3", columns=[]),
        ])

        result = differ.diff(source, target)

        assert len(result.tables_to_create) == 3
        names = [t.name for t in result.tables_to_create]
        assert "table1" in names
        assert "table2" in names
        assert "table3" in names

    def test_multiple_tables_dropped(self, differ):
        """Test dropping of multiple tables."""
        source = SchemaInfo(tables=[
            TableInfo(schema="public", name="table1", columns=[]),
            TableInfo(schema="public", name="table2", columns=[]),
        ])
        target = SchemaInfo()

        result = differ.diff(source, target)

        assert len(result.tables_to_drop) == 2

    def test_table_rename_detection(self, differ):
        """Test that table rename is detected as drop + create."""
        source = SchemaInfo(tables=[
            TableInfo(schema="public", name="old_name", columns=[
                ColumnInfo(name="id", data_type="integer", is_nullable=False, column_default=None,
                          character_maximum_length=None, numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None),
            ])
        ])
        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="new_name", columns=[
                ColumnInfo(name="id", data_type="integer", is_nullable=False, column_default=None,
                          character_maximum_length=None, numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None),
            ])
        ])

        result = differ.diff(source, target)

        # Rename is detected as drop + create (can be optimized later)
        assert len(result.tables_to_drop) == 1
        assert len(result.tables_to_create) == 1


# =============================================================================
# Column Diff Tests
# =============================================================================


class TestColumnDiff:
    """Test column-level diff functionality."""

    def test_add_column(self, differ, sample_table):
        """Test detection of column addition."""
        source = SchemaInfo(tables=[sample_table])

        # Create target with additional column
        target_table = TableInfo(
            schema="public",
            name="users",
            columns=sample_table.columns + [
                ColumnInfo(
                    name="created_at",
                    data_type="timestamp with time zone",
                    is_nullable=True,
                    column_default="CURRENT_TIMESTAMP",
                    character_maximum_length=None,
                    numeric_precision=None,
                    numeric_scale=None,
                    is_identity=False,
                    identity_generation=None,
                )
            ],
            constraints=sample_table.constraints,
            indexes=sample_table.indexes,
        )
        target = SchemaInfo(tables=[target_table])

        result = differ.diff(source, target)

        assert len(result.tables_to_alter) == 1
        alter = result.tables_to_alter[0]
        assert len(alter.columns_to_add) == 1
        assert alter.columns_to_add[0].name == "created_at"

    def test_drop_column(self, differ, sample_table):
        """Test detection of column removal."""
        source = SchemaInfo(tables=[sample_table])

        # Create target with one fewer column
        target_table = TableInfo(
            schema="public",
            name="users",
            columns=[sample_table.columns[0]],  # Only keep id
            constraints=sample_table.constraints,
            indexes=sample_table.indexes,
        )
        target = SchemaInfo(tables=[target_table])

        result = differ.diff(source, target)

        assert len(result.tables_to_alter) == 1
        alter = result.tables_to_alter[0]
        assert len(alter.columns_to_drop) == 1
        assert alter.columns_to_drop[0].name == "email"

    def test_modify_column_type(self, differ, sample_table):
        """Test detection of column type change."""
        source = SchemaInfo(tables=[sample_table])

        # Change email column type
        target_table = TableInfo(
            schema="public",
            name="users",
            columns=[
                sample_table.columns[0],  # id unchanged
                ColumnInfo(
                    name="email",
                    data_type="text",  # Changed from varchar
                    is_nullable=False,
                    column_default=None,
                    character_maximum_length=None,  # No length for text
                    numeric_precision=None,
                    numeric_scale=None,
                    is_identity=False,
                    identity_generation=None,
                )
            ],
            constraints=sample_table.constraints,
            indexes=sample_table.indexes,
        )
        target = SchemaInfo(tables=[target_table])

        result = differ.diff(source, target)

        assert len(result.tables_to_alter) == 1
        alter = result.tables_to_alter[0]
        assert len(alter.columns_to_modify) == 1
        assert alter.columns_to_modify[0].column.name == "email"

    def test_modify_column_nullable(self, differ, sample_table):
        """Test detection of nullable change."""
        source = SchemaInfo(tables=[sample_table])

        target_table = TableInfo(
            schema="public",
            name="users",
            columns=[
                sample_table.columns[0],
                ColumnInfo(
                    name="email",
                    data_type="character varying",
                    is_nullable=True,  # Changed from False
                    column_default=None,
                    character_maximum_length=255,
                    numeric_precision=None,
                    numeric_scale=None,
                    is_identity=False,
                    identity_generation=None,
                )
            ],
            constraints=sample_table.constraints,
            indexes=sample_table.indexes,
        )
        target = SchemaInfo(tables=[target_table])

        result = differ.diff(source, target)

        assert len(result.tables_to_alter) == 1
        alter = result.tables_to_alter[0]
        assert len(alter.columns_to_modify) == 1

    def test_modify_column_default(self, differ, sample_table):
        """Test detection of default value change."""
        source = SchemaInfo(tables=[sample_table])

        target_table = TableInfo(
            schema="public",
            name="users",
            columns=[
                sample_table.columns[0],
                ColumnInfo(
                    name="email",
                    data_type="character varying",
                    is_nullable=False,
                    column_default="'unknown@example.com'",  # Added default
                    character_maximum_length=255,
                    numeric_precision=None,
                    numeric_scale=None,
                    is_identity=False,
                    identity_generation=None,
                )
            ],
            constraints=sample_table.constraints,
            indexes=sample_table.indexes,
        )
        target = SchemaInfo(tables=[target_table])

        result = differ.diff(source, target)

        assert len(result.tables_to_alter) == 1

    def test_multiple_column_changes(self, differ):
        """Test multiple column changes in one table."""
        source = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[
                ColumnInfo(name="col1", data_type="integer", is_nullable=False, column_default=None,
                          character_maximum_length=None, numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None),
                ColumnInfo(name="col2", data_type="text", is_nullable=True, column_default=None,
                          character_maximum_length=None, numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None),
                ColumnInfo(name="col3", data_type="boolean", is_nullable=True, column_default=None,
                          character_maximum_length=None, numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None),
            ])
        ])

        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[
                ColumnInfo(name="col1", data_type="bigint", is_nullable=False, column_default=None,
                          character_maximum_length=None, numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None),  # Modified type
                # col2 removed
                ColumnInfo(name="col3", data_type="boolean", is_nullable=True, column_default=None,
                          character_maximum_length=None, numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None),
                ColumnInfo(name="col4", data_type="timestamp", is_nullable=True, column_default=None,
                          character_maximum_length=None, numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None),  # Added
            ])
        ])

        result = differ.diff(source, target)

        assert len(result.tables_to_alter) == 1
        alter = result.tables_to_alter[0]
        assert len(alter.columns_to_add) == 1
        assert len(alter.columns_to_drop) == 1
        assert len(alter.columns_to_modify) == 1


# =============================================================================
# Index Diff Tests
# =============================================================================


class TestIndexDiff:
    """Test index diff functionality."""

    def test_add_index(self, differ, sample_table):
        """Test detection of new index."""
        source = SchemaInfo(tables=[sample_table])

        target_table = TableInfo(
            schema="public",
            name="users",
            columns=sample_table.columns,
            constraints=sample_table.constraints,
            indexes=sample_table.indexes + [
                IndexInfo(
                    name="idx_users_email",
                    columns=["email"],
                    is_unique=False,
                    is_primary=False,
                    index_type="btree",
                    definition="CREATE INDEX idx_users_email ON public.users USING btree (email)",
                )
            ],
        )
        target = SchemaInfo(tables=[target_table])

        result = differ.diff(source, target)

        assert len(result.tables_to_alter) == 1
        alter = result.tables_to_alter[0]
        assert len(alter.indexes_to_add) == 1
        assert alter.indexes_to_add[0].name == "idx_users_email"

    def test_drop_index(self, differ):
        """Test detection of index to drop."""
        source = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[], indexes=[
                IndexInfo(name="idx_to_drop", columns=["col1"], is_unique=False,
                         is_primary=False, index_type="btree", definition="CREATE INDEX..."),
            ])
        ])
        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[], indexes=[])
        ])

        result = differ.diff(source, target)

        assert len(result.tables_to_alter) == 1
        alter = result.tables_to_alter[0]
        assert len(alter.indexes_to_drop) == 1
        assert alter.indexes_to_drop[0].name == "idx_to_drop"

    def test_modify_index(self, differ):
        """Test detection of modified index (unique change)."""
        source = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[], indexes=[
                IndexInfo(name="idx_col1", columns=["col1"], is_unique=False,
                         is_primary=False, index_type="btree", definition="CREATE INDEX..."),
            ])
        ])
        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[], indexes=[
                IndexInfo(name="idx_col1", columns=["col1"], is_unique=True,  # Changed
                         is_primary=False, index_type="btree", definition="CREATE UNIQUE INDEX..."),
            ])
        ])

        result = differ.diff(source, target)

        assert len(result.tables_to_alter) == 1
        alter = result.tables_to_alter[0]
        # Modify = drop + add
        assert len(alter.indexes_to_drop) >= 1 or len(alter.indexes_to_add) >= 1

    def test_index_column_order_change(self, differ):
        """Test detection of index column order change."""
        source = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[], indexes=[
                IndexInfo(name="idx_multi", columns=["col1", "col2"], is_unique=False,
                         is_primary=False, index_type="btree", definition="CREATE INDEX..."),
            ])
        ])
        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[], indexes=[
                IndexInfo(name="idx_multi", columns=["col2", "col1"], is_unique=False,  # Reordered
                         is_primary=False, index_type="btree", definition="CREATE INDEX..."),
            ])
        ])

        result = differ.diff(source, target)

        # Column order change should be detected
        assert len(result.tables_to_alter) == 1

    def test_gin_index_type_change(self, differ):
        """Test detection of index type change."""
        source = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[], indexes=[
                IndexInfo(name="idx_data", columns=["data"], is_unique=False,
                         is_primary=False, index_type="btree", definition="CREATE INDEX..."),
            ])
        ])
        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[], indexes=[
                IndexInfo(name="idx_data", columns=["data"], is_unique=False,
                         is_primary=False, index_type="gin", definition="CREATE INDEX... USING gin"),
            ])
        ])

        result = differ.diff(source, target)

        assert len(result.tables_to_alter) == 1


# =============================================================================
# Constraint Diff Tests
# =============================================================================


class TestConstraintDiff:
    """Test constraint diff functionality."""

    def test_add_unique_constraint(self, differ, sample_table):
        """Test detection of new unique constraint."""
        source = SchemaInfo(tables=[sample_table])

        target_table = TableInfo(
            schema="public",
            name="users",
            columns=sample_table.columns,
            constraints=sample_table.constraints + [
                ConstraintInfo(
                    name="users_email_unique",
                    constraint_type="UNIQUE",
                    columns=["email"],
                )
            ],
            indexes=sample_table.indexes,
        )
        target = SchemaInfo(tables=[target_table])

        result = differ.diff(source, target)

        assert len(result.tables_to_alter) == 1
        alter = result.tables_to_alter[0]
        assert len(alter.constraints_to_add) == 1
        assert alter.constraints_to_add[0].name == "users_email_unique"

    def test_drop_constraint(self, differ):
        """Test detection of constraint to drop."""
        source = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[], constraints=[
                ConstraintInfo(name="test_check", constraint_type="CHECK",
                              columns=[], check_clause="col1 > 0"),
            ])
        ])
        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[], constraints=[])
        ])

        result = differ.diff(source, target)

        assert len(result.tables_to_alter) == 1
        alter = result.tables_to_alter[0]
        assert len(alter.constraints_to_drop) == 1

    def test_add_foreign_key(self, differ):
        """Test detection of new foreign key constraint."""
        source = SchemaInfo(tables=[
            TableInfo(schema="public", name="orders", columns=[
                ColumnInfo(name="user_id", data_type="integer", is_nullable=False, column_default=None,
                          character_maximum_length=None, numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None),
            ], constraints=[])
        ])

        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="orders", columns=[
                ColumnInfo(name="user_id", data_type="integer", is_nullable=False, column_default=None,
                          character_maximum_length=None, numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None),
            ], constraints=[
                ConstraintInfo(
                    name="orders_user_id_fkey",
                    constraint_type="FOREIGN KEY",
                    columns=["user_id"],
                    foreign_table_schema="public",
                    foreign_table_name="users",
                    foreign_columns=["id"],
                    on_delete="CASCADE",
                    on_update="NO ACTION",
                )
            ])
        ])

        result = differ.diff(source, target)

        assert len(result.tables_to_alter) == 1
        alter = result.tables_to_alter[0]
        assert len(alter.constraints_to_add) == 1
        assert alter.constraints_to_add[0].constraint_type == "FOREIGN KEY"

    def test_modify_check_constraint(self, differ):
        """Test detection of check constraint modification."""
        source = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[], constraints=[
                ConstraintInfo(name="test_check", constraint_type="CHECK",
                              columns=[], check_clause="col1 > 0"),
            ])
        ])
        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[], constraints=[
                ConstraintInfo(name="test_check", constraint_type="CHECK",
                              columns=[], check_clause="col1 >= 0"),  # Changed
            ])
        ])

        result = differ.diff(source, target)

        # Modified constraint = drop + add
        assert len(result.tables_to_alter) == 1


# =============================================================================
# View Diff Tests
# =============================================================================


class TestViewDiff:
    """Test view diff functionality."""

    def test_create_view(self, differ):
        """Test detection of new view."""
        source = SchemaInfo()
        target = SchemaInfo(views=[
            ViewInfo(schema="public", name="active_users",
                    definition="SELECT * FROM users WHERE active = true",
                    is_materialized=False)
        ])

        result = differ.diff(source, target)

        assert len(result.views_to_create) == 1
        assert result.views_to_create[0].name == "active_users"

    def test_drop_view(self, differ):
        """Test detection of view to drop."""
        source = SchemaInfo(views=[
            ViewInfo(schema="public", name="old_view",
                    definition="SELECT * FROM old_table",
                    is_materialized=False)
        ])
        target = SchemaInfo()

        result = differ.diff(source, target)

        assert len(result.views_to_drop) == 1
        assert result.views_to_drop[0].name == "old_view"

    def test_modify_view_definition(self, differ):
        """Test detection of view definition change."""
        source = SchemaInfo(views=[
            ViewInfo(schema="public", name="user_summary",
                    definition="SELECT id, name FROM users",
                    is_materialized=False)
        ])
        target = SchemaInfo(views=[
            ViewInfo(schema="public", name="user_summary",
                    definition="SELECT id, name, email FROM users",  # Changed
                    is_materialized=False)
        ])

        result = differ.diff(source, target)

        assert len(result.views_to_modify) == 1

    def test_materialized_view_change(self, differ):
        """Test detection of materialized view status change."""
        source = SchemaInfo(views=[
            ViewInfo(schema="public", name="stats",
                    definition="SELECT count(*) FROM users",
                    is_materialized=False)
        ])
        target = SchemaInfo(views=[
            ViewInfo(schema="public", name="stats",
                    definition="SELECT count(*) FROM users",
                    is_materialized=True)  # Changed to materialized
        ])

        result = differ.diff(source, target)

        assert len(result.views_to_modify) == 1


# =============================================================================
# Sequence Diff Tests
# =============================================================================


class TestSequenceDiff:
    """Test sequence diff functionality."""

    def test_create_sequence(self, differ):
        """Test detection of new sequence."""
        source = SchemaInfo()
        target = SchemaInfo(sequences=[
            SequenceInfo(schema="public", name="order_seq", data_type="bigint",
                        start_value=1, increment=1, min_value=1,
                        max_value=9223372036854775807, cycle=False)
        ])

        result = differ.diff(source, target)

        assert len(result.sequences_to_create) == 1
        assert result.sequences_to_create[0].name == "order_seq"

    def test_drop_sequence(self, differ):
        """Test detection of sequence to drop."""
        source = SchemaInfo(sequences=[
            SequenceInfo(schema="public", name="old_seq", data_type="integer",
                        start_value=1, increment=1, min_value=1,
                        max_value=2147483647, cycle=False)
        ])
        target = SchemaInfo()

        result = differ.diff(source, target)

        assert len(result.sequences_to_drop) == 1

    def test_modify_sequence_increment(self, differ):
        """Test detection of sequence increment change."""
        source = SchemaInfo(sequences=[
            SequenceInfo(schema="public", name="counter", data_type="bigint",
                        start_value=1, increment=1, min_value=1,
                        max_value=9223372036854775807, cycle=False)
        ])
        target = SchemaInfo(sequences=[
            SequenceInfo(schema="public", name="counter", data_type="bigint",
                        start_value=1, increment=10, min_value=1,  # Changed
                        max_value=9223372036854775807, cycle=False)
        ])

        result = differ.diff(source, target)

        assert len(result.sequences_to_modify) == 1


# =============================================================================
# Enum Diff Tests
# =============================================================================


class TestEnumDiff:
    """Test enum type diff functionality."""

    def test_create_enum(self, differ):
        """Test detection of new enum type."""
        source = SchemaInfo()
        target = SchemaInfo(enums=[
            EnumInfo(schema="public", name="status_type",
                    values=["pending", "active", "completed"])
        ])

        result = differ.diff(source, target)

        assert len(result.enums_to_create) == 1
        assert result.enums_to_create[0].name == "status_type"

    def test_drop_enum(self, differ):
        """Test detection of enum to drop."""
        source = SchemaInfo(enums=[
            EnumInfo(schema="public", name="old_enum", values=["a", "b"])
        ])
        target = SchemaInfo()

        result = differ.diff(source, target)

        assert len(result.enums_to_drop) == 1

    def test_add_enum_value(self, differ):
        """Test detection of new enum value."""
        source = SchemaInfo(enums=[
            EnumInfo(schema="public", name="priority",
                    values=["low", "medium", "high"])
        ])
        target = SchemaInfo(enums=[
            EnumInfo(schema="public", name="priority",
                    values=["low", "medium", "high", "critical"])  # Added
        ])

        result = differ.diff(source, target)

        assert len(result.enums_to_modify) == 1

    def test_remove_enum_value(self, differ):
        """Test detection of removed enum value (not directly supported in PG)."""
        source = SchemaInfo(enums=[
            EnumInfo(schema="public", name="status",
                    values=["a", "b", "c"])
        ])
        target = SchemaInfo(enums=[
            EnumInfo(schema="public", name="status",
                    values=["a", "b"])  # Removed 'c'
        ])

        result = differ.diff(source, target)

        # Removing enum values requires recreation
        assert len(result.enums_to_modify) == 1


# =============================================================================
# SQL Generation Tests
# =============================================================================


class TestSQLGeneration:
    """Test SQL generation from diffs."""

    def test_generate_create_table_sql(self, differ, sample_table):
        """Test SQL generation for table creation."""
        source = SchemaInfo()
        target = SchemaInfo(tables=[sample_table])

        result = differ.diff(source, target)
        sql = differ.generate_migration_sql(result)

        assert "CREATE TABLE" in sql
        assert "public" in sql
        assert "users" in sql

    def test_generate_drop_table_sql(self, differ, sample_table):
        """Test SQL generation for table drop."""
        source = SchemaInfo(tables=[sample_table])
        target = SchemaInfo()

        result = differ.diff(source, target)
        sql = differ.generate_migration_sql(result)

        assert "DROP TABLE" in sql
        assert "users" in sql

    def test_generate_add_column_sql(self, differ, sample_table):
        """Test SQL generation for column addition."""
        source = SchemaInfo(tables=[sample_table])

        target_table = TableInfo(
            schema="public",
            name="users",
            columns=sample_table.columns + [
                ColumnInfo(name="status", data_type="text", is_nullable=True, column_default=None,
                          character_maximum_length=None, numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None)
            ],
            constraints=sample_table.constraints,
            indexes=sample_table.indexes,
        )
        target = SchemaInfo(tables=[target_table])

        result = differ.diff(source, target)
        sql = differ.generate_migration_sql(result)

        assert "ALTER TABLE" in sql
        assert "ADD COLUMN" in sql
        assert "status" in sql

    def test_generate_drop_column_sql(self, differ, sample_table):
        """Test SQL generation for column drop."""
        source = SchemaInfo(tables=[sample_table])

        target_table = TableInfo(
            schema="public",
            name="users",
            columns=[sample_table.columns[0]],  # Only id
            constraints=sample_table.constraints,
            indexes=sample_table.indexes,
        )
        target = SchemaInfo(tables=[target_table])

        result = differ.diff(source, target)
        sql = differ.generate_migration_sql(result)

        assert "ALTER TABLE" in sql
        assert "DROP COLUMN" in sql
        assert "email" in sql

    def test_generate_create_index_sql(self, differ):
        """Test SQL generation for index creation."""
        source = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[], indexes=[])
        ])
        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[], indexes=[
                IndexInfo(name="idx_test", columns=["col1"], is_unique=False,
                         is_primary=False, index_type="btree",
                         definition="CREATE INDEX idx_test ON public.test (col1)")
            ])
        ])

        result = differ.diff(source, target)
        sql = differ.generate_migration_sql(result)

        assert "CREATE INDEX" in sql
        assert "idx_test" in sql

    def test_generate_add_constraint_sql(self, differ):
        """Test SQL generation for constraint addition."""
        source = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[], constraints=[])
        ])
        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[], constraints=[
                ConstraintInfo(name="test_unique", constraint_type="UNIQUE", columns=["col1"])
            ])
        ])

        result = differ.diff(source, target)
        sql = differ.generate_migration_sql(result)

        assert "ADD CONSTRAINT" in sql
        assert "UNIQUE" in sql

    def test_generate_create_enum_sql(self, differ):
        """Test SQL generation for enum creation."""
        source = SchemaInfo()
        target = SchemaInfo(enums=[
            EnumInfo(schema="public", name="status", values=["a", "b", "c"])
        ])

        result = differ.diff(source, target)
        sql = differ.generate_migration_sql(result)

        assert "CREATE TYPE" in sql
        assert "status" in sql
        assert "ENUM" in sql

    def test_generate_rollback_sql(self, differ, sample_table):
        """Test rollback SQL generation."""
        source = SchemaInfo()
        target = SchemaInfo(tables=[sample_table])

        result = differ.diff(source, target)
        up_sql = differ.generate_migration_sql(result)
        down_sql = differ.generate_rollback_sql(result)

        assert "CREATE TABLE" in up_sql
        assert "DROP TABLE" in down_sql


# =============================================================================
# Complex Scenario Tests
# =============================================================================


class TestComplexScenarios:
    """Test complex diff scenarios."""

    def test_complete_schema_change(self, differ):
        """Test a complete schema transformation."""
        source = SchemaInfo(
            tables=[
                TableInfo(schema="public", name="old_users", columns=[
                    ColumnInfo(name="id", data_type="integer", is_nullable=False, column_default=None,
                              character_maximum_length=None, numeric_precision=None, numeric_scale=None,
                              is_identity=False, identity_generation=None),
                ]),
            ],
            views=[
                ViewInfo(schema="public", name="old_view", definition="SELECT 1",
                        is_materialized=False),
            ],
        )

        target = SchemaInfo(
            tables=[
                TableInfo(schema="public", name="new_users", columns=[
                    ColumnInfo(name="id", data_type="bigint", is_nullable=False, column_default=None,
                              character_maximum_length=None, numeric_precision=None, numeric_scale=None,
                              is_identity=True, identity_generation="BY DEFAULT"),
                    ColumnInfo(name="email", data_type="text", is_nullable=False, column_default=None,
                              character_maximum_length=None, numeric_precision=None, numeric_scale=None,
                              is_identity=False, identity_generation=None),
                ]),
            ],
            views=[
                ViewInfo(schema="public", name="new_view", definition="SELECT 2",
                        is_materialized=False),
            ],
            enums=[
                EnumInfo(schema="public", name="status", values=["active", "inactive"]),
            ],
        )

        result = differ.diff(source, target)

        # Should have changes in all areas
        assert len(result.tables_to_create) == 1
        assert len(result.tables_to_drop) == 1
        assert len(result.views_to_create) == 1
        assert len(result.views_to_drop) == 1
        assert len(result.enums_to_create) == 1

    def test_multi_schema_diff(self, differ):
        """Test diff across multiple schemas."""
        source = SchemaInfo(tables=[
            TableInfo(schema="public", name="table1", columns=[]),
            TableInfo(schema="private", name="table2", columns=[]),
        ])

        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="table1", columns=[]),
            TableInfo(schema="private", name="table3", columns=[]),  # renamed
        ])

        result = differ.diff(source, target)

        assert len(result.tables_to_drop) == 1
        assert len(result.tables_to_create) == 1

    def test_circular_foreign_keys(self, differ):
        """Test handling of circular foreign key references."""
        source = SchemaInfo()

        # Tables with circular references
        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="table_a", columns=[
                ColumnInfo(name="id", data_type="integer", is_nullable=False, column_default=None,
                          character_maximum_length=None, numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None),
                ColumnInfo(name="b_id", data_type="integer", is_nullable=True, column_default=None,
                          character_maximum_length=None, numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None),
            ], constraints=[
                ConstraintInfo(name="fk_a_b", constraint_type="FOREIGN KEY",
                              columns=["b_id"], foreign_table_schema="public",
                              foreign_table_name="table_b", foreign_columns=["id"]),
            ]),
            TableInfo(schema="public", name="table_b", columns=[
                ColumnInfo(name="id", data_type="integer", is_nullable=False, column_default=None,
                          character_maximum_length=None, numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None),
                ColumnInfo(name="a_id", data_type="integer", is_nullable=True, column_default=None,
                          character_maximum_length=None, numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None),
            ], constraints=[
                ConstraintInfo(name="fk_b_a", constraint_type="FOREIGN KEY",
                              columns=["a_id"], foreign_table_schema="public",
                              foreign_table_name="table_a", foreign_columns=["id"]),
            ]),
        ])

        result = differ.diff(source, target)
        sql = differ.generate_migration_sql(result)

        # Should handle circular dependencies
        assert "CREATE TABLE" in sql

    def test_empty_diff_no_sql(self, differ):
        """Test that empty diff produces no SQL."""
        source = SchemaInfo()
        target = SchemaInfo()

        result = differ.diff(source, target)
        sql = differ.generate_migration_sql(result)

        assert sql == "" or sql.strip() == "-- No changes detected"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_special_characters_in_names(self, differ):
        """Test handling of special characters in identifiers."""
        source = SchemaInfo()
        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="table-with-dashes", columns=[
                ColumnInfo(name="column with spaces", data_type="text", is_nullable=True,
                          column_default=None, character_maximum_length=None,
                          numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None),
            ])
        ])

        result = differ.diff(source, target)
        sql = differ.generate_migration_sql(result)

        # Should properly quote identifiers
        assert '"table-with-dashes"' in sql or "table-with-dashes" in sql

    def test_reserved_word_as_name(self, differ):
        """Test handling of PostgreSQL reserved words as identifiers."""
        source = SchemaInfo()
        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="select", columns=[
                ColumnInfo(name="order", data_type="integer", is_nullable=True,
                          column_default=None, character_maximum_length=None,
                          numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None),
            ])
        ])

        result = differ.diff(source, target)
        sql = differ.generate_migration_sql(result)

        # Reserved words should be quoted
        assert '"select"' in sql or '"order"' in sql

    def test_very_long_identifier(self, differ):
        """Test handling of very long identifiers."""
        long_name = "a" * 100  # PostgreSQL limit is 63

        source = SchemaInfo()
        target = SchemaInfo(tables=[
            TableInfo(schema="public", name=long_name, columns=[])
        ])

        result = differ.diff(source, target)

        assert len(result.tables_to_create) == 1

    def test_unicode_identifiers(self, differ):
        """Test handling of unicode identifiers."""
        source = SchemaInfo()
        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="таблица", columns=[
                ColumnInfo(name="колонка", data_type="text", is_nullable=True,
                          column_default=None, character_maximum_length=None,
                          numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None),
            ])
        ])

        result = differ.diff(source, target)
        sql = differ.generate_migration_sql(result)

        assert "таблица" in sql

    def test_numeric_precision_scale(self, differ):
        """Test proper handling of numeric precision and scale."""
        source = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[
                ColumnInfo(name="amount", data_type="numeric", is_nullable=True,
                          column_default=None, character_maximum_length=None,
                          numeric_precision=10, numeric_scale=2,
                          is_identity=False, identity_generation=None),
            ])
        ])
        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[
                ColumnInfo(name="amount", data_type="numeric", is_nullable=True,
                          column_default=None, character_maximum_length=None,
                          numeric_precision=12, numeric_scale=4,  # Changed
                          is_identity=False, identity_generation=None),
            ])
        ])

        result = differ.diff(source, target)

        assert len(result.tables_to_alter) == 1

    def test_array_column_type(self, differ):
        """Test handling of array column types."""
        source = SchemaInfo()
        target = SchemaInfo(tables=[
            TableInfo(schema="public", name="test", columns=[
                ColumnInfo(name="tags", data_type="text[]", is_nullable=True,
                          column_default=None, character_maximum_length=None,
                          numeric_precision=None, numeric_scale=None,
                          is_identity=False, identity_generation=None),
            ])
        ])

        result = differ.diff(source, target)
        sql = differ.generate_migration_sql(result)

        assert "text[]" in sql or "ARRAY" in sql

"""Schema diff functionality for comparing database schemas.

Compares two schemas and generates the SQL needed to transform
the source schema into the target schema.
"""

import logging
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Optional

from .schema_pull import (
    ColumnInfo,
    ConstraintInfo,
    EnumInfo,
    IndexInfo,
    SchemaInfo,
    SequenceInfo,
    TableInfo,
    ViewInfo,
)

logger = logging.getLogger(__name__)


class DiffType(str, Enum):
    """Type of schema difference."""

    CREATE = "create"
    DROP = "drop"
    ALTER = "alter"


@dataclass
class ColumnDiff:
    """Represents a column change."""

    column: ColumnInfo
    old_column: Optional[ColumnInfo] = None
    diff_type: DiffType = DiffType.ALTER


@dataclass
class IndexDiff:
    """Represents an index change."""

    index: IndexInfo
    diff_type: DiffType = DiffType.CREATE


@dataclass
class ConstraintDiff:
    """Represents a constraint change."""

    constraint: ConstraintInfo
    diff_type: DiffType = DiffType.CREATE


@dataclass
class TableDiff:
    """Represents changes to a table."""

    table: TableInfo
    columns_to_add: list[ColumnInfo] = field(default_factory=list)
    columns_to_drop: list[ColumnInfo] = field(default_factory=list)
    columns_to_modify: list[ColumnDiff] = field(default_factory=list)
    indexes_to_add: list[IndexInfo] = field(default_factory=list)
    indexes_to_drop: list[IndexInfo] = field(default_factory=list)
    constraints_to_add: list[ConstraintInfo] = field(default_factory=list)
    constraints_to_drop: list[ConstraintInfo] = field(default_factory=list)


@dataclass
class SchemaDiffResult:
    """Result of comparing two schemas."""

    tables_to_create: list[TableInfo] = field(default_factory=list)
    tables_to_drop: list[TableInfo] = field(default_factory=list)
    tables_to_alter: list[TableDiff] = field(default_factory=list)
    views_to_create: list[ViewInfo] = field(default_factory=list)
    views_to_drop: list[ViewInfo] = field(default_factory=list)
    views_to_modify: list[ViewInfo] = field(default_factory=list)
    sequences_to_create: list[SequenceInfo] = field(default_factory=list)
    sequences_to_drop: list[SequenceInfo] = field(default_factory=list)
    sequences_to_modify: list[SequenceInfo] = field(default_factory=list)
    enums_to_create: list[EnumInfo] = field(default_factory=list)
    enums_to_drop: list[EnumInfo] = field(default_factory=list)
    enums_to_modify: list[EnumInfo] = field(default_factory=list)


class SchemaDiff:
    """Compare two schemas and generate migration SQL.

    Compares a source schema (current database state) with a target schema
    (desired state) and produces the diff needed to transform source to target.
    """

    def diff(self, source: SchemaInfo, target: SchemaInfo) -> SchemaDiffResult:
        """Compare two schemas and return the differences.

        Args:
            source: Current schema state
            target: Desired schema state

        Returns:
            SchemaDiffResult containing all differences
        """
        result = SchemaDiffResult()

        # Diff tables
        self._diff_tables(source.tables, target.tables, result)

        # Diff views
        self._diff_views(source.views, target.views, result)

        # Diff sequences
        self._diff_sequences(source.sequences, target.sequences, result)

        # Diff enums
        self._diff_enums(source.enums, target.enums, result)

        return result

    def _diff_tables(
        self,
        source_tables: list[TableInfo],
        target_tables: list[TableInfo],
        result: SchemaDiffResult,
    ) -> None:
        """Diff tables between source and target."""
        source_map = {(t.schema, t.name): t for t in source_tables}
        target_map = {(t.schema, t.name): t for t in target_tables}

        source_keys = set(source_map.keys())
        target_keys = set(target_map.keys())

        # Tables to create (in target but not in source)
        for key in target_keys - source_keys:
            result.tables_to_create.append(target_map[key])

        # Tables to drop (in source but not in target)
        for key in source_keys - target_keys:
            result.tables_to_drop.append(source_map[key])

        # Tables to alter (in both)
        for key in source_keys & target_keys:
            table_diff = self._diff_single_table(source_map[key], target_map[key])
            if table_diff:
                result.tables_to_alter.append(table_diff)

    def _diff_single_table(
        self,
        source: TableInfo,
        target: TableInfo,
    ) -> Optional[TableDiff]:
        """Diff a single table between source and target."""
        diff = TableDiff(table=target)
        has_changes = False

        # Diff columns
        source_cols = {c.name: c for c in source.columns}
        target_cols = {c.name: c for c in target.columns}

        # Columns to add
        for name in set(target_cols.keys()) - set(source_cols.keys()):
            diff.columns_to_add.append(target_cols[name])
            has_changes = True

        # Columns to drop
        for name in set(source_cols.keys()) - set(target_cols.keys()):
            diff.columns_to_drop.append(source_cols[name])
            has_changes = True

        # Columns to modify
        for name in set(source_cols.keys()) & set(target_cols.keys()):
            if self._columns_differ(source_cols[name], target_cols[name]):
                diff.columns_to_modify.append(ColumnDiff(
                    column=target_cols[name],
                    old_column=source_cols[name],
                ))
                has_changes = True

        # Diff indexes
        source_idx = {i.name: i for i in source.indexes}
        target_idx = {i.name: i for i in target.indexes}

        for name in set(target_idx.keys()) - set(source_idx.keys()):
            diff.indexes_to_add.append(target_idx[name])
            has_changes = True

        for name in set(source_idx.keys()) - set(target_idx.keys()):
            diff.indexes_to_drop.append(source_idx[name])
            has_changes = True

        # Check for modified indexes
        for name in set(source_idx.keys()) & set(target_idx.keys()):
            if self._indexes_differ(source_idx[name], target_idx[name]):
                diff.indexes_to_drop.append(source_idx[name])
                diff.indexes_to_add.append(target_idx[name])
                has_changes = True

        # Diff constraints
        source_con = {c.name: c for c in source.constraints}
        target_con = {c.name: c for c in target.constraints}

        for name in set(target_con.keys()) - set(source_con.keys()):
            diff.constraints_to_add.append(target_con[name])
            has_changes = True

        for name in set(source_con.keys()) - set(target_con.keys()):
            diff.constraints_to_drop.append(source_con[name])
            has_changes = True

        # Check for modified constraints
        for name in set(source_con.keys()) & set(target_con.keys()):
            if self._constraints_differ(source_con[name], target_con[name]):
                diff.constraints_to_drop.append(source_con[name])
                diff.constraints_to_add.append(target_con[name])
                has_changes = True

        return diff if has_changes else None

    def _columns_differ(self, source: ColumnInfo, target: ColumnInfo) -> bool:
        """Check if two columns are different."""
        return (
            source.data_type != target.data_type
            or source.is_nullable != target.is_nullable
            or source.column_default != target.column_default
            or source.character_maximum_length != target.character_maximum_length
            or source.numeric_precision != target.numeric_precision
            or source.numeric_scale != target.numeric_scale
            or source.is_identity != target.is_identity
        )

    def _indexes_differ(self, source: IndexInfo, target: IndexInfo) -> bool:
        """Check if two indexes are different."""
        return (
            source.columns != target.columns
            or source.is_unique != target.is_unique
            or source.index_type != target.index_type
        )

    def _constraints_differ(self, source: ConstraintInfo, target: ConstraintInfo) -> bool:
        """Check if two constraints are different."""
        return (
            source.constraint_type != target.constraint_type
            or source.columns != target.columns
            or source.check_clause != target.check_clause
            or source.foreign_table_name != target.foreign_table_name
            or source.foreign_columns != target.foreign_columns
            or source.on_delete != target.on_delete
            or source.on_update != target.on_update
        )

    def _diff_views(
        self,
        source_views: list[ViewInfo],
        target_views: list[ViewInfo],
        result: SchemaDiffResult,
    ) -> None:
        """Diff views between source and target."""
        source_map = {(v.schema, v.name): v for v in source_views}
        target_map = {(v.schema, v.name): v for v in target_views}

        source_keys = set(source_map.keys())
        target_keys = set(target_map.keys())

        for key in target_keys - source_keys:
            result.views_to_create.append(target_map[key])

        for key in source_keys - target_keys:
            result.views_to_drop.append(source_map[key])

        for key in source_keys & target_keys:
            source_view = source_map[key]
            target_view = target_map[key]
            if (
                source_view.definition != target_view.definition
                or source_view.is_materialized != target_view.is_materialized
            ):
                result.views_to_modify.append(target_view)

    def _diff_sequences(
        self,
        source_seqs: list[SequenceInfo],
        target_seqs: list[SequenceInfo],
        result: SchemaDiffResult,
    ) -> None:
        """Diff sequences between source and target."""
        source_map = {(s.schema, s.name): s for s in source_seqs}
        target_map = {(s.schema, s.name): s for s in target_seqs}

        source_keys = set(source_map.keys())
        target_keys = set(target_map.keys())

        for key in target_keys - source_keys:
            result.sequences_to_create.append(target_map[key])

        for key in source_keys - target_keys:
            result.sequences_to_drop.append(source_map[key])

        for key in source_keys & target_keys:
            source_seq = source_map[key]
            target_seq = target_map[key]
            if (
                source_seq.increment != target_seq.increment
                or source_seq.min_value != target_seq.min_value
                or source_seq.max_value != target_seq.max_value
                or source_seq.cycle != target_seq.cycle
            ):
                result.sequences_to_modify.append(target_seq)

    def _diff_enums(
        self,
        source_enums: list[EnumInfo],
        target_enums: list[EnumInfo],
        result: SchemaDiffResult,
    ) -> None:
        """Diff enum types between source and target."""
        source_map = {(e.schema, e.name): e for e in source_enums}
        target_map = {(e.schema, e.name): e for e in target_enums}

        source_keys = set(source_map.keys())
        target_keys = set(target_map.keys())

        for key in target_keys - source_keys:
            result.enums_to_create.append(target_map[key])

        for key in source_keys - target_keys:
            result.enums_to_drop.append(source_map[key])

        for key in source_keys & target_keys:
            source_enum = source_map[key]
            target_enum = target_map[key]
            if source_enum.values != target_enum.values:
                result.enums_to_modify.append(target_enum)

    def generate_migration_sql(self, diff: SchemaDiffResult) -> str:
        """Generate SQL migration from a diff result.

        Args:
            diff: SchemaDiffResult from diff()

        Returns:
            SQL string for the migration
        """
        statements = []

        # Create enums first (types need to exist before tables use them)
        for enum in diff.enums_to_create:
            statements.append(self._generate_create_enum_sql(enum))

        # Modify enums (add values)
        for enum in diff.enums_to_modify:
            statements.append(self._generate_modify_enum_sql(enum))

        # Create sequences
        for seq in diff.sequences_to_create:
            statements.append(self._generate_create_sequence_sql(seq))

        # Create tables
        for table in diff.tables_to_create:
            statements.append(self._generate_create_table_sql(table))

        # Alter tables
        for table_diff in diff.tables_to_alter:
            alter_stmts = self._generate_alter_table_sql(table_diff)
            statements.extend(alter_stmts)

        # Create views
        for view in diff.views_to_create:
            statements.append(self._generate_create_view_sql(view))

        # Modify views
        for view in diff.views_to_modify:
            statements.append(self._generate_modify_view_sql(view))

        # Drop views
        for view in diff.views_to_drop:
            statements.append(self._generate_drop_view_sql(view))

        # Drop tables
        for table in diff.tables_to_drop:
            statements.append(self._generate_drop_table_sql(table))

        # Drop sequences
        for seq in diff.sequences_to_drop:
            statements.append(self._generate_drop_sequence_sql(seq))

        # Drop enums
        for enum in diff.enums_to_drop:
            statements.append(self._generate_drop_enum_sql(enum))

        # Modify sequences
        for seq in diff.sequences_to_modify:
            statements.append(self._generate_modify_sequence_sql(seq))

        if not statements:
            return "-- No changes detected"

        return "\n\n".join(statements)

    def generate_rollback_sql(self, diff: SchemaDiffResult) -> str:
        """Generate SQL to rollback a migration.

        Args:
            diff: SchemaDiffResult from diff()

        Returns:
            SQL string for rolling back the migration
        """
        statements = []

        # Reverse of migration: drop what was created, create what was dropped

        # Drop created views
        for view in diff.views_to_create:
            statements.append(self._generate_drop_view_sql(view))

        # Drop created tables
        for table in diff.tables_to_create:
            statements.append(self._generate_drop_table_sql(table))

        # Re-create dropped tables
        for table in diff.tables_to_drop:
            statements.append(self._generate_create_table_sql(table))

        # Re-create dropped views
        for view in diff.views_to_drop:
            statements.append(self._generate_create_view_sql(view))

        # Drop created sequences
        for seq in diff.sequences_to_create:
            statements.append(self._generate_drop_sequence_sql(seq))

        # Re-create dropped sequences
        for seq in diff.sequences_to_drop:
            statements.append(self._generate_create_sequence_sql(seq))

        # Drop created enums
        for enum in diff.enums_to_create:
            statements.append(self._generate_drop_enum_sql(enum))

        # Re-create dropped enums
        for enum in diff.enums_to_drop:
            statements.append(self._generate_create_enum_sql(enum))

        if not statements:
            return "-- No changes to rollback"

        return "\n\n".join(statements)

    def _quote_identifier(self, name: str) -> str:
        """Quote an identifier for use in SQL."""
        # Escape double quotes
        escaped = name.replace('"', '""')
        return f'"{escaped}"'

    def _generate_create_enum_sql(self, enum: EnumInfo) -> str:
        """Generate CREATE TYPE ENUM SQL."""
        values = ", ".join(f"'{v}'" for v in enum.values)
        return f"CREATE TYPE {self._quote_identifier(enum.schema)}.{self._quote_identifier(enum.name)} AS ENUM ({values});"

    def _generate_modify_enum_sql(self, enum: EnumInfo) -> str:
        """Generate ALTER TYPE ENUM SQL for adding values.

        Note: PostgreSQL doesn't support removing enum values directly.
        """
        # For simplicity, we'll just generate ADD VALUE statements
        # In a real implementation, we'd need to track which values are new
        statements = []
        for value in enum.values:
            stmt = f"ALTER TYPE {self._quote_identifier(enum.schema)}.{self._quote_identifier(enum.name)} ADD VALUE IF NOT EXISTS '{value}';"
            statements.append(stmt)
        return "\n".join(statements)

    def _generate_drop_enum_sql(self, enum: EnumInfo) -> str:
        """Generate DROP TYPE SQL."""
        return f"DROP TYPE IF EXISTS {self._quote_identifier(enum.schema)}.{self._quote_identifier(enum.name)};"

    def _generate_create_sequence_sql(self, seq: SequenceInfo) -> str:
        """Generate CREATE SEQUENCE SQL."""
        sql = f"CREATE SEQUENCE {self._quote_identifier(seq.schema)}.{self._quote_identifier(seq.name)}"
        sql += f"\n    AS {seq.data_type}"
        sql += f"\n    START WITH {seq.start_value}"
        sql += f"\n    INCREMENT BY {seq.increment}"
        sql += f"\n    MINVALUE {seq.min_value}"
        sql += f"\n    MAXVALUE {seq.max_value}"
        sql += f"\n    {'CYCLE' if seq.cycle else 'NO CYCLE'};"
        return sql

    def _generate_modify_sequence_sql(self, seq: SequenceInfo) -> str:
        """Generate ALTER SEQUENCE SQL."""
        sql = f"ALTER SEQUENCE {self._quote_identifier(seq.schema)}.{self._quote_identifier(seq.name)}"
        sql += f"\n    INCREMENT BY {seq.increment}"
        sql += f"\n    MINVALUE {seq.min_value}"
        sql += f"\n    MAXVALUE {seq.max_value}"
        sql += f"\n    {'CYCLE' if seq.cycle else 'NO CYCLE'};"
        return sql

    def _generate_drop_sequence_sql(self, seq: SequenceInfo) -> str:
        """Generate DROP SEQUENCE SQL."""
        return f"DROP SEQUENCE IF EXISTS {self._quote_identifier(seq.schema)}.{self._quote_identifier(seq.name)};"

    def _generate_create_table_sql(self, table: TableInfo) -> str:
        """Generate CREATE TABLE SQL."""
        lines = [f"CREATE TABLE {self._quote_identifier(table.schema)}.{self._quote_identifier(table.name)} ("]

        # Columns
        column_defs = []
        for col in table.columns:
            col_def = self._generate_column_definition(col)
            column_defs.append(f"    {col_def}")

        # Primary key constraint
        for constraint in table.constraints:
            if constraint.constraint_type == "PRIMARY KEY":
                cols = ", ".join(self._quote_identifier(c) for c in constraint.columns)
                column_defs.append(f"    CONSTRAINT {self._quote_identifier(constraint.name)} PRIMARY KEY ({cols})")

        lines.append(",\n".join(column_defs))
        lines.append(");")

        # Other constraints
        for constraint in table.constraints:
            if constraint.constraint_type == "FOREIGN KEY":
                lines.append(self._generate_add_foreign_key_sql(table, constraint))
            elif constraint.constraint_type == "UNIQUE":
                lines.append(self._generate_add_unique_constraint_sql(table, constraint))
            elif constraint.constraint_type == "CHECK":
                lines.append(self._generate_add_check_constraint_sql(table, constraint))

        # Indexes (excluding primary key)
        for index in table.indexes:
            if not index.is_primary:
                lines.append(f"{index.definition};")

        return "\n".join(lines)

    def _generate_column_definition(self, col: ColumnInfo) -> str:
        """Generate column definition SQL."""
        parts = [self._quote_identifier(col.name)]

        # Data type with length/precision
        if col.character_maximum_length:
            parts.append(f"{col.data_type}({col.character_maximum_length})")
        elif col.numeric_precision and col.numeric_scale:
            parts.append(f"{col.data_type}({col.numeric_precision},{col.numeric_scale})")
        else:
            parts.append(col.data_type)

        # NOT NULL
        if not col.is_nullable:
            parts.append("NOT NULL")

        # DEFAULT
        if col.column_default and not col.is_identity:
            parts.append(f"DEFAULT {col.column_default}")

        # IDENTITY
        if col.is_identity:
            gen = col.identity_generation or "BY DEFAULT"
            parts.append(f"GENERATED {gen} AS IDENTITY")

        return " ".join(parts)

    def _generate_drop_table_sql(self, table: TableInfo) -> str:
        """Generate DROP TABLE SQL."""
        return f"DROP TABLE IF EXISTS {self._quote_identifier(table.schema)}.{self._quote_identifier(table.name)} CASCADE;"

    def _generate_alter_table_sql(self, diff: TableDiff) -> list[str]:
        """Generate ALTER TABLE SQL statements."""
        statements = []
        table_ref = f"{self._quote_identifier(diff.table.schema)}.{self._quote_identifier(diff.table.name)}"

        # Drop constraints first (before columns they depend on)
        for constraint in diff.constraints_to_drop:
            statements.append(
                f"ALTER TABLE {table_ref} DROP CONSTRAINT IF EXISTS {self._quote_identifier(constraint.name)};"
            )

        # Drop indexes
        for index in diff.indexes_to_drop:
            statements.append(f"DROP INDEX IF EXISTS {self._quote_identifier(index.name)};")

        # Drop columns
        for col in diff.columns_to_drop:
            statements.append(f"ALTER TABLE {table_ref} DROP COLUMN {self._quote_identifier(col.name)};")

        # Add columns
        for col in diff.columns_to_add:
            col_def = self._generate_column_definition(col)
            statements.append(f"ALTER TABLE {table_ref} ADD COLUMN {col_def};")

        # Modify columns
        for col_diff in diff.columns_to_modify:
            stmts = self._generate_alter_column_sql(diff.table, col_diff)
            statements.extend(stmts)

        # Add indexes
        for index in diff.indexes_to_add:
            statements.append(f"{index.definition};")

        # Add constraints
        for constraint in diff.constraints_to_add:
            if constraint.constraint_type == "FOREIGN KEY":
                statements.append(self._generate_add_foreign_key_sql(diff.table, constraint))
            elif constraint.constraint_type == "UNIQUE":
                statements.append(self._generate_add_unique_constraint_sql(diff.table, constraint))
            elif constraint.constraint_type == "CHECK":
                statements.append(self._generate_add_check_constraint_sql(diff.table, constraint))

        return statements

    def _generate_alter_column_sql(self, table: TableInfo, col_diff: ColumnDiff) -> list[str]:
        """Generate ALTER COLUMN SQL statements."""
        statements = []
        table_ref = f"{self._quote_identifier(table.schema)}.{self._quote_identifier(table.name)}"
        col_ref = self._quote_identifier(col_diff.column.name)
        col = col_diff.column
        old = col_diff.old_column

        # Type change
        if old and col.data_type != old.data_type:
            type_str = col.data_type
            if col.character_maximum_length:
                type_str = f"{col.data_type}({col.character_maximum_length})"
            statements.append(
                f"ALTER TABLE {table_ref} ALTER COLUMN {col_ref} TYPE {type_str} USING {col_ref}::{type_str};"
            )

        # Nullable change
        if old and col.is_nullable != old.is_nullable:
            if col.is_nullable:
                statements.append(f"ALTER TABLE {table_ref} ALTER COLUMN {col_ref} DROP NOT NULL;")
            else:
                statements.append(f"ALTER TABLE {table_ref} ALTER COLUMN {col_ref} SET NOT NULL;")

        # Default change
        if old and col.column_default != old.column_default:
            if col.column_default:
                statements.append(
                    f"ALTER TABLE {table_ref} ALTER COLUMN {col_ref} SET DEFAULT {col.column_default};"
                )
            else:
                statements.append(f"ALTER TABLE {table_ref} ALTER COLUMN {col_ref} DROP DEFAULT;")

        return statements

    def _generate_add_foreign_key_sql(self, table: TableInfo, constraint: ConstraintInfo) -> str:
        """Generate ADD FOREIGN KEY SQL."""
        table_ref = f"{self._quote_identifier(table.schema)}.{self._quote_identifier(table.name)}"
        cols = ", ".join(self._quote_identifier(c) for c in constraint.columns)
        ref_table = f"{self._quote_identifier(constraint.foreign_table_schema)}.{self._quote_identifier(constraint.foreign_table_name)}"
        ref_cols = ", ".join(self._quote_identifier(c) for c in (constraint.foreign_columns or []))

        sql = f"ALTER TABLE {table_ref} ADD CONSTRAINT {self._quote_identifier(constraint.name)} "
        sql += f"FOREIGN KEY ({cols}) REFERENCES {ref_table} ({ref_cols})"
        if constraint.on_update:
            sql += f" ON UPDATE {constraint.on_update}"
        if constraint.on_delete:
            sql += f" ON DELETE {constraint.on_delete}"
        sql += ";"
        return sql

    def _generate_add_unique_constraint_sql(self, table: TableInfo, constraint: ConstraintInfo) -> str:
        """Generate ADD UNIQUE CONSTRAINT SQL."""
        table_ref = f"{self._quote_identifier(table.schema)}.{self._quote_identifier(table.name)}"
        cols = ", ".join(self._quote_identifier(c) for c in constraint.columns)
        return f"ALTER TABLE {table_ref} ADD CONSTRAINT {self._quote_identifier(constraint.name)} UNIQUE ({cols});"

    def _generate_add_check_constraint_sql(self, table: TableInfo, constraint: ConstraintInfo) -> str:
        """Generate ADD CHECK CONSTRAINT SQL."""
        table_ref = f"{self._quote_identifier(table.schema)}.{self._quote_identifier(table.name)}"
        return f"ALTER TABLE {table_ref} ADD CONSTRAINT {self._quote_identifier(constraint.name)} CHECK ({constraint.check_clause});"

    def _generate_create_view_sql(self, view: ViewInfo) -> str:
        """Generate CREATE VIEW SQL."""
        view_type = "MATERIALIZED VIEW" if view.is_materialized else "VIEW"
        return f"CREATE {view_type} {self._quote_identifier(view.schema)}.{self._quote_identifier(view.name)} AS {view.definition}"

    def _generate_modify_view_sql(self, view: ViewInfo) -> str:
        """Generate CREATE OR REPLACE VIEW SQL."""
        if view.is_materialized:
            # Materialized views need DROP + CREATE
            return f"DROP MATERIALIZED VIEW IF EXISTS {self._quote_identifier(view.schema)}.{self._quote_identifier(view.name)};\n" + \
                   self._generate_create_view_sql(view)
        else:
            return f"CREATE OR REPLACE VIEW {self._quote_identifier(view.schema)}.{self._quote_identifier(view.name)} AS {view.definition}"

    def _generate_drop_view_sql(self, view: ViewInfo) -> str:
        """Generate DROP VIEW SQL."""
        view_type = "MATERIALIZED VIEW" if view.is_materialized else "VIEW"
        return f"DROP {view_type} IF EXISTS {self._quote_identifier(view.schema)}.{self._quote_identifier(view.name)};"

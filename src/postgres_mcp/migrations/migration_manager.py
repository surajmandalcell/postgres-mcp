"""Migration manager for orchestrating database migrations.

Provides Drizzle/Supabase-style migration capabilities:
- Discover and parse migration files
- Execute migrations up/down
- Batch operations
- Dry run support
- Status reporting
- Rollback operations
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..sql import SqlDriver
from .migration_tracker import MigrationRecord
from .migration_tracker import MigrationTracker

logger = logging.getLogger(__name__)


@dataclass
class MigrationFile:
    """Represents a migration file on disk."""

    name: str
    path: Path
    up_sql: str
    down_sql: Optional[str]
    checksum: str
    timestamp: str


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    success: bool
    applied_count: int = 0
    rolled_back_count: int = 0
    pending_count: int = 0
    batch: int = 0
    error: Optional[str] = None
    dry_run: bool = False
    sql_preview: str = ""
    migrations: list[str] = field(default_factory=list)


class MigrationError(Exception):
    """Error during migration execution."""

    pass


class MigrationManager:
    """Manages database migrations.

    Provides functionality for:
    - Discovering migration files from a directory
    - Applying pending migrations (migrate up)
    - Rolling back migrations (migrate down)
    - Checking migration status
    - Generating new migration files
    - Dry run support
    """

    def __init__(
        self,
        sql_driver: SqlDriver,
        migrations_dir: Path,
        schema: str = "public",
    ):
        """Initialize the migration manager.

        Args:
            sql_driver: SQL driver for database access
            migrations_dir: Directory containing migration files
            schema: Schema for migration tracking table
        """
        self.sql_driver = sql_driver
        self.migrations_dir = Path(migrations_dir)
        self.tracker = MigrationTracker(sql_driver=sql_driver, schema=schema)

    async def discover_migrations(self) -> list[MigrationFile]:
        """Discover all migration files in the migrations directory.

        Supports two formats:
        1. Directory format: migrations_dir/{timestamp}_{name}/up.sql and down.sql
        2. Single file format: migrations_dir/{timestamp}_{name}.sql with -- migrate:up/down markers

        Returns:
            List of MigrationFile objects sorted by timestamp
        """
        migrations = []

        if not self.migrations_dir.exists():
            logger.warning(f"Migrations directory does not exist: {self.migrations_dir}")
            return migrations

        # Check for directory-based migrations
        for item in sorted(self.migrations_dir.iterdir()):
            if item.is_dir() and self._is_valid_migration_name(item.name):
                migration = self._parse_directory_migration(item)
                if migration:
                    migrations.append(migration)

            elif item.is_file() and item.suffix == ".sql" and self._is_valid_migration_name(item.stem):
                migration = self._parse_single_file_migration(item)
                if migration:
                    migrations.append(migration)

        # Sort by name (which includes timestamp)
        migrations.sort(key=lambda m: m.name)

        return migrations

    def _is_valid_migration_name(self, name: str) -> bool:
        """Check if a name follows the migration naming convention.

        Valid format: {timestamp}_{description}
        Where timestamp is YYYYMMDDHHmmss (14 digits)
        """
        if name.startswith(".") or name.startswith("_"):
            return False

        # Check for timestamp prefix
        pattern = r"^\d{14}_\w+"
        return bool(re.match(pattern, name))

    def _parse_directory_migration(self, path: Path) -> Optional[MigrationFile]:
        """Parse a directory-based migration."""
        up_file = path / "up.sql"

        if not up_file.exists():
            logger.warning(f"Migration directory missing up.sql: {path}")
            return None

        try:
            up_sql = up_file.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to read migration file {up_file}: {e}")
            raise

        down_sql = None
        down_file = path / "down.sql"
        if down_file.exists():
            try:
                down_sql = down_file.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Failed to read down migration {down_file}: {e}")

        checksum = hashlib.sha256(up_sql.encode()).hexdigest()
        timestamp = path.name[:14] if len(path.name) >= 14 else path.name

        return MigrationFile(
            name=path.name,
            path=path,
            up_sql=up_sql,
            down_sql=down_sql,
            checksum=checksum,
            timestamp=timestamp,
        )

    def _parse_single_file_migration(self, path: Path) -> Optional[MigrationFile]:
        """Parse a single-file migration with -- migrate:up/down markers."""
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to read migration file {path}: {e}")
            raise

        up_sql = None
        down_sql = None

        # Parse sections
        up_match = re.search(r"--\s*migrate:up\s*\n(.*?)(?=--\s*migrate:down|$)", content, re.DOTALL | re.IGNORECASE)
        down_match = re.search(r"--\s*migrate:down\s*\n(.*?)$", content, re.DOTALL | re.IGNORECASE)

        if up_match:
            up_sql = up_match.group(1).strip()
        else:
            # If no markers, treat entire file as up migration
            up_sql = content.strip()

        if down_match:
            down_sql = down_match.group(1).strip()

        if not up_sql:
            logger.warning(f"Migration file has no up SQL: {path}")
            return None

        checksum = hashlib.sha256(up_sql.encode()).hexdigest()
        timestamp = path.stem[:14] if len(path.stem) >= 14 else path.stem

        return MigrationFile(
            name=path.stem,
            path=path,
            up_sql=up_sql,
            down_sql=down_sql,
            checksum=checksum,
            timestamp=timestamp,
        )

    async def migrate_up(
        self,
        steps: Optional[int] = None,
        target: Optional[str] = None,
        dry_run: bool = False,
    ) -> MigrationResult:
        """Apply pending migrations.

        Args:
            steps: Number of migrations to apply (default: all)
            target: Target migration name to migrate up to (inclusive)
            dry_run: If True, don't actually execute migrations

        Returns:
            MigrationResult with operation details
        """
        try:
            await self.tracker.ensure_migration_table()

            # Discover all migrations
            all_migrations = await self.discover_migrations()
            applied_records = await self.tracker.get_applied_migrations()
            applied_names = {r.name for r in applied_records}

            # Find pending migrations
            pending = [m for m in all_migrations if m.name not in applied_names]

            # Apply step limit
            if target:
                target_idx = next(
                    (i for i, m in enumerate(pending) if m.name == target or m.name.endswith(target)),
                    len(pending),
                )
                pending = pending[: target_idx + 1]
            elif steps:
                pending = pending[:steps]

            if dry_run:
                sql_preview = "\n\n".join([f"-- Migration: {m.name}\n{m.up_sql}" for m in pending])
                return MigrationResult(
                    success=True,
                    pending_count=len(pending),
                    dry_run=True,
                    sql_preview=sql_preview,
                    migrations=[m.name for m in pending],
                )

            if not pending:
                logger.info("No pending migrations to apply")
                return MigrationResult(success=True, applied_count=0)

            # Get next batch number
            latest_batch = await self.tracker.get_latest_batch()
            new_batch = latest_batch + 1

            # Apply each migration
            applied = []
            for migration in pending:
                try:
                    logger.info(f"Applying migration: {migration.name}")
                    await self.sql_driver.execute_query(migration.up_sql)
                    await self.tracker.record_migration(
                        name=migration.name,
                        checksum=migration.checksum,
                        batch=new_batch,
                        executed_sql=migration.up_sql,
                    )
                    applied.append(migration.name)
                    logger.info(f"Applied migration: {migration.name}")
                except Exception as e:
                    logger.error(f"Failed to apply migration {migration.name}: {e}")
                    return MigrationResult(
                        success=False,
                        applied_count=len(applied),
                        batch=new_batch,
                        error=str(e),
                        migrations=applied,
                    )

            return MigrationResult(
                success=True,
                applied_count=len(applied),
                batch=new_batch,
                migrations=applied,
            )

        except Exception as e:
            logger.error(f"Migration error: {e}")
            return MigrationResult(success=False, error=str(e))

    async def rollback(
        self,
        steps: Optional[int] = None,
        dry_run: bool = False,
    ) -> MigrationResult:
        """Rollback migrations.

        Args:
            steps: Number of migrations to rollback (default: latest batch)
            dry_run: If True, don't actually execute rollbacks

        Returns:
            MigrationResult with operation details
        """
        try:
            await self.tracker.ensure_migration_table()

            # Discover all migrations for down SQL
            all_migrations = await self.discover_migrations()
            migration_map = {m.name: m for m in all_migrations}

            if steps:
                # Rollback specific number of migrations
                applied = await self.tracker.get_applied_migrations()
                # Get most recent N migrations (reverse order)
                to_rollback = list(reversed(applied))[:steps]
            else:
                # Rollback latest batch
                latest_batch = await self.tracker.get_latest_batch()
                if latest_batch == 0:
                    return MigrationResult(success=True, rolled_back_count=0)
                to_rollback = await self.tracker.get_migrations_in_batch(latest_batch)

            if not to_rollback:
                logger.info("No migrations to rollback")
                return MigrationResult(success=True, rolled_back_count=0)

            # Check all have down migrations
            for record in to_rollback:
                migration = migration_map.get(record.name)
                if not migration or not migration.down_sql:
                    return MigrationResult(
                        success=False,
                        error=f"No down migration found for: {record.name}. Cannot rollback.",
                    )

            if dry_run:
                sql_preview = "\n\n".join([
                    f"-- Rollback: {r.name}\n{migration_map[r.name].down_sql}"
                    for r in to_rollback
                ])
                return MigrationResult(
                    success=True,
                    rolled_back_count=len(to_rollback),
                    dry_run=True,
                    sql_preview=sql_preview,
                    migrations=[r.name for r in to_rollback],
                )

            # Execute rollbacks
            rolled_back = []
            for record in to_rollback:
                migration = migration_map[record.name]
                try:
                    logger.info(f"Rolling back migration: {record.name}")
                    await self.sql_driver.execute_query(migration.down_sql)
                    await self.tracker.remove_migration(record.name)
                    rolled_back.append(record.name)
                    logger.info(f"Rolled back migration: {record.name}")
                except Exception as e:
                    logger.error(f"Failed to rollback {record.name}: {e}")
                    return MigrationResult(
                        success=False,
                        rolled_back_count=len(rolled_back),
                        error=str(e),
                        migrations=rolled_back,
                    )

            return MigrationResult(
                success=True,
                rolled_back_count=len(rolled_back),
                migrations=rolled_back,
            )

        except Exception as e:
            logger.error(f"Rollback error: {e}")
            return MigrationResult(success=False, error=str(e))

    async def rollback_all(self) -> MigrationResult:
        """Rollback all applied migrations.

        Returns:
            MigrationResult with operation details
        """
        try:
            await self.tracker.ensure_migration_table()

            # Get all applied migrations
            applied = await self.tracker.get_applied_migrations()

            if not applied:
                return MigrationResult(success=True, rolled_back_count=0)

            # Rollback all (most recent first)
            return await self.rollback(steps=len(applied))

        except Exception as e:
            logger.error(f"Rollback all error: {e}")
            return MigrationResult(success=False, error=str(e))

    async def fresh(self) -> MigrationResult:
        """Drop all tables and re-run all migrations.

        WARNING: This will delete all data!

        Returns:
            MigrationResult with operation details
        """
        try:
            # First rollback all
            rollback_result = await self.rollback_all()
            if not rollback_result.success:
                return rollback_result

            # Then migrate up
            return await self.migrate_up()

        except Exception as e:
            logger.error(f"Fresh migration error: {e}")
            return MigrationResult(success=False, error=str(e))

    async def reset(self) -> None:
        """Drop the migration tracking table.

        WARNING: This will lose all migration history!
        """
        try:
            await self.sql_driver.execute_query(
                f"DROP TABLE IF EXISTS {self.tracker.table_name} CASCADE"
            )
            logger.info("Reset migration tracking table")
        except Exception as e:
            logger.error(f"Failed to reset migration table: {e}")
            raise

    async def status(self) -> dict:
        """Get migration status.

        Returns:
            Dictionary with migration status information
        """
        try:
            await self.tracker.ensure_migration_table()

            # Discover all migrations
            all_migrations = await self.discover_migrations()
            applied_records = await self.tracker.get_applied_migrations()
            applied_map = {r.name: r for r in applied_records}
            latest_batch = await self.tracker.get_latest_batch()

            # Build status for each migration
            migrations_status = []
            for migration in all_migrations:
                record = applied_map.get(migration.name)
                if record:
                    # Check for checksum mismatch
                    checksum_mismatch = record.checksum != migration.checksum
                    migrations_status.append({
                        "name": migration.name,
                        "status": "applied",
                        "applied_at": record.applied_at.isoformat(),
                        "batch": record.batch,
                        "checksum_mismatch": checksum_mismatch,
                    })
                else:
                    migrations_status.append({
                        "name": migration.name,
                        "status": "pending",
                    })

            applied_count = len([m for m in migrations_status if m["status"] == "applied"])
            pending_count = len([m for m in migrations_status if m["status"] == "pending"])

            return {
                "total_migrations": len(all_migrations),
                "applied_count": applied_count,
                "pending_count": pending_count,
                "latest_batch": latest_batch,
                "migrations": migrations_status,
            }

        except Exception as e:
            logger.error(f"Failed to get migration status: {e}")
            raise

    async def generate_migration(
        self,
        name: str,
        up_sql: str = "",
        down_sql: str = "",
    ) -> Path:
        """Generate a new migration file.

        Args:
            name: Migration name/description
            up_sql: Up migration SQL (optional)
            down_sql: Down migration SQL (optional)

        Returns:
            Path to the created migration directory
        """
        # Ensure migrations directory exists
        self.migrations_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Sanitize name
        safe_name = self._sanitize_name(name)

        # Create migration directory
        migration_name = f"{timestamp}_{safe_name}"
        migration_path = self.migrations_dir / migration_name
        migration_path.mkdir(exist_ok=True)

        # Write up.sql
        up_file = migration_path / "up.sql"
        if up_sql:
            up_file.write_text(up_sql, encoding="utf-8")
        else:
            up_file.write_text("-- Add migration SQL here\n", encoding="utf-8")

        # Write down.sql
        down_file = migration_path / "down.sql"
        if down_sql:
            down_file.write_text(down_sql, encoding="utf-8")
        else:
            down_file.write_text("-- Add rollback SQL here\n", encoding="utf-8")

        logger.info(f"Created migration: {migration_name}")

        return migration_path

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a migration name to be filesystem-safe."""
        # Replace spaces and special chars with underscores
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        # Remove consecutive underscores
        safe = re.sub(r"_+", "_", safe)
        # Remove leading/trailing underscores
        safe = safe.strip("_")
        return safe.lower() or "migration"

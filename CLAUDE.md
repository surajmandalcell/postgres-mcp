# CLAUDE.md - Project Guidelines

## Project Overview

**pgsql-mcp** is a PostgreSQL MCP (Model Context Protocol) server providing index tuning, query analysis, explain plans, and database health monitoring.

## Build & Test Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest -v

# Run linting
uv run ruff format --check .
uv run ruff check .

# Run type checking
uv run pyright

# Run the server locally
uv run pgsql-mcp --help
```

## Code Style

- Python 3.12+
- Line length: 150 characters
- Formatter: ruff/black
- Type hints required
- Google-style docstrings

## Architecture

```
src/postgres_mcp/
├── __init__.py          # Entry point (main function)
├── sql/                 # SQL execution & safety
├── index/               # Index optimization (DTA algorithm)
├── explain/             # Query plan analysis
├── database_health/     # Health check modules
├── migrations/          # Schema migration support
└── top_queries/         # pg_stat_statements analysis
```

## Development Rules

### SSOT (Single Source of Truth) - 100% Compliance

1. **Configuration**: All config lives in `pyproject.toml` - no duplicate config files
2. **Constants**: Define once, import everywhere - no magic strings/numbers scattered in code
3. **Types**: Single type definition per concept - reuse via imports
4. **Database schemas**: Schema definitions in one place, reference elsewhere
5. **Error messages**: Centralized error definitions when used in multiple places
6. **Version**: Single version in `pyproject.toml` only

### DRY (Don't Repeat Yourself) - 100% Compliance

1. **No copy-paste code**: Extract to functions/classes when logic repeats
2. **Shared utilities**: Common operations go in dedicated utility modules
3. **SQL patterns**: Reusable SQL builders instead of string concatenation
4. **Test fixtures**: Shared fixtures in `conftest.py`, not duplicated per test
5. **Configuration**: Environment handling in one module, imported by others

### File Hygiene - Always Remove Unnecessary Files

1. **Delete, don't comment**: Remove dead code entirely, don't comment it out
2. **No orphan files**: Delete files that are no longer imported/used
3. **No placeholder files**: Remove empty `__init__.py` if not needed for packaging
4. **Clean imports**: Remove unused imports immediately
5. **No backup files**: No `.bak`, `.old`, `.backup` files in repo
6. **No generated files in git**: Add build artifacts to `.gitignore`
7. **Remove deprecated code**: When replacing functionality, delete the old implementation

### Before Every Commit

- [ ] No duplicate logic exists
- [ ] No unused imports
- [ ] No commented-out code
- [ ] No orphan files
- [ ] All constants centralized
- [ ] Types reused via imports
- [ ] Tests pass: `uv run pytest`
- [ ] Lint passes: `uv run ruff check .`
- [ ] Types pass: `uv run pyright`

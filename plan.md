# Migration Plan: Python → Node.js (TypeScript)

## Overview

Migrate pgsql-mcp from Python to Node.js/TypeScript with Drizzle ORM integration and expanded Supabase MCP-like features.

**Current State:** Python 3.12+, FastMCP, psycopg3, pglast
**Target State:** Node.js 20+, TypeScript, MCP SDK, postgres.js/Drizzle, pgsql-ast-parser

---

## Estimated Effort

| Metric | Value |
|--------|-------|
| Total AI Hours | 24-40 hours |
| Total Tokens | ~6 million |
| Estimated Cost | $50-100 (Sonnet) / $250-300 (Opus) |

---

## Phase 1: Project Setup & Infrastructure

### 1.1 Initialize Node.js Project
- Create package.json with TypeScript configuration
- Setup tsconfig.json with strict mode
- Configure ESLint + Prettier
- Setup vitest for testing
- Add Docker compose for local PostgreSQL

### 1.2 Core Dependencies
```
Runtime:
- @modelcontextprotocol/sdk (MCP server framework)
- postgres (postgres.js - fast PostgreSQL driver)
- drizzle-orm + drizzle-kit (schema management & migrations)
- pgsql-ast-parser (SQL parsing - closest to pglast)
- zod (validation, replaces Pydantic)
- commander (CLI parsing)
- dotenv (environment config)

Development:
- typescript
- vitest + @vitest/coverage-v8
- eslint + prettier
- tsx (TypeScript execution)
```

### 1.3 Project Structure
```
src/
├── index.ts                 # Entry point + CLI
├── server.ts                # MCP server setup
├── config.ts                # Environment & configuration
├── types/                   # Shared TypeScript types
│   ├── index.ts
│   ├── database.ts
│   └── mcp.ts
├── db/                      # Database layer
│   ├── connection.ts        # Connection pool management
│   ├── driver.ts            # Query execution wrapper
│   └── safe-driver.ts       # Read-only safe execution
├── sql/                     # SQL utilities
│   ├── parser.ts            # SQL parsing & validation
│   ├── bind-params.ts       # Parameter substitution
│   └── validators.ts        # Safety validators
├── tools/                   # MCP tool implementations
│   ├── schema/              # Schema introspection tools
│   ├── query/               # Query execution tools
│   ├── explain/             # EXPLAIN plan tools
│   ├── index-advisor/       # Index optimization tools
│   ├── health/              # Database health tools
│   ├── migrations/          # Migration tools
│   ├── crud/                # CRUD operation tools (NEW)
│   ├── ddl/                 # DDL management tools (NEW)
│   ├── auth/                # User management tools (NEW)
│   └── policies/            # RLS policy tools (NEW)
└── utils/                   # Shared utilities
    ├── logger.ts
    ├── errors.ts
    └── formatting.ts

tests/
├── setup.ts                 # Test configuration
├── fixtures/                # Shared test fixtures
├── unit/                    # Unit tests
└── integration/             # Integration tests with real DB

drizzle/
├── schema.ts                # Drizzle schema definitions
└── migrations/              # Generated migrations
```

---

## Phase 2: Database Layer

### 2.1 Connection Management
- Implement connection pool using postgres.js built-in pooling
- Support both connection string and individual params
- Graceful shutdown with connection draining
- Health check endpoint for pool status

### 2.2 Query Drivers
- **StandardDriver**: Direct query execution for unrestricted mode
- **SafeDriver**: Read-only wrapper with validation
  - Parse SQL with pgsql-ast-parser
  - Whitelist: SELECT, EXPLAIN, ANALYZE, SHOW, VACUUM (analyze only)
  - Blacklist: INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE
  - Enforce read-only transaction wrapper
  - Query timeout support (default 30s)

### 2.3 Drizzle Integration
- Define schema introspection queries
- Support for pull existing schema (drizzle-kit introspect)
- Migration generation and tracking
- Type-safe query builders for CRUD operations

---

## Phase 3: SQL Parsing & Safety

### 3.1 SQL Parser Selection
**Challenge:** No direct Node.js equivalent to Python's pglast

**Options evaluated:**
1. `pgsql-ast-parser` - Best PostgreSQL support, actively maintained
2. `node-sql-parser` - Broader SQL support, less PG-specific
3. `libpg-query-node` - Node bindings to actual PG parser (most accurate)

**Recommendation:** Use `libpg-query-node` for accuracy (same parser as PostgreSQL) with `pgsql-ast-parser` as fallback

### 3.2 Safety Validation
- Statement type detection (DML vs DDL vs DQL)
- Dangerous keyword scanning
- Subquery analysis for hidden mutations
- Function call validation (block unsafe functions)
- Comment stripping before analysis

### 3.3 Bind Parameter Handling
- Detect $1, $2, :name parameter styles
- Type inference from context
- Sample value generation for EXPLAIN
- Support for LIKE pattern parameters

---

## Phase 4: Port Existing MCP Tools

### 4.1 Schema Introspection Tools
| Python | Node.js | Notes |
|--------|---------|-------|
| `list_schemas()` | `listSchemas` | Direct port |
| `list_objects()` | `listObjects` | Direct port |
| `get_object_details()` | `getObjectDetails` | Direct port |

### 4.2 Query Tools
| Python | Node.js | Notes |
|--------|---------|-------|
| `execute_sql()` | `executeSql` | Add Drizzle option |
| `explain_query()` | `explainQuery` | Direct port |

### 4.3 Index Advisor Tools
| Python | Node.js | Notes |
|--------|---------|-------|
| `analyze_workload_indexes()` | `analyzeWorkloadIndexes` | Complex - DTA algorithm |
| `analyze_query_indexes()` | `analyzeQueryIndexes` | Complex - DTA algorithm |

**DTA Algorithm Port:**
- Implement candidate index generation
- Port seed selection strategy
- Port greedy optimization loop
- Integrate HypoPG for hypothetical indexes
- Cost model calculations
- Time-bounded execution (anytime algorithm)
- Pareto optimization for recommendations

### 4.4 Health Check Tools
| Python | Node.js | Notes |
|--------|---------|-------|
| `analyze_db_health()` | `analyzeDbHealth` | 7 sub-calculators |

Health calculators to port:
- Index health (invalid, duplicate, bloated, unused)
- Replication status (lag, slots)
- Sequence health (overflow risk)
- Connection utilization
- Constraint validity
- Buffer/cache hit rates
- Vacuum/transaction ID wraparound

### 4.5 Query Analytics Tools
| Python | Node.js | Notes |
|--------|---------|-------|
| `get_top_queries()` | `getTopQueries` | pg_stat_statements |

---

## Phase 5: New Supabase-like Features

### 5.1 Enhanced CRUD Operations

**New Tools:**
- `createRecord(table, data, returning?)` - Insert with optional return
- `readRecords(table, options)` - Select with filters, pagination, sorting
- `updateRecords(table, data, filters)` - Update with conditions
- `deleteRecords(table, filters)` - Delete with conditions
- `upsertRecord(table, data, conflictColumns)` - Insert or update

**Query Options:**
```typescript
interface QueryOptions {
  select?: string[];           // Column selection
  filter?: FilterCondition[];  // WHERE conditions
  order?: OrderBy[];           // ORDER BY
  limit?: number;              // LIMIT
  offset?: number;             // OFFSET
  count?: 'exact' | 'planned'; // Include count
}
```

### 5.2 DDL Management Tools

**Table Management:**
- `createTable(schema, name, columns, constraints)`
- `alterTable(schema, name, changes)`
- `dropTable(schema, name, cascade?)`
- `renameTable(schema, oldName, newName)`

**Column Management:**
- `addColumn(table, column)`
- `alterColumn(table, column, changes)`
- `dropColumn(table, column)`
- `renameColumn(table, oldName, newName)`

**Index Management:**
- `createIndex(table, columns, options)`
- `dropIndex(name)`
- `reindex(target)`

**Constraint Management:**
- `addConstraint(table, constraint)`
- `dropConstraint(table, name)`

### 5.3 User & Auth Management

**Tools:**
- `listUsers(options)` - List with pagination/filters
- `getUser(id)` - Get user details
- `createUser(email, password, metadata?)`
- `updateUser(id, changes)`
- `deleteUser(id)`
- `listUserRoles(userId)`
- `assignRole(userId, role)`
- `revokeRole(userId, role)`

### 5.4 RLS Policy Management

**Tools:**
- `listPolicies(table?)`
- `getPolicy(table, name)`
- `createPolicy(table, name, options)`
- `alterPolicy(table, name, changes)`
- `dropPolicy(table, name)`
- `enableRls(table)`
- `disableRls(table)`

**Policy Options:**
```typescript
interface PolicyOptions {
  command: 'ALL' | 'SELECT' | 'INSERT' | 'UPDATE' | 'DELETE';
  using?: string;      // USING expression
  withCheck?: string;  // WITH CHECK expression
  roles?: string[];    // Target roles
}
```

### 5.5 Type Generation

**Tools:**
- `generateTypes(options)` - Generate TypeScript types from schema
- `generateDrizzleSchema(options)` - Generate Drizzle schema file

**Output formats:**
- TypeScript interfaces
- Zod schemas
- Drizzle table definitions

### 5.6 Extension Management

**Tools:**
- `listExtensions()`
- `installExtension(name, schema?)`
- `dropExtension(name, cascade?)`
- `getExtensionDetails(name)`

### 5.7 Function & Trigger Management

**Tools:**
- `listFunctions(schema?)`
- `getFunctionDefinition(schema, name)`
- `listTriggers(table?)`
- `getTriggerDefinition(table, name)`

---

## Phase 6: Testing Strategy

### 6.1 Unit Tests
- SQL parser validation
- Safety checker edge cases
- Bind parameter substitution
- Cost calculations
- Health check scoring
- Type generation

### 6.2 Integration Tests
- Full CRUD operations against real PostgreSQL
- DDL operations with rollback
- Index advisor with HypoPG
- Health checks on known states
- Migration tracking

### 6.3 Test Infrastructure
- Docker Compose for PostgreSQL 15/16
- Isolated test databases per suite
- Fixture data generators
- Snapshot testing for type generation

### 6.4 Coverage Target
- Minimum 80% line coverage
- 100% coverage on safety-critical paths (SQL validation)

---

## Phase 7: Configuration & Security

### 7.1 Access Modes
- **UNRESTRICTED**: Full SQL execution (development only)
- **RESTRICTED**: Read-only with SafeDriver (default)
- **READONLY**: Only SELECT statements allowed

### 7.2 Feature Flags
Enable/disable tool groups:
```typescript
interface FeatureFlags {
  schema: boolean;      // Schema introspection
  query: boolean;       // SQL execution
  explain: boolean;     // EXPLAIN plans
  indexAdvisor: boolean;// Index recommendations
  health: boolean;      // Health checks
  crud: boolean;        // CRUD operations
  ddl: boolean;         // DDL management
  auth: boolean;        // User management
  policies: boolean;    // RLS policies
  types: boolean;       // Type generation
}
```

### 7.3 Connection Security
- SSL/TLS support with certificate validation
- Connection string sanitization in logs
- Credential rotation support

---

## Phase 8: Transport & Deployment

### 8.1 Transport Options
- **stdio**: Standard MCP over stdin/stdout (default)
- **SSE**: Server-Sent Events for web clients
- **WebSocket**: For bidirectional communication (future)

### 8.2 Deployment Options
- npm package (global install)
- Docker image
- Standalone binary (pkg or bun compile)

### 8.3 CLI Interface
```bash
pgsql-mcp [options]

Options:
  --database-url <url>     PostgreSQL connection string
  --access-mode <mode>     unrestricted | restricted | readonly
  --transport <type>       stdio | sse
  --port <number>          Port for SSE transport
  --features <list>        Comma-separated feature flags
  --timeout <seconds>      Query timeout (default: 30)
  --help                   Show help
  --version                Show version
```

---

## Phase 9: Documentation

### 9.1 User Documentation
- README with quick start
- Tool reference with examples
- Configuration guide
- Security best practices

### 9.2 Developer Documentation
- Architecture overview
- Contributing guide
- API documentation (TypeDoc)

---

## Migration Sequence

### Week 1 (AI Hours 1-15)
1. Project setup and infrastructure
2. Database connection layer
3. SQL parsing and safety validation
4. Port schema introspection tools
5. Port query execution tools

### Week 2 (AI Hours 16-30)
6. Port EXPLAIN plan tools
7. Port index advisor (DTA algorithm)
8. Port health check tools
9. Port top queries tools
10. Unit tests for ported features

### Week 3 (AI Hours 31-40)
11. Implement new CRUD operations
12. Implement DDL management tools
13. Implement user/auth management
14. Implement RLS policy tools
15. Implement type generation
16. Integration tests
17. Documentation and polish

---

## Risk Mitigation

### High Risk: SQL Parser Accuracy
- **Risk:** pgsql-ast-parser may not handle all PostgreSQL syntax
- **Mitigation:** Use libpg-query-node (actual PG parser) as primary, fallback to regex for edge cases
- **Fallback:** Maintain allowlist of known-safe query patterns

### High Risk: DTA Algorithm Complexity
- **Risk:** Complex optimization algorithm may have subtle bugs
- **Mitigation:** Comprehensive test suite with known workloads
- **Fallback:** Start with simpler greedy-only approach, add sophistication iteratively

### Medium Risk: HypoPG Compatibility
- **Risk:** HypoPG behavior differences across PG versions
- **Mitigation:** Test against PG 14, 15, 16
- **Fallback:** Graceful degradation when HypoPG unavailable

### Medium Risk: Breaking Changes
- **Risk:** Behavior differences between Python and Node implementations
- **Mitigation:** Document all intentional changes, maintain compatibility mode option

---

## Success Criteria

### Functional Requirements
- [ ] All 11 existing MCP tools ported and working
- [ ] 15+ new Supabase-like tools implemented
- [ ] Works with PostgreSQL 14, 15, 16
- [ ] Passes all ported tests + new tests
- [ ] Compatible with Claude Desktop, Cursor, VS Code

### Non-Functional Requirements
- [ ] <100ms response time for simple queries
- [ ] <5s response time for index analysis
- [ ] Memory usage <256MB for typical workloads
- [ ] Clean startup/shutdown with no resource leaks

### Quality Requirements
- [ ] 80%+ test coverage
- [ ] Zero high/critical security vulnerabilities
- [ ] TypeScript strict mode with no `any` escapes
- [ ] Documented public API

---

## Post-Migration Enhancements (Future)

1. **Real-time subscriptions** - Listen to database changes
2. **Edge function management** - If targeting Supabase specifically
3. **Storage integration** - File/blob management
4. **Branching support** - Database branching for preview environments
5. **Query caching** - Intelligent result caching
6. **Audit logging** - Track all operations
7. **Multi-database support** - Connect to multiple databases
8. **GraphQL layer** - Auto-generate GraphQL from schema

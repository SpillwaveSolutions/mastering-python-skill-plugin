# Advanced Python (Chapters 1-18)

Source: `corpus/Advanced_python/`

---

## Chapter 1 — Scripts to Systems

**File:** `chapter01.md`

- **When to refactor**: Scripts with >500 lines, multiple entry points, or shared state
- **Priorities**: Modularity, error handling, config, logging, tests
- **Modern typing**: `TypeVarTuple`, `Concatenate`, structural pattern matching
- **Exceptions**: `add_note`, `ExceptionGroup` for enriched error context
- **Pydantic v2**: `Annotated` validators, model_config, computed fields
- **3.12 gains**: 5-10% faster; better error messages

---

## Chapter 2 — Modern Project Structure

**File:** `chapter02.md`

- **Layout**: `src/<pkg>/`, `tests/`, `docs/`, `pyproject.toml`
- **Config**: `.env` + Pydantic/Dynaconf; separate from code
- **Metadata**: Centralize in `pyproject.toml`
- **Pattern**:
  ```
  project/
  ├── src/mypackage/
  │   ├── __init__.py
  │   └── core.py
  ├── tests/
  ├── pyproject.toml
  └── .env
  ```

---

## Chapter 3 — Code Quality, Linting, Formatting

**File:** `chapter03.md`

- **Style**: Black + isort for formatting
- **Lint**: Ruff (fast) or Pylint (comprehensive)
- **Metrics**: Radon/Xenon for complexity; pytest-cov for coverage
- **Automation**: pre-commit + CI enforcement
- **Balance**: Strictness vs productivity; measure maintainability

---

## Chapter 4 — Type Hints & Static Analysis

**File:** `chapter04.md`

- **3.12 typing**: Built-in generics, `|` unions, PEP 695 type params/aliases
- **Protocols**: Structural typing for interface-like semantics
- **Tools**: mypy/pyright for static; Pydantic for runtime
- **Benefits**: IDE support, refactoring safety, early error detection

---

## Chapter 5 — Comprehensive Testing

**File:** `chapter05.md`

- **Pyramid**: Unit > integration > E2E > contract
- **pytest**: Fixtures, mocking, markers, parametrize
- **Property-based**: Hypothesis for edge case discovery
- **Async testing**: pytest-asyncio for coroutine tests
- **Quality**: coverage.py + mutmut for mutation testing

---

## Chapter 6 — Error Handling, Debugging, Logging

**File:** `chapter06.md`

- **Exceptions**: Notes, ExceptionGroup, layered handling
- **Resilience**: Circuit breakers, retries (tenacity)
- **Logging**: structlog for structured output
- **Observability**: OpenTelemetry/Prometheus for tracing/metrics
- **Debugging**: ipdb, debugpy; Sentry for production errors

---

## Chapter 7 — Documentation as Code

**File:** `chapter07.md`

- **Tools**: Sphinx/MyST/MkDocs for docs
- **Doc tests**: pytest/doctest integration
- **Coverage**: pdoc/docstr-coverage for docstring analysis
- **API docs**: OpenAPI/Swagger, GraphQL schemas
- **Versioning**: Keep docs fresh; track staleness

---

## Chapter 8 — Performance & Profiling

**File:** `chapter08.md`

- **Profile first**: cProfile, py-spy, scalene, line-level tools
- **CPU/memory/I/O**: Targeted profiling per bottleneck type
- **Optimize**: Better algorithms, stdlib, vectorization, Numba
- **GIL**: Understand trade-offs; 3.12+ speedups
- **Benchmark**: Validate changes with measurements

---

## Chapter 9 — Advanced Concurrency

**File:** `chapter09.md`

- **Choose**: threading (I/O) vs multiprocessing (CPU) vs asyncio
- **Executors**: `ThreadPoolExecutor`, `ProcessPoolExecutor`
- **Structured async**: `TaskGroup` for cancellation/cleanup
- **Patterns**: High-concurrency service architectures

---

## Chapter 10 — Configuration Management

**File:** `chapter10.md`

- **Tools**: Pydantic Settings, Dynaconf, python-dotenv
- **Secrets**: Environment variables, secret managers (never in code)
- **Multi-env**: Dev/staging/prod config separation
- **Validation**: Type-safe config with defaults and overrides

---

## Chapter 11 — Database Interactions & ORMs

**File:** `chapter11.md`

- **Options**: Raw drivers (`asyncpg`), query builders (Core), ORMs (SQLAlchemy 2.0)
- **Async**: SQLAlchemy async sessions, connection pooling
- **Transactions**: Isolation levels, proper commit/rollback
- **Migrations**: Alembic for schema evolution
- **Microservices**: db-per-service, eventual consistency patterns

---

## Chapter 12 — Data Validation & Pydantic

**File:** `chapter12.md`

- **Pydantic v2**: Performance improvements, `model_validator`
- **Annotated**: Combine types with validation metadata
- **Settings**: `BaseSettings` for env-based config
- **Integration**: FastAPI, CLI tools, data pipelines

---

## Chapter 13 — Dependency Management

**File:** `chapter13.md`

- **Tools**: Poetry, PDM, Hatch comparison
- **Virtual envs**: Isolation strategies, venv vs conda
- **Pinning**: Lockfiles for reproducibility
- **CI**: Reproducible builds, dependency caching

---

## Chapter 14 — Containerization & Deployment

**File:** `chapter14.md`

- **Docker**: Slim images, multi-stage builds, .dockerignore
- **Optimization**: Layer caching, minimal base images (distroless)
- **Config**: Environment variables, secrets mounting
- **Targets**: Kubernetes, serverless (Lambda, Cloud Run)
- **Pattern**:
  ```dockerfile
  FROM python:3.12-slim AS builder
  WORKDIR /app
  COPY pyproject.toml poetry.lock ./
  RUN pip install poetry && poetry install --no-dev

  FROM python:3.12-slim
  COPY --from=builder /app/.venv /app/.venv
  COPY src/ /app/src/
  CMD ["/app/.venv/bin/python", "-m", "mypackage"]
  ```

---

## Chapter 15 — Performance in Production

**File:** `chapter15.md`

- **Load profiling**: py-spy/scalene under realistic traffic
- **Database**: Query optimization, connection pooling, read replicas
- **Caching**: Redis/memcached for hot paths; cache invalidation strategies
- **Async batching**: Gather I/O operations for throughput
- **Monitoring**: APM tools (Datadog, New Relic), custom metrics
- **Scaling**: Horizontal (replicas) vs vertical (resources)

---

## Chapter 16 — Security Fundamentals

**File:** `chapter16.md`

- **Secure coding**: Input validation, output encoding, least privilege
- **Dependencies**: Safety/pip-audit for vulnerability scanning
- **Secrets**: Never commit; use env vars or secret managers
- **Authentication**: OAuth2/JWT patterns; secure password hashing (argon2)
- **Runtime**: Container hardening, non-root users, read-only filesystems

---

## Chapter 17 — Python Security Best Practices

**File:** `chapter17.md`

- **Static analysis**: semgrep, Snyk for code scanning
- **Dangerous patterns**: Avoid `eval`; use `ast.literal_eval` and safe serializers (JSON over binary formats)
- **OWASP Top 10**: Injection (parameterized queries), XSS (templating escapes), CSRF
- **Threat modeling**: Identify attack surfaces, data flows, trust boundaries
- **Secure defaults**: TLS everywhere, strict CORS, security headers

---

## Chapter 18 — Advanced Topics

**File:** `chapter18.md`

- **Subinterpreters**: True parallelism without multiprocessing overhead (3.12+)
- **C extensions**: Cython, pybind11 for performance-critical code
- **Memory management**: `__slots__`, weakrefs, gc tuning
- **Metaprogramming**: Metaclasses, descriptors (use sparingly)
- **Plugin systems**: Entry points, dynamic imports, registry patterns

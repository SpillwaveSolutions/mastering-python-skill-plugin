# Part 2: Data, Collections, and Automation (Chapters 1-14)

Source: `corpus/modern_python_series/part2/src/chapters/`

---

## Chapter 1 — Working with Data and Documents

**File:** `chapter01.md`

- **pypdf**: `PdfReader` over binary; `decrypt('')` for encrypted; pair with OCR for scanned
- **Pandas 2.2.3**: `read_csv`, boolean masks, `groupby`/`mean`; Arrow backend via `dtype_backend='pyarrow'`
- **DuckDB 1.2.0**: Register DataFrames (`con.register('df', df)`), query with SQL, `fetchdf()`
- **Pattern**: Mix Pandas for transforms, DuckDB for SQL-heavy analytics

---

## Chapter 2 — Advanced Collections

**File:** `chapter02.md`

- **OrderedDict**: `move_to_end`, `popitem(last=...)` for histories/undo
- **Sorted**: `sortedcontainers.SortedDict` for sorted maps; `bisect` for sorted lists
- **Heaps**: `heapq` for min-heaps; `queue.PriorityQueue` for thread-safe
- **Queues**: `queue.Queue`/`LifoQueue` for FIFO/LIFO; `collections.deque` for O(1) ends
- **Helpers**: `defaultdict` for grouping, `Counter` for frequency, `namedtuple`/`dataclass`

---

## Chapter 3 — Relational Databases

**File:** `chapter03.md`

- **SQLite**: `sqlite3` for embedded; parameterized queries, context managers
- **PostgreSQL/MySQL**: Drivers with pooling; JSONB/arrays (Postgres)
- **SQLAlchemy ORM**: `declarative_base`, `Column`, `relationship`; eager loading; indexes
- **Async DB**: `sqlalchemy[asyncio]` + async engines/sessions
- **Testing**: In-memory SQLite; create/drop schema per test

---

## Chapter 4 — Data Validation with Pydantic

**File:** `chapter04.md`

- **Models**: `BaseModel` with type enforcement; nested models
- **Validators**: `@validator` for rules; `Field` for constraints (`min_length`, `regex`)
- **Integration**: Natural fit with FastAPI for request/response validation
- **Usage**: Enforce positive numbers, email patterns; auto type conversion

---

## Chapter 5 — JSON and Data Serialization

**File:** `chapter05.md`

- **Built-in**: `json.dumps/loads` for strings; `json.dump/load` for files
- **Complex**: Nested dicts/lists; watch for circular refs
- **Custom types**: Custom encoders or Pydantic models; JSON Schema validation
- **Pitfalls**: No datetime/set/custom objects; convert to strings/lists

---

## Chapter 6 — Introduction to Async

**File:** `chapter06.md`

- **When**: I/O-bound with `asyncio`; CPU-bound with process pools
- **Core**: `async def`, `await`, event loop; `asyncio.run`, `gather`, `sleep`
- **Thread pools**: `ThreadPoolExecutor` for sync I/O libs
- **Process pools**: `ProcessPoolExecutor` for CPU-heavy work
- **Patterns**: Keep async non-blocking; prefer awaitable I/O

---

## Chapter 7 — Logging Best Practices

**File:** `chapter07.md`

- **Levels**: DEBUG/INFO/WARNING/ERROR/CRITICAL; avoid `print` in production
- **Architecture**: Loggers emit, Handlers route, Formatters shape
- **Config**: `basicConfig` or `dictConfig`; include timestamps
- **Best practices**: Structured messages; never log secrets; rotation handlers

---

## Chapter 8 — REST Backends and APIs

**File:** `chapter08.md`

- **REST**: HTTP verbs on resources; JSON responses
- **FastAPI**: Type-driven routes with Pydantic; `uvicorn`; auto OpenAPI docs
  ```python
  @app.get("/health")
  def health(): return {"status": "ok"}
  ```
- **Django REST Framework**: Models/serializers/viewsets for Django ecosystems
- **Patterns**: Validate inputs; FastAPI for async, DRF for full-stack Django

---

## Chapter 9 — Functional Pipelines with pipe

**File:** `chapter09.md`

- **Pipe library**: `pip install pipe`; chain with `|` operator
- **Core ops**: `map`, `filter`, `reduce` in pipelines; lazy eval
- **Usage**: Small composable lambdas; short chains; good for data cleaning

---

## Chapter 10 — UI Development

**File:** `chapter10.md`

- **Streamlit**: Widgets, charts, layout, `@st.cache_data`, session state
  ```bash
  streamlit run app.py
  ```
- **PySide6**: Desktop Qt for native apps, offline, rich widgets
- **Choice**: Streamlit for dashboards; PySide6 for desktop richness

---

## Chapter 11 — Type Checking and Static Analysis

**File:** `chapter11.md`

- **Hints**: `list[str]`, `dict[str, int]`, `|` unions, `Optional`
- **Tools**: `mypy` for mismatches; IDE autocomplete improvements
- **Patterns**: Self-document and prevent runtime surprises; public surfaces first

---

## Chapter 12 — Packaging and Distribution

**File:** `chapter12.md`

- **Structure**: `src/` layout; metadata in `pyproject.toml`
- **Build**: sdist/wheel with modern backends; `project.scripts` for entry points
- **Publish**: Upload to PyPI with twine; semver versioning
- **CI/QA**: Tests/lint/build in CI; matrix builds

---

## Chapter 13 — uv (Fast Dependency Management)

**File:** `chapter13.md`

- **What**: Drop-in faster `pip` with parallel installs and caching
- **Usage**: `uv venv`, `uv install <pkg>`, `-r requirements.txt`
- **Fit**: Lightweight speed without full project tooling

---

## Chapter 14 — PDM vs Poetry

**File:** `chapter14.md`

- **Poetry**: Dependency groups (`--with`), `poetry install/run/shell`
- **PDM**: `tool.pdm.dev-dependencies`, `pdm install --dev`, `pdm run`
- **Choice**: Poetry for opinionated all-in-one; PDM for standards-focused

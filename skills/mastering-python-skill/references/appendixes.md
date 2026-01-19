# Appendixes

---

## Part 1: Appendix A — Quick Reference (Cheat Sheet)

**Source:** `corpus/modern_python_series/part1/src/appendixes/appendixA.md`

### Syntax at a Glance

```python
# Multiple assignment
a, b = 1, 2
a, b = b, a  # swap

# Ternary
status = "adult" if age >= 18 else "minor"

# List comprehension
squares = [x*x for x in range(10)]
evens = [x for x in nums if x % 2 == 0]

# Dict comprehension
d = {k: v for k, v in pairs}

# Lambda
double = lambda x: x * 2
```

### Common Collection Operations

```python
# List
lst.append(x)      # add to end
lst.extend(other)  # add all
lst.pop()          # remove last
lst[::2]           # every 2nd

# Dict
d.get(k, default)  # safe access
d.items()          # key-value pairs
k in d             # membership

# Set
s1 | s2  # union
s1 & s2  # intersection
s1 - s2  # difference
```

### Stdlib Mini-Tour

| Module | Purpose |
|--------|---------|
| `os`/`sys` | Process, env, paths |
| `pathlib` | Modern file paths |
| `datetime` | Date/time handling |
| `json` | JSON encode/decode |
| `re` | Regular expressions |
| `itertools` | Iterator combinators |
| `functools` | Function utilities |
| `collections` | Specialized containers |

---

## Part 2: Appendix A — Tool Ecosystem Reference

**Source:** `corpus/modern_python_series/part2/src/appendixes/appendixA.md`

### Environment Tools

| Tool | Purpose |
|------|---------|
| `venv` | Built-in virtual envs |
| `pyenv` | Python version management |
| `pipx` | Global CLI tool isolation |
| `uv` | Fast pip replacement |
| `poetry` | All-in-one project management |
| `pdm` | PEP-compliant manager |

### Quality Tools

| Tool | Purpose |
|------|---------|
| `black` | Code formatter |
| `ruff` | Fast linter |
| `mypy` | Type checker |
| `pytest` | Test runner |
| `pre-commit` | Git hooks |

---

## Part 2: Appendix B — Migration Guide

**Source:** `corpus/modern_python_series/part2/src/appendixes/appendixB.md`

### Moving to Modern Python

1. **Update Python**: 3.10+ for match, 3.11+ for speed, 3.12+ for typing
2. **Add type hints**: Start with function signatures
3. **Adopt `pyproject.toml`**: Replace setup.py/setup.cfg
4. **Use `src/` layout**: Avoid import confusion
5. **Add lockfile**: Poetry lock or requirements.txt with hashes

---

## Part 2: Appendix C — Poetry vs Hatch

**Source:** `corpus/modern_python_series/part2/src/appendixes/appendixC.md`

| Feature | Poetry | Hatch |
|---------|--------|-------|
| Config | `pyproject.toml` | `pyproject.toml` |
| Lockfile | `poetry.lock` | No lockfile |
| Env management | Built-in | Built-in |
| Build backend | Poetry | Hatchling |
| Matrix CI | Plugin | Native |
| Philosophy | Opinionated | Flexible |

### GitHub Actions Example

```yaml
- uses: actions/setup-python@v5
  with:
    python-version: '3.12'
- run: pip install poetry
- run: poetry install
- run: poetry run pytest
```

---

## Part 2: Appendix D — Task Runner Patterns

**Source:** `corpus/modern_python_series/part2/src/appendixes/appendixD.md`

### Taskfile Example

```yaml
version: '3'
tasks:
  dev:
    cmds:
      - poetry install
      - poetry run pre-commit install

  test:
    cmds:
      - poetry run pytest -v

  lint:
    cmds:
      - poetry run ruff check .
      - poetry run mypy src/

  build:
    cmds:
      - poetry build
```

### Common Tasks

- **dev**: Setup development environment
- **test**: Run test suite
- **lint**: Format and lint checks
- **build**: Create distribution packages
- **db**: Database migrations/seeding
- **run**: Start application

---

## Part 2: Appendix E — Additional Reference

**Source:** `corpus/modern_python_series/part2/src/appendixes/appendixE.md`

### Useful Patterns

```python
# Context manager for timing
from contextlib import contextmanager
import time

@contextmanager
def timer(label: str):
    start = time.perf_counter()
    yield
    print(f"{label}: {time.perf_counter() - start:.3f}s")

# Retry decorator
from functools import wraps

def retry(times: int = 3):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception:
                    if attempt == times - 1:
                        raise
        return wrapper
    return decorator
```

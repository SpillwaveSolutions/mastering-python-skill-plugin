---
name: mastering-python-skill
description: Modern Python coaching covering language foundations through advanced production patterns. Use when asked to "write Python code", "explain Python concepts", "set up a Python project", "configure Poetry or PDM", "write pytest tests", "create a FastAPI endpoint", "process data with pandas", or "debug Python errors". Triggers on "Python best practices", "type hints", "async Python", "packaging", "virtual environments", "Pydantic validation".
allowed-tools:
  - Read
  - Write
  - Bash
  - Edit
metadata:
  version: 2.0.0
  domains:
    - python
    - testing
    - packaging
    - web-development
    - async
    - security
---

# Mastering Python Skill

Production-ready Python patterns with runnable code examples.

## Contents

- [Quick Navigation](#quick-navigation)
- [Quickstart Checklist](#quickstart-checklist)
- [Reference Files](#reference-files)
- [Sample CLI Tools](#sample-cli-tools)
- [Full Table of Contents](TOC.md)

---

## Quick Navigation

| Topic | Reference |
|-------|-----------|
| Type hints, Pydantic | [type-systems.md](references/foundations/type-systems.md) |
| Project layout | [project-structure.md](references/foundations/project-structure.md) |
| Linting, formatting | [code-quality.md](references/foundations/code-quality.md) |
| Async/await patterns | [async-programming.md](references/patterns/async-programming.md) |
| Error handling | [error-handling.md](references/patterns/error-handling.md) |
| Decorators | [decorators.md](references/patterns/decorators.md) |
| Context managers | [context-managers.md](references/patterns/context-managers.md) |
| Generators | [generators.md](references/patterns/generators.md) |
| pytest, fixtures | [pytest-essentials.md](references/testing/pytest-essentials.md) |
| Mocking | [mocking-strategies.md](references/testing/mocking-strategies.md) |
| Property testing | [property-testing.md](references/testing/property-testing.md) |
| FastAPI patterns | [fastapi-patterns.md](references/web-apis/fastapi-patterns.md) |
| Pydantic validation | [pydantic-validation.md](references/web-apis/pydantic-validation.md) |
| SQLAlchemy async | [database-access.md](references/web-apis/database-access.md) |
| Poetry workflow | [poetry-workflow.md](references/packaging/poetry-workflow.md) |
| pyproject.toml | [pyproject-config.md](references/packaging/pyproject-config.md) |
| Docker deployment | [docker-deployment.md](references/packaging/docker-deployment.md) |
| CI/CD pipelines | [ci-cd-pipelines.md](references/production/ci-cd-pipelines.md) |
| Monitoring | [monitoring.md](references/production/monitoring.md) |
| Security | [security.md](references/production/security.md) |

---

## Quickstart Checklist

```
- [ ] Verify Python: `python --version` (prefer 3.12+)
- [ ] Create venv: `python -m venv .venv && source .venv/bin/activate`
- [ ] Install deps: `poetry install` or `pip install -r requirements.txt`
- [ ] Quality checks: `ruff check . && mypy src/`
- [ ] Run tests: `pytest -v`
```

### Validation by Domain

| Domain | Smoke Test |
|--------|------------|
| FastAPI | `uvicorn main:app --reload` → GET `/health` |
| Async | `asyncio.run(main())` with awaited I/O |
| Packaging | `python -m build` → install in temp venv |
| Database | `alembic upgrade head` → run migration |

---

## Reference Files

### Foundations
| File | Topics |
|------|--------|
| [syntax-essentials.md](references/foundations/syntax-essentials.md) | Variables, data types, control flow |
| [type-systems.md](references/foundations/type-systems.md) | Type hints, generics, protocols |
| [project-structure.md](references/foundations/project-structure.md) | src layout, __init__.py patterns |
| [code-quality.md](references/foundations/code-quality.md) | Ruff, Black, mypy configuration |

### Patterns
| File | Topics |
|------|--------|
| [async-programming.md](references/patterns/async-programming.md) | async/await, asyncio, httpx |
| [error-handling.md](references/patterns/error-handling.md) | Exceptions, Result types |
| [decorators.md](references/patterns/decorators.md) | Function/class decorators |
| [context-managers.md](references/patterns/context-managers.md) | with statements, contextlib |
| [generators.md](references/patterns/generators.md) | yield, itertools patterns |

### Testing
| File | Topics |
|------|--------|
| [pytest-essentials.md](references/testing/pytest-essentials.md) | Fixtures, markers, parametrize |
| [mocking-strategies.md](references/testing/mocking-strategies.md) | unittest.mock, pytest-mock |
| [property-testing.md](references/testing/property-testing.md) | Hypothesis strategies |

### Web APIs
| File | Topics |
|------|--------|
| [fastapi-patterns.md](references/web-apis/fastapi-patterns.md) | Dependency injection, middleware |
| [pydantic-validation.md](references/web-apis/pydantic-validation.md) | Models, validators, settings |
| [database-access.md](references/web-apis/database-access.md) | SQLAlchemy async, Alembic |

### Packaging
| File | Topics |
|------|--------|
| [poetry-workflow.md](references/packaging/poetry-workflow.md) | Dependency management, publishing |
| [pyproject-config.md](references/packaging/pyproject-config.md) | PEP 621, tool configuration |
| [docker-deployment.md](references/packaging/docker-deployment.md) | Multi-stage builds, compose |

### Production
| File | Topics |
|------|--------|
| [ci-cd-pipelines.md](references/production/ci-cd-pipelines.md) | GitHub Actions, matrix testing |
| [monitoring.md](references/production/monitoring.md) | Logging, metrics, tracing |
| [security.md](references/production/security.md) | OWASP, auth, secrets |

---

## Sample CLI Tools

Runnable examples in [sample-cli/](sample-cli/):

| Tool | Demonstrates |
|------|-------------|
| [async_fetcher.py](sample-cli/async_fetcher.py) | Async HTTP with rate limiting |
| [config_loader.py](sample-cli/config_loader.py) | Pydantic settings, .env files |
| [db_cli.py](sample-cli/db_cli.py) | SQLAlchemy async CRUD |

```bash
# Quick test
python sample-cli/async_fetcher.py https://httpbin.org/get
python sample-cli/config_loader.py
python sample-cli/db_cli.py init && python sample-cli/db_cli.py list
```

---

## When NOT to Use

- **Non-Python languages**: Use language-specific skills
- **ML/AI model internals**: Use PyTorch/TensorFlow skills
- **Cloud infrastructure**: Use AWS/GCP skills for infra (this covers code)
- **Legacy Python 2**: Focus is Python 3.10+

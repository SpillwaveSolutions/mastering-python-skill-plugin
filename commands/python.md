---
name: python
description: Get Python coaching on syntax, patterns, testing, packaging, and best practices
arguments:
  - name: topic
    description: The Python topic or question to get help with
    required: false
---

# /python Command

Invoke the mastering-python-skill for Python coaching and guidance.

## Usage

```
/python                          # General Python help
/python async await patterns     # Specific topic
/python FastAPI endpoint         # Framework guidance
/python pytest fixtures          # Testing help
```

## What This Command Does

1. Loads the mastering-python-skill with comprehensive Python knowledge
2. Applies modern Python best practices (3.10+)
3. Provides guidance based on the topic:
   - **Foundations**: Syntax, types, OOP, control flow (Part 1)
   - **Data & APIs**: Pandas, async, REST, packaging (Part 2)
   - **Production**: Security, performance, deployment (Advanced)

## Topics Covered

| Domain | Examples |
|--------|----------|
| Syntax & Types | Type hints, dataclasses, protocols |
| Testing | pytest, fixtures, mocking, parametrize |
| Packaging | Poetry, PDM, pyproject.toml, uv |
| Web | FastAPI, Django REST, Pydantic |
| Data | Pandas, DuckDB, JSON serialization |
| Async | asyncio, concurrency, TaskGroup |
| Quality | Black, Ruff, mypy, pre-commit |

## Examples

```
/python How do I write a context manager?
/python Best practices for FastAPI error handling
/python Poetry vs PDM comparison
/python Type hints for generic functions
```

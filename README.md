# Mastering Python Plugin

Modern Python coaching covering language foundations through advanced production patterns.

## Installation

```bash
cd ~/.claude/skills
git clone https://github.com/SpillwaveSolutions/mastering-python-skill-plugin.git
```

## Features

### Skill: mastering-python-skill

Comprehensive Python guidance triggered automatically when you:
- Ask to "write Python code" or "explain Python concepts"
- Request help with "pytest tests", "FastAPI endpoints", or "Pydantic validation"
- Need guidance on "Poetry", "PDM", "type hints", or "async Python"

### Command: /python

```bash
/python                          # General Python help
/python async await patterns     # Specific topic
/python FastAPI endpoint         # Framework guidance
```

### Agent: python-expert

Expert assistant for production-quality Python development:
- Code writing with modern patterns
- Refactoring to best practices
- Testing strategies
- Package configuration

## Topics Covered

| Domain | Content |
|--------|---------|
| **Foundations** | Syntax, types, OOP, control flow, modules |
| **Testing** | pytest, fixtures, mocking, parametrize |
| **Packaging** | Poetry, PDM, uv, pyproject.toml |
| **Web** | FastAPI, Django REST, Pydantic |
| **Data** | Pandas, DuckDB, JSON serialization |
| **Async** | asyncio, concurrency, TaskGroup |
| **Quality** | Black, Ruff, mypy, pre-commit |
| **Production** | Docker, security, performance, deployment |

## Content Sources

- **Part 1** (13 chapters): Python foundations
- **Part 2** (14 chapters): Data, async, REST, packaging
- **Advanced** (18 chapters): Systems design, security, deployment

## License

MIT

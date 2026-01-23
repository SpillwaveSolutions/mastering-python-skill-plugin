# Mastering Python Plugin

Modern Python coaching covering language foundations through advanced production patterns.

## Installation

### skilz Installation for Agentic Skill
View all platforms →

#### Claude Code (CLI)
**Fast**
```bash
skilz install SpillwaveSolutions/mastering-python-skill-plugin/mastering-python-skill
```

#### OpenCode (CLI)
**Fast**
```bash
skilz install SpillwaveSolutions/mastering-python-skill-plugin/mastering-python-skill --agent opencode
```

#### OpenAI Codex (CLI)
**Native**
```bash
skilz install SpillwaveSolutions/mastering-python-skill-plugin/mastering-python-skill --agent codex
```

#### Gemini CLI (Project)
**Project**
```bash
skilz install SpillwaveSolutions/mastering-python-skill-plugin/mastering-python-skill --agent gemini
```

**First time? Install Skilz:** `pip install skilz`

Works with 14 AI coding assistants including Cursor, Aider, Copilot, Windsurf, Qwen, Kimi, and more. [View All Agents](https://github.com/SpillwaveSolutions/skilz-cli/blob/main/docs/COMPREHENSIVE_USER_GUIDE.md).

### For Claude Desktop
**Easy**

Download Agent Skill ZIP, extract and copy to `~/.claude/skills/` then restart Claude Desktop

### Manual Installation
1. Clone the repository:
```bash
git clone https://github.com/SpillwaveSolutions/mastering-python-skill-plugin
```

2. Copy the agent skill directory:
```bash
cp -r mastering-python-skill-plugin/skills/mastering-python-skill ~/.claude/skills/
```

### Alternate Method (GitHub)
```bash
skilz install -g https://github.com/SpillwaveSolutions/mastering-python-skill-plugin
```

### Additional Resources
- [Skilz CLI User Guide](https://github.com/SpillwaveSolutions/skilz-cli/blob/main/docs/COMPREHENSIVE_USER_GUIDE.md)
- [Marketplace Link](https://skillzwave.ai/agent-skill/spillwavesolutions__mastering-python-skill-plugin__mastering-python-skill/)

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

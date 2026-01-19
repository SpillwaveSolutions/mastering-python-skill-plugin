---
name: mastering-python-skill
description: Modern Python coaching covering language foundations through advanced production patterns. Use when asked to "write Python code", "explain Python concepts", "set up a Python project", "configure Poetry or PDM", "write pytest tests", "create a FastAPI endpoint", "process data with pandas", or "debug Python errors". Triggers on "Python best practices", "type hints", "async Python", "packaging", "virtual environments", "Pydantic validation".
allowed-tools:
  - Read
  - Write
  - Bash
  - Edit
metadata:
  version: 1.0.0
  domains:
    - python
    - testing
    - packaging
    - web-development
    - data-processing
    - async
---

# Mastering Python Skill

Modern Python coaching from Part 1 (chapters 1-13), Part 2 (chapters 1-14), and Advanced Python (chapters 1-18).

## Contents

- [How to Navigate](#how-to-navigate)
- [Quickstart Checklist](#quickstart-checklist)
- [Content Organization](#content-organization)
- [Finding Source Material](#finding-source-material)
- [When NOT to Use](#when-not-to-use)

---

## How to Navigate

1. **Identify the domain** (e.g., async, REST, packaging, testing)
2. **Read the matching reference** — see [Content Organization](#content-organization)
3. **Open corpus chapter** for full details/examples
4. **Apply changes** with validation — see [Quickstart Checklist](#quickstart-checklist)

Load only what you need. Keep this overview in context; pull specifics from reference files.

---

## Quickstart Checklist

Copy and track progress:

```
- [ ] Verify Python version: `python --version` (prefer 3.12+)
- [ ] Create/activate venv: `python -m venv .venv && source .venv/bin/activate`
- [ ] Install dependencies: `pip install -r requirements.txt` (or `poetry install`)
- [ ] Run quality checks: `black . && ruff check . && mypy src/`
- [ ] Run tests: `pytest -v`
- [ ] Smoke test the feature (see validation patterns below)
```

### Validation Patterns

| Domain | Smoke Test |
|--------|------------|
| REST (FastAPI) | `uvicorn main:app --reload` → hit `/health` |
| REST (Django) | `python manage.py runserver` → hit endpoint |
| Packaging | `python -m build` → install wheel in temp venv |
| Data pipeline | Run small dataset through transform |
| UI (Streamlit) | `streamlit run app.py` → click key path |
| Async | `asyncio.run(main())` with awaited I/O |

---

## Content Organization

| Content | Reference | Corpus Source |
|---------|-----------|---------------|
| **Part 1** (13 chapters): Syntax, OOP, functions, modules, testing, Poetry | [references/part1-chapters.md](references/part1-chapters.md) | `corpus/modern_python_series/part1/src/chapters/` |
| **Part 2** (14 chapters): Data, async, REST, packaging, uv, PDM | [references/part2-chapters.md](references/part2-chapters.md) | `corpus/modern_python_series/part2/src/chapters/` |
| **Advanced** (18 chapters): Systems design, concurrency, security, deployment | [references/advanced-python.md](references/advanced-python.md) | `corpus/Advanced_python/` |
| **Appendixes**: Quick reference, tools, migration, task runners | [references/appendixes.md](references/appendixes.md) | `corpus/modern_python_series/*/src/appendixes/` |

### Quick Topic Lookup

| Topic | Reference | Chapter |
|-------|-----------|---------|
| Type hints | [Part 1](references/part1-chapters.md) | Ch 2, 5 |
| Classes/OOP | [Part 1](references/part1-chapters.md) | Ch 3 |
| Testing/pytest | [Part 1](references/part1-chapters.md) | Ch 8 |
| Poetry workflow | [Part 1](references/part1-chapters.md) | Ch 13 |
| Pandas/DuckDB | [Part 2](references/part2-chapters.md) | Ch 1 |
| Async/concurrency | [Part 2](references/part2-chapters.md) + [Advanced](references/advanced-python.md) | P2 Ch 6, Adv Ch 9 |
| FastAPI/REST | [Part 2](references/part2-chapters.md) | Ch 8 |
| Pydantic | [Part 2](references/part2-chapters.md) + [Advanced](references/advanced-python.md) | P2 Ch 4, Adv Ch 12 |
| Docker/deployment | [Advanced](references/advanced-python.md) | Ch 14 |
| Security | [Advanced](references/advanced-python.md) | Ch 16, 17 |
| Performance | [Advanced](references/advanced-python.md) | Ch 8, 15 |

---

## Finding Source Material

| Content | Path |
|---------|------|
| Part 1 chapters | `corpus/modern_python_series/part1/src/chapters/chapter01.md` – `chapter13.md` |
| Part 1 appendix | `corpus/modern_python_series/part1/src/appendixes/appendixA.md` |
| Part 2 chapters | `corpus/modern_python_series/part2/src/chapters/chapter01.md` – `chapter14.md` |
| Part 2 appendixes | `corpus/modern_python_series/part2/src/appendixes/appendixA.md` – `appendixE.md` |
| Advanced Python | `corpus/Advanced_python/chapter01.md` – `chapter18.md` |

---

## When NOT to Use

- **Non-Python languages**: Use language-specific skills for JavaScript, Go, Rust, Java
- **ML/AI model architecture**: Use specialized skills for PyTorch/TensorFlow internals
- **Cloud infrastructure**: Use AWS/GCP skills for deployment config (this skill covers code, not infra)
- **Legacy Python 2**: This skill focuses on Python 3.10+ patterns
- **Framework-specific deep dives**: For Django admin customization or Flask blueprints, consult framework docs

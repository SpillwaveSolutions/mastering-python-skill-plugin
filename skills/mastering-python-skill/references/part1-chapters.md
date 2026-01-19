# Part 1: Python Foundations (Chapters 1-13)

Source: `corpus/modern_python_series/part1/src/chapters/`

---

## Chapter 1 — Python Landscape

**File:** `chapter01.md`

- **3.13 Highlights**: Friendlier tracebacks, faster CPython, better typing (TypedDict/generics), subinterpreters
- **Ecosystem**: PyPI scale; data (`numpy`, `pandas`), AI (`torch`, `sklearn`), web (`fastapi`, `django`), automation (`pathlib`, `subprocess`, `rich`)
- **Tooling**: Envs (`venv`, `poetry`, `conda`, `uv`), formatters (`black`), linters (`ruff`), typing (`mypy`), testing (`pytest`)
- **Trade-offs**: GIL mitigations (multiprocessing/async/subinterpreters); memory via generators, `__slots__`, streaming

---

## Chapter 2 — Syntax and Data Types

**File:** `chapter02.md`

- **Indentation**: 4 spaces; blocks defined by indent
- **Names**: `snake_case` for vars/functions, `CamelCase` for classes, `UPPER_CASE` for constants
- **Numbers**: `int` unbounded, `float` for decimals; use `decimal.Decimal` for money
- **Strings**: f-strings for formatting; slices `s[::]`, methods (`lower`, `split`, `join`, `replace`)
- **Collections**:
  - `list`: ordered, mutable; `append`, `extend`, `pop`, slicing
  - `tuple`: ordered, immutable; packing/unpacking
  - `set`: unique, unordered; `|`, `&`, `-`, `^`
  - `dict`: key/value; `.get`, `.items`, membership
- **Type Hints**:
  ```python
  from typing import Iterable

  def average(nums: Iterable[float]) -> float:
      return sum(nums) / len(nums) if nums else 0.0
  ```

---

## Chapter 3 — Classes and OOP

**File:** `chapter03.md`

- **Define/Init**:
  ```python
  class Dog:
      def __init__(self, name: str, breed: str):
          self.name = name
          self.breed = breed
      def bark(self) -> str:
          return f"{self.name} says woof!"
  ```
- **Inheritance**: Reuse via `super()`; override selectively
- **Duck Typing**: Depend on behavior, not type; add tests to guard expectations
- **Dunder Methods**: `__len__`, `__iter__`, `__getitem__`, `__repr__` for native feel
- **Dataclasses/Protocols (3.10+)**: `@dataclass` for boilerplate; `Protocol` for structural typing

---

## Chapter 4 — Control Flow and Functions

**File:** `chapter04.md`

- **Conditionals**: `if`/`elif`/`else`; ternary: `status = "adult" if age >= 18 else "minor"`
- **Loops**:
  - `for` with `enumerate`, `range`
  - `while` for condition-driven
  - Comprehensions: `[x*x for x in nums if x % 2 == 0]`
- **Match (3.10+)**: Structural pattern matching for clear branching
- **Functions**:
  ```python
  def greet(name: str, loud: bool = False) -> str:
      msg = f"Hello, {name}"
      return msg.upper() if loud else msg
  ```
- **Error Handling**:
  ```python
  try:
      risky()
  except ValueError as exc:
      handle(exc)
  finally:
      cleanup()
  ```

---

## Chapter 5 — Advanced Python Features

**File:** `chapter05.md`

- **Comprehensions**: Lists/sets/dicts; `{name: age for name, age in zip(names, ages)}`
- **Generators**:
  ```python
  def read_lines(path: str):
      with open(path) as fh:
          for line in fh:
              yield line.rstrip("\n")
  ```
- **Context Managers**:
  ```python
  from contextlib import contextmanager

  @contextmanager
  def opened(path):
      fh = open(path)
      try:
          yield fh
      finally:
          fh.close()
  ```
- **Decorators**:
  ```python
  from functools import wraps

  def log_calls(fn):
      @wraps(fn)
      def wrapper(*args, **kwargs):
          print(f"calling {fn.__name__}")
          return fn(*args, **kwargs)
      return wrapper
  ```
- **Typing**: `Iterable[int]`, `Mapping[str, str]`, `TypedDict`, `Literal`, `Union`, `Optional`

---

## Chapter 6 — Modules and Packages

**File:** `chapter06.md`

- **Modules**: `.py` file; use `if __name__ == "__main__":` for runnable examples
- **Packages**: Directory with `__init__.py`; expose clean API, avoid stdlib collisions
- **Imports**: Explicit imports; avoid `from x import *`; alias common libs (`import numpy as np`)
- **Structure**: Group imports: stdlib, third-party, local; avoid circular imports
- **Absolute vs Relative**: Prefer absolute; use relative (`from .foo import bar`) for siblings

---

## Chapter 7 — Your First Python Project

**File:** `chapter07.md`

- **Layout**: `src/` layout: `project/{src/<pkg>/..., tests/, README.md, pyproject.toml}`
- **Environments**: `python -m venv .venv && source .venv/bin/activate`
- **Modern managers**: Poetry or PDM (both create lockfiles)
- **Script pattern**:
  ```python
  def main():
      ...
  if __name__ == "__main__":
      main()
  ```

---

## Chapter 8 — Testing Your Python Code

**File:** `chapter08.md`

- **Pytest basics**: Files `test_*.py`, functions `test_*`, plain `assert`
  ```python
  def test_divide_by_zero():
      with pytest.raises(ValueError):
          divide(1, 0)
  ```
- **Fixtures**: `@pytest.fixture`; `yield` for cleanup; scopes (`function`, `module`, `session`)
- **Parametrize**: `@pytest.mark.parametrize(...)` for case coverage
- **Mocking**: `unittest.mock` (`mock.patch`, `Mock`); `assert_called_once_with`

---

## Chapter 9 — Setting Up Your Python Environment

**File:** `chapter09.md`

- **Version management**: `pyenv` to install/switch versions
- **Isolation**: `python -m venv .venv` per project; `pipx` for global CLI tools
- **Dependencies**: `pip freeze > requirements.txt`; `pip-tools` for locking
- **Tooling**: IDE (VS Code/PyCharm); `black`, `ruff`, `mypy`, `pytest`; pre-commit hooks

---

## Chapter 10 — Files, Paths, and Filesystem

**File:** `chapter10.md`

- **I/O**: `with open(path, mode, encoding="utf-8")`; stream large files line-by-line
- **pathlib**: `Path` for OS-agnostic paths; `/` joins, `.read_text()`, `.glob("**/*.py")`
- **Directory ops**: `Path.mkdir(parents=True, exist_ok=True)`; `shutil.copy2/move/rmtree`
- **Temp files**: `tempfile` for safe temp files/dirs

---

## Chapter 11 — Standard Library Essentials

**File:** `chapter11.md`

- **datetime/time**: `datetime.now()`, `strftime`/`strptime`; prefer aware datetimes with `zoneinfo`
- **Regex**: `re.search`, `match`, `findall`, `sub`; raw strings; capture groups
- **CSV**: `csv.reader`/`DictReader`; handle dialects/encoding
- **CLIs**: `argparse` for positional/optional args, flags, subcommands

---

## Chapter 12 — Automation, System Tools, Configuration

**File:** `chapter12.md`

- **subprocess**: `subprocess.run([...], check=True, capture_output=True, text=True)`
- **Filesystem**: `pathlib`/`shutil` for mkdir/copy/move; `glob` patterns
- **Env/config**: `os.environ`; load `.env`; parse YAML; config hierarchy (defaults → file → env → CLI)

---

## Chapter 13 — Poetry Workflow

**File:** `chapter13.md`

- **Setup**: `poetry new/init`; edit `pyproject.toml`
- **Dependencies**: `poetry add <pkg>`; dev deps via `--group dev`
- **Environments**: `poetry env use`, `poetry shell`, `poetry run`
- **QA**: Configure `pytest`, `ruff`, `black`, `mypy`; pre-commit hooks
- **Packaging**: `poetry build` for wheel/sdist; publish to PyPI

---

## Quick Patterns

- **Choose obvious way**: Prefer built-ins and stdlib (`pathlib`, `itertools`, `functools`)
- **Immutability**: Tuples/`dataclass(frozen=True)` for stable data
- **Comprehensions**: When clarity improves; otherwise explicit loops
- **Testing**: When using duck typing, add unit tests for behavior
- **Performance**: Stream with generators; prefer `numpy`/`pandas` for heavy math

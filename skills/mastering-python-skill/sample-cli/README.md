# Sample CLI Tools

Runnable examples demonstrating Python patterns from the reference documentation.

## Prerequisites

Install dependencies:

```bash
pip install httpx pydantic pydantic-settings sqlalchemy aiosqlite python-dotenv
```

Or with Poetry:

```bash
poetry add httpx pydantic pydantic-settings sqlalchemy aiosqlite python-dotenv
```

## Tools

### 1. Async Fetcher (`async_fetcher.py`)

Demonstrates async HTTP client patterns from [async-programming.md](../references/patterns/async-programming.md).

**Features:**
- Async/await with httpx
- Concurrent requests with asyncio.gather()
- Rate limiting and timeout handling
- Structured error handling

**Usage:**

```bash
# Fetch single URL
python sample-cli/async_fetcher.py https://api.github.com

# Fetch multiple URLs concurrently
python sample-cli/async_fetcher.py https://httpbin.org/get https://api.github.com

# With rate limit (max concurrent requests)
python sample-cli/async_fetcher.py --concurrency 3 url1 url2 url3 url4 url5
```

---

### 2. Config Loader (`config_loader.py`)

Demonstrates Pydantic settings patterns from [pydantic-validation.md](../references/web-apis/pydantic-validation.md).

**Features:**
- Environment variable loading
- .env file support
- Type validation and coercion
- Nested configuration models
- Secret handling with SecretStr

**Usage:**

```bash
# Show default configuration
python sample-cli/config_loader.py

# With environment variables
APP_DEBUG=true APP_LOG_LEVEL=DEBUG python sample-cli/config_loader.py

# With .env file (create .env in current directory)
echo "APP_DEBUG=true" > .env
echo "DATABASE_URL=postgresql://localhost/mydb" >> .env
python sample-cli/config_loader.py
```

---

### 3. Database CLI (`db_cli.py`)

Demonstrates SQLAlchemy async patterns from [database-access.md](../references/web-apis/database-access.md).

**Features:**
- Async SQLAlchemy with aiosqlite
- Repository pattern
- CRUD operations
- Transaction management

**Usage:**

```bash
# Initialize database and add sample data
python sample-cli/db_cli.py init

# List all users
python sample-cli/db_cli.py list

# Add a user
python sample-cli/db_cli.py add "John Doe" john@example.com

# Get user by ID
python sample-cli/db_cli.py get 1

# Update user
python sample-cli/db_cli.py update 1 --name "Jane Doe" --email jane@example.com

# Delete user
python sample-cli/db_cli.py delete 1
```

---

## Running Tests

Each CLI tool includes basic validation. Run them to verify your environment:

```bash
# Quick smoke tests
python sample-cli/async_fetcher.py https://httpbin.org/get
python sample-cli/config_loader.py
python sample-cli/db_cli.py init && python sample-cli/db_cli.py list
```

## Related Documentation

| Tool | Reference |
|------|-----------|
| async_fetcher.py | [async-programming.md](../references/patterns/async-programming.md) |
| config_loader.py | [pydantic-validation.md](../references/web-apis/pydantic-validation.md) |
| db_cli.py | [database-access.md](../references/web-apis/database-access.md) |

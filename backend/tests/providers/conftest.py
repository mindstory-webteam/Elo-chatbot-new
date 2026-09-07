"""
Fixtures specific to provider and database tests.

Provider tests focus on storage, vector, and database providers.
Global fixtures from the parent conftest.py are automatically available.
"""
import os
import uuid

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from app.config import settings


@pytest.fixture
def mock_connection_string():
    """Mock database connection string."""
    return "postgresql://postgres:postgres@localhost:5432/test_db"


@pytest.fixture
def mock_storage_provider():
    """Mock storage provider for testing."""
    mock = MagicMock()
    mock.upload = AsyncMock(return_value="https://storage.example.com/file.pdf")
    mock.download = AsyncMock(return_value=b"file content")
    mock.delete = AsyncMock(return_value=True)
    return mock


@pytest.fixture(scope="session")
def postgres_url() -> str:
    """DSN for the scratch database used by the data-layer tests."""
    return os.environ.get(
        "TEST_POSTGRES_URL",
        "postgresql://postgres:postgres@localhost:5432/elo_test",
    )


@pytest.fixture(scope="session")
def _postgres_ready(postgres_url) -> bool:
    """
    Apply migrations once per session.

    Skips the whole data-layer suite when no PostgreSQL server is reachable, so
    the rest of the test run still works on a machine without a database.
    """
    import asyncio
    import asyncpg

    async def _prepare():
        conn = await asyncpg.connect(postgres_url)
        await conn.close()
        from app.database.migrate import apply_migrations
        await apply_migrations(postgres_url)

    try:
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(_prepare())
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"PostgreSQL not available at {postgres_url}: {exc}",
                    allow_module_level=True)
    return True


_TEST_TABLES = (
    "conversations", "pages", "crawl_jobs", "long_term_memory", "users",
    "sites", "trigger_events", "handoff_sessions", "qa_pairs", "leads",
    "documents", "platform_settings",
)


@pytest.fixture
async def pg_db(postgres_url, _postgres_ready):
    """
    A connected PostgresDB against a truncated scratch database.

    Each test starts from an empty schema so assertions on counts are exact.
    """
    from app.database.postgres import PostgresDB

    db = PostgresDB()
    with patch.object(settings, "POSTGRES_URL", postgres_url), \
         patch.object(settings, "POSTGRES_AUTO_MIGRATE", False):
        await db.connect()
    async with db.pool.acquire() as conn:
        await conn.execute(f"TRUNCATE {', '.join(_TEST_TABLES)} RESTART IDENTITY CASCADE")
    try:
        yield db
    finally:
        await db.disconnect()


@pytest.fixture
async def pg_provider(postgres_url, _postgres_ready):
    """The same scratch database, exposed through the provider interface."""
    from app.providers.database.postgres_provider import PostgresProvider

    provider = PostgresProvider()
    with patch.object(settings, "POSTGRES_URL", postgres_url), \
         patch.object(settings, "POSTGRES_AUTO_MIGRATE", False):
        await provider.connect()
    async with provider.pool.acquire() as conn:
        await conn.execute(f"TRUNCATE {', '.join(_TEST_TABLES)} RESTART IDENTITY CASCADE")
    try:
        yield provider
    finally:
        await provider.disconnect()


@pytest.fixture
async def a_site(pg_db) -> str:
    """A persisted site to hang site-scoped fixtures off."""
    site_id = f"site_{uuid.uuid4().hex[:8]}"
    await pg_db.create_site({
        "site_id": site_id,
        "name": "Fixture Site",
        "url": "https://fixture.example.com",
        "status": "ready",
    })
    return site_id


@pytest.fixture
async def a_user(pg_db) -> dict:
    """A persisted user."""
    email = f"user_{uuid.uuid4().hex[:8]}@example.com"
    user_id = await pg_db.create_user({
        "email": email,
        "name": "Fixture User",
        "password_hash": "hashed",
        "role": "user",
    })
    return await pg_db.get_user_by_id(user_id)


@pytest.fixture
def a_session() -> str:
    """A unique chat session id."""
    return f"sess_{uuid.uuid4().hex[:8]}"

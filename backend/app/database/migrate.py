"""
PostgreSQL migration runner.

Applies the ``.sql`` files in ``app/database/migrations`` in filename order and
records each one in a ``schema_migrations`` table so they are only ever applied
once. Each migration runs inside a transaction.

Usage::

    python -m app.database.migrate            # apply pending migrations
    python -m app.database.migrate status     # show applied / pending
"""
import asyncio
import hashlib
import os
import sys
from pathlib import Path
from typing import List, Tuple

import asyncpg
from loguru import logger

from app.config import settings

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_CREATE_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    checksum    TEXT NOT NULL,
    applied_at  TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc')
);
"""


def discover_migrations() -> List[Tuple[str, Path]]:
    """Return (version, path) for every migration file, sorted by version."""
    if not MIGRATIONS_DIR.is_dir():
        return []
    files = sorted(MIGRATIONS_DIR.glob("*.sql"), key=lambda p: p.name)
    return [(p.stem, p) for p in files]


def _checksum(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()[:16]


async def apply_migrations(dsn: str = None) -> int:
    """
    Apply all pending migrations. Returns the number applied.

    Safe to call repeatedly and safe to call concurrently: an advisory lock
    serialises competing workers so two processes starting at once cannot both
    try to create the same tables.
    """
    dsn = dsn or settings.postgres_dsn
    conn = await asyncpg.connect(dsn)
    applied = 0
    try:
        await conn.execute(_CREATE_MIGRATIONS_TABLE)
        # Serialise migration runs across processes.
        await conn.execute("SELECT pg_advisory_lock(982451653)")
        try:
            rows = await conn.fetch("SELECT version, checksum FROM schema_migrations")
            done = {r["version"]: r["checksum"] for r in rows}

            for version, path in discover_migrations():
                sql = path.read_text(encoding="utf-8")
                digest = _checksum(sql)

                if version in done:
                    if done[version] != digest:
                        logger.warning(
                            f"Migration {version} has changed since it was applied "
                            f"(recorded {done[version]}, file {digest}). "
                            "Add a new migration instead of editing an applied one."
                        )
                    continue

                logger.info(f"Applying migration {version}...")
                async with conn.transaction():
                    await conn.execute(sql)
                    await conn.execute(
                        "INSERT INTO schema_migrations (version, checksum) VALUES ($1, $2)",
                        version,
                        digest,
                    )
                applied += 1
                logger.info(f"Applied migration {version}")

            if applied == 0:
                logger.info("Database schema is up to date")
            else:
                logger.info(f"Applied {applied} migration(s)")
        finally:
            await conn.execute("SELECT pg_advisory_unlock(982451653)")
    finally:
        await conn.close()
    return applied


async def migration_status(dsn: str = None) -> None:
    """Print which migrations are applied and which are pending."""
    dsn = dsn or settings.postgres_dsn
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(_CREATE_MIGRATIONS_TABLE)
        rows = await conn.fetch("SELECT version, applied_at FROM schema_migrations")
        done = {r["version"]: r["applied_at"] for r in rows}
        print(f"{'STATUS':<10} {'VERSION':<32} APPLIED AT")
        print("-" * 72)
        for version, _ in discover_migrations():
            if version in done:
                print(f"{'applied':<10} {version:<32} {done[version]}")
            else:
                print(f"{'pending':<10} {version:<32} -")
    finally:
        await conn.close()


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "up"

    if command in ("status", "--status"):
        asyncio.run(migration_status())
        return 0

    if command in ("up", "--up"):
        try:
            asyncio.run(apply_migrations())
        except Exception as exc:  # pragma: no cover - CLI surface
            logger.error(f"Migration failed: {exc}")
            return 1
        return 0

    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

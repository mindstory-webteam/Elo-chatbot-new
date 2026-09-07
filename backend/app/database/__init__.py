"""
Database connections and operations.

Backed by PostgreSQL. This module maintains backward compatibility while
supporting the provider architecture. For new code, prefer using:
  - app.core.dependencies for FastAPI dependency injection
  - app.providers for direct provider access

The vector store re-exports are resolved lazily so database-only entry points
(notably `python -m app.database.migrate`) need not import the FAISS/LangChain
stack just to run schema migrations.
"""
from .postgres import PostgresDB, get_postgres, reset_postgres

# Canonical accessor for the application database.
get_database = get_postgres

__all__ = [
    "PostgresDB",
    "get_postgres",
    "get_database",
    "reset_postgres",
    "VectorStore",
    "get_vector_store",
]

_LAZY = {"VectorStore", "get_vector_store"}


def __getattr__(name):
    if name in _LAZY:
        from . import vector_store as _vs
        return getattr(_vs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

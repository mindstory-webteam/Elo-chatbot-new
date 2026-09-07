"""
PostgreSQL connection and operations for conversations, pages, and memory.

This module replaces the previous MongoDB implementation. Every public method
keeps the exact signature and return shape the rest of the application already
depends on -- including the ``_id`` key that used to hold a MongoDB ObjectId and
now holds the row's UUID primary key as a string.

Storage model
-------------
Ordered, index-addressable embedded arrays (conversation ``messages`` and
``notes``, site ``triggers``, handoff ``messages`` and ``ai_conversation``) are
stored as JSONB. The application mutates them positionally and reads them whole,
so JSONB preserves the existing semantics exactly. Everything the app filters or
sorts on is a real indexed column.

Each table also carries a JSONB ``data`` column. Any key written through
``update_site``/``update_user``/etc. that has no dedicated column lands there and
is merged back to the top level on read, which preserves MongoDB's schemaless
"whatever you set comes back" behaviour that several routes rely on.
"""
import asyncio
import json
import re
import uuid as _uuid_module
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
from loguru import logger

from app.config import settings

# ---------------------------------------------------------------------------
# JSON encoding helpers
#
# MongoDB round-tripped datetimes inside embedded documents natively. JSONB
# cannot, so datetimes are stored as ISO-8601 strings and revived on read for
# the well-known timestamp keys. Callers keep receiving datetime objects and all
# existing datetime arithmetic in the routes keeps working.
# ---------------------------------------------------------------------------

_DATETIME_KEYS = frozenset({
    "timestamp",
    "created_at",
    "updated_at",
    "feedback_at",
    "resolved_at",
    "captured_at",
    "uploaded_at",
    "indexed_at",
    "last_crawled",
    "completed_at",
    "started_at",
    "first_response_at",
    "last_crawl_at",
    "next_crawl_at",
    "assigned_at",
    "deleted_at",
})

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:?\d{2}|Z)?$")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, _uuid_module.UUID):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default)


def _revive(value: Any, key: Optional[str] = None) -> Any:
    """Recursively turn ISO strings under known timestamp keys back into datetimes."""
    if isinstance(value, dict):
        return {k: _revive(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_revive(v, key) for v in value]
    if isinstance(value, str) and key in _DATETIME_KEYS and _ISO_RE.match(value):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            # Keep everything naive-UTC, matching the app's datetime.utcnow() convention.
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(tz=None).replace(tzinfo=None)
            return parsed
        except ValueError:
            return value
    return value


def _loads(raw: str) -> Any:
    return _revive(json.loads(raw))


# ---------------------------------------------------------------------------
# Table metadata
#
# Declares which keys map to real columns. Anything else goes to `data`.
# ---------------------------------------------------------------------------

class _Table:
    def __init__(self, name: str, columns: set, json_columns: set = frozenset(),
                 has_data: bool = True):
        self.name = name
        self.columns = columns
        self.json_columns = json_columns
        self.has_data = has_data


_TABLES = {
    "sites": _Table(
        "sites",
        {"site_id", "user_id", "name", "url", "status", "has_documents", "config",
         "triggers", "global_cooldown_ms", "handoff_config", "crawl_schedule",
         "created_at", "updated_at"},
        {"config", "triggers", "handoff_config", "crawl_schedule"},
    ),
    "users": _Table(
        "users",
        {"user_id", "email", "name", "password_hash", "role", "owner_id",
         "assigned_site_ids", "must_change_password", "is_active",
         "created_at", "updated_at"},
        {"assigned_site_ids"},
    ),
    "documents": _Table(
        "documents",
        {"doc_id", "site_id", "filename", "file_type", "word_count", "char_count",
         "chunks_created", "metadata", "status", "error", "uploaded_by",
         "uploaded_at", "indexed_at"},
        {"metadata"},
    ),
    "leads": _Table(
        "leads",
        {"lead_id", "site_id", "session_id", "email", "name", "source",
         "captured_at", "metadata"},
        {"metadata"},
    ),
    "qa_pairs": _Table(
        "qa_pairs",
        {"id", "site_id", "question", "answer", "enabled", "use_count",
         "created_at", "updated_at"},
    ),
    "pages": _Table(
        "pages",
        {"url", "title", "content", "chunk_count", "metadata", "status",
         "last_crawled", "created_at"},
        {"metadata"},
    ),
    "crawl_jobs": _Table(
        "crawl_jobs",
        {"site_id", "target_url", "status", "pages_crawled", "pages_indexed",
         "errors", "trigger", "created_at", "updated_at", "completed_at"},
        {"errors"},
    ),
    "handoff_sessions": _Table(
        "handoff_sessions",
        {"handoff_id", "session_id", "site_id", "status", "visitor_email",
         "visitor_name", "reason", "ai_summary", "ai_conversation", "messages",
         "assigned_agent_id", "assigned_agent_name", "visitor_queue_signals",
         "created_at", "updated_at", "resolved_at"},
        {"ai_conversation", "messages"},
    ),
}

# Columns that exist for internal bookkeeping and must never surface to callers.
_INTERNAL_COLUMNS = frozenset({"search_text", "search_vector"})

# Sort fields the conversation list endpoint is allowed to order by. Whitelisted
# because the value reaches SQL as an identifier and cannot be parameterised.
_CONVERSATION_SORT_FIELDS = frozenset({
    "updated_at", "created_at", "session_id", "status", "priority",
    "satisfaction_rating", "site_id",
})


def _split_doc(table: _Table, doc: Dict) -> Tuple[Dict, Dict]:
    """Split a document into (column values, leftover keys destined for `data`)."""
    cols, extra = {}, {}
    for key, value in doc.items():
        if key in ("_id", "id") and key not in table.columns:
            continue
        if key in table.columns:
            cols[key] = value
        else:
            extra[key] = value
    return cols, extra


def _row_to_dict(row, *, id_key: Optional[str] = "_id") -> Optional[Dict]:
    """Convert a row to the dict shape the application expects."""
    if row is None:
        return None
    out = dict(row)
    for internal in _INTERNAL_COLUMNS:
        out.pop(internal, None)
    data = out.pop("data", None)
    if isinstance(data, dict):
        out.update(data)
    if id_key and "id" in out:
        out[id_key] = str(out.pop("id"))
    return out


def _rows_to_dicts(rows, *, id_key: Optional[str] = "_id") -> List[Dict]:
    return [_row_to_dict(r, id_key=id_key) for r in rows]


def _as_uuid(value: Any) -> Optional[_uuid_module.UUID]:
    """Best-effort UUID parse; returns None for non-UUID input rather than raising."""
    if isinstance(value, _uuid_module.UUID):
        return value
    try:
        return _uuid_module.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def _messages_text(messages: List[Dict]) -> str:
    """Concatenate message contents for the full-text search column."""
    return " ".join(str(m.get("content", "")) for m in messages if m.get("content"))


class PostgresDB:
    """PostgreSQL client for managing conversations, pages, and long-term memory."""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    # ==================== Connection ====================

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:
        """Register JSON codecs so JSONB columns behave like Mongo sub-documents."""
        for type_name in ("json", "jsonb"):
            await conn.set_type_codec(
                type_name,
                encoder=_dumps,
                decoder=_loads,
                schema="pg_catalog",
            )

    async def connect(self):
        """Connect to PostgreSQL and ensure the schema is present."""
        try:
            self.pool = await asyncpg.create_pool(
                dsn=settings.postgres_dsn,
                min_size=settings.POSTGRES_POOL_MIN_SIZE,
                max_size=settings.POSTGRES_POOL_MAX_SIZE,
                command_timeout=settings.POSTGRES_COMMAND_TIMEOUT,
                init=self._init_connection,
            )
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
            logger.info(f"Connected to PostgreSQL: {settings.postgres_database_name}")

            if settings.POSTGRES_AUTO_MIGRATE:
                from app.database.migrate import apply_migrations
                await apply_migrations(settings.postgres_dsn)
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def disconnect(self):
        """Close the PostgreSQL connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Disconnected from PostgreSQL")

    async def ping(self) -> bool:
        """Liveness check (replaces the Mongo `ping` admin command)."""
        async with self.pool.acquire() as conn:
            await conn.execute("SELECT 1")
        return True

    async def health_check(self) -> bool:
        try:
            return await self.ping()
        except Exception:
            return False

    # ==================== Conversations ====================

    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: List[Dict] = None,
        metadata: Dict = None,
        site_id: str = None,
        response_time_ms: int = None
    ) -> str:
        """Save a message to a conversation."""
        now = datetime.utcnow()
        message = {
            "role": role,
            "content": content,
            "sources": sources or [],
            "metadata": metadata or {},
            "timestamp": now,
            "message_id": f"{session_id}_{now.timestamp()}"
        }

        if response_time_ms is not None:
            message["response_time_ms"] = response_time_ms

        appended = [message]
        search_fragment = f" {content}" if content else ""

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO conversations (
                    session_id, site_id, messages, search_text,
                    created_at, updated_at, status, priority, tags, unread, notes
                )
                VALUES ($1, $2, $3::jsonb, $4, $5, $5, 'open', 'medium',
                        '[]'::jsonb, TRUE, '[]'::jsonb)
                ON CONFLICT (session_id) DO UPDATE SET
                    messages    = conversations.messages || $3::jsonb,
                    search_text = conversations.search_text || $4,
                    updated_at  = $5,
                    site_id     = COALESCE($2, conversations.site_id)
                """,
                session_id, site_id, appended, search_fragment, now,
            )

            # Track first assistant response time.
            if role == "assistant":
                await conn.execute(
                    """
                    UPDATE conversations SET first_response_at = $2
                    WHERE session_id = $1 AND first_response_at IS NULL
                    """,
                    session_id, now,
                )

        return message["message_id"]

    async def add_message_feedback(
        self,
        session_id: str,
        message_id: str,
        feedback: str
    ) -> bool:
        """Add feedback to a specific message (positive/negative)."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE conversations c
                SET messages = (
                    SELECT jsonb_agg(
                        CASE WHEN elem->>'message_id' = $2
                             THEN elem || jsonb_build_object(
                                 'feedback', $3::text,
                                 'feedback_at', $4::text
                             )
                             ELSE elem END
                        ORDER BY ord
                    )
                    FROM jsonb_array_elements(c.messages) WITH ORDINALITY AS t(elem, ord)
                )
                WHERE c.session_id = $1
                  AND EXISTS (
                      SELECT 1 FROM jsonb_array_elements(c.messages) e
                      WHERE e->>'message_id' = $2
                  )
                """,
                session_id, message_id, feedback, datetime.utcnow().isoformat(),
            )
        return _affected(result) > 0

    async def add_feedback_by_index(
        self,
        session_id: str,
        message_index: int,
        feedback: str
    ) -> bool:
        """Add feedback to a message by its index in the conversation."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE conversations
                SET messages = jsonb_set(
                        jsonb_set(messages, ARRAY[$2::text, 'feedback'],
                                  to_jsonb($3::text), true),
                        ARRAY[$2::text, 'feedback_at'], to_jsonb($4::text), true
                    )
                WHERE session_id = $1
                  AND $2::int >= 0
                  AND jsonb_array_length(messages) > $2::int
                """,
                session_id, message_index, feedback, datetime.utcnow().isoformat(),
            )
        return _affected(result) > 0

    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = None
    ) -> List[Dict]:
        """Get conversation history for a session."""
        limit = limit or settings.CONVERSATION_WINDOW_SIZE

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT messages FROM conversations WHERE session_id = $1",
                session_id,
            )
        if not row:
            return []

        messages = row["messages"] or []
        return messages[-limit:] if limit else messages

    async def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM conversations WHERE session_id = $1", session_id
            )
        return _affected(result) > 0

    async def get_all_sessions(self, limit: int = 50) -> List[Dict]:
        """Get all conversation sessions."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, session_id, created_at, updated_at
                FROM conversations ORDER BY updated_at DESC LIMIT $1
                """,
                limit,
            )
        return _rows_to_dicts(rows)

    async def get_conversations_paginated(
        self,
        site_id: str = None,
        site_ids: Optional[List[str]] = None,
        page: int = 1,
        limit: int = 20,
        sort_by: str = "updated_at",
        order: int = -1,
        date_from: datetime = None,
        date_to: datetime = None,
        status: str = None,
        priority: str = None,
        tag: str = None
    ) -> Tuple[List[Dict], int]:
        """Get paginated conversations with filters."""
        where, params = [], []

        def add(clause: str, value: Any):
            params.append(value)
            where.append(clause.format(n=len(params)))

        if site_id:
            add("site_id = ${n}", site_id)
        elif site_ids is not None:
            add("site_id = ANY(${n}::text[])", list(site_ids))

        if date_from:
            add("created_at >= ${n}", date_from)
        if date_to:
            add("created_at <= ${n}", date_to)
        if status:
            add("status = ${n}", status)
        if priority:
            add("priority = ${n}", priority)
        if tag:
            add("tags @> ${n}::jsonb", [tag])

        clause = f"WHERE {' AND '.join(where)}" if where else ""

        sort_field = sort_by if sort_by in _CONVERSATION_SORT_FIELDS else "updated_at"
        direction = "ASC" if order and order > 0 else "DESC"
        offset = max(0, (page - 1) * limit)

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(
                f"SELECT COUNT(*) FROM conversations {clause}", *params
            )
            rows = await conn.fetch(
                f"""
                SELECT id, session_id, site_id, created_at, updated_at, status,
                       priority, tags, unread, visitor_name, visitor_email,
                       satisfaction_rating,
                       jsonb_array_length(messages) AS message_count,
                       messages->0->>'content'      AS _first_message
                FROM conversations {clause}
                ORDER BY {sort_field} {direction}
                LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
                """,
                *params, limit, offset,
            )

        conversations = []
        for row in rows:
            conv = _row_to_dict(row)
            first = conv.pop("_first_message", None)
            conv["first_message"] = (first or "")[:100]
            conversations.append(conv)

        return conversations, total or 0

    async def get_conversation_full(self, session_id: str) -> Optional[Dict]:
        """Get full conversation with all messages and stats."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM conversations WHERE session_id = $1", session_id
            )
        conv = _row_to_dict(row)
        if not conv:
            return None

        messages = conv.get("messages") or []

        positive_feedback = sum(1 for m in messages if m.get("feedback") == "positive")
        negative_feedback = sum(1 for m in messages if m.get("feedback") == "negative")
        response_times = [m.get("response_time_ms") for m in messages if m.get("response_time_ms")]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        stats = {
            "message_count": len(messages),
            "user_messages": sum(1 for m in messages if m.get("role") == "user"),
            "assistant_messages": sum(1 for m in messages if m.get("role") == "assistant"),
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "avg_response_time_ms": round(avg_response_time, 2),
            "first_response_time_ms": None,
            "resolution_time_ms": None
        }

        # Calculate sentiment from feedback ratio
        if positive_feedback + negative_feedback > 0:
            conv["sentiment"] = round(
                (positive_feedback - negative_feedback) / (positive_feedback + negative_feedback), 2
            )
        else:
            conv["sentiment"] = None

        # First response time (time from first user message to first assistant message)
        messages_sorted = sorted(messages, key=lambda m: m.get("timestamp", datetime.utcnow()))
        first_user = next((m for m in messages_sorted if m.get("role") == "user"), None)
        first_assistant = next((m for m in messages_sorted if m.get("role") == "assistant"), None)
        if first_user and first_assistant:
            user_ts = first_user.get("timestamp")
            asst_ts = first_assistant.get("timestamp")
            if user_ts and asst_ts:
                diff = (asst_ts - user_ts).total_seconds() * 1000
                stats["first_response_time_ms"] = max(0, int(diff))

        # Resolution time
        if conv.get("resolved_at") and conv.get("created_at"):
            diff = (conv["resolved_at"] - conv["created_at"]).total_seconds() * 1000
            stats["resolution_time_ms"] = max(0, int(diff))

        conv["stats"] = stats

        return conv

    async def search_conversations(
        self,
        query: str,
        site_id: str = None,
        site_ids: Optional[List[str]] = None,
        page: int = 1,
        limit: int = 20
    ) -> Tuple[List[Dict], int]:
        """Search conversations by message content."""
        where, params = [], []

        def add(clause: str, value: Any):
            params.append(value)
            where.append(clause.format(n=len(params)))

        if query:
            # MongoDB's $text matched ANY of the search terms; mirror that by
            # OR-ing a tsquery per term rather than requiring all of them.
            terms = [t for t in re.split(r"\s+", query.strip()) if t]
            if terms:
                placeholders = []
                for term in terms:
                    params.append(term)
                    placeholders.append(f"plainto_tsquery('english', ${len(params)})")
                where.append(f"search_vector @@ ({' || '.join(placeholders)})")

        if site_id:
            add("site_id = ${n}", site_id)
        elif site_ids is not None:
            add("site_id = ANY(${n}::text[])", list(site_ids))

        clause = f"WHERE {' AND '.join(where)}" if where else ""
        offset = max(0, (page - 1) * limit)

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(
                f"SELECT COUNT(*) FROM conversations {clause}", *params
            )
            rows = await conn.fetch(
                f"""
                SELECT id, session_id, site_id, created_at, updated_at, messages,
                       status, priority, tags, unread, visitor_name, visitor_email,
                       satisfaction_rating
                FROM conversations {clause}
                ORDER BY updated_at DESC
                LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
                """,
                *params, limit, offset,
            )

        conversations = []
        for row in rows:
            conv = _row_to_dict(row)
            messages = conv.get("messages") or []
            conv["message_count"] = len(messages)

            matching_snippet = ""
            if query:
                for msg in messages:
                    content = msg.get("content", "")
                    if query.lower() in content.lower():
                        idx = content.lower().find(query.lower())
                        start = max(0, idx - 30)
                        end = min(len(content), idx + len(query) + 30)
                        matching_snippet = "..." + content[start:end] + "..."
                        break

            conv["matching_snippet"] = matching_snippet
            conv["first_message"] = messages[0].get("content", "")[:100] if messages else ""
            del conv["messages"]
            conversations.append(conv)

        return conversations, total or 0

    async def delete_conversations_bulk(self, session_ids: List[str]) -> int:
        """Delete multiple conversations at once."""
        if not session_ids:
            return 0
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM conversations WHERE session_id = ANY($1::text[])",
                list(session_ids),
            )
        return _affected(result)

    async def get_conversations_for_export(
        self,
        session_ids: List[str] = None,
        site_id: str = None
    ) -> List[Dict]:
        """Get conversations for export."""
        async with self.pool.acquire() as conn:
            if session_ids:
                rows = await conn.fetch(
                    """
                    SELECT * FROM conversations WHERE session_id = ANY($1::text[])
                    ORDER BY created_at DESC LIMIT 1000
                    """,
                    list(session_ids),
                )
            elif site_id:
                rows = await conn.fetch(
                    """
                    SELECT * FROM conversations WHERE site_id = $1
                    ORDER BY created_at DESC LIMIT 1000
                    """,
                    site_id,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM conversations ORDER BY created_at DESC LIMIT 1000"
                )
        return _rows_to_dicts(rows)

    async def get_all_conversations(
        self,
        site_id: Optional[str] = None,
        limit: int = 10000
    ) -> List[Dict]:
        """
        Fetch full conversation documents, optionally filtered by site.

        Backs the analytics endpoints, which previously reached into the Mongo
        collection directly and iterated the raw documents.
        """
        async with self.pool.acquire() as conn:
            if site_id:
                rows = await conn.fetch(
                    "SELECT * FROM conversations WHERE site_id = $1 LIMIT $2",
                    site_id, limit,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM conversations LIMIT $1", limit
                )
        return _rows_to_dicts(rows)

    async def get_recent_conversations(
        self,
        site_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """Most recently updated conversations, newest first."""
        async with self.pool.acquire() as conn:
            if site_id:
                rows = await conn.fetch(
                    """
                    SELECT * FROM conversations WHERE site_id = $1
                    ORDER BY updated_at DESC LIMIT $2
                    """,
                    site_id, limit,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT $1",
                    limit,
                )
        return _rows_to_dicts(rows)

    async def get_conversation_counts_by_site(self, limit: int = 20) -> List[Dict]:
        """Conversation and message totals grouped by site, busiest first."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT site_id,
                       COUNT(*)                              AS conversation_count,
                       COALESCE(SUM(jsonb_array_length(messages)), 0) AS message_count
                FROM conversations
                GROUP BY site_id
                ORDER BY conversation_count DESC
                LIMIT $1
                """,
                limit,
            )
        return [
            {
                "site_id": r["site_id"],
                "conversation_count": r["conversation_count"],
                "message_count": int(r["message_count"]),
            }
            for r in rows
        ]

    async def count_conversations(self, site_id: Optional[str] = None) -> int:
        """Total conversations, optionally scoped to a site."""
        async with self.pool.acquire() as conn:
            if site_id:
                return await conn.fetchval(
                    "SELECT COUNT(*) FROM conversations WHERE site_id = $1", site_id
                ) or 0
            return await conn.fetchval("SELECT COUNT(*) FROM conversations") or 0

    # ==================== Conversation Feature Methods ====================

    async def _update_conversation(self, session_id: str, assignments: str, *params) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE conversations SET {assignments} WHERE session_id = $1",
                session_id, *params,
            )
        return _affected(result) > 0

    async def update_conversation_status(self, session_id: str, status: str) -> bool:
        now = datetime.utcnow()
        if status in ("resolved", "closed"):
            return await self._update_conversation(
                session_id, "status = $2, updated_at = $3, resolved_at = $3", status, now
            )
        return await self._update_conversation(
            session_id, "status = $2, updated_at = $3", status, now
        )

    async def update_conversation_priority(self, session_id: str, priority: str) -> bool:
        return await self._update_conversation(
            session_id, "priority = $2, updated_at = $3", priority, datetime.utcnow()
        )

    async def update_conversation_tags(self, session_id: str, tags: list) -> bool:
        return await self._update_conversation(
            session_id, "tags = $2::jsonb, updated_at = $3",
            list(tags), datetime.utcnow()
        )

    async def add_conversation_note(self, session_id: str, content: str) -> dict:
        now = datetime.utcnow()
        note = {
            "note_id": str(_uuid_module.uuid4()),
            "content": content,
            "created_at": now,
            "updated_at": now
        }
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE conversations
                SET notes = notes || $2::jsonb, updated_at = $3
                WHERE session_id = $1
                """,
                session_id, [note], now,
            )
        return note

    async def update_conversation_note(self, session_id: str, note_id: str, content: str) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE conversations c
                SET notes = (
                        SELECT jsonb_agg(
                            CASE WHEN elem->>'note_id' = $2
                                 THEN elem || jsonb_build_object(
                                     'content', $3::text, 'updated_at', $4::text)
                                 ELSE elem END
                            ORDER BY ord
                        )
                        FROM jsonb_array_elements(c.notes) WITH ORDINALITY AS t(elem, ord)
                    ),
                    updated_at = $4::timestamp
                WHERE c.session_id = $1
                  AND EXISTS (
                      SELECT 1 FROM jsonb_array_elements(c.notes) e
                      WHERE e->>'note_id' = $2
                  )
                """,
                session_id, note_id, content, datetime.utcnow().isoformat(),
            )
        return _affected(result) > 0

    async def delete_conversation_note(self, session_id: str, note_id: str) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE conversations c
                SET notes = COALESCE((
                        SELECT jsonb_agg(elem ORDER BY ord)
                        FROM jsonb_array_elements(c.notes) WITH ORDINALITY AS t(elem, ord)
                        WHERE elem->>'note_id' IS DISTINCT FROM $2
                    ), '[]'::jsonb)
                WHERE c.session_id = $1
                  AND EXISTS (
                      SELECT 1 FROM jsonb_array_elements(c.notes) e
                      WHERE e->>'note_id' = $2
                  )
                """,
                session_id, note_id,
            )
        return _affected(result) > 0

    async def update_conversation_visitor(
        self, session_id: str, visitor_name: str = None, visitor_email: str = None
    ) -> bool:
        assignments = ["updated_at = $2"]
        params: List[Any] = [datetime.utcnow()]
        if visitor_name is not None:
            params.append(visitor_name)
            assignments.append(f"visitor_name = ${len(params) + 1}")
        if visitor_email is not None:
            params.append(visitor_email)
            assignments.append(f"visitor_email = ${len(params) + 1}")
        return await self._update_conversation(session_id, ", ".join(assignments), *params)

    async def mark_conversation_read(self, session_id: str) -> bool:
        return await self._update_conversation(session_id, "unread = FALSE")

    async def set_conversation_rating(self, session_id: str, rating: int) -> bool:
        return await self._update_conversation(
            session_id, "satisfaction_rating = $2, updated_at = $3",
            rating, datetime.utcnow()
        )

    async def auto_close_inactive_conversations(self, days_inactive: int = 7) -> int:
        cutoff = datetime.utcnow() - timedelta(days=days_inactive)
        now = datetime.utcnow()
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE conversations
                SET status = 'closed', resolved_at = $2
                WHERE updated_at < $1 AND (status = 'open' OR status IS NULL)
                """,
                cutoff, now,
            )
        return _affected(result)

    # ==================== Pages ====================

    async def save_page(
        self,
        url: str,
        title: str,
        content: str,
        chunk_count: int = 0,
        metadata: Dict = None
    ) -> str:
        """Save a crawled page."""
        now = datetime.utcnow()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO pages (url, title, content, chunk_count, metadata,
                                   last_crawled, status, created_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, 'indexed', $6)
                ON CONFLICT (url) DO UPDATE SET
                    title        = EXCLUDED.title,
                    content      = EXCLUDED.content,
                    chunk_count  = EXCLUDED.chunk_count,
                    metadata     = EXCLUDED.metadata,
                    last_crawled = EXCLUDED.last_crawled,
                    status       = 'indexed'
                RETURNING id, (xmax = 0) AS inserted
                """,
                url, title, content, chunk_count, metadata or {}, now,
            )
        return str(row["id"]) if row["inserted"] else url

    async def get_page(self, url: str) -> Optional[Dict]:
        """Get a page by URL."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM pages WHERE url = $1", url)
        return _row_to_dict(row)

    async def get_all_pages(self, status: str = None) -> List[Dict]:
        """Get all pages."""
        async with self.pool.acquire() as conn:
            if status:
                rows = await conn.fetch(
                    """
                    SELECT * FROM pages WHERE status = $1
                    ORDER BY last_crawled DESC NULLS LAST LIMIT 1000
                    """,
                    status,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM pages ORDER BY last_crawled DESC NULLS LAST LIMIT 1000"
                )
        return _rows_to_dicts(rows)

    async def delete_page(self, url: str) -> bool:
        """Delete a page."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("DELETE FROM pages WHERE url = $1", url)
        return _affected(result) > 0

    async def get_page_count(self) -> int:
        """Get total number of indexed pages."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM pages WHERE status = 'indexed'"
            ) or 0

    # ==================== Crawl Jobs ====================

    async def create_crawl_job(self, target_url: str) -> str:
        """Create a new crawl job."""
        now = datetime.utcnow()
        async with self.pool.acquire() as conn:
            job_id = await conn.fetchval(
                """
                INSERT INTO crawl_jobs (target_url, status, pages_crawled,
                                        pages_indexed, errors, created_at, updated_at)
                VALUES ($1, 'running', 0, 0, '[]'::jsonb, $2, $2)
                RETURNING id
                """,
                target_url, now,
            )
        return str(job_id)

    async def update_crawl_job(
        self,
        job_id: str,
        status: str = None,
        pages_crawled: int = None,
        pages_indexed: int = None,
        error: str = None
    ):
        """Update a crawl job."""
        job_uuid = _as_uuid(job_id)
        if job_uuid is None:
            return

        assignments = ["updated_at = $2"]
        params: List[Any] = [job_uuid, datetime.utcnow()]

        def add(column: str, value: Any, cast: str = ""):
            params.append(value)
            assignments.append(f"{column} = ${len(params)}{cast}")

        if status:
            add("status", status)
        if pages_crawled is not None:
            add("pages_crawled", pages_crawled)
        if pages_indexed is not None:
            add("pages_indexed", pages_indexed)
        if error:
            params.append([error])
            assignments.append(f"errors = errors || ${len(params)}::jsonb")

        async with self.pool.acquire() as conn:
            await conn.execute(
                f"UPDATE crawl_jobs SET {', '.join(assignments)} WHERE id = $1",
                *params,
            )

    async def mark_crawl_job_completed(self, job_id: str) -> bool:
        """Stamp completed_at on a finished crawl job."""
        job_uuid = _as_uuid(job_id)
        if job_uuid is None:
            return False
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE crawl_jobs SET completed_at = $2, updated_at = $2 WHERE id = $1",
                job_uuid, datetime.utcnow(),
            )
        return _affected(result) > 0

    async def get_crawl_job(self, job_id: str) -> Optional[Dict]:
        """Get a crawl job by ID."""
        job_uuid = _as_uuid(job_id)
        if job_uuid is None:
            return None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM crawl_jobs WHERE id = $1", job_uuid)
        return _row_to_dict(row)

    async def get_latest_crawl_job(self) -> Optional[Dict]:
        """Get the latest crawl job."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM crawl_jobs ORDER BY created_at DESC LIMIT 1"
            )
        return _row_to_dict(row)

    async def get_crawl_job_by_url(self, url: str) -> Optional[Dict]:
        """Get crawl job by target URL."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM crawl_jobs WHERE target_url = $1
                ORDER BY created_at DESC LIMIT 1
                """,
                url,
            )
        return _row_to_dict(row)

    # ==================== Long-term Memory ====================

    async def save_user_memory(
        self,
        user_id: str,
        key: str,
        value: Any
    ):
        """Save long-term memory for a user."""
        now = datetime.utcnow()
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO long_term_memory (user_id, memory, created_at, updated_at)
                VALUES ($1, jsonb_build_object($2::text, $3::jsonb), $4, $4)
                ON CONFLICT (user_id) DO UPDATE SET
                    memory     = jsonb_set(long_term_memory.memory,
                                           ARRAY[$2::text], $3::jsonb, true),
                    updated_at = $4
                """,
                user_id, key, value, now,
            )

    async def get_user_memory(self, user_id: str) -> Dict:
        """Get all long-term memory for a user."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT memory FROM long_term_memory WHERE user_id = $1", user_id
            )
        return (row["memory"] or {}) if row else {}

    async def clear_user_memory(self, user_id: str) -> bool:
        """Clear long-term memory for a user."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM long_term_memory WHERE user_id = $1", user_id
            )
        return _affected(result) > 0

    # ==================== User Management ====================

    async def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM users WHERE email = $1", email)
        return _row_to_dict(row)

    async def create_user(self, user_data: Dict) -> str:
        """Create a new user."""
        now = datetime.utcnow()
        user_id = str(_uuid_module.uuid4())

        doc = dict(user_data)
        doc["user_id"] = user_id
        doc.setdefault("created_at", now)
        doc["updated_at"] = now

        cols, extra = _split_doc(_TABLES["users"], doc)
        # `id` mirrors `user_id` so str(user["_id"]) and user["user_id"] always agree.
        cols["id"] = _uuid_module.UUID(user_id)

        names = list(cols.keys())
        placeholders = [f"${i + 1}" for i in range(len(names))]
        values = [cols[n] for n in names]

        if extra:
            values.append(extra)
            names.append("data")
            placeholders.append(f"${len(values)}::jsonb")

        async with self.pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO users ({', '.join(names)}) VALUES ({', '.join(placeholders)})",
                *values,
            )

        # Mirror the previous behaviour of mutating the caller's dict in place.
        user_data["user_id"] = user_id
        user_data["created_at"] = doc["created_at"]
        user_data["updated_at"] = now
        return user_id

    async def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Get user by ID (searches both the user_id field and the row primary key)."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM users WHERE user_id = $1", user_id)
            if not row:
                as_uuid = _as_uuid(user_id)
                if as_uuid is not None:
                    row = await conn.fetchrow("SELECT * FROM users WHERE id = $1", as_uuid)
        return _row_to_dict(row)

    async def update_user(self, user_id: str, updates: Dict) -> bool:
        """Update user fields by user_id or row primary key."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False

        row_id = _as_uuid(user["_id"])
        if row_id is None:
            return False

        patch = dict(updates)
        patch["updated_at"] = datetime.utcnow()
        return await self._update_row("users", "id", row_id, patch)

    async def get_all_users(self) -> List[Dict]:
        """List users (admin tooling)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM users ORDER BY created_at DESC LIMIT 500"
            )
        return _rows_to_dicts(rows)

    async def delete_user(self, user_id: str) -> bool:
        """Delete user by logical id."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        row_id = _as_uuid(user["_id"])
        if row_id is None:
            return False
        async with self.pool.acquire() as conn:
            result = await conn.execute("DELETE FROM users WHERE id = $1", row_id)
        return _affected(result) > 0

    async def list_users_agents_for_owner(self, owner_id: str) -> List[Dict]:
        """Support agents created by this owner (owner_id)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM users WHERE role = 'agent' AND owner_id = $1
                ORDER BY created_at DESC LIMIT 200
                """,
                owner_id,
            )
        return _rows_to_dicts(rows)

    async def list_all_agents(self) -> List[Dict]:
        """List all support agents across all owners (admin use)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM users WHERE role = 'agent'
                ORDER BY created_at DESC LIMIT 500
                """
            )
        return _rows_to_dicts(rows)

    async def transfer_sites_to_user(self, from_user_id: str, to_user_id: str) -> int:
        """Reassign all sites owned by from_user_id to to_user_id. Returns count."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE sites SET user_id = $2, updated_at = $3 WHERE user_id = $1",
                from_user_id, to_user_id, datetime.utcnow(),
            )
        return _affected(result)

    async def transfer_agents_to_user(self, from_owner_id: str, to_owner_id: str) -> int:
        """Reassign all agents owned by from_owner_id to to_owner_id. Returns count."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE users SET owner_id = $2, updated_at = $3
                WHERE role = 'agent' AND owner_id = $1
                """,
                from_owner_id, to_owner_id, datetime.utcnow(),
            )
        return _affected(result)

    # ==================== Generic row update helper ====================

    async def _update_row(self, table_name: str, key_column: str, key_value: Any,
                          updates: Dict) -> bool:
        """
        Apply a partial update, routing known keys to columns and everything else
        into the JSONB `data` catch-all (mirroring MongoDB's $set on a schemaless
        document).
        """
        table = _TABLES[table_name]
        cols, extra = _split_doc(table, updates)

        assignments, params = [], [key_value]

        for name, value in cols.items():
            params.append(value)
            cast = "::jsonb" if name in table.json_columns else ""
            assignments.append(f"{name} = ${len(params)}{cast}")

        if extra and table.has_data:
            params.append(extra)
            assignments.append(f"data = data || ${len(params)}::jsonb")

        if not assignments:
            return False

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE {table.name} SET {', '.join(assignments)} "
                f"WHERE {key_column} = $1",
                *params,
            )
        return _affected(result) > 0

    # ==================== Site Management ====================

    async def list_sites(self, user_id: Optional[str] = None) -> List[Dict]:
        """List all sites."""
        async with self.pool.acquire() as conn:
            if user_id:
                rows = await conn.fetch(
                    """
                    SELECT * FROM sites WHERE user_id = $1
                    ORDER BY created_at DESC LIMIT 100
                    """,
                    user_id,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM sites ORDER BY created_at DESC LIMIT 100"
                )
        return _rows_to_dicts(rows)

    async def list_sites_by_site_ids(self, site_ids: List[str]) -> List[Dict]:
        """List sites whose site_id is in the given list."""
        if not site_ids:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM sites WHERE site_id = ANY($1::text[])
                ORDER BY created_at DESC LIMIT 100
                """,
                list(site_ids),
            )
        return _rows_to_dicts(rows)

    async def get_site(self, site_id: str) -> Optional[Dict]:
        """Get site by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM sites WHERE site_id = $1", site_id)
        return _row_to_dict(row)

    async def create_site(self, site_data: Dict) -> str:
        """Create a new site."""
        now = datetime.utcnow()
        site_data["created_at"] = now
        site_data["updated_at"] = now

        cols, extra = _split_doc(_TABLES["sites"], site_data)
        names = list(cols.keys())
        values = [cols[n] for n in names]
        placeholders = [
            f"${i + 1}::jsonb" if names[i] in _TABLES["sites"].json_columns else f"${i + 1}"
            for i in range(len(names))
        ]

        if extra:
            values.append(extra)
            names.append("data")
            placeholders.append(f"${len(values)}::jsonb")

        async with self.pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO sites ({', '.join(names)}) VALUES ({', '.join(placeholders)})",
                *values,
            )
        return site_data.get("site_id", "")

    async def update_site(self, site_id: str, updates: Dict) -> bool:
        """Update site data."""
        updates["updated_at"] = datetime.utcnow()
        return await self._update_row("sites", "site_id", site_id, updates)

    async def delete_site(self, site_id: str) -> bool:
        """Delete a site."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("DELETE FROM sites WHERE site_id = $1", site_id)
        return _affected(result) > 0

    async def count_sites(self, status: Optional[str] = None) -> int:
        """Count sites, optionally filtered by status."""
        async with self.pool.acquire() as conn:
            if status:
                return await conn.fetchval(
                    "SELECT COUNT(*) FROM sites WHERE status = $1", status
                ) or 0
            return await conn.fetchval("SELECT COUNT(*) FROM sites") or 0

    async def get_site_names(self, site_ids: List[str]) -> Dict[str, str]:
        """Map site_id -> display name for the given ids."""
        if not site_ids:
            return {}
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT site_id, name FROM sites WHERE site_id = ANY($1::text[])",
                list(site_ids),
            )
        return {r["site_id"]: (r["name"] or r["site_id"]) for r in rows}

    # ==================== Proactive Chat Triggers ====================

    async def get_site_triggers(self, site_id: str) -> Dict:
        """Get all triggers for a site."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT triggers, global_cooldown_ms FROM sites WHERE site_id = $1",
                site_id,
            )
        if not row:
            return {"triggers": [], "global_cooldown_ms": 30000}

        return {
            "triggers": row["triggers"] or [],
            "global_cooldown_ms": row["global_cooldown_ms"]
            if row["global_cooldown_ms"] is not None else 30000
        }

    async def save_trigger(self, site_id: str, trigger: Dict) -> Dict:
        """Save a new trigger or update existing one."""
        now = datetime.utcnow()

        if not trigger.get("id"):
            trigger["id"] = str(_uuid_module.uuid4())[:8]
            trigger["created_at"] = now

        trigger["updated_at"] = now
        payload = trigger

        async with self.pool.acquire() as conn:
            replaced = await conn.execute(
                """
                UPDATE sites s
                SET triggers = (
                        SELECT jsonb_agg(
                            CASE WHEN elem->>'id' = $2 THEN $3::jsonb ELSE elem END
                            ORDER BY ord
                        )
                        FROM jsonb_array_elements(s.triggers) WITH ORDINALITY AS t(elem, ord)
                    )
                WHERE s.site_id = $1
                  AND EXISTS (
                      SELECT 1 FROM jsonb_array_elements(s.triggers) e
                      WHERE e->>'id' = $2
                  )
                """,
                site_id, trigger["id"], payload,
            )

            if _affected(replaced) == 0:
                appended = await conn.execute(
                    """
                    UPDATE sites
                    SET triggers = COALESCE(triggers, '[]'::jsonb) || jsonb_build_array($2::jsonb)
                    WHERE site_id = $1
                    """,
                    site_id, payload,
                )
                if _affected(appended) == 0:
                    # Preserves the previous upsert-on-push behaviour.
                    await conn.execute(
                        """
                        INSERT INTO sites (site_id, triggers, created_at, updated_at)
                        VALUES ($1, jsonb_build_array($2::jsonb), $3, $3)
                        ON CONFLICT (site_id) DO UPDATE SET
                            triggers = COALESCE(sites.triggers, '[]'::jsonb)
                                       || jsonb_build_array($2::jsonb)
                        """,
                        site_id, payload, now,
                    )

        return trigger

    async def update_trigger(self, site_id: str, trigger_id: str, updates: Dict) -> Optional[Dict]:
        """Update specific fields of a trigger."""
        updates["updated_at"] = datetime.utcnow()

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE sites s
                SET triggers = (
                        SELECT jsonb_agg(
                            CASE WHEN elem->>'id' = $2 THEN elem || $3::jsonb ELSE elem END
                            ORDER BY ord
                        )
                        FROM jsonb_array_elements(s.triggers) WITH ORDINALITY AS t(elem, ord)
                    )
                WHERE s.site_id = $1
                  AND EXISTS (
                      SELECT 1 FROM jsonb_array_elements(s.triggers) e
                      WHERE e->>'id' = $2
                  )
                """,
                site_id, trigger_id, updates,
            )

            if _affected(result) == 0:
                return None

            row = await conn.fetchrow(
                """
                SELECT elem FROM sites s,
                     jsonb_array_elements(s.triggers) AS elem
                WHERE s.site_id = $1 AND elem->>'id' = $2
                LIMIT 1
                """,
                site_id, trigger_id,
            )

        return row["elem"] if row else None

    async def delete_trigger(self, site_id: str, trigger_id: str) -> bool:
        """Delete a trigger from a site."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE sites s
                SET triggers = COALESCE((
                        SELECT jsonb_agg(elem ORDER BY ord)
                        FROM jsonb_array_elements(s.triggers) WITH ORDINALITY AS t(elem, ord)
                        WHERE elem->>'id' IS DISTINCT FROM $2
                    ), '[]'::jsonb)
                WHERE s.site_id = $1
                  AND EXISTS (
                      SELECT 1 FROM jsonb_array_elements(s.triggers) e
                      WHERE e->>'id' = $2
                  )
                """,
                site_id, trigger_id,
            )
        return _affected(result) > 0

    async def reorder_triggers(self, site_id: str, trigger_ids: List[str]) -> bool:
        """Reorder triggers by setting priority based on list order."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT triggers FROM sites WHERE site_id = $1", site_id
            )
            if not row:
                return False

            triggers = row["triggers"] or []
            trigger_map = {t["id"]: t for t in triggers}

            for idx, trigger_id in enumerate(trigger_ids):
                if trigger_id in trigger_map:
                    trigger_map[trigger_id]["priority"] = len(trigger_ids) - idx
                    trigger_map[trigger_id]["updated_at"] = datetime.utcnow()

            reordered = [trigger_map[tid] for tid in trigger_ids if tid in trigger_map]
            remaining = [t for t in triggers if t["id"] not in trigger_ids]
            all_triggers = reordered + remaining

            await conn.execute(
                "UPDATE sites SET triggers = $2::jsonb WHERE site_id = $1",
                site_id, all_triggers,
            )
        return True

    async def set_global_cooldown(self, site_id: str, cooldown_ms: int) -> bool:
        """Set global cooldown between triggers for a site."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE sites SET global_cooldown_ms = $2 WHERE site_id = $1",
                site_id, cooldown_ms,
            )
        return _affected(result) > 0

    async def log_trigger_event(
        self,
        site_id: str,
        trigger_id: str,
        session_id: str,
        event_type: str,
        metadata: Dict = None
    ) -> str:
        """Log a trigger event for analytics."""
        async with self.pool.acquire() as conn:
            event_id = await conn.fetchval(
                """
                INSERT INTO trigger_events (site_id, trigger_id, session_id,
                                            event_type, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                RETURNING id
                """,
                site_id, trigger_id, session_id, event_type,
                datetime.utcnow(), metadata or {},
            )
        return str(event_id)

    async def get_trigger_analytics(
        self,
        site_id: str,
        period_days: int = 7
    ) -> List[Dict]:
        """Get trigger analytics for a site."""
        start_date = datetime.utcnow() - timedelta(days=period_days)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT trigger_id, event_type, COUNT(*) AS count
                FROM trigger_events
                WHERE site_id = $1 AND timestamp >= $2
                GROUP BY trigger_id, event_type
                """,
                site_id, start_date,
            )
            site_row = await conn.fetchrow(
                "SELECT triggers FROM sites WHERE site_id = $1", site_id
            )

        grouped: Dict[str, Dict[str, int]] = {}
        for row in rows:
            grouped.setdefault(row["trigger_id"], {})[row["event_type"]] = row["count"]

        trigger_names = (
            {t["id"]: t.get("name") for t in (site_row["triggers"] or [])}
            if site_row else {}
        )

        analytics = []
        for trigger_id, events in grouped.items():
            shown = events.get("shown", 0)
            clicked = events.get("clicked", 0)
            dismissed = events.get("dismissed", 0)
            converted = events.get("converted", 0)

            analytics.append({
                "trigger_id": trigger_id,
                "trigger_name": trigger_names.get(trigger_id, "Unknown"),
                "shown_count": shown,
                "clicked_count": clicked,
                "dismissed_count": dismissed,
                "converted_count": converted,
                "click_rate": round((clicked / shown * 100) if shown > 0 else 0, 1),
                "conversion_rate": round((converted / shown * 100) if shown > 0 else 0, 1)
            })

        return analytics

    # ==================== Human Handoff ====================

    async def create_handoff_session(
        self,
        session_id: str,
        site_id: str,
        reason: str = "user_request",
        visitor_email: str = None,
        visitor_name: str = None,
        ai_conversation: List[Dict] = None,
        ai_summary: str = None
    ) -> Dict:
        """Create a new handoff session."""
        handoff_id = str(_uuid_module.uuid4())[:12]
        now = datetime.utcnow()

        handoff = {
            "handoff_id": handoff_id,
            "session_id": session_id,
            "site_id": site_id,
            "status": "pending",
            "visitor_email": visitor_email,
            "visitor_name": visitor_name,
            "reason": reason,
            "ai_summary": ai_summary,
            "ai_conversation": ai_conversation or [],
            "messages": [],
            "assigned_agent_id": None,
            "assigned_agent_name": None,
            "created_at": now,
            "updated_at": now,
            "resolved_at": None,
            "visitor_queue_signals": 0,
        }

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO handoff_sessions (
                    handoff_id, session_id, site_id, status, visitor_email,
                    visitor_name, reason, ai_summary, ai_conversation, messages,
                    assigned_agent_id, assigned_agent_name, created_at, updated_at,
                    resolved_at, visitor_queue_signals
                )
                VALUES ($1, $2, $3, 'pending', $4, $5, $6, $7, $8::jsonb,
                        '[]'::jsonb, NULL, NULL, $9, $9, NULL, 0)
                """,
                handoff_id, session_id, site_id, visitor_email, visitor_name,
                reason, ai_summary, ai_conversation or [], now,
            )
        return handoff

    async def get_handoff_session(self, handoff_id: str) -> Optional[Dict]:
        """Get a handoff session by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM handoff_sessions WHERE handoff_id = $1", handoff_id
            )
        return _row_to_dict(row)

    async def get_handoff_by_session(self, session_id: str, active_only: bool = True) -> Optional[Dict]:
        """Get active handoff for a chat session."""
        async with self.pool.acquire() as conn:
            if active_only:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM handoff_sessions
                    WHERE session_id = $1 AND status = ANY(ARRAY['pending', 'active'])
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    session_id,
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM handoff_sessions WHERE session_id = $1
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    session_id,
                )
        return _row_to_dict(row)

    async def bump_handoff_visitor_requeue_pending(self, handoff_id: str) -> None:
        """Visitor requested human again for an existing pending handoff (widget re-post)."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE handoff_sessions
                SET visitor_queue_signals = visitor_queue_signals + 1, updated_at = $2
                WHERE handoff_id = $1 AND status = 'pending'
                """,
                handoff_id, datetime.utcnow(),
            )

    async def get_handoff_queue(
        self,
        site_id: Optional[str] = None,
        site_ids: Optional[List[str]] = None,
        status: List[str] = None,
        page: int = 1,
        limit: int = 20,
        agent_queue_user_id: Optional[str] = None,
    ) -> Tuple[List[Dict], int, int, int]:
        """Get handoff queue with counts. Use site_ids for multiple sites, or site_id for one, or neither for all sites.

        When ``agent_queue_user_id`` is set (support agent dashboard), rows are limited to:
        unassigned handoffs (no ``assigned_agent_id``) or handoffs assigned to that agent.
        Admins and site owners omit ``agent_queue_user_id`` and see all rows for the site(s).
        """
        base_clauses: List[str] = []
        params: List[Any] = []

        if site_ids is not None:
            params.append(list(site_ids))
            base_clauses.append(f"site_id = ANY(${len(params)}::text[])")
        elif site_id:
            params.append(site_id)
            base_clauses.append(f"site_id = ${len(params)}")

        if agent_queue_user_id:
            params.append(str(agent_queue_user_id))
            base_clauses.append(
                f"(assigned_agent_id IS NULL OR assigned_agent_id = '' "
                f"OR assigned_agent_id = ${len(params)})"
            )

        base_sql = f"WHERE {' AND '.join(base_clauses)}" if base_clauses else ""

        statuses = list(status) if status else ["pending", "active"]
        params.append(statuses)
        status_param = len(params)
        joined = base_clauses + [f"status = ANY(${status_param}::text[])"]
        query_sql = f"WHERE {' AND '.join(joined)}"

        offset = max(0, (page - 1) * limit)

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(
                f"SELECT COUNT(*) FROM handoff_sessions {query_sql}", *params
            )

            count_params = params[:status_param - 1]
            counts = await conn.fetchrow(
                f"""
                SELECT
                    COUNT(*) FILTER (WHERE status = 'pending') AS pending,
                    COUNT(*) FILTER (WHERE status = 'active')  AS active
                FROM handoff_sessions {base_sql}
                """,
                *count_params,
            )

            rows = await conn.fetch(
                f"""
                SELECT id, handoff_id, session_id, site_id, status, visitor_email,
                       visitor_name, reason, ai_summary, assigned_agent_id,
                       assigned_agent_name, visitor_queue_signals, created_at,
                       updated_at, resolved_at, data,
                       jsonb_array_length(messages)     AS message_count,
                       messages -> -1 ->> 'content'     AS _last_message
                FROM handoff_sessions {query_sql}
                ORDER BY
                    CASE status
                        WHEN 'pending'   THEN 0
                        WHEN 'active'    THEN 1
                        WHEN 'resolved'  THEN 2
                        WHEN 'abandoned' THEN 3
                        ELSE 99
                    END ASC,
                    updated_at DESC,
                    created_at DESC
                LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
                """,
                *params, limit, offset,
            )

        now = datetime.utcnow()
        handoffs = []
        for row in rows:
            h = _row_to_dict(row)
            last_message = h.pop("_last_message", None)
            h["last_message_preview"] = (last_message or "")[:100]
            created = h.get("created_at") or now
            h["wait_time_seconds"] = int((now - created).total_seconds())
            handoffs.append(h)

        pending_count = counts["pending"] if counts else 0
        active_count = counts["active"] if counts else 0

        return handoffs, total or 0, pending_count, active_count

    async def update_handoff_status(
        self,
        handoff_id: str,
        status: str,
        agent_id: str = None,
        agent_name: str = None
    ) -> Optional[Dict]:
        """Update handoff status and optionally assign agent."""
        now = datetime.utcnow()
        assignments = ["status = $2", "updated_at = $3"]
        params: List[Any] = [handoff_id, status, now]

        if agent_id:
            params.append(agent_id)
            assignments.append(f"assigned_agent_id = ${len(params)}")
        if agent_name:
            params.append(agent_name)
            assignments.append(f"assigned_agent_name = ${len(params)}")
        if status == "resolved":
            params.append(now)
            assignments.append(f"resolved_at = ${len(params)}")

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE handoff_sessions SET {', '.join(assignments)} WHERE handoff_id = $1",
                *params,
            )

        if _affected(result) > 0:
            return await self.get_handoff_session(handoff_id)
        return None

    async def assign_handoff_agent(
        self,
        handoff_id: str,
        agent_id: str,
        agent_name: str,
    ) -> Optional[Dict]:
        """Set assigned agent without changing status (admin routing)."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE handoff_sessions
                SET assigned_agent_id = $2, assigned_agent_name = $3, updated_at = $4
                WHERE handoff_id = $1
                """,
                handoff_id, agent_id, agent_name, datetime.utcnow(),
            )
        if _affected(result) > 0:
            return await self.get_handoff_session(handoff_id)
        return None

    async def add_handoff_message(
        self,
        handoff_id: str,
        role: str,
        content: str,
        sender_name: str = None
    ) -> Optional[Dict]:
        """Add a message to a handoff session."""
        now = datetime.utcnow()
        message = {
            "id": str(_uuid_module.uuid4())[:8],
            "role": role,
            "content": content,
            "sender_name": sender_name,
            "timestamp": now
        }

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE handoff_sessions
                SET messages = messages || $2::jsonb, updated_at = $3
                WHERE handoff_id = $1
                """,
                handoff_id, [message], now,
            )

        if _affected(result) > 0:
            return message
        return None

    async def get_handoff_messages(
        self,
        handoff_id: str,
        since: datetime = None
    ) -> List[Dict]:
        """Get messages from a handoff session, optionally filtered by timestamp."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT messages, status, assigned_agent_name
                FROM handoff_sessions WHERE handoff_id = $1
                """,
                handoff_id,
            )

        if not row:
            return []

        messages = row["messages"] or []

        if since:
            messages = [m for m in messages if m.get("timestamp", datetime.min) > since]

        return {
            "messages": messages,
            "status": row["status"],
            "agent_name": row["assigned_agent_name"]
        }

    async def count_handoffs(
        self,
        site_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> int:
        """Count handoff sessions, optionally filtered by site and/or status."""
        clauses, params = [], []
        if site_id:
            params.append(site_id)
            clauses.append(f"site_id = ${len(params)}")
        if status:
            params.append(status)
            clauses.append(f"status = ${len(params)}")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                f"SELECT COUNT(*) FROM handoff_sessions {where}", *params
            ) or 0

    # ==================== Business Hours ====================

    async def get_site_handoff_config(self, site_id: str) -> Optional[Dict]:
        """Get handoff configuration for a site."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT handoff_config FROM sites WHERE site_id = $1", site_id
            )

        if not row:
            return None

        config = row["handoff_config"]
        if config:
            return config

        return {
            "enabled": True,
            "confidence_threshold": 0.3,
            "business_hours": {
                "enabled": False,
                "timezone": "UTC",
                "schedule": {
                    "mon": {"enabled": True, "start": "09:00", "end": "17:00"},
                    "tue": {"enabled": True, "start": "09:00", "end": "17:00"},
                    "wed": {"enabled": True, "start": "09:00", "end": "17:00"},
                    "thu": {"enabled": True, "start": "09:00", "end": "17:00"},
                    "fri": {"enabled": True, "start": "09:00", "end": "17:00"},
                    "sat": {"enabled": False, "start": "09:00", "end": "17:00"},
                    "sun": {"enabled": False, "start": "09:00", "end": "17:00"}
                },
                "offline_message": "We're currently offline. Leave your email and we'll get back to you."
            },
            "auto_suggest_phrases": [
                "I'm not sure",
                "I don't have information",
                "I cannot help with",
                "please contact support"
            ]
        }

    async def update_site_handoff_config(self, site_id: str, config: Dict) -> bool:
        """Update handoff configuration for a site."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE sites SET handoff_config = $2::jsonb, updated_at = $3
                WHERE site_id = $1
                """,
                site_id, config, datetime.utcnow(),
            )
        return _affected(result) > 0

    async def check_business_hours(self, site_id: str) -> Dict:
        """Check if site is within business hours."""
        import pytz

        config = await self.get_site_handoff_config(site_id)
        if not config:
            return {"available": True, "is_within_hours": True}

        bh = config.get("business_hours", {})
        if not bh.get("enabled", False):
            return {"available": True, "is_within_hours": True}

        tz_name = bh.get("timezone", "UTC")
        try:
            tz = pytz.timezone(tz_name)
        except Exception:
            tz = pytz.UTC

        now = datetime.now(tz)
        day_key = now.strftime("%a").lower()[:3]

        schedule = bh.get("schedule", {})
        day_schedule = schedule.get(day_key, {"enabled": False})

        if not day_schedule.get("enabled", False):
            next_day = self._find_next_working_day(schedule, day_key)
            return {
                "available": False,
                "is_within_hours": False,
                "offline_message": bh.get("offline_message"),
                "next_available": next_day
            }

        start_str = day_schedule.get("start", "09:00")
        end_str = day_schedule.get("end", "17:00")

        try:
            start_h, start_m = map(int, start_str.split(":"))
            end_h, end_m = map(int, end_str.split(":"))

            start_time = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
            end_time = now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)

            if start_time <= now <= end_time:
                return {"available": True, "is_within_hours": True}
            else:
                if now < start_time:
                    next_available = f"Today at {start_time.strftime('%H:%M')}"
                else:
                    next_day = self._find_next_working_day(schedule, day_key)
                    next_available = next_day

                return {
                    "available": False,
                    "is_within_hours": False,
                    "offline_message": bh.get("offline_message"),
                    "next_available": next_available
                }
        except Exception:
            return {"available": True, "is_within_hours": True}

    def _find_next_working_day(self, schedule: Dict, current_day: str) -> str:
        """Find the next working day."""
        days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        current_idx = days.index(current_day) if current_day in days else 0

        for i in range(1, 8):
            next_idx = (current_idx + i) % 7
            day_key = days[next_idx]
            day_schedule = schedule.get(day_key, {})
            if day_schedule.get("enabled", False):
                day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                start_time = day_schedule.get("start", "09:00")
                return f"{day_names[next_idx]} at {start_time}"

        return "Unknown"

    # ==================== Lead Management ====================

    async def save_lead(self, lead_data: Dict) -> Dict:
        """Save a new lead."""
        lead_id = str(_uuid_module.uuid4())[:12]
        now = datetime.utcnow()

        lead = {
            "lead_id": lead_id,
            "site_id": lead_data.get("site_id"),
            "session_id": lead_data.get("session_id"),
            "email": lead_data.get("email"),
            "name": lead_data.get("name"),
            "source": lead_data.get("source", "chat"),
            "captured_at": now,
            "metadata": lead_data.get("metadata", {})
        }

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO leads (lead_id, site_id, session_id, email, name,
                                   source, captured_at, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                """,
                lead_id, lead["site_id"], lead["session_id"], lead["email"],
                lead["name"], lead["source"], now, lead["metadata"],
            )

        lead["id"] = lead_id
        return lead

    async def get_leads(
        self,
        site_id: str,
        page: int = 1,
        limit: int = 20,
        search: str = None
    ) -> Tuple[List[Dict], int]:
        """Get paginated leads for a site."""
        clauses = ["site_id = $1"]
        params: List[Any] = [site_id]

        if search:
            params.append(f"%{search}%")
            clauses.append(f"(email ILIKE ${len(params)} OR name ILIKE ${len(params)})")

        where = f"WHERE {' AND '.join(clauses)}"
        offset = max(0, (page - 1) * limit)

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(f"SELECT COUNT(*) FROM leads {where}", *params)
            rows = await conn.fetch(
                f"""
                SELECT * FROM leads {where}
                ORDER BY captured_at DESC
                LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
                """,
                *params, limit, offset,
            )

        leads = []
        for row in rows:
            lead = _row_to_dict(row)
            lead["id"] = lead.get("lead_id") or lead["_id"]
            leads.append(lead)

        return leads, total or 0

    async def get_lead_by_id(self, lead_id: str) -> Optional[Dict]:
        """Get a lead by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM leads WHERE lead_id = $1", lead_id)
        lead = _row_to_dict(row)
        if lead:
            lead["id"] = lead["lead_id"]
        return lead

    async def get_lead_by_session(self, site_id: str, session_id: str) -> Optional[Dict]:
        """Get a lead by session ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM leads WHERE site_id = $1 AND session_id = $2 LIMIT 1",
                site_id, session_id,
            )
        lead = _row_to_dict(row)
        if lead:
            lead["id"] = lead.get("lead_id") or lead["_id"]
        return lead

    async def delete_lead(self, lead_id: str) -> bool:
        """Delete a lead by ID."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("DELETE FROM leads WHERE lead_id = $1", lead_id)
        return _affected(result) > 0

    async def get_all_leads_for_export(self, site_id: str) -> List[Dict]:
        """Get all leads for a site (for CSV export)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM leads WHERE site_id = $1
                ORDER BY captured_at DESC LIMIT 10000
                """,
                site_id,
            )
        leads = []
        for row in rows:
            lead = _row_to_dict(row)
            lead["id"] = lead.get("lead_id") or lead["_id"]
            leads.append(lead)
        return leads

    async def get_leads_count(self, site_id: str) -> int:
        """Get total leads count for a site."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM leads WHERE site_id = $1", site_id
            ) or 0

    # ==================== Scheduled Crawling Methods ====================

    async def get_crawl_schedule(self, site_id: str) -> Optional[Dict]:
        """Get crawl schedule configuration for a site."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT crawl_schedule FROM sites WHERE site_id = $1", site_id
            )
        if not row:
            return None

        return row["crawl_schedule"] or {
            "enabled": False,
            "frequency": "weekly",
            "custom_cron": None,
            "max_pages": 50,
            "include_patterns": [],
            "exclude_patterns": [],
            "notify_on_completion": True,
            "last_crawl_at": None,
            "next_crawl_at": None
        }

    async def update_crawl_schedule(self, site_id: str, schedule_config: Dict) -> bool:
        """Update crawl schedule configuration for a site."""
        schedule_config["updated_at"] = datetime.utcnow()
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE sites SET crawl_schedule = $2::jsonb, updated_at = $3
                WHERE site_id = $1
                """,
                site_id, schedule_config, datetime.utcnow(),
            )
        return _affected(result) > 0

    async def get_sites_with_schedules(self) -> List[Dict]:
        """Get all sites that have scheduled crawling enabled."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, site_id, url, crawl_schedule FROM sites
                WHERE crawl_schedule IS NOT NULL
                  AND (crawl_schedule -> 'enabled') = 'true'::jsonb
                LIMIT 1000
                """
            )
        return _rows_to_dicts(rows)

    async def get_crawl_history(self, site_id: str, limit: int = 10) -> List[Dict]:
        """Get crawl history for a site."""
        async with self.pool.acquire() as conn:
            site_row = await conn.fetchrow(
                "SELECT url FROM sites WHERE site_id = $1", site_id
            )
            if not site_row:
                return []

            site_url = site_row["url"]
            if not site_url:
                return []

            jobs = await conn.fetch(
                """
                SELECT * FROM crawl_jobs
                WHERE site_id = $1 OR target_url = $2
                ORDER BY created_at DESC LIMIT $3
                """,
                site_id, site_url, limit,
            )

        history = []
        for job in jobs:
            created_at = job["created_at"]
            completed_at = job["completed_at"]
            duration = None

            if created_at and completed_at:
                duration = int((completed_at - created_at).total_seconds())

            history.append({
                "job_id": str(job["id"]),
                "status": job["status"] or "unknown",
                "pages_crawled": job["pages_crawled"] or 0,
                "pages_indexed": job["pages_indexed"] or 0,
                "errors": job["errors"] or [],
                "trigger": job["trigger"] or "manual",
                "started_at": created_at,
                "completed_at": completed_at,
                "duration_seconds": duration
            })

        return history

    async def get_running_crawl_job(self, site_id: str) -> Optional[Dict]:
        """Check if there's a running crawl job for a site."""
        async with self.pool.acquire() as conn:
            site_row = await conn.fetchrow(
                "SELECT url FROM sites WHERE site_id = $1", site_id
            )
            if not site_row:
                return None

            row = await conn.fetchrow(
                """
                SELECT * FROM crawl_jobs
                WHERE (site_id = $1 OR target_url = $2) AND status = 'running'
                LIMIT 1
                """,
                site_id, site_row["url"],
            )
        return _row_to_dict(row)

    async def create_scheduled_crawl_job(
        self,
        site_id: str,
        target_url: str,
        trigger: str = "manual"
    ) -> str:
        """Create a new crawl job with site_id and trigger tracking."""
        now = datetime.utcnow()
        async with self.pool.acquire() as conn:
            job_id = await conn.fetchval(
                """
                INSERT INTO crawl_jobs (site_id, target_url, status, pages_crawled,
                                        pages_indexed, errors, trigger,
                                        created_at, updated_at)
                VALUES ($1, $2, 'running', 0, 0, '[]'::jsonb, $3, $4, $4)
                RETURNING id
                """,
                site_id, target_url, trigger, now,
            )
        return str(job_id)

    # ==================== Platform White-label Methods ====================

    async def get_platform_whitelabel(self) -> Optional[Dict]:
        """Get platform white-label configuration."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT config, updated_at FROM platform_settings WHERE type = 'whitelabel'"
            )
        if not row:
            return None

        config = dict(row["config"] or {})
        config["updated_at"] = row["updated_at"]
        return config

    async def update_platform_whitelabel(self, config: Dict) -> Dict:
        """Update platform white-label configuration."""
        now = datetime.utcnow()
        payload = {k: v for k, v in config.items() if k not in ("type", "updated_at")}

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO platform_settings (type, config, updated_at)
                VALUES ('whitelabel', $1::jsonb, $2)
                ON CONFLICT (type) DO UPDATE SET
                    config     = platform_settings.config || EXCLUDED.config,
                    updated_at = EXCLUDED.updated_at
                """,
                payload, now,
            )

        return await self.get_platform_whitelabel()

    # ==================== Q&A Training Methods ====================

    async def create_qa_pair(self, qa_data: Dict) -> Dict:
        """Create a new Q&A pair."""
        now = datetime.utcnow()
        qa_data["id"] = qa_data.get("id") or str(_uuid_module.uuid4())
        qa_data["created_at"] = now
        qa_data["updated_at"] = now
        qa_data["enabled"] = qa_data.get("enabled", True)
        qa_data["use_count"] = qa_data.get("use_count", 0)

        cols, extra = _split_doc(_TABLES["qa_pairs"], qa_data)
        names = list(cols.keys())
        values = [cols[n] for n in names]
        placeholders = [f"${i + 1}" for i in range(len(names))]

        if extra:
            values.append(extra)
            names.append("data")
            placeholders.append(f"${len(values)}::jsonb")

        async with self.pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO qa_pairs ({', '.join(names)}) VALUES ({', '.join(placeholders)})",
                *values,
            )
        return qa_data

    async def get_qa_pairs(
        self,
        site_id: str,
        page: int = 1,
        limit: int = 20,
        search: str = None,
        enabled_only: bool = False
    ) -> Tuple[List[Dict], int]:
        """Get paginated Q&A pairs for a site."""
        clauses = ["site_id = $1"]
        params: List[Any] = [site_id]

        if enabled_only:
            clauses.append("enabled = TRUE")

        if search:
            params.append(f"%{search}%")
            clauses.append(
                f"(question ILIKE ${len(params)} OR answer ILIKE ${len(params)})"
            )

        where = f"WHERE {' AND '.join(clauses)}"
        offset = max(0, (page - 1) * limit)

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(f"SELECT COUNT(*) FROM qa_pairs {where}", *params)
            rows = await conn.fetch(
                f"""
                SELECT * FROM qa_pairs {where}
                ORDER BY created_at DESC
                LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
                """,
                *params, limit, offset,
            )

        return _rows_to_dicts(rows, id_key=None), total or 0

    async def get_qa_pair(self, qa_id: str) -> Optional[Dict]:
        """Get a single Q&A pair by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM qa_pairs WHERE id = $1", qa_id)
        return _row_to_dict(row, id_key=None)

    async def update_qa_pair(self, qa_id: str, updates: Dict) -> Optional[Dict]:
        """Update a Q&A pair."""
        updates["updated_at"] = datetime.utcnow()
        updated = await self._update_row("qa_pairs", "id", qa_id, updates)
        if updated:
            return await self.get_qa_pair(qa_id)
        return None

    async def delete_qa_pair(self, qa_id: str) -> bool:
        """Delete a Q&A pair."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("DELETE FROM qa_pairs WHERE id = $1", qa_id)
        return _affected(result) > 0

    async def increment_qa_use_count(self, qa_id: str) -> bool:
        """Increment the use count for a Q&A pair."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE qa_pairs SET use_count = use_count + 1 WHERE id = $1", qa_id
            )
        return _affected(result) > 0

    async def get_qa_for_rag(self, site_id: str) -> List[Dict]:
        """Get all enabled Q&A pairs for RAG retrieval."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM qa_pairs WHERE site_id = $1 AND enabled = TRUE LIMIT 1000",
                site_id,
            )
        return _rows_to_dicts(rows, id_key=None)

    async def get_qa_stats(self, site_id: str) -> Dict:
        """Get Q&A statistics for a site."""
        async with self.pool.acquire() as conn:
            totals = await conn.fetchrow(
                """
                SELECT COUNT(*)                                        AS total_pairs,
                       COUNT(*) FILTER (WHERE enabled)                 AS enabled_pairs,
                       COALESCE(SUM(use_count), 0)                     AS total_uses
                FROM qa_pairs WHERE site_id = $1
                """,
                site_id,
            )
            most_used_rows = await conn.fetch(
                """
                SELECT id, question, use_count FROM qa_pairs
                WHERE site_id = $1 AND use_count > 0
                ORDER BY use_count DESC LIMIT 5
                """,
                site_id,
            )

        stats = {
            "total_pairs": totals["total_pairs"] if totals else 0,
            "enabled_pairs": totals["enabled_pairs"] if totals else 0,
            "total_uses": int(totals["total_uses"]) if totals else 0,
            "most_used": []
        }

        for qa in most_used_rows:
            question = qa["question"] or ""
            stats["most_used"].append({
                "id": qa["id"],
                "question": question[:50] + "..." if len(question) > 50 else question,
                "use_count": qa["use_count"]
            })

        return stats

    async def get_message_by_index(self, session_id: str, message_index: int) -> Optional[Dict]:
        """Get a specific message from a conversation by its index."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT messages FROM conversations WHERE session_id = $1", session_id
            )
        if not row:
            return None

        messages = row["messages"] or []
        if message_index < 0 or message_index >= len(messages):
            return None

        return messages[message_index]

    async def mark_message_has_qa(self, session_id: str, message_index: int, qa_id: str) -> bool:
        """Mark a message as having a Q&A pair created from it."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE conversations
                SET messages = jsonb_set(messages, ARRAY[$2::text, 'qa_pair_id'],
                                         to_jsonb($3::text), true)
                WHERE session_id = $1
                  AND $2::int >= 0
                  AND jsonb_array_length(messages) > $2::int
                """,
                session_id, message_index, qa_id,
            )
        return _affected(result) > 0

    # ==================== Documents ====================

    async def create_document(self, doc_record: Dict) -> str:
        """Insert an uploaded document's metadata row."""
        cols, extra = _split_doc(_TABLES["documents"], doc_record)
        names = list(cols.keys())
        values = [cols[n] for n in names]
        placeholders = [
            f"${i + 1}::jsonb" if names[i] in _TABLES["documents"].json_columns
            else f"${i + 1}"
            for i in range(len(names))
        ]

        if extra:
            values.append(extra)
            names.append("data")
            placeholders.append(f"${len(values)}::jsonb")

        async with self.pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO documents ({', '.join(names)}) "
                f"VALUES ({', '.join(placeholders)})",
                *values,
            )
        return doc_record.get("doc_id", "")

    async def update_document(self, doc_id: str, updates: Dict) -> bool:
        """Update a document metadata row."""
        return await self._update_row("documents", "doc_id", doc_id, updates)

    async def get_documents(self, site_id: str, limit: int = 1000) -> List[Dict]:
        """List a site's documents."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM documents WHERE site_id = $1 "
                "ORDER BY uploaded_at DESC LIMIT $2",
                site_id, limit,
            )
        return _rows_to_dicts(rows)

    async def get_document(self, doc_id: str, site_id: Optional[str] = None) -> Optional[Dict]:
        """Fetch a single document, optionally scoped to a site."""
        async with self.pool.acquire() as conn:
            if site_id:
                row = await conn.fetchrow(
                    "SELECT * FROM documents WHERE doc_id = $1 AND site_id = $2",
                    doc_id, site_id,
                )
            else:
                row = await conn.fetchrow(
                    "SELECT * FROM documents WHERE doc_id = $1", doc_id
                )
        return _row_to_dict(row)

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document metadata row."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM documents WHERE doc_id = $1", doc_id
            )
        return _affected(result) > 0

    async def count_documents(self, site_id: str) -> int:
        """Count a site's documents."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM documents WHERE site_id = $1", site_id
            ) or 0

    # ==================== Maintenance ====================

    async def clear_all_data(self) -> None:
        """
        Delete conversations, pages, crawl jobs and long-term memory.

        Backs the destructive admin endpoint. Sites, users, leads and Q&A pairs
        are left intact, exactly as before.
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                "TRUNCATE conversations, pages, crawl_jobs, long_term_memory"
            )


def _affected(status: str) -> int:
    """Extract the affected-row count from an asyncpg command tag."""
    if not status:
        return 0
    parts = status.split()
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return 0


# Singleton instance
_postgres: Optional[PostgresDB] = None
_postgres_lock = asyncio.Lock()


async def get_postgres() -> PostgresDB:
    """Get or create the PostgreSQL client."""
    global _postgres
    if _postgres is None:
        async with _postgres_lock:
            if _postgres is None:
                instance = PostgresDB()
                await instance.connect()
                _postgres = instance
    return _postgres


async def reset_postgres() -> None:
    """Drop the cached client (used by tests and on shutdown)."""
    global _postgres
    if _postgres is not None:
        await _postgres.disconnect()
        _postgres = None

"""
Tests for database providers and operations.

Tests PostgreSQL connection, CRUD operations for sites, users, conversations,
and vector store.

The database tests run against a real PostgreSQL instance rather than mocks:
the data layer's behaviour lives largely in SQL and JSONB semantics, which
mocks cannot verify. Point TEST_POSTGRES_URL at a scratch database; the whole
module skips cleanly when no server is reachable.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for optional packages that may cause import issues
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestPostgresConnection:
    """Tests for PostgreSQL connection handling."""

    async def test_connect_success(self, pg_db):
        """Test successful connection."""
        assert pg_db.pool is not None
        assert await pg_db.ping() is True

    async def test_connect_failure(self):
        """Test connection failure raises."""
        from app.database.postgres import PostgresDB

        db = PostgresDB()
        with patch("app.database.postgres.asyncpg.create_pool",
                   side_effect=Exception("Connection failed")):
            with pytest.raises(Exception):
                await db.connect()

    async def test_disconnect(self):
        """Test disconnect closes the pool."""
        from app.database.postgres import PostgresDB

        db = PostgresDB()
        pool = MagicMock()
        pool.close = AsyncMock()
        db.pool = pool

        await db.disconnect()

        pool.close.assert_called_once()
        assert db.pool is None

    async def test_disconnect_no_pool(self):
        """Test disconnect with no pool does not raise."""
        from app.database.postgres import PostgresDB

        db = PostgresDB()
        db.pool = None
        await db.disconnect()  # must not raise


class TestPostgresProviderConnection:
    """Tests for the PostgresProvider health checks."""

    async def test_provider_health_check_success(self, pg_db):
        """Health check returns True when reachable."""
        assert await pg_db.health_check() is True

    async def test_provider_health_check_failure(self):
        """Health check returns False when unreachable."""
        from app.providers.database.postgres_provider import PostgresProvider

        provider = PostgresProvider()
        provider.pool = MagicMock()
        provider.pool.acquire = MagicMock(side_effect=Exception("down"))

        assert await provider.health_check() is False


class TestConversationOperations:
    """Tests for conversation CRUD operations."""

    async def test_save_message(self, pg_db, a_session):
        """Test saving a message to a conversation."""
        message_id = await pg_db.save_message(
            session_id=a_session, role="user", content="Hello", site_id="site_1"
        )

        assert message_id.startswith(a_session)
        history = await pg_db.get_conversation_history(a_session)
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert isinstance(history[0]["timestamp"], datetime)

    async def test_save_message_with_sources(self, pg_db, a_session):
        """Test saving a message with source documents."""
        sources = [{"url": "https://example.com", "title": "Example"}]
        await pg_db.save_message(
            session_id=a_session, role="assistant", content="Answer",
            sources=sources, response_time_ms=250,
        )

        history = await pg_db.get_conversation_history(a_session)
        assert history[0]["sources"] == sources
        assert history[0]["response_time_ms"] == 250

    async def test_save_message_sets_conversation_defaults(self, pg_db, a_session):
        """A new conversation gets the documented default fields."""
        await pg_db.save_message(a_session, "user", "Hi", site_id="site_1")

        conv = await pg_db.get_conversation_full(a_session)
        assert conv["status"] == "open"
        assert conv["priority"] == "medium"
        assert conv["unread"] is True
        assert conv["tags"] == []
        assert conv["notes"] == []

    async def test_save_message_records_first_response(self, pg_db, a_session):
        """first_response_at is stamped once, on the first assistant reply."""
        await pg_db.save_message(a_session, "user", "Q")
        await pg_db.save_message(a_session, "assistant", "A1")
        first = (await pg_db.get_conversation_full(a_session))["first_response_at"]
        assert first is not None

        await pg_db.save_message(a_session, "assistant", "A2")
        assert (await pg_db.get_conversation_full(a_session))["first_response_at"] == first

    async def test_get_conversation_history(self, pg_db, a_session):
        """Test retrieving conversation history."""
        await pg_db.save_message(a_session, "user", "Hello")
        await pg_db.save_message(a_session, "assistant", "Hi there")

        history = await pg_db.get_conversation_history(a_session)

        assert len(history) == 2
        assert [m["role"] for m in history] == ["user", "assistant"]

    async def test_get_conversation_history_empty(self, pg_db):
        """Test retrieving history for a non-existent session."""
        assert await pg_db.get_conversation_history("nonexistent") == []

    async def test_get_conversation_history_with_limit(self, pg_db, a_session):
        """Test the limit returns the most recent messages."""
        for i in range(5):
            await pg_db.save_message(a_session, "user", f"Message {i}")

        history = await pg_db.get_conversation_history(a_session, limit=2)

        assert len(history) == 2
        assert history[-1]["content"] == "Message 4"

    async def test_clear_conversation(self, pg_db, a_session):
        """Test clearing a conversation."""
        await pg_db.save_message(a_session, "user", "Hello")

        assert await pg_db.clear_conversation(a_session) is True
        assert await pg_db.get_conversation_history(a_session) == []

    async def test_clear_nonexistent_conversation(self, pg_db):
        """Test clearing a non-existent conversation returns False."""
        assert await pg_db.clear_conversation("nonexistent") is False

    async def test_add_message_feedback(self, pg_db, a_session):
        """Test adding feedback to a message."""
        await pg_db.save_message(a_session, "user", "Q")
        message_id = await pg_db.save_message(a_session, "assistant", "A")

        assert await pg_db.add_message_feedback(a_session, message_id, "positive") is True

        history = await pg_db.get_conversation_history(a_session)
        assert history[1]["feedback"] == "positive"
        assert isinstance(history[1]["feedback_at"], datetime)

    async def test_add_message_feedback_unknown_message(self, pg_db, a_session):
        """Feedback for an unknown message id returns False."""
        await pg_db.save_message(a_session, "user", "Q")
        assert await pg_db.add_message_feedback(a_session, "missing", "positive") is False

    async def test_add_feedback_by_index(self, pg_db, a_session):
        """Test adding feedback by message index."""
        await pg_db.save_message(a_session, "user", "Q")
        await pg_db.save_message(a_session, "assistant", "A")

        assert await pg_db.add_feedback_by_index(a_session, 1, "negative") is True
        assert (await pg_db.get_message_by_index(a_session, 1))["feedback"] == "negative"

    async def test_add_feedback_by_index_out_of_range(self, pg_db, a_session):
        """Out-of-range index returns False rather than corrupting the array."""
        await pg_db.save_message(a_session, "user", "Q")
        assert await pg_db.add_feedback_by_index(a_session, 99, "negative") is False

    async def test_delete_conversations_bulk(self, pg_db):
        """Test bulk deleting conversations."""
        sessions = [f"bulk_{uuid.uuid4().hex[:8]}" for _ in range(3)]
        for s in sessions:
            await pg_db.save_message(s, "user", "Hello")

        assert await pg_db.delete_conversations_bulk(sessions) == 3

    async def test_delete_conversations_bulk_empty(self, pg_db):
        """Bulk delete with no ids is a no-op."""
        assert await pg_db.delete_conversations_bulk([]) == 0

    async def test_get_conversations_paginated(self, pg_db, a_site):
        """Paginated listing returns counts and the first message preview."""
        session = f"page_{uuid.uuid4().hex[:8]}"
        await pg_db.save_message(session, "user", "First question", site_id=a_site)
        await pg_db.save_message(session, "assistant", "Reply", site_id=a_site)

        convs, total = await pg_db.get_conversations_paginated(site_id=a_site)

        assert total == 1
        assert convs[0]["message_count"] == 2
        assert convs[0]["first_message"] == "First question"

    async def test_get_conversations_paginated_filters(self, pg_db, a_site):
        """Status, priority and tag filters narrow the result set."""
        session = f"filt_{uuid.uuid4().hex[:8]}"
        await pg_db.save_message(session, "user", "Hi", site_id=a_site)
        await pg_db.update_conversation_status(session, "resolved")
        await pg_db.update_conversation_tags(session, ["vip"])

        _, resolved = await pg_db.get_conversations_paginated(site_id=a_site, status="resolved")
        _, open_ = await pg_db.get_conversations_paginated(site_id=a_site, status="open")
        _, tagged = await pg_db.get_conversations_paginated(site_id=a_site, tag="vip")

        assert (resolved, open_, tagged) == (1, 0, 1)

    async def test_get_conversations_paginated_rejects_bad_sort(self, pg_db, a_site):
        """An unknown sort field falls back to the default instead of reaching SQL."""
        await pg_db.save_message(f"s_{uuid.uuid4().hex[:8]}", "user", "Hi", site_id=a_site)

        convs, total = await pg_db.get_conversations_paginated(
            site_id=a_site, sort_by="updated_at; DROP TABLE users"
        )

        assert total == 1 and len(convs) == 1

    async def test_get_conversation_full_stats(self, pg_db, a_session):
        """Full conversation includes derived stats and sentiment."""
        await pg_db.save_message(a_session, "user", "Q")
        mid = await pg_db.save_message(a_session, "assistant", "A", response_time_ms=100)
        await pg_db.add_message_feedback(a_session, mid, "positive")

        conv = await pg_db.get_conversation_full(a_session)

        assert conv["stats"]["message_count"] == 2
        assert conv["stats"]["user_messages"] == 1
        assert conv["stats"]["assistant_messages"] == 1
        assert conv["stats"]["positive_feedback"] == 1
        assert conv["stats"]["avg_response_time_ms"] == 100
        assert conv["sentiment"] == 1.0

    async def test_get_conversation_full_missing(self, pg_db):
        """Unknown session returns None."""
        assert await pg_db.get_conversation_full("nope") is None

    async def test_search_conversations(self, pg_db, a_site):
        """Full-text search finds messages and returns a snippet."""
        session = f"search_{uuid.uuid4().hex[:8]}"
        await pg_db.save_message(session, "user", "How do I reset my password", site_id=a_site)

        results, total = await pg_db.search_conversations("password", site_id=a_site)

        assert total == 1
        assert "password" in results[0]["matching_snippet"].lower()

    async def test_search_conversations_or_semantics(self, pg_db, a_site):
        """Multi-word search matches ANY term, as MongoDB's $text did."""
        session = f"search_{uuid.uuid4().hex[:8]}"
        await pg_db.save_message(session, "user", "billing question", site_id=a_site)

        _, total = await pg_db.search_conversations("billing kangaroo", site_id=a_site)

        assert total == 1

    async def test_conversation_notes_lifecycle(self, pg_db, a_session):
        """Notes can be added, edited and removed."""
        await pg_db.save_message(a_session, "user", "Hi")

        note = await pg_db.add_conversation_note(a_session, "internal")
        assert await pg_db.update_conversation_note(a_session, note["note_id"], "edited") is True
        assert (await pg_db.get_conversation_full(a_session))["notes"][0]["content"] == "edited"
        assert await pg_db.delete_conversation_note(a_session, note["note_id"]) is True
        assert (await pg_db.get_conversation_full(a_session))["notes"] == []

    async def test_auto_close_inactive_conversations(self, pg_db, a_session):
        """Stale open conversations are closed."""
        await pg_db.save_message(a_session, "user", "Hi")
        async with pg_db.pool.acquire() as conn:
            await conn.execute(
                "UPDATE conversations SET updated_at = $2 WHERE session_id = $1",
                a_session, datetime.utcnow() - timedelta(days=30),
            )

        assert await pg_db.auto_close_inactive_conversations(7) == 1
        assert (await pg_db.get_conversation_full(a_session))["status"] == "closed"


class TestSiteOperations:
    """Tests for site CRUD operations."""

    async def test_create_site(self, pg_db):
        """Test creating a new site."""
        site_id = f"site_{uuid.uuid4().hex[:8]}"
        returned = await pg_db.create_site({
            "site_id": site_id, "name": "Test Site",
            "url": "https://test.com", "user_id": "user_1",
        })

        assert returned == site_id
        site = await pg_db.get_site(site_id)
        assert site["name"] == "Test Site"
        assert site["created_at"] is not None

    async def test_create_site_preserves_unknown_fields(self, pg_db):
        """Schemaless extras survive the round trip, as they did in MongoDB."""
        site_id = f"site_{uuid.uuid4().hex[:8]}"
        await pg_db.create_site({
            "site_id": site_id, "name": "S", "custom_setting": {"nested": True},
        })

        assert (await pg_db.get_site(site_id))["custom_setting"] == {"nested": True}

    async def test_get_site(self, pg_db, a_site):
        """Test retrieving a site by ID."""
        site = await pg_db.get_site(a_site)

        assert site["site_id"] == a_site
        assert "_id" in site

    async def test_get_site_not_found(self, pg_db):
        """Test retrieving a non-existent site."""
        assert await pg_db.get_site("nonexistent") is None

    async def test_list_sites(self, pg_db, a_site):
        """Test listing all sites."""
        sites = await pg_db.list_sites()

        assert any(s["site_id"] == a_site for s in sites)

    async def test_list_sites_by_user(self, pg_db):
        """Test listing sites filtered by user."""
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        await pg_db.create_site({"site_id": f"s_{uuid.uuid4().hex[:8]}", "user_id": user_id})

        sites = await pg_db.list_sites(user_id=user_id)

        assert len(sites) == 1
        assert sites[0]["user_id"] == user_id

    async def test_list_sites_by_site_ids(self, pg_db, a_site):
        """Test listing sites by an explicit id list."""
        assert len(await pg_db.list_sites_by_site_ids([a_site])) == 1
        assert await pg_db.list_sites_by_site_ids([]) == []

    async def test_update_site(self, pg_db, a_site):
        """Test updating site data."""
        assert await pg_db.update_site(a_site, {"name": "Updated"}) is True
        assert (await pg_db.get_site(a_site))["name"] == "Updated"

    async def test_update_site_unknown_field(self, pg_db, a_site):
        """Unknown update keys land in the JSONB catch-all."""
        await pg_db.update_site(a_site, {"brand_new_key": 42})

        assert (await pg_db.get_site(a_site))["brand_new_key"] == 42

    async def test_delete_site(self, pg_db, a_site):
        """Test deleting a site."""
        assert await pg_db.delete_site(a_site) is True
        assert await pg_db.get_site(a_site) is None


class TestUserOperations:
    """Tests for user CRUD operations."""

    async def test_create_user(self, pg_db):
        """Test creating a new user."""
        email = f"u{uuid.uuid4().hex[:8]}@example.com"
        user_id = await pg_db.create_user({
            "email": email, "name": "Test User",
            "password_hash": "hashed", "role": "user",
        })

        assert user_id
        user = await pg_db.get_user_by_email(email)
        assert user["name"] == "Test User"

    async def test_create_user_id_matches_row_id(self, pg_db):
        """str(user["_id"]) and user["user_id"] agree, so ownership checks are stable."""
        email = f"u{uuid.uuid4().hex[:8]}@example.com"
        user_id = await pg_db.create_user({"email": email, "password_hash": "h"})

        user = await pg_db.get_user_by_email(email)
        assert str(user["_id"]) == user["user_id"] == user_id

    async def test_get_user_by_email(self, pg_db, a_user):
        """Test retrieving a user by email."""
        user = await pg_db.get_user_by_email(a_user["email"])

        assert user["user_id"] == a_user["user_id"]

    async def test_get_user_by_email_not_found(self, pg_db):
        """Test retrieving a non-existent user."""
        assert await pg_db.get_user_by_email("nobody@example.com") is None

    async def test_get_user_by_id(self, pg_db, a_user):
        """Test retrieving a user by either identifier."""
        assert (await pg_db.get_user_by_id(a_user["user_id"]))["email"] == a_user["email"]
        assert (await pg_db.get_user_by_id(str(a_user["_id"])))["email"] == a_user["email"]

    async def test_get_user_by_id_invalid(self, pg_db):
        """A non-UUID id is handled without raising."""
        assert await pg_db.get_user_by_id("not-a-uuid") is None

    async def test_update_user(self, pg_db, a_user):
        """Test updating a user."""
        assert await pg_db.update_user(a_user["user_id"], {"name": "Renamed"}) is True
        assert (await pg_db.get_user_by_id(a_user["user_id"]))["name"] == "Renamed"

    async def test_delete_user(self, pg_db, a_user):
        """Test deleting a user."""
        assert await pg_db.delete_user(a_user["user_id"]) is True
        assert await pg_db.get_user_by_id(a_user["user_id"]) is None

    async def test_agent_listing_and_transfer(self, pg_db, a_user):
        """Agents are scoped by owner and can be reassigned in bulk."""
        await pg_db.create_user({
            "email": f"a{uuid.uuid4().hex[:8]}@example.com", "role": "agent",
            "owner_id": a_user["user_id"], "password_hash": "h",
        })
        new_owner = await pg_db.create_user({
            "email": f"o{uuid.uuid4().hex[:8]}@example.com", "password_hash": "h",
        })

        assert len(await pg_db.list_users_agents_for_owner(a_user["user_id"])) == 1
        assert await pg_db.transfer_agents_to_user(a_user["user_id"], new_owner) == 1
        assert len(await pg_db.list_users_agents_for_owner(new_owner)) == 1


class TestCrawlJobOperations:
    """Tests for crawl job operations."""

    async def test_create_crawl_job(self, pg_db):
        """Test creating a crawl job."""
        job_id = await pg_db.create_crawl_job("https://example.com")

        job = await pg_db.get_crawl_job(job_id)
        assert job["status"] == "running"
        assert job["target_url"] == "https://example.com"

    async def test_update_crawl_job(self, pg_db):
        """Test updating a crawl job."""
        job_id = await pg_db.create_crawl_job("https://example.com")

        await pg_db.update_crawl_job(job_id, status="completed", pages_crawled=10, pages_indexed=8)

        job = await pg_db.get_crawl_job(job_id)
        assert job["status"] == "completed"
        assert job["pages_crawled"] == 10
        assert job["pages_indexed"] == 8

    async def test_update_crawl_job_with_error(self, pg_db):
        """Test errors accumulate on the job."""
        job_id = await pg_db.create_crawl_job("https://example.com")

        await pg_db.update_crawl_job(job_id, error="first")
        await pg_db.update_crawl_job(job_id, error="second")

        assert (await pg_db.get_crawl_job(job_id))["errors"] == ["first", "second"]

    async def test_update_crawl_job_invalid_id(self, pg_db):
        """An unparseable job id is ignored rather than raising."""
        await pg_db.update_crawl_job("not-a-uuid", status="completed")
        assert await pg_db.get_crawl_job("not-a-uuid") is None

    async def test_mark_crawl_job_completed(self, pg_db):
        """completed_at is stamped so crawl history can report duration."""
        job_id = await pg_db.create_crawl_job("https://example.com")

        assert await pg_db.mark_crawl_job_completed(job_id) is True
        assert (await pg_db.get_crawl_job(job_id))["completed_at"] is not None

    async def test_get_crawl_job_by_url(self, pg_db):
        """Test finding the latest job for a URL."""
        await pg_db.create_crawl_job("https://example.com")
        job_id = await pg_db.create_crawl_job("https://example.com")

        assert (await pg_db.get_crawl_job_by_url("https://example.com"))["_id"] == job_id


class TestPageOperations:
    """Tests for page operations."""

    async def test_save_page(self, pg_db):
        """Test saving a crawled page."""
        await pg_db.save_page(
            url="https://example.com/page", title="Test Page",
            content="Page content", chunk_count=5, metadata={"lang": "en"},
        )

        page = await pg_db.get_page("https://example.com/page")
        assert page["title"] == "Test Page"
        assert page["chunk_count"] == 5
        assert page["metadata"] == {"lang": "en"}
        assert page["status"] == "indexed"

    async def test_save_page_upserts(self, pg_db):
        """Re-saving the same URL updates rather than duplicates."""
        await pg_db.save_page("https://example.com/p", "Old", "c")
        await pg_db.save_page("https://example.com/p", "New", "c2")

        assert (await pg_db.get_page("https://example.com/p"))["title"] == "New"
        assert await pg_db.get_page_count() == 1

    async def test_get_page(self, pg_db):
        """Test retrieving a page by URL."""
        await pg_db.save_page("https://example.com/x", "X", "content")

        assert (await pg_db.get_page("https://example.com/x"))["url"] == "https://example.com/x"

    async def test_get_page_not_found(self, pg_db):
        """Test retrieving a non-existent page."""
        assert await pg_db.get_page("https://nope.example.com") is None

    async def test_get_all_pages(self, pg_db):
        """Test retrieving all pages."""
        await pg_db.save_page("https://example.com/1", "1", "c")
        await pg_db.save_page("https://example.com/2", "2", "c")

        assert len(await pg_db.get_all_pages()) == 2
        assert len(await pg_db.get_all_pages(status="indexed")) == 2

    async def test_delete_page(self, pg_db):
        """Test deleting a page."""
        await pg_db.save_page("https://example.com/d", "D", "c")

        assert await pg_db.delete_page("https://example.com/d") is True
        assert await pg_db.delete_page("https://example.com/d") is False

    async def test_get_page_count(self, pg_db):
        """Test counting indexed pages."""
        assert await pg_db.get_page_count() == 0
        await pg_db.save_page("https://example.com/c", "C", "c")
        assert await pg_db.get_page_count() == 1


class TestTriggerOperations:
    """Tests for proactive chat trigger operations."""

    async def test_get_site_triggers(self, pg_db, a_site):
        """Test retrieving triggers for a site."""
        await pg_db.save_trigger(a_site, {"name": "Welcome", "message": "Hi"})

        result = await pg_db.get_site_triggers(a_site)

        assert len(result["triggers"]) == 1
        assert result["triggers"][0]["name"] == "Welcome"

    async def test_get_site_triggers_not_found(self, pg_db):
        """Test retrieving triggers for a non-existent site returns defaults."""
        result = await pg_db.get_site_triggers("nonexistent")

        assert result == {"triggers": [], "global_cooldown_ms": 30000}

    async def test_save_trigger_new(self, pg_db, a_site):
        """Test saving a new trigger generates an id."""
        trigger = await pg_db.save_trigger(a_site, {"name": "New Trigger"})

        assert trigger["id"]
        assert trigger["created_at"] is not None
        assert len((await pg_db.get_site_triggers(a_site))["triggers"]) == 1

    async def test_save_trigger_replaces_existing(self, pg_db, a_site):
        """Saving a trigger with a known id replaces it in place."""
        trigger = await pg_db.save_trigger(a_site, {"name": "First"})
        await pg_db.save_trigger(a_site, {**trigger, "name": "Second"})

        triggers = (await pg_db.get_site_triggers(a_site))["triggers"]
        assert len(triggers) == 1
        assert triggers[0]["name"] == "Second"

    async def test_update_trigger(self, pg_db, a_site):
        """Test partially updating a trigger."""
        trigger = await pg_db.save_trigger(a_site, {"name": "T", "delay": 1000})

        updated = await pg_db.update_trigger(a_site, trigger["id"], {"delay": 5000})

        assert updated["delay"] == 5000
        assert updated["name"] == "T"

    async def test_update_trigger_missing(self, pg_db, a_site):
        """Updating an unknown trigger returns None."""
        assert await pg_db.update_trigger(a_site, "missing", {"delay": 1}) is None

    async def test_reorder_triggers(self, pg_db, a_site):
        """Test reordering triggers by the supplied id order."""
        t1 = await pg_db.save_trigger(a_site, {"name": "One"})
        t2 = await pg_db.save_trigger(a_site, {"name": "Two"})

        assert await pg_db.reorder_triggers(a_site, [t2["id"], t1["id"]]) is True

        order = [t["id"] for t in (await pg_db.get_site_triggers(a_site))["triggers"]]
        assert order == [t2["id"], t1["id"]]

    async def test_set_global_cooldown(self, pg_db, a_site):
        """Test setting the global cooldown."""
        assert await pg_db.set_global_cooldown(a_site, 5000) is True
        assert (await pg_db.get_site_triggers(a_site))["global_cooldown_ms"] == 5000

    async def test_delete_trigger(self, pg_db, a_site):
        """Test deleting a trigger."""
        trigger = await pg_db.save_trigger(a_site, {"name": "T"})

        assert await pg_db.delete_trigger(a_site, trigger["id"]) is True
        assert await pg_db.delete_trigger(a_site, trigger["id"]) is False
        assert (await pg_db.get_site_triggers(a_site))["triggers"] == []


class TestHandoffOperations:
    """Tests for human handoff operations."""

    async def test_create_handoff_session(self, pg_db, a_site):
        """Test creating a handoff session."""
        handoff = await pg_db.create_handoff_session(
            session_id="sess_1", site_id=a_site, reason="user_request",
            visitor_email="visitor@example.com",
        )

        assert handoff["status"] == "pending"
        assert handoff["visitor_email"] == "visitor@example.com"
        assert handoff["handoff_id"]

    async def test_get_handoff_session(self, pg_db, a_site):
        """Test retrieving a handoff session."""
        created = await pg_db.create_handoff_session(
            session_id="sess_1", site_id=a_site,
            ai_conversation=[{"role": "user", "content": "hi"}],
        )

        handoff = await pg_db.get_handoff_session(created["handoff_id"])

        assert handoff["site_id"] == a_site
        assert handoff["ai_conversation"][0]["content"] == "hi"

    async def test_get_handoff_by_session(self, pg_db, a_site):
        """Test finding the active handoff for a chat session."""
        created = await pg_db.create_handoff_session(session_id="sess_x", site_id=a_site)

        found = await pg_db.get_handoff_by_session("sess_x")

        assert found["handoff_id"] == created["handoff_id"]

    async def test_update_handoff_status(self, pg_db, a_site):
        """Test updating handoff status and assigning an agent."""
        created = await pg_db.create_handoff_session(session_id="sess_1", site_id=a_site)

        updated = await pg_db.update_handoff_status(
            created["handoff_id"], "active", agent_id="agent_1", agent_name="Agent",
        )

        assert updated["status"] == "active"
        assert updated["assigned_agent_id"] == "agent_1"

    async def test_update_handoff_status_resolved_stamps_time(self, pg_db, a_site):
        """Resolving a handoff records resolved_at."""
        created = await pg_db.create_handoff_session(session_id="sess_1", site_id=a_site)

        updated = await pg_db.update_handoff_status(created["handoff_id"], "resolved")

        assert updated["resolved_at"] is not None

    async def test_add_handoff_message(self, pg_db, a_site):
        """Test adding a message to a handoff session."""
        created = await pg_db.create_handoff_session(session_id="sess_1", site_id=a_site)

        message = await pg_db.add_handoff_message(
            created["handoff_id"], "agent", "How can I help?", "Agent Smith",
        )

        assert message["content"] == "How can I help?"
        result = await pg_db.get_handoff_messages(created["handoff_id"])
        assert len(result["messages"]) == 1
        assert result["status"] == "pending"

    async def test_get_handoff_messages_since(self, pg_db, a_site):
        """The `since` filter excludes older messages."""
        created = await pg_db.create_handoff_session(session_id="sess_1", site_id=a_site)
        await pg_db.add_handoff_message(created["handoff_id"], "agent", "Hi")

        result = await pg_db.get_handoff_messages(
            created["handoff_id"], since=datetime.utcnow() + timedelta(minutes=1)
        )

        assert result["messages"] == []

    async def test_get_handoff_queue(self, pg_db, a_site):
        """Queue returns rows plus pending/active counts."""
        await pg_db.create_handoff_session(session_id="s1", site_id=a_site)
        second = await pg_db.create_handoff_session(session_id="s2", site_id=a_site)
        await pg_db.update_handoff_status(second["handoff_id"], "active")

        rows, total, pending, active = await pg_db.get_handoff_queue(site_id=a_site)

        assert (total, pending, active) == (2, 1, 1)
        assert rows[0]["status"] == "pending"  # pending sorts ahead of active
        assert "messages" not in rows[0]

    async def test_get_handoff_queue_agent_visibility(self, pg_db, a_site):
        """Agents see unassigned handoffs and their own, not other agents'."""
        mine = await pg_db.create_handoff_session(session_id="s1", site_id=a_site)
        theirs = await pg_db.create_handoff_session(session_id="s2", site_id=a_site)
        await pg_db.assign_handoff_agent(mine["handoff_id"], "agent_1", "Me")
        await pg_db.assign_handoff_agent(theirs["handoff_id"], "agent_2", "Them")

        _, total, _, _ = await pg_db.get_handoff_queue(
            site_id=a_site, agent_queue_user_id="agent_1"
        )

        assert total == 1

    async def test_bump_handoff_visitor_requeue(self, pg_db, a_site):
        """Re-requesting a human increments the queue signal counter."""
        created = await pg_db.create_handoff_session(session_id="s1", site_id=a_site)

        await pg_db.bump_handoff_visitor_requeue_pending(created["handoff_id"])

        assert (await pg_db.get_handoff_session(created["handoff_id"]))["visitor_queue_signals"] == 1
class TestVectorStoreOperations:
    """Tests for vector store operations."""
    
    @pytest.fixture(autouse=True)
    def mock_torch_multiprocessing(self):
        """Mock torch multiprocessing to avoid torch_shm_manager errors."""
        with patch.dict('sys.modules', {
            'torch': MagicMock(),
            'torch.multiprocessing': MagicMock(),
            'torch.multiprocessing.reductions': MagicMock(),
        }):
            yield
    
    def test_vector_store_initialize(self, mock_torch_multiprocessing):
        """Test vector store initialization."""
        with patch.dict('sys.modules', {
            'langchain_community.embeddings': MagicMock(),
            'sentence_transformers': MagicMock(),
        }):
            with patch("langchain_community.embeddings.HuggingFaceEmbeddings") as mock_embeddings:
                with patch("langchain_community.vectorstores.FAISS") as mock_faiss:
                    with patch("os.path.exists", return_value=False):
                        with patch("os.makedirs"):
                            mock_embeddings.return_value = MagicMock()
                            mock_faiss.from_texts.return_value = MagicMock()
                            
                            from app.database.vector_store import VectorStore
                            
                            vs = VectorStore()
                            vs.embeddings = mock_embeddings.return_value
                            vs.vector_store = mock_faiss.from_texts.return_value
                            vs._initialized = True
                            
                            assert vs._initialized is True
    
    def test_vector_store_add_documents(self, mock_torch_multiprocessing):
        """Test adding documents to vector store."""
        with patch.dict('sys.modules', {
            'langchain_community.embeddings': MagicMock(),
            'sentence_transformers': MagicMock(),
        }):
            with patch("langchain_community.embeddings.HuggingFaceEmbeddings") as mock_embeddings:
                with patch("langchain_community.vectorstores.FAISS") as mock_faiss:
                    with patch("os.path.exists", return_value=False):
                        with patch("os.makedirs"):
                            mock_vs_instance = MagicMock()
                            mock_vs_instance.add_documents.return_value = ["id1", "id2"]
                            mock_faiss.from_texts.return_value = mock_vs_instance
                            mock_embeddings.return_value = MagicMock()
                            
                            from app.database.vector_store import VectorStore
                            
                            vs = VectorStore()
                            vs.embeddings = mock_embeddings.return_value
                            vs.vector_store = mock_vs_instance
                            vs._initialized = True
                            
                            mock_doc1 = MagicMock()
                            mock_doc1.page_content = "Doc 1"
                            mock_doc1.metadata = {"source": "test"}
                            mock_doc2 = MagicMock()
                            mock_doc2.page_content = "Doc 2"
                            mock_doc2.metadata = {"source": "test"}
                            
                            ids = vs.add_documents([mock_doc1, mock_doc2])
                            
                            assert ids == ["id1", "id2"]
    
    def test_vector_store_similarity_search(self, mock_torch_multiprocessing):
        """Test similarity search."""
        with patch.dict('sys.modules', {
            'langchain_community.embeddings': MagicMock(),
            'sentence_transformers': MagicMock(),
        }):
            with patch("langchain_community.embeddings.HuggingFaceEmbeddings") as mock_embeddings:
                with patch("langchain_community.vectorstores.FAISS") as mock_faiss:
                    with patch("os.path.exists", return_value=False):
                        with patch("os.makedirs"):
                            mock_doc1 = MagicMock()
                            mock_doc1.page_content = "Result 1"
                            mock_doc1.metadata = {}
                            mock_doc2 = MagicMock()
                            mock_doc2.page_content = "Result 2"
                            mock_doc2.metadata = {}
                            
                            mock_vs_instance = MagicMock()
                            mock_vs_instance.similarity_search.return_value = [mock_doc1, mock_doc2]
                            mock_faiss.from_texts.return_value = mock_vs_instance
                            mock_embeddings.return_value = MagicMock()
                            
                            from app.database.vector_store import VectorStore
                            
                            vs = VectorStore()
                            vs.embeddings = mock_embeddings.return_value
                            vs.vector_store = mock_vs_instance
                            vs._initialized = True
                            
                            results = vs.similarity_search("test query", k=2)
                            
                            assert len(results) == 2
                            mock_vs_instance.similarity_search.assert_called_once_with("test query", k=2)
    
    def test_vector_store_get_collection_stats(self, mock_torch_multiprocessing):
        """Test getting collection stats."""
        with patch.dict('sys.modules', {
            'langchain_community.embeddings': MagicMock(),
            'sentence_transformers': MagicMock(),
        }):
            with patch("langchain_community.embeddings.HuggingFaceEmbeddings") as mock_embeddings:
                with patch("langchain_community.vectorstores.FAISS") as mock_faiss:
                    with patch("os.path.exists", return_value=False):
                        with patch("os.makedirs"):
                            mock_vs_instance = MagicMock()
                            mock_vs_instance.index.ntotal = 100
                            mock_faiss.from_texts.return_value = mock_vs_instance
                            mock_embeddings.return_value = MagicMock()
                            
                            from app.database.vector_store import VectorStore
                            
                            vs = VectorStore()
                            vs.embeddings = mock_embeddings.return_value
                            vs.vector_store = mock_vs_instance
                            vs._initialized = True
                            vs.index_path = "faiss_index"
                            
                            stats = vs.get_collection_stats()
                            
                            assert stats["name"] == "faiss_index"
                            assert stats["count"] == 100




class TestAnalyticsOperations:
    """Tests for analytics aggregation."""

    async def test_get_analytics_overview(self, pg_provider, a_site):
        """Test the provider's analytics overview aggregation."""
        await pg_provider.save_message("s1", "user", "Hello", site_id=a_site)
        await pg_provider.save_message("s1", "assistant", "Hi", site_id=a_site)
        await pg_provider.save_message("s2", "user", "Question", site_id=a_site)

        result = await pg_provider.get_analytics_overview(site_id=a_site)

        assert result["total_conversations"] == 2
        assert result["total_messages"] == 3

    async def test_get_trigger_analytics(self, pg_db, a_site):
        """Test trigger analytics aggregation and derived rates."""
        trigger = await pg_db.save_trigger(a_site, {"name": "Welcome"})
        for event in ("shown", "shown", "clicked", "converted"):
            await pg_db.log_trigger_event(a_site, trigger["id"], "sess_1", event)

        analytics = await pg_db.get_trigger_analytics(a_site, period_days=7)

        assert len(analytics) == 1
        assert analytics[0]["trigger_name"] == "Welcome"
        assert analytics[0]["shown_count"] == 2
        assert analytics[0]["clicked_count"] == 1
        assert analytics[0]["click_rate"] == 50.0
        assert analytics[0]["conversion_rate"] == 50.0

    async def test_conversation_counts_by_site(self, pg_db, a_site):
        """Conversation and message totals group by site."""
        await pg_db.save_message("s1", "user", "a", site_id=a_site)
        await pg_db.save_message("s1", "assistant", "b", site_id=a_site)

        rows = await pg_db.get_conversation_counts_by_site()

        assert rows[0]["site_id"] == a_site
        assert rows[0]["conversation_count"] == 1
        assert rows[0]["message_count"] == 2

    async def test_count_helpers(self, pg_db, a_site):
        """Counting helpers used by the analytics routes."""
        await pg_db.save_message("s1", "user", "a", site_id=a_site)
        await pg_db.create_handoff_session(session_id="s1", site_id=a_site)

        assert await pg_db.count_conversations(a_site) == 1
        assert await pg_db.count_handoffs(site_id=a_site) == 1
        assert await pg_db.count_handoffs(site_id=a_site, status="resolved") == 0
        assert await pg_db.count_sites() >= 1


class TestPlatformSettings:
    """Tests for platform white-label settings."""

    async def test_get_platform_whitelabel(self, pg_db):
        """Unset white-label config returns None."""
        assert await pg_db.get_platform_whitelabel() is None

    async def test_update_platform_whitelabel(self, pg_db):
        """Test updating white-label configuration."""
        result = await pg_db.update_platform_whitelabel({
            "brand_name": "Custom Brand", "logo_url": "https://example.com/logo.png",
        })

        assert result["brand_name"] == "Custom Brand"
        assert "type" not in result

    async def test_update_platform_whitelabel_merges(self, pg_db):
        """Partial updates preserve previously stored keys."""
        await pg_db.update_platform_whitelabel({"brand_name": "A", "logo_url": "/l.png"})

        result = await pg_db.update_platform_whitelabel({"brand_name": "B"})

        assert result["brand_name"] == "B"
        assert result["logo_url"] == "/l.png"


class TestLongTermMemory:
    """Tests for long-term memory operations."""

    async def test_save_user_memory(self, pg_db):
        """Test saving a memory key."""
        await pg_db.save_user_memory("user_1", "preference", "dark_mode")

        assert await pg_db.get_user_memory("user_1") == {"preference": "dark_mode"}

    async def test_save_user_memory_merges_keys(self, pg_db):
        """Saving a second key does not clobber the first."""
        await pg_db.save_user_memory("user_1", "name", "Bob")
        await pg_db.save_user_memory("user_1", "plan", {"tier": "pro"})

        assert await pg_db.get_user_memory("user_1") == {"name": "Bob", "plan": {"tier": "pro"}}

    async def test_get_user_memory(self, pg_db):
        """Test retrieving user memory."""
        await pg_db.save_user_memory("user_1", "language", "en")

        assert (await pg_db.get_user_memory("user_1"))["language"] == "en"

    async def test_get_user_memory_empty(self, pg_db):
        """Test retrieving memory for an unknown user."""
        assert await pg_db.get_user_memory("unknown") == {}

    async def test_clear_user_memory(self, pg_db):
        """Test clearing user memory."""
        await pg_db.save_user_memory("user_1", "k", "v")

        assert await pg_db.clear_user_memory("user_1") is True
        assert await pg_db.get_user_memory("user_1") == {}


class TestBusinessHours:
    """Tests for business hours configuration."""

    async def test_get_site_handoff_config(self, pg_db, a_site):
        """Test retrieving a stored handoff config."""
        config = {"enabled": True, "confidence_threshold": 0.5}
        await pg_db.update_site_handoff_config(a_site, config)

        result = await pg_db.get_site_handoff_config(a_site)

        assert result["confidence_threshold"] == 0.5

    async def test_get_site_handoff_config_default(self, pg_db, a_site):
        """A site with no stored config gets the documented defaults."""
        result = await pg_db.get_site_handoff_config(a_site)

        assert result["enabled"] is True
        assert result["business_hours"]["enabled"] is False
        assert "auto_suggest_phrases" in result

    async def test_get_site_handoff_config_missing_site(self, pg_db):
        """An unknown site returns None rather than defaults."""
        assert await pg_db.get_site_handoff_config("nonexistent") is None

    async def test_update_site_handoff_config(self, pg_db, a_site):
        """Test updating the handoff configuration."""
        assert await pg_db.update_site_handoff_config(a_site, {"enabled": False}) is True
        assert (await pg_db.get_site_handoff_config(a_site))["enabled"] is False

    async def test_check_business_hours_disabled(self, pg_db, a_site):
        """With business hours disabled the site is always available."""
        result = await pg_db.check_business_hours(a_site)

        assert result["available"] is True

    async def test_check_business_hours_closed_all_week(self, pg_db, a_site):
        """With every day disabled the site reports offline."""
        await pg_db.update_site_handoff_config(a_site, {
            "enabled": True,
            "business_hours": {
                "enabled": True, "timezone": "UTC",
                "schedule": {d: {"enabled": False, "start": "09:00", "end": "17:00"}
                             for d in ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]},
                "offline_message": "We are closed",
            },
        })

        result = await pg_db.check_business_hours(a_site)

        assert result["available"] is False
        assert result["offline_message"] == "We are closed"


class TestLeadOperations:
    """Tests for lead capture operations."""

    async def test_save_and_get_lead(self, pg_db, a_site):
        """Leads round trip with their metadata."""
        lead = await pg_db.save_lead({
            "site_id": a_site, "session_id": "s1",
            "email": "lead@example.com", "name": "Lee", "metadata": {"utm": "ads"},
        })

        fetched = await pg_db.get_lead_by_id(lead["lead_id"])
        assert fetched["email"] == "lead@example.com"
        assert fetched["metadata"]["utm"] == "ads"
        assert fetched["id"] == lead["lead_id"]

    async def test_get_leads_paginated_and_search(self, pg_db, a_site):
        """Listing supports search across email and name."""
        await pg_db.save_lead({"site_id": a_site, "email": "alice@example.com", "name": "Alice"})
        await pg_db.save_lead({"site_id": a_site, "email": "bob@example.com", "name": "Bob"})

        _, total = await pg_db.get_leads(a_site)
        _, found = await pg_db.get_leads(a_site, search="alice")
        _, missing = await pg_db.get_leads(a_site, search="nobody")

        assert (total, found, missing) == (2, 1, 0)

    async def test_get_lead_by_session(self, pg_db, a_site):
        """Leads can be looked up by chat session."""
        await pg_db.save_lead({"site_id": a_site, "session_id": "s1", "email": "x@example.com"})

        assert (await pg_db.get_lead_by_session(a_site, "s1"))["email"] == "x@example.com"

    async def test_delete_lead(self, pg_db, a_site):
        """Test deleting a lead."""
        lead = await pg_db.save_lead({"site_id": a_site, "email": "x@example.com"})

        assert await pg_db.delete_lead(lead["lead_id"]) is True
        assert await pg_db.get_leads_count(a_site) == 0


class TestQAOperations:
    """Tests for Q&A training pairs."""

    async def test_create_and_get_qa_pair(self, pg_db, a_site):
        """Q&A pairs round trip, including ad-hoc fields."""
        qa = await pg_db.create_qa_pair({
            "site_id": a_site, "question": "Refund policy?",
            "answer": "30 days.", "tags": ["billing"],
        })

        fetched = await pg_db.get_qa_pair(qa["id"])
        assert fetched["answer"] == "30 days."
        assert fetched["tags"] == ["billing"]
        assert fetched["enabled"] is True
        assert "_id" not in fetched

    async def test_get_qa_pairs_search_and_filter(self, pg_db, a_site):
        """Listing supports search and an enabled-only filter."""
        qa = await pg_db.create_qa_pair({
            "site_id": a_site, "question": "Refund policy?", "answer": "30 days.",
        })
        await pg_db.create_qa_pair({
            "site_id": a_site, "question": "Shipping?", "answer": "2 days.",
        })
        await pg_db.update_qa_pair(qa["id"], {"enabled": False})

        _, total = await pg_db.get_qa_pairs(a_site)
        _, searched = await pg_db.get_qa_pairs(a_site, search="refund")
        _, enabled = await pg_db.get_qa_pairs(a_site, enabled_only=True)

        assert (total, searched, enabled) == (2, 1, 1)

    async def test_get_qa_for_rag_excludes_disabled(self, pg_db, a_site):
        """Disabled pairs are not offered to retrieval."""
        qa = await pg_db.create_qa_pair({"site_id": a_site, "question": "Q", "answer": "A"})

        assert len(await pg_db.get_qa_for_rag(a_site)) == 1
        await pg_db.update_qa_pair(qa["id"], {"enabled": False})
        assert await pg_db.get_qa_for_rag(a_site) == []

    async def test_qa_stats(self, pg_db, a_site):
        """Stats report totals and the most-used pairs."""
        qa = await pg_db.create_qa_pair({"site_id": a_site, "question": "Q", "answer": "A"})
        await pg_db.increment_qa_use_count(qa["id"])

        stats = await pg_db.get_qa_stats(a_site)

        assert stats["total_pairs"] == 1
        assert stats["enabled_pairs"] == 1
        assert stats["total_uses"] == 1
        assert stats["most_used"][0]["id"] == qa["id"]

    async def test_delete_qa_pair(self, pg_db, a_site):
        """Test deleting a Q&A pair."""
        qa = await pg_db.create_qa_pair({"site_id": a_site, "question": "Q", "answer": "A"})

        assert await pg_db.delete_qa_pair(qa["id"]) is True
        assert await pg_db.get_qa_pair(qa["id"]) is None


class TestDocumentOperations:
    """Tests for uploaded document metadata."""

    async def test_document_lifecycle(self, pg_db, a_site):
        """Documents can be created, updated, listed and deleted."""
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        await pg_db.create_document({
            "doc_id": doc_id, "site_id": a_site, "filename": "a.pdf",
            "file_type": "pdf", "word_count": 10, "status": "processing",
            "uploaded_at": datetime.utcnow(),
        })

        assert (await pg_db.get_document(doc_id, site_id=a_site))["filename"] == "a.pdf"
        assert await pg_db.update_document(doc_id, {"status": "indexed"}) is True
        assert (await pg_db.get_document(doc_id))["status"] == "indexed"
        assert await pg_db.count_documents(a_site) == 1
        assert await pg_db.delete_document(doc_id) is True
        assert await pg_db.get_documents(a_site) == []

    async def test_save_document_via_provider(self, pg_provider, a_site):
        """The provider interface generates an id when one is not supplied."""
        doc_id = await pg_provider.save_document({"site_id": a_site, "filename": "b.pdf"})

        assert doc_id
        assert (await pg_provider.get_document(doc_id))["filename"] == "b.pdf"


class TestScheduleOperations:
    """Tests for scheduled crawling configuration."""

    async def test_default_schedule(self, pg_db, a_site):
        """A site with no schedule returns the documented defaults."""
        schedule = await pg_db.get_crawl_schedule(a_site)

        assert schedule["enabled"] is False
        assert schedule["frequency"] == "weekly"

    async def test_update_and_list_schedules(self, pg_db, a_site):
        """Enabled schedules are discoverable by the scheduler."""
        assert await pg_db.update_crawl_schedule(a_site, {
            "enabled": True, "frequency": "daily",
        }) is True

        sites = await pg_db.get_sites_with_schedules()

        assert len(sites) == 1
        assert sites[0]["site_id"] == a_site

    async def test_crawl_history_and_running_job(self, pg_db, a_site):
        """History reports duration; a running job is discoverable."""
        job_id = await pg_db.create_scheduled_crawl_job(a_site, "https://example.com", "scheduled")

        assert (await pg_db.get_running_crawl_job(a_site))["_id"] == job_id

        await pg_db.mark_crawl_job_completed(job_id)
        history = await pg_db.get_crawl_history(a_site)

        assert history[0]["trigger"] == "scheduled"
        assert history[0]["duration_seconds"] is not None


class TestMaintenanceOperations:
    """Tests for the destructive admin helper."""

    async def test_clear_all_data(self, pg_db, a_site):
        """Conversations and pages are wiped; sites are preserved."""
        await pg_db.save_message("s1", "user", "hi", site_id=a_site)
        await pg_db.save_page("https://example.com/p", "P", "c")

        await pg_db.clear_all_data()

        assert await pg_db.count_conversations() == 0
        assert await pg_db.get_page_count() == 0
        assert await pg_db.get_site(a_site) is not None

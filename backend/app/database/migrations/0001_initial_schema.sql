-- =============================================================================
-- elo :: 0001_initial_schema
-- Initial PostgreSQL schema (replaces the previous MongoDB collections).
--
-- Design notes
-- ------------
-- * Every former MongoDB collection becomes a table with first-class columns
--   for the fields that are filtered/sorted on, plus a JSONB `data` column that
--   preserves MongoDB's "any extra field you set comes back on read" behaviour.
--   Application code that writes ad-hoc keys therefore keeps working unchanged.
-- * Embedded arrays that the application always reads/writes as a whole
--   (conversation `messages`, `notes`, site `triggers`, handoff `messages`,
--   `ai_conversation`) stay as JSONB. They are ordered, index-addressable
--   collections that the app mutates positionally, so JSONB preserves the exact
--   existing semantics with no behavioural drift.
-- * Timestamps are TIMESTAMP (without time zone) holding UTC, matching the
--   application's existing `datetime.utcnow()` convention so all datetime
--   arithmetic in the routes continues to work on naive UTC values.
-- * No cross-table FOREIGN KEYs are declared. MongoDB never enforced them and
--   several flows (chat widget, trigger events) legitimately write rows whose
--   site_id has not been created yet. Adding constraints here would change
--   runtime behaviour, so integrity stays at the application layer as before.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================================
-- conversations
-- =============================================================================
CREATE TABLE IF NOT EXISTS conversations (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          TEXT NOT NULL UNIQUE,
    site_id             TEXT,
    status              TEXT      DEFAULT 'open',
    priority            TEXT      DEFAULT 'medium',
    tags                JSONB     NOT NULL DEFAULT '[]'::jsonb,
    unread              BOOLEAN   DEFAULT TRUE,
    visitor_name        TEXT,
    visitor_email       TEXT,
    satisfaction_rating INTEGER,
    messages            JSONB     NOT NULL DEFAULT '[]'::jsonb,
    notes               JSONB     NOT NULL DEFAULT '[]'::jsonb,
    created_at          TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    updated_at          TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    first_response_at   TIMESTAMP,
    resolved_at         TIMESTAMP,
    -- Denormalised concatenation of message contents, maintained by the app on
    -- every write. Backs full-text search (formerly a MongoDB $text index).
    search_text         TEXT      NOT NULL DEFAULT '',
    search_vector       tsvector GENERATED ALWAYS AS (to_tsvector('english', search_text)) STORED,
    data                JSONB     NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_conversations_site_id    ON conversations (site_id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations (updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_status     ON conversations (status);
CREATE INDEX IF NOT EXISTS idx_conversations_priority   ON conversations (priority);
CREATE INDEX IF NOT EXISTS idx_conversations_site_upd   ON conversations (site_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_search     ON conversations USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS idx_conversations_tags       ON conversations USING GIN (tags jsonb_path_ops);

-- =============================================================================
-- pages
-- =============================================================================
CREATE TABLE IF NOT EXISTS pages (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url          TEXT NOT NULL UNIQUE,
    title        TEXT,
    content      TEXT,
    chunk_count  INTEGER   NOT NULL DEFAULT 0,
    metadata     JSONB     NOT NULL DEFAULT '{}'::jsonb,
    status       TEXT,
    last_crawled TIMESTAMP,
    created_at   TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    data         JSONB     NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_pages_status       ON pages (status);
CREATE INDEX IF NOT EXISTS idx_pages_last_crawled ON pages (last_crawled DESC);

-- =============================================================================
-- crawl_jobs
-- =============================================================================
CREATE TABLE IF NOT EXISTS crawl_jobs (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    site_id       TEXT,
    target_url    TEXT,
    status        TEXT      NOT NULL DEFAULT 'running',
    pages_crawled INTEGER   NOT NULL DEFAULT 0,
    pages_indexed INTEGER   NOT NULL DEFAULT 0,
    errors        JSONB     NOT NULL DEFAULT '[]'::jsonb,
    trigger       TEXT,
    created_at    TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    updated_at    TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    completed_at  TIMESTAMP,
    data          JSONB     NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_crawl_jobs_created_at ON crawl_jobs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_crawl_jobs_site_id    ON crawl_jobs (site_id);
CREATE INDEX IF NOT EXISTS idx_crawl_jobs_target_url ON crawl_jobs (target_url);
CREATE INDEX IF NOT EXISTS idx_crawl_jobs_status     ON crawl_jobs (status);

-- =============================================================================
-- long_term_memory
-- =============================================================================
CREATE TABLE IF NOT EXISTS long_term_memory (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    TEXT NOT NULL UNIQUE,
    memory     JSONB     NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    updated_at TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc')
);

-- =============================================================================
-- users
--
-- `id` is surfaced to the application as `_id` (previously the MongoDB
-- ObjectId). `user_id` is populated with the same value so that the two
-- identifiers the codebase uses interchangeably -- str(user["_id"]) for site
-- ownership / JWT subject, and user_id for lookups -- always agree.
-- =============================================================================
CREATE TABLE IF NOT EXISTS users (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id              TEXT NOT NULL UNIQUE,
    email                TEXT NOT NULL UNIQUE,
    name                 TEXT,
    password_hash        TEXT,
    role                 TEXT,
    owner_id             TEXT,
    assigned_site_ids    JSONB     NOT NULL DEFAULT '[]'::jsonb,
    must_change_password BOOLEAN,
    is_active            BOOLEAN,
    created_at           TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    updated_at           TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    data                 JSONB     NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_users_owner_id   ON users (owner_id);
CREATE INDEX IF NOT EXISTS idx_users_role_owner ON users (role, owner_id);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users (created_at DESC);

-- =============================================================================
-- sites
-- =============================================================================
CREATE TABLE IF NOT EXISTS sites (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    site_id           TEXT NOT NULL UNIQUE,
    user_id           TEXT,
    name              TEXT,
    url               TEXT,
    status            TEXT,
    has_documents     BOOLEAN,
    config            JSONB     NOT NULL DEFAULT '{}'::jsonb,
    triggers          JSONB     NOT NULL DEFAULT '[]'::jsonb,
    global_cooldown_ms INTEGER,
    handoff_config    JSONB,
    crawl_schedule    JSONB,
    created_at        TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    updated_at        TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    data              JSONB     NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_sites_status     ON sites (status);
CREATE INDEX IF NOT EXISTS idx_sites_user_id    ON sites (user_id);
CREATE INDEX IF NOT EXISTS idx_sites_created_at ON sites (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sites_url        ON sites (url);
CREATE INDEX IF NOT EXISTS idx_sites_schedule_enabled
    ON sites (((crawl_schedule -> 'enabled')::boolean))
    WHERE crawl_schedule IS NOT NULL;

-- =============================================================================
-- trigger_events
-- =============================================================================
CREATE TABLE IF NOT EXISTS trigger_events (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    site_id    TEXT,
    trigger_id TEXT,
    session_id TEXT,
    event_type TEXT,
    timestamp  TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    metadata   JSONB     NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_trigger_events_site_id    ON trigger_events (site_id);
CREATE INDEX IF NOT EXISTS idx_trigger_events_trigger_id ON trigger_events (trigger_id);
CREATE INDEX IF NOT EXISTS idx_trigger_events_timestamp  ON trigger_events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trigger_events_site_ts    ON trigger_events (site_id, timestamp DESC);

-- =============================================================================
-- handoff_sessions
-- =============================================================================
CREATE TABLE IF NOT EXISTS handoff_sessions (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    handoff_id             TEXT NOT NULL UNIQUE,
    session_id             TEXT,
    site_id                TEXT,
    status                 TEXT      NOT NULL DEFAULT 'pending',
    visitor_email          TEXT,
    visitor_name           TEXT,
    reason                 TEXT,
    ai_summary             TEXT,
    ai_conversation        JSONB     NOT NULL DEFAULT '[]'::jsonb,
    messages               JSONB     NOT NULL DEFAULT '[]'::jsonb,
    assigned_agent_id      TEXT,
    assigned_agent_name    TEXT,
    visitor_queue_signals  INTEGER   NOT NULL DEFAULT 0,
    created_at             TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    updated_at             TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    resolved_at            TIMESTAMP,
    data                   JSONB     NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_handoff_session_id  ON handoff_sessions (session_id);
CREATE INDEX IF NOT EXISTS idx_handoff_site_id     ON handoff_sessions (site_id);
CREATE INDEX IF NOT EXISTS idx_handoff_status      ON handoff_sessions (status);
CREATE INDEX IF NOT EXISTS idx_handoff_site_status ON handoff_sessions (site_id, status, created_at);
CREATE INDEX IF NOT EXISTS idx_handoff_agent       ON handoff_sessions (assigned_agent_id);

-- =============================================================================
-- qa_pairs
-- =============================================================================
CREATE TABLE IF NOT EXISTS qa_pairs (
    id         TEXT PRIMARY KEY,
    site_id    TEXT,
    question   TEXT,
    answer     TEXT,
    enabled    BOOLEAN   NOT NULL DEFAULT TRUE,
    use_count  INTEGER   NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    updated_at TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    data       JSONB     NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_qa_pairs_site_id      ON qa_pairs (site_id);
CREATE INDEX IF NOT EXISTS idx_qa_pairs_site_enabled ON qa_pairs (site_id, enabled);
CREATE INDEX IF NOT EXISTS idx_qa_pairs_created_at   ON qa_pairs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_qa_pairs_search
    ON qa_pairs USING GIN (to_tsvector('english', coalesce(question, '') || ' ' || coalesce(answer, '')));

-- =============================================================================
-- leads
-- =============================================================================
CREATE TABLE IF NOT EXISTS leads (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_id     TEXT NOT NULL UNIQUE,
    site_id     TEXT,
    session_id  TEXT,
    email       TEXT,
    name        TEXT,
    source      TEXT,
    captured_at TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    metadata    JSONB     NOT NULL DEFAULT '{}'::jsonb,
    data        JSONB     NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_leads_site_id     ON leads (site_id);
CREATE INDEX IF NOT EXISTS idx_leads_session_id  ON leads (session_id);
CREATE INDEX IF NOT EXISTS idx_leads_email       ON leads (email);
CREATE INDEX IF NOT EXISTS idx_leads_site_captur ON leads (site_id, captured_at DESC);

-- =============================================================================
-- documents
-- =============================================================================
CREATE TABLE IF NOT EXISTS documents (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id         TEXT NOT NULL UNIQUE,
    site_id        TEXT,
    filename       TEXT,
    file_type      TEXT,
    word_count     INTEGER   NOT NULL DEFAULT 0,
    char_count     INTEGER   NOT NULL DEFAULT 0,
    chunks_created INTEGER   NOT NULL DEFAULT 0,
    metadata       JSONB     NOT NULL DEFAULT '{}'::jsonb,
    status         TEXT,
    error          TEXT,
    uploaded_by    TEXT,
    uploaded_at    TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    indexed_at     TIMESTAMP,
    data           JSONB     NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_documents_site_id     ON documents (site_id);
CREATE INDEX IF NOT EXISTS idx_documents_uploaded_at ON documents (uploaded_at DESC);

-- =============================================================================
-- platform_settings
-- =============================================================================
CREATE TABLE IF NOT EXISTS platform_settings (
    type       TEXT PRIMARY KEY,
    config     JSONB     NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc')
);

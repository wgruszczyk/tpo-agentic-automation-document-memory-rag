CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS system_state (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    source_path TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source_modified_at TIMESTAMPTZ NOT NULL,
    effective_at TIMESTAMPTZ NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    indexed_profile_hash TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_active_idx ON documents (is_active);
CREATE INDEX IF NOT EXISTS documents_effective_at_idx ON documents (effective_at DESC);
CREATE INDEX IF NOT EXISTS documents_content_hash_idx ON documents (content_hash);
CREATE INDEX IF NOT EXISTS documents_metadata_idx ON documents USING gin (metadata);
CREATE INDEX IF NOT EXISTS documents_title_trgm_idx ON documents USING gin (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS documents_source_path_trgm_idx ON documents USING gin (source_path gin_trgm_ops);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    approx_tokens INTEGER NOT NULL,
    embedding vector NOT NULL,
    embedding_profile_hash TEXT NOT NULL,
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id, chunk_index);
CREATE INDEX IF NOT EXISTS chunks_profile_idx ON chunks (embedding_profile_hash);
CREATE INDEX IF NOT EXISTS chunks_search_idx ON chunks USING gin (search_vector);
CREATE INDEX IF NOT EXISTS chunks_content_trgm_idx ON chunks USING gin (content gin_trgm_ops);

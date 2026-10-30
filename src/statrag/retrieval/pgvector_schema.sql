-- Schéma PostgreSQL + pgvector sécurisé par Row-Level Security (RLS)
-- Garantit l'isolation stricte des corpus documentaires directement au niveau du moteur SQL.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    chunk_id      UUID PRIMARY KEY,
    tenant_id     UUID NOT NULL,
    document_ref  TEXT NOT NULL,
    chunk_index   INTEGER NOT NULL CHECK (chunk_index >= 0),
    content       TEXT NOT NULL CHECK (char_length(content) > 0),
    embedding     vector(384) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (tenant_id, document_ref, chunk_index)
);

-- Index HNSW sur distance cosinus pour maintenir une latence sous-milliseconde à grande échelle
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding_hnsw
    ON knowledge_chunks
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_tenant
    ON knowledge_chunks (tenant_id);

-- Verrouillage Row-Level Security obligatoire même pour le propriétaire applicatif
ALTER TABLE knowledge_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_chunks FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation_policy ON knowledge_chunks;
CREATE POLICY tenant_isolation_policy ON knowledge_chunks
    USING (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::UUID)
    WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::UUID);

-- Migration: Add vector embeddings to component_patterns for semantic search
-- Purpose: Enable semantic similarity search (pgvector)
-- Requires: pgvector extension installed
-- Created: 2026-01-23

BEGIN;

-- 1. Enable pgvector extension (skip if already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Add embedding column to component_patterns
-- Assuming Llama 3.2 embeddings (768 dimensions)
-- Adjust dimension if using different model
ALTER TABLE component_patterns
ADD COLUMN IF NOT EXISTS embedding vector(768);

-- 3. Create index for fast similarity search
-- Using ivfflat (Inverted File with Flat compression)
-- lists parameter: sqrt(row_count) is a good starting point
-- Adjust after data population based on table size

-- Note: Index creation might take time if table has many rows
CREATE INDEX IF NOT EXISTS idx_component_patterns_embedding
ON component_patterns
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- For smaller datasets, you can also use:
-- CREATE INDEX idx_component_patterns_embedding
-- ON component_patterns
-- USING hnsw (embedding vector_cosine_ops);  -- Hierarchical Navigable Small World

-- 4. Create function for semantic search
CREATE OR REPLACE FUNCTION search_similar_components(
    p_query_embedding vector(768),
    p_similarity_threshold FLOAT DEFAULT 0.80,
    p_category TEXT DEFAULT NULL,
    p_department TEXT DEFAULT NULL,
    p_limit INT DEFAULT 10
)
RETURNS TABLE (
    id INT,
    name TEXT,
    category TEXT,
    avg_hours_3d_layout FLOAT,
    avg_hours_3d_detail FLOAT,
    avg_hours_2d FLOAT,
    confidence FLOAT,
    occurrence_count INT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        cp.id,
        cp.name,
        cp.category,
        cp.avg_hours_3d_layout,
        cp.avg_hours_3d_detail,
        cp.avg_hours_2d,
        cp.confidence,
        cp.occurrence_count,
        1 - (cp.embedding <=> p_query_embedding) AS similarity
    FROM component_patterns cp
    WHERE cp.embedding IS NOT NULL
      AND 1 - (cp.embedding <=> p_query_embedding) > p_similarity_threshold
      AND (p_category IS NULL OR cp.category = p_category)
      AND (p_department IS NULL OR cp.department_code = p_department)
    ORDER BY cp.embedding <=> p_query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- 5. Create function to get weighted average from similar components
CREATE OR REPLACE FUNCTION get_weighted_estimate_from_similar(
    p_query_embedding vector(768),
    p_similarity_threshold FLOAT DEFAULT 0.85,
    p_category TEXT DEFAULT NULL,
    p_department TEXT DEFAULT NULL,
    p_top_k INT DEFAULT 3
)
RETURNS TABLE (
    avg_hours_3d_layout FLOAT,
    avg_hours_3d_detail FLOAT,
    avg_hours_2d FLOAT,
    confidence FLOAT,
    matched_components TEXT,
    avg_similarity FLOAT
) AS $$
DECLARE
    total_weight FLOAT;
BEGIN
    -- Get weighted averages from top-k similar components
    RETURN QUERY
    WITH similar_comps AS (
        SELECT * FROM search_similar_components(
            p_query_embedding,
            p_similarity_threshold,
            p_category,
            p_department,
            p_top_k
        )
    ),
    weighted_calcs AS (
        SELECT
            SUM(sc.avg_hours_3d_layout * sc.similarity) AS weighted_layout,
            SUM(sc.avg_hours_3d_detail * sc.similarity) AS weighted_detail,
            SUM(sc.avg_hours_2d * sc.similarity) AS weighted_2d,
            SUM(sc.similarity) AS total_sim,
            AVG(sc.similarity) AS avg_sim,
            STRING_AGG(sc.name || ' (' || ROUND(sc.similarity::numeric, 2) || ')', ', ') AS matched
        FROM similar_comps sc
    )
    SELECT
        wc.weighted_layout / NULLIF(wc.total_sim, 0) AS avg_hours_3d_layout,
        wc.weighted_detail / NULLIF(wc.total_sim, 0) AS avg_hours_3d_detail,
        wc.weighted_2d / NULLIF(wc.total_sim, 0) AS avg_hours_2d,
        LEAST(wc.avg_sim, 0.95) AS confidence,  -- Cap at 0.95
        wc.matched AS matched_components,
        wc.avg_sim AS avg_similarity
    FROM weighted_calcs wc
    WHERE wc.total_sim > 0;
END;
$$ LANGUAGE plpgsql;

-- 6. Add metadata columns for tracking embedding generation
ALTER TABLE component_patterns
ADD COLUMN IF NOT EXISTS embedding_model TEXT,
ADD COLUMN IF NOT EXISTS embedding_generated_at TIMESTAMP;

-- 7. Create view for components missing embeddings
CREATE OR REPLACE VIEW components_needing_embeddings AS
SELECT
    id,
    name,
    category,
    department_code,
    occurrence_count
FROM component_patterns
WHERE embedding IS NULL
ORDER BY occurrence_count DESC;

COMMIT;

-- Usage examples (after running batch embedding generation):
--
-- Example 1: Direct search
-- SELECT * FROM search_similar_components(
--     '[0.1, 0.2, ...]'::vector(768),  -- query embedding
--     0.85,  -- 85% similarity threshold
--     'Mechanical',  -- category filter
--     '131',  -- department filter
--     10  -- limit
-- );
--
-- Example 2: Get weighted estimate
-- SELECT * FROM get_weighted_estimate_from_similar(
--     '[0.1, 0.2, ...]'::vector(768),
--     0.85,
--     'Mechanical',
--     '131',
--     3  -- top 3 matches
-- );
--
-- Example 3: Check how many components need embeddings
-- SELECT COUNT(*) FROM components_needing_embeddings;

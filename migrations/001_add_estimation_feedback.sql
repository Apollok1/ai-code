-- Migration: Add estimation_feedback table for continuous learning
-- Purpose: Collect actual hours vs estimated hours to improve model accuracy
-- Created: 2026-01-23

BEGIN;

-- 1. Create estimation_feedback table
CREATE TABLE IF NOT EXISTS estimation_feedback (
    id SERIAL PRIMARY KEY,

    -- Project reference
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,

    -- Component info
    component_name TEXT NOT NULL,
    component_category TEXT,
    department_code TEXT,

    -- AI Estimation (what the model predicted)
    estimated_hours_3d_layout FLOAT NOT NULL,
    estimated_hours_3d_detail FLOAT NOT NULL,
    estimated_hours_2d FLOAT NOT NULL,
    estimated_total_hours FLOAT GENERATED ALWAYS AS (
        estimated_hours_3d_layout + estimated_hours_3d_detail + estimated_hours_2d
    ) STORED,
    estimated_confidence FLOAT CHECK (estimated_confidence >= 0 AND estimated_confidence <= 1),

    -- Actual values (filled in during/after project execution)
    actual_hours_3d_layout FLOAT,
    actual_hours_3d_detail FLOAT,
    actual_hours_2d FLOAT,
    actual_total_hours FLOAT GENERATED ALWAYS AS (
        COALESCE(actual_hours_3d_layout, 0) +
        COALESCE(actual_hours_3d_detail, 0) +
        COALESCE(actual_hours_2d, 0)
    ) STORED,

    -- Error metrics (computed)
    error_percentage FLOAT GENERATED ALWAYS AS (
        CASE
            WHEN actual_total_hours > 0 THEN
                100.0 * ABS(estimated_total_hours - actual_total_hours) / actual_total_hours
            ELSE NULL
        END
    ) STORED,

    mae FLOAT GENERATED ALWAYS AS (
        CASE
            WHEN actual_hours_3d_layout IS NOT NULL THEN (
                ABS(estimated_hours_3d_layout - COALESCE(actual_hours_3d_layout, 0)) +
                ABS(estimated_hours_3d_detail - COALESCE(actual_hours_3d_detail, 0)) +
                ABS(estimated_hours_2d - COALESCE(actual_hours_2d, 0))
            ) / 3.0
            ELSE NULL
        END
    ) STORED,

    -- Model metadata
    model_used TEXT,
    complexity_level TEXT,
    complexity_multiplier FLOAT,
    pattern_matched BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    actual_hours_updated_at TIMESTAMP,

    -- Notes
    notes TEXT
);

-- 2. Create indexes for performance
CREATE INDEX idx_feedback_project ON estimation_feedback(project_id);
CREATE INDEX idx_feedback_component ON estimation_feedback(component_name);
CREATE INDEX idx_feedback_department ON estimation_feedback(department_code);
CREATE INDEX idx_feedback_model ON estimation_feedback(model_used);
CREATE INDEX idx_feedback_created ON estimation_feedback(created_at DESC);

-- Index for finding high-quality examples (for few-shot learning)
CREATE INDEX idx_feedback_high_quality ON estimation_feedback(error_percentage)
WHERE actual_hours_3d_layout IS NOT NULL AND error_percentage < 15.0;

-- 3. Create view for analytics
CREATE OR REPLACE VIEW estimation_accuracy_summary AS
SELECT
    model_used,
    complexity_level,
    COUNT(*) AS total_estimates,
    COUNT(actual_hours_3d_layout) AS completed_estimates,
    AVG(error_percentage) AS avg_error_pct,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY error_percentage) AS median_error_pct,
    AVG(mae) AS avg_mae,
    AVG(CASE WHEN error_percentage < 10 THEN 1.0 ELSE 0.0 END) AS accuracy_at_10_pct,
    AVG(CASE WHEN error_percentage < 20 THEN 1.0 ELSE 0.0 END) AS accuracy_at_20_pct,
    AVG(estimated_confidence) AS avg_estimated_confidence,
    MIN(created_at) AS first_estimate,
    MAX(created_at) AS last_estimate
FROM estimation_feedback
WHERE actual_hours_3d_layout IS NOT NULL
GROUP BY model_used, complexity_level;

-- 4. Create function to get best examples for few-shot learning
CREATE OR REPLACE FUNCTION get_best_estimation_examples(
    p_department TEXT DEFAULT NULL,
    p_limit INT DEFAULT 5,
    p_min_accuracy FLOAT DEFAULT 0.9  -- 90% accuracy = <10% error
)
RETURNS TABLE (
    component_name TEXT,
    category TEXT,
    complexity TEXT,
    est_layout FLOAT,
    est_detail FLOAT,
    est_2d FLOAT,
    actual_layout FLOAT,
    actual_detail FLOAT,
    actual_2d FLOAT,
    accuracy FLOAT,
    reasoning TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        f.component_name,
        f.component_category,
        f.complexity_level,
        f.estimated_hours_3d_layout,
        f.estimated_hours_3d_detail,
        f.estimated_hours_2d,
        f.actual_hours_3d_layout,
        f.actual_hours_3d_detail,
        f.actual_hours_2d,
        100.0 - f.error_percentage AS accuracy,
        f.notes AS reasoning
    FROM estimation_feedback f
    WHERE f.actual_hours_3d_layout IS NOT NULL
      AND f.error_percentage IS NOT NULL
      AND f.error_percentage < (100.0 - p_min_accuracy * 100.0)
      AND (p_department IS NULL OR f.department_code = p_department)
    ORDER BY f.error_percentage ASC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- 5. Add trigger to update actual_hours_updated_at
CREATE OR REPLACE FUNCTION update_actual_hours_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.actual_hours_3d_layout IS DISTINCT FROM OLD.actual_hours_3d_layout
       OR NEW.actual_hours_3d_detail IS DISTINCT FROM OLD.actual_hours_3d_detail
       OR NEW.actual_hours_2d IS DISTINCT FROM OLD.actual_hours_2d
    THEN
        NEW.actual_hours_updated_at = NOW();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_actual_hours_timestamp
BEFORE UPDATE ON estimation_feedback
FOR EACH ROW
EXECUTE FUNCTION update_actual_hours_timestamp();

-- 6. Insert sample data (optional - for testing)
-- Uncomment to populate with test data
/*
INSERT INTO estimation_feedback (
    project_id, component_name, component_category, department_code,
    estimated_hours_3d_layout, estimated_hours_3d_detail, estimated_hours_2d,
    estimated_confidence, model_used, complexity_level
) VALUES
    (1, 'Main Housing', 'Enclosure', '131', 5.0, 15.0, 8.0, 0.75, 'llama3.2', 'Medium'),
    (1, 'Motor Mount', 'Support', '131', 3.0, 10.0, 5.0, 0.80, 'llama3.2', 'Low'),
    (2, 'Bearing Assembly', 'Mechanical', '132', 2.0, 8.0, 4.0, 0.85, 'gpt-4', 'Medium');
*/

COMMIT;

-- Verification queries
-- SELECT * FROM estimation_feedback LIMIT 5;
-- SELECT * FROM estimation_accuracy_summary;
-- SELECT * FROM get_best_estimation_examples('131', 5, 0.9);

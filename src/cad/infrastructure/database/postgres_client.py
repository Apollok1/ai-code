"""
CAD Estimator Pro - PostgreSQL Database Client

Implementation of DatabaseClient protocol using psycopg2.
"""
import logging
from contextlib import contextmanager
from typing import Any, Generator
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
import json

from ...domain.models import Project, ProjectVersion, ComponentPattern, Component, Estimate, DepartmentCode, EstimatePhases
from ...domain.models.config import DatabaseConfig
from ...domain.exceptions import DatabaseError, ConnectionError as CADConnectionError, QueryError, NotFoundError

logger = logging.getLogger(__name__)


class PostgresClient:
    """
    PostgreSQL database client implementation.

    Implements DatabaseClient protocol using psycopg2 with connection pooling.
    """

    def __init__(self, config: DatabaseConfig):
        """
        Initialize PostgreSQL client.

        Args:
            config: Database configuration

        Raises:
            ConnectionError: If connection pool creation fails
        """
        self.config = config
        try:
            self.pool = SimpleConnectionPool(
                config.pool_min_size,
                config.pool_max_size,
                host=config.host,
                port=config.port,
                database=config.database,
                user=config.user,
                password=config.password
            )
            logger.info(f"PostgreSQL connection pool created (min={config.pool_min_size}, max={config.pool_max_size})")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}", exc_info=True)
            raise CADConnectionError(f"Failed to connect to database: {e}", query=None)

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """
        Get database connection from pool (context manager).

        Yields:
            psycopg2 connection

        Raises:
            ConnectionError: If connection cannot be obtained
        """
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)
            if conn:
                conn.rollback()
            raise CADConnectionError(f"Database connection error: {e}", query=None)
        finally:
            if conn:
                self.pool.putconn(conn)

    def init_schema(self) -> bool:
        """
        Initialize database schema (tables, indexes, extensions).

        Returns:
            True if successful

        Raises:
            DatabaseError: If initialization fails
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Enable extensions
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

                    # Projects table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS projects (
                            id SERIAL PRIMARY KEY,
                            name VARCHAR(500) NOT NULL,
                            client VARCHAR(200),
                            department VARCHAR(3) NOT NULL CHECK (department IN ('131', '132', '133', '134', '135')),
                            description TEXT,
                            cad_system VARCHAR(50),
                            components JSONB,
                            estimated_hours NUMERIC(10,2),
                            estimated_hours_3d_layout NUMERIC(10,2),
                            estimated_hours_3d_detail NUMERIC(10,2),
                            estimated_hours_2d NUMERIC(10,2),
                            actual_hours NUMERIC(10,2),
                            accuracy NUMERIC(5,4),
                            ai_analysis TEXT,
                            description_embedding vector(768),
                            is_historical BOOLEAN DEFAULT FALSE,
                            created_at TIMESTAMP DEFAULT NOW(),
                            updated_at TIMESTAMP DEFAULT NOW()
                        )
                    """)

                    # Component patterns table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS component_patterns (
                            id SERIAL PRIMARY KEY,
                            name VARCHAR(500) NOT NULL,
                            pattern_key VARCHAR(500) NOT NULL,
                            department VARCHAR(3) NOT NULL CHECK (department IN ('131', '132', '133', '134', '135')),
                            avg_hours_3d_layout NUMERIC(10,2) DEFAULT 0,
                            avg_hours_3d_detail NUMERIC(10,2) DEFAULT 0,
                            avg_hours_2d NUMERIC(10,2) DEFAULT 0,
                            avg_hours_total NUMERIC(10,2) DEFAULT 0,
                            proportion_layout NUMERIC(5,4) DEFAULT 0,
                            proportion_detail NUMERIC(5,4) DEFAULT 0,
                            proportion_doc NUMERIC(5,4) DEFAULT 0,
                            occurrences INTEGER DEFAULT 0,
                            m2_layout NUMERIC(15,5) DEFAULT 0,
                            m2_detail NUMERIC(15,5) DEFAULT 0,
                            m2_doc NUMERIC(15,5) DEFAULT 0,
                            confidence NUMERIC(5,4) DEFAULT 0,
                            source VARCHAR(50) DEFAULT 'actual',
                            name_embedding vector(768),
                            last_updated TIMESTAMP DEFAULT NOW(),
                            last_actual_sample_at TIMESTAMP,
                            UNIQUE(pattern_key, department)
                        )
                    """)

                    # Project versions table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS project_versions (
                            id SERIAL PRIMARY KEY,
                            project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                            version VARCHAR(20) NOT NULL,
                            components JSONB,
                            estimated_hours NUMERIC(10,2),
                            estimated_hours_3d_layout NUMERIC(10,2),
                            estimated_hours_3d_detail NUMERIC(10,2),
                            estimated_hours_2d NUMERIC(10,2),
                            change_description TEXT,
                            changed_by VARCHAR(100) DEFAULT 'System',
                            is_approved BOOLEAN DEFAULT FALSE,
                            created_at TIMESTAMP DEFAULT NOW(),
                            UNIQUE(project_id, version)
                        )
                    """)

                    # Category baselines table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS category_baselines (
                            id SERIAL PRIMARY KEY,
                            department VARCHAR(3) NOT NULL CHECK (department IN ('131', '132', '133', '134', '135')),
                            category VARCHAR(100) NOT NULL,
                            mean_layout NUMERIC(10,2) DEFAULT 0,
                            mean_detail NUMERIC(10,2) DEFAULT 0,
                            mean_doc NUMERIC(10,2) DEFAULT 0,
                            m2_layout NUMERIC(15,5) DEFAULT 0,
                            m2_detail NUMERIC(15,5) DEFAULT 0,
                            m2_doc NUMERIC(15,5) DEFAULT 0,
                            occurrences INTEGER DEFAULT 0,
                            confidence NUMERIC(5,4) DEFAULT 0,
                            UNIQUE(department, category)
                        )
                    """)

                    # Component bundles table (parent→sub relations)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS component_bundles (
                            id SERIAL PRIMARY KEY,
                            department VARCHAR(3) NOT NULL CHECK (department IN ('131', '132', '133', '134', '135')),
                            parent_key VARCHAR(500) NOT NULL,
                            parent_name VARCHAR(500) NOT NULL,
                            sub_key VARCHAR(500) NOT NULL,
                            sub_name VARCHAR(500) NOT NULL,
                            occurrences INTEGER DEFAULT 0,
                            total_qty NUMERIC(10,2) DEFAULT 0,
                            confidence NUMERIC(5,4) DEFAULT 0,
                            UNIQUE(department, parent_key, sub_key)
                        )
                    """)

                    # Indexes
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_projects_department ON projects(department)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_projects_created_at ON projects(created_at DESC)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_patterns_department ON component_patterns(department)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_patterns_key ON component_patterns(pattern_key)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_versions_project ON project_versions(project_id)")

                    # HNSW indexes for vector search (pgvector)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_projects_embedding
                        ON projects USING hnsw (description_embedding vector_cosine_ops)
                        WHERE description_embedding IS NOT NULL
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_patterns_embedding
                        ON component_patterns USING hnsw (name_embedding vector_cosine_ops)
                        WHERE name_embedding IS NOT NULL
                    """)

                    # GIN indexes for text search
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_projects_description_gin
                        ON projects USING gin (to_tsvector('simple', coalesce(description, '')))
                    """)

                    conn.commit()
                    logger.info("✅ Database schema initialized successfully")
                    return True

        except Exception as e:
            logger.error(f"Schema initialization failed: {e}", exc_info=True)
            raise DatabaseError(f"Failed to initialize schema: {e}", query=None)

    # Project operations

    def get_project(self, project_id: int) -> Project | None:
        """Get project by ID."""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT id, name, client, department, description, cad_system,
                               components, estimated_hours, actual_hours,
                               created_at, updated_at, is_historical
                        FROM projects WHERE id = %s
                    """, (project_id,))
                    row = cur.fetchone()

                    if not row:
                        return None

                    return self._project_from_row(row)

        except Exception as e:
            logger.error(f"Failed to get project {project_id}: {e}", exc_info=True)
            raise QueryError(f"Failed to get project: {e}", query="SELECT FROM projects")

    def save_project(self, project: Project) -> int:
        """Save project (insert or update)."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Convert estimate to JSONB
                    components_json = json.dumps([c.to_dict() for c in project.estimate.components], ensure_ascii=False)
                    phases = project.estimate.phases

                    if project.id is None:
                        # INSERT
                        cur.execute("""
                            INSERT INTO projects (
                                name, client, department, description, cad_system,
                                components, estimated_hours,
                                estimated_hours_3d_layout, estimated_hours_3d_detail, estimated_hours_2d,
                                actual_hours, is_historical
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING id
                        """, (
                            project.name, project.client, project.department.value, project.description,
                            project.cad_system, components_json, project.estimated_hours,
                            phases.layout, phases.detail, phases.documentation,
                            project.actual_hours, project.is_historical
                        ))
                        project_id = cur.fetchone()[0]
                    else:
                        # UPDATE
                        cur.execute("""
                            UPDATE projects SET
                                name = %s, client = %s, department = %s, description = %s, cad_system = %s,
                                components = %s, estimated_hours = %s,
                                estimated_hours_3d_layout = %s, estimated_hours_3d_detail = %s, estimated_hours_2d = %s,
                                actual_hours = %s, is_historical = %s, updated_at = NOW()
                            WHERE id = %s
                        """, (
                            project.name, project.client, project.department.value, project.description,
                            project.cad_system, components_json, project.estimated_hours,
                            phases.layout, phases.detail, phases.documentation,
                            project.actual_hours, project.is_historical, project.id
                        ))
                        project_id = project.id

                    conn.commit()
                    logger.info(f"✅ Saved project ID {project_id}: {project.name}")
                    return project_id

        except Exception as e:
            logger.error(f"Failed to save project: {e}", exc_info=True)
            raise QueryError(f"Failed to save project: {e}", query="INSERT/UPDATE projects")

    def delete_project(self, project_id: int) -> bool:
        """Delete project by ID."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM projects WHERE id = %s", (project_id,))
                    deleted = cur.rowcount > 0
                    conn.commit()
                    if deleted:
                        logger.info(f"✅ Deleted project ID {project_id}")
                    return deleted

        except Exception as e:
            logger.error(f"Failed to delete project {project_id}: {e}", exc_info=True)
            raise QueryError(f"Failed to delete project: {e}", query="DELETE FROM projects")

    def search_projects(
        self,
        query: str | None = None,
        department: str | None = None,
        is_historical: bool | None = None,
        limit: int = 100
    ) -> list[Project]:
        """Search projects with filters."""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    sql = """
                        SELECT id, name, client, department, description, cad_system,
                               components, estimated_hours, actual_hours,
                               created_at, updated_at, is_historical
                        FROM projects WHERE 1=1
                    """
                    params = []

                    if query:
                        sql += " AND to_tsvector('simple', coalesce(name,'') || ' ' || coalesce(client,'') || ' ' || coalesce(description,'')) @@ websearch_to_tsquery('simple', %s)"
                        params.append(query)

                    if department:
                        sql += " AND department = %s"
                        params.append(department)

                    if is_historical is not None:
                        sql += " AND is_historical = %s"
                        params.append(is_historical)

                    sql += " ORDER BY created_at DESC LIMIT %s"
                    params.append(limit)

                    cur.execute(sql, params)
                    rows = cur.fetchall()

                    return [self._project_from_row(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to search projects: {e}", exc_info=True)
            raise QueryError(f"Failed to search projects: {e}", query="SELECT FROM projects")

    # Project versions

    def save_project_version(self, version: ProjectVersion) -> int:
        """Save project version."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    components_json = json.dumps([c.to_dict() for c in version.estimate.components], ensure_ascii=False)
                    phases = version.estimate.phases

                    cur.execute("""
                        INSERT INTO project_versions (
                            project_id, version, components,
                            estimated_hours, estimated_hours_3d_layout,
                            estimated_hours_3d_detail, estimated_hours_2d,
                            change_description, changed_by, is_approved
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (project_id, version) DO UPDATE SET
                            components = EXCLUDED.components,
                            estimated_hours = EXCLUDED.estimated_hours,
                            estimated_hours_3d_layout = EXCLUDED.estimated_hours_3d_layout,
                            estimated_hours_3d_detail = EXCLUDED.estimated_hours_3d_detail,
                            estimated_hours_2d = EXCLUDED.estimated_hours_2d,
                            change_description = EXCLUDED.change_description,
                            is_approved = EXCLUDED.is_approved
                        RETURNING id
                    """, (
                        version.project_id, version.version, components_json,
                        version.estimate.total_hours, phases.layout, phases.detail, phases.documentation,
                        version.change_description, version.changed_by, version.is_approved
                    ))
                    version_id = cur.fetchone()[0]
                    conn.commit()
                    logger.info(f"✅ Saved version {version.version} for project {version.project_id}")
                    return version_id

        except Exception as e:
            logger.error(f"Failed to save project version: {e}", exc_info=True)
            raise QueryError(f"Failed to save project version: {e}", query="INSERT project_versions")

    def get_project_versions(self, project_id: int) -> list[ProjectVersion]:
        """Get all versions for a project."""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT id, project_id, version, components,
                               estimated_hours, estimated_hours_3d_layout,
                               estimated_hours_3d_detail, estimated_hours_2d,
                               change_description, changed_by, is_approved, created_at
                        FROM project_versions
                        WHERE project_id = %s
                        ORDER BY created_at DESC
                    """, (project_id,))
                    rows = cur.fetchall()

                    versions = []
                    for row in rows:
                        # Parse components JSON
                        components_data = row['components'] or []
                        components = [Component.from_dict(c) for c in components_data]

                        phases = EstimatePhases(
                            layout=float(row['estimated_hours_3d_layout'] or 0),
                            detail=float(row['estimated_hours_3d_detail'] or 0),
                            documentation=float(row['estimated_hours_2d'] or 0)
                        )

                        estimate = Estimate.from_components(components)

                        versions.append(ProjectVersion(
                            id=row['id'],
                            project_id=row['project_id'],
                            version=row['version'],
                            estimate=estimate,
                            change_description=row['change_description'] or "",
                            changed_by=row['changed_by'] or "System",
                            is_approved=row['is_approved'] or False,
                            created_at=row['created_at']
                        ))

                    return versions

        except Exception as e:
            logger.error(f"Failed to get project versions: {e}", exc_info=True)
            raise QueryError(f"Failed to get project versions: {e}", query="SELECT FROM project_versions")

    # Component patterns

    def get_pattern(self, pattern_key: str, department: str) -> ComponentPattern | None:
        """Get component pattern by key and department."""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT name, pattern_key, department,
                               avg_hours_3d_layout, avg_hours_3d_detail, avg_hours_2d,
                               confidence, occurrences, source
                        FROM component_patterns
                        WHERE pattern_key = %s AND department = %s
                    """, (pattern_key, department))
                    row = cur.fetchone()

                    if not row:
                        return None

                    return ComponentPattern(
                        name=row['name'],
                        pattern_key=row['pattern_key'],
                        department_code=row['department'],
                        avg_hours_layout=float(row['avg_hours_3d_layout'] or 0),
                        avg_hours_detail=float(row['avg_hours_3d_detail'] or 0),
                        avg_hours_doc=float(row['avg_hours_2d'] or 0),
                        confidence=float(row['confidence'] or 0),
                        occurrences=int(row['occurrences'] or 0),
                        source=row['source'] or "actual"
                    )

        except Exception as e:
            logger.error(f"Failed to get pattern {pattern_key}: {e}", exc_info=True)
            raise QueryError(f"Failed to get pattern: {e}", query="SELECT FROM component_patterns")

    def save_pattern(self, pattern: ComponentPattern) -> bool:
        """Save or update component pattern."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO component_patterns (
                            name, pattern_key, department,
                            avg_hours_3d_layout, avg_hours_3d_detail, avg_hours_2d, avg_hours_total,
                            confidence, occurrences, source
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (pattern_key, department) DO UPDATE SET
                            name = EXCLUDED.name,
                            avg_hours_3d_layout = EXCLUDED.avg_hours_3d_layout,
                            avg_hours_3d_detail = EXCLUDED.avg_hours_3d_detail,
                            avg_hours_2d = EXCLUDED.avg_hours_2d,
                            avg_hours_total = EXCLUDED.avg_hours_total,
                            confidence = EXCLUDED.confidence,
                            occurrences = EXCLUDED.occurrences,
                            source = EXCLUDED.source,
                            last_updated = NOW()
                    """, (
                        pattern.name, pattern.pattern_key, pattern.department_code,
                        pattern.avg_hours_layout, pattern.avg_hours_detail, pattern.avg_hours_doc,
                        pattern.total_hours, pattern.confidence, pattern.occurrences, pattern.source
                    ))
                    conn.commit()
                    logger.info(f"✅ Saved pattern: {pattern.name} ({pattern.department_code})")
                    return True

        except Exception as e:
            logger.error(f"Failed to save pattern: {e}", exc_info=True)
            raise QueryError(f"Failed to save pattern: {e}", query="INSERT component_patterns")

    def get_patterns_by_department(self, department: str, min_occurrences: int = 1) -> list[ComponentPattern]:
        """Get all patterns for a department."""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT name, pattern_key, department,
                               avg_hours_3d_layout, avg_hours_3d_detail, avg_hours_2d,
                               confidence, occurrences, source
                        FROM component_patterns
                        WHERE department = %s AND occurrences >= %s
                        ORDER BY occurrences DESC
                    """, (department, min_occurrences))
                    rows = cur.fetchall()

                    return [
                        ComponentPattern(
                            name=row['name'],
                            pattern_key=row['pattern_key'],
                            department_code=row['department'],
                            avg_hours_layout=float(row['avg_hours_3d_layout'] or 0),
                            avg_hours_detail=float(row['avg_hours_3d_detail'] or 0),
                            avg_hours_doc=float(row['avg_hours_2d'] or 0),
                            confidence=float(row['confidence'] or 0),
                            occurrences=int(row['occurrences'] or 0),
                            source=row['source'] or "actual"
                        )
                        for row in rows
                    ]

        except Exception as e:
            logger.error(f"Failed to get patterns: {e}", exc_info=True)
            raise QueryError(f"Failed to get patterns: {e}", query="SELECT FROM component_patterns")

    def delete_patterns_with_low_confidence(self, min_confidence: float = 0.1) -> int:
        """Delete patterns with confidence below threshold."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM component_patterns WHERE confidence < %s", (min_confidence,))
                    deleted = cur.rowcount
                    conn.commit()
                    logger.info(f"✅ Deleted {deleted} low-confidence patterns")
                    return deleted

        except Exception as e:
            logger.error(f"Failed to delete patterns: {e}", exc_info=True)
            raise QueryError(f"Failed to delete patterns: {e}", query="DELETE FROM component_patterns")

    # Helper methods

    def _project_from_row(self, row: dict) -> Project:
        """Convert database row to Project model."""
        # Parse components JSON
        components_data = row['components'] or []
        components = [Component.from_dict(c) for c in components_data]

        # Create estimate
        estimate = Estimate.from_components(components)

        return Project(
            id=row['id'],
            name=row['name'],
            department=DepartmentCode(row['department']),
            estimate=estimate,
            description=row.get('description', ''),
            client=row.get('client', ''),
            cad_system=row.get('cad_system', ''),
            actual_hours=row.get('actual_hours'),
            created_at=row['created_at'],
            updated_at=row.get('updated_at', row['created_at']),
            is_historical=row.get('is_historical', False)
        )

"""
CAD Estimator Pro - PgVector Embeddings Service

Service for generating and managing embeddings using pgvector for semantic search.
"""
import logging
from typing import Any

from ...domain.exceptions import EmbeddingError
from ...infrastructure.ai.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class PgVectorService:
    """
    PgVector embeddings service for semantic search.

    Manages embeddings for projects and component patterns using pgvector extension.
    """

    def __init__(self, db_client: Any, ai_client: OllamaClient):
        """
        Initialize PgVectorService.

        Args:
            db_client: Database client (DatabaseClient protocol)
            ai_client: AI client for embedding generation (OllamaClient)
        """
        self.db = db_client
        self.ai = ai_client

    def ensure_project_embedding(
        self,
        project_id: int,
        description: str
    ) -> bool:
        """
        Ensure project has embedding vector.

        Generates embedding if not exists or description changed.

        Args:
            project_id: Project ID
            description: Project description

        Returns:
            True if successful

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not description or len(description.strip()) < 10:
            logger.warning(f"Project {project_id} has empty/short description, skipping embedding")
            return False

        try:
            # Generate embedding
            embedding = self.ai.generate_embedding(description)

            # Convert to pgvector format
            embedding_str = self._to_pgvector(embedding)

            # Update database
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE projects
                        SET description_embedding = %s::vector
                        WHERE id = %s
                    """, (embedding_str, project_id))
                    conn.commit()

            logger.debug(f"✅ Updated embedding for project {project_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update project embedding: {e}", exc_info=True)
            raise EmbeddingError(f"Project embedding failed: {e}", text=description[:100])

    def ensure_pattern_embedding(
        self,
        pattern_key: str,
        department: str,
        name: str
    ) -> bool:
        """
        Ensure pattern has embedding vector.

        Args:
            pattern_key: Pattern key (canonicalized name)
            department: Department code
            name: Pattern name

        Returns:
            True if successful

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not name or len(name.strip()) < 3:
            logger.warning(f"Pattern '{pattern_key}' has empty/short name, skipping embedding")
            return False

        try:
            # Generate embedding from name
            embedding = self.ai.generate_embedding(name)

            # Convert to pgvector format
            embedding_str = self._to_pgvector(embedding)

            # Update database
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE component_patterns
                        SET name_embedding = %s::vector
                        WHERE pattern_key = %s AND department = %s
                    """, (embedding_str, pattern_key, department))
                    conn.commit()

            logger.debug(f"✅ Updated embedding for pattern '{name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to update pattern embedding: {e}", exc_info=True)
            raise EmbeddingError(f"Pattern embedding failed: {e}", text=name)

    def find_similar_projects(
        self,
        description: str,
        department: str,
        limit: int = 10,
        similarity_threshold: float = 0.6
    ) -> list[dict]:
        """
        Find similar projects using semantic search.

        Args:
            description: Query description
            department: Department code
            limit: Max results
            similarity_threshold: Min similarity (0.0-1.0)

        Returns:
            List of similar projects with similarity scores
        """
        if not description or len(description.strip()) < 10:
            return []

        try:
            # Generate query embedding
            query_embedding = self.ai.generate_embedding(description)
            embedding_str = self._to_pgvector(query_embedding)

            # Search using cosine similarity
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT
                            id, name, client, estimated_hours, actual_hours,
                            1 - (description_embedding <=> %s::vector) as similarity
                        FROM projects
                        WHERE department = %s
                          AND description_embedding IS NOT NULL
                          AND 1 - (description_embedding <=> %s::vector) >= %s
                        ORDER BY description_embedding <=> %s::vector
                        LIMIT %s
                    """, (embedding_str, department, embedding_str, similarity_threshold, embedding_str, limit))

                    rows = cur.fetchall()

                    results = []
                    for row in rows:
                        results.append({
                            'id': row[0],
                            'name': row[1],
                            'client': row[2],
                            'estimated_hours': float(row[3] or 0),
                            'actual_hours': float(row[4]) if row[4] else None,
                            'similarity': float(row[5])
                        })

                    logger.info(f"🔍 Found {len(results)} similar projects (threshold={similarity_threshold})")
                    return results

        except Exception as e:
            logger.error(f"Semantic project search failed: {e}", exc_info=True)
            return []

    def find_similar_components(
        self,
        name: str,
        department: str,
        limit: int = 10,
        similarity_threshold: float = 0.6
    ) -> list[dict]:
        """
        Find similar component patterns using semantic search.

        Args:
            name: Query component name
            department: Department code
            limit: Max results
            similarity_threshold: Min similarity (0.0-1.0)

        Returns:
            List of similar patterns with similarity scores
        """
        if not name or len(name.strip()) < 3:
            return []

        try:
            # Generate query embedding
            query_embedding = self.ai.generate_embedding(name)
            embedding_str = self._to_pgvector(query_embedding)

            # Search using cosine similarity
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT
                            name, pattern_key, avg_hours_total,
                            avg_hours_3d_layout, avg_hours_3d_detail, avg_hours_2d,
                            occurrences, confidence,
                            1 - (name_embedding <=> %s::vector) as similarity
                        FROM component_patterns
                        WHERE department = %s
                          AND name_embedding IS NOT NULL
                          AND 1 - (name_embedding <=> %s::vector) >= %s
                        ORDER BY name_embedding <=> %s::vector
                        LIMIT %s
                    """, (embedding_str, department, embedding_str, similarity_threshold, embedding_str, limit))

                    rows = cur.fetchall()

                    results = []
                    for row in rows:
                        results.append({
                            'name': row[0],
                            'pattern_key': row[1],
                            'avg_hours_total': float(row[2] or 0),
                            'avg_hours_3d_layout': float(row[3] or 0),
                            'avg_hours_3d_detail': float(row[4] or 0),
                            'avg_hours_2d': float(row[5] or 0),
                            'occurrences': int(row[6] or 0),
                            'confidence': float(row[7] or 0),
                            'similarity': float(row[8])
                        })

                    logger.info(f"🔍 Found {len(results)} similar components for '{name}'")
                    return results

        except Exception as e:
            logger.error(f"Semantic component search failed: {e}", exc_info=True)
            return []

    def _to_pgvector(self, embedding: list[float]) -> str:
        """
        Convert embedding list to pgvector format string.

        Args:
            embedding: Embedding vector

        Returns:
            pgvector format string: '[0.1, 0.2, 0.3, ...]'
        """
        return '[' + ','.join(str(x) for x in embedding) + ']'

    def batch_update_embeddings(
        self,
        update_projects: bool = True,
        update_patterns: bool = True
    ) -> dict[str, int]:
        """
        Batch update embeddings for all entities.

        Args:
            update_projects: Update project embeddings
            update_patterns: Update pattern embeddings

        Returns:
            Dict with counts: {'projects': 10, 'patterns': 50}
        """
        counts = {'projects': 0, 'patterns': 0}

        try:
            if update_projects:
                with self.db.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT id, description FROM projects
                            WHERE description IS NOT NULL
                              AND description != ''
                              AND description_embedding IS NULL
                        """)
                        rows = cur.fetchall()

                        for row in rows:
                            try:
                                self.ensure_project_embedding(row[0], row[1])
                                counts['projects'] += 1
                            except Exception as e:
                                logger.warning(f"Failed to embed project {row[0]}: {e}")
                                continue

            if update_patterns:
                with self.db.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT pattern_key, department, name FROM component_patterns
                            WHERE name IS NOT NULL
                              AND name != ''
                              AND name_embedding IS NULL
                        """)
                        rows = cur.fetchall()

                        for row in rows:
                            try:
                                self.ensure_pattern_embedding(row[0], row[1], row[2])
                                counts['patterns'] += 1
                            except Exception as e:
                                logger.warning(f"Failed to embed pattern {row[0]}: {e}")
                                continue

            logger.info(f"✅ Batch embedding update complete: {counts}")
            return counts

        except Exception as e:
            logger.error(f"Batch embedding update failed: {e}", exc_info=True)
            return counts

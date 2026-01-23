#!/usr/bin/env python3
"""
Generate embeddings for all components in component_patterns table.

This is a one-time batch job (or periodic update) to populate
the embedding column for semantic search.

Usage:
    python scripts/generate_component_embeddings.py

Requirements:
    - pgvector extension enabled
    - component_patterns table exists
    - Ollama running with nomic-embed-text model
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cad.infrastructure.database.postgres_client import PostgresClient
from src.cad.infrastructure.ai.ollama_client import OllamaClient
from src.cad.domain.models.config import DatabaseConfig, OllamaConfig
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_embeddings_batch(
    db: PostgresClient,
    ai: OllamaClient,
    batch_size: int = 50,
    embedding_model: str = "nomic-embed-text"
):
    """
    Generate embeddings for all components missing them.

    Args:
        db: Database client
        ai: AI client for embeddings
        batch_size: How many components to process at once
        embedding_model: Model to use for embeddings
    """

    # 1. Get components that need embeddings
    logger.info("Fetching components needing embeddings...")

    query = """
    SELECT id, name, category, department_code
    FROM component_patterns
    WHERE embedding IS NULL
    ORDER BY occurrence_count DESC
    """

    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            components = cur.fetchall()

    total = len(components)
    logger.info(f"Found {total} components needing embeddings")

    if total == 0:
        logger.info("✅ All components already have embeddings!")
        return

    # 2. Process in batches
    processed = 0
    errors = 0

    for i in range(0, total, batch_size):
        batch = components[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")

        for comp_id, name, category, dept in batch:
            try:
                # Generate embedding
                # Combine name + category for better semantic representation
                text = f"{name}"
                if category:
                    text += f" ({category})"

                logger.info(f"  Generating embedding for: {text}")
                embedding = ai.generate_embedding(text, model=embedding_model)

                # Update database
                update_query = """
                UPDATE component_patterns
                SET embedding = %s::vector,
                    embedding_model = %s,
                    embedding_generated_at = NOW()
                WHERE id = %s
                """

                with db.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(update_query, (embedding, embedding_model, comp_id))
                    conn.commit()

                processed += 1

                # Rate limiting (if needed)
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"  ❌ Failed to generate embedding for '{name}': {e}")
                errors += 1
                continue

        logger.info(f"  Processed {processed}/{total} components ({errors} errors)")

    # 3. Summary
    logger.info("=" * 60)
    logger.info(f"✅ Embedding generation complete!")
    logger.info(f"   Total: {total}")
    logger.info(f"   Successful: {processed}")
    logger.info(f"   Errors: {errors}")
    logger.info("=" * 60)

    # 4. Verify
    verify_query = """
    SELECT
        COUNT(*) FILTER (WHERE embedding IS NOT NULL) AS with_embeddings,
        COUNT(*) FILTER (WHERE embedding IS NULL) AS without_embeddings,
        COUNT(*) AS total
    FROM component_patterns
    """

    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(verify_query)
            result = cur.fetchone()

    logger.info(f"\nCurrent state:")
    logger.info(f"  With embeddings: {result[0]}/{result[2]} ({result[0]/result[2]*100:.1f}%)")
    logger.info(f"  Without embeddings: {result[1]}/{result[2]}")


def main():
    """Main entry point."""
    try:
        # Initialize clients
        logger.info("Initializing clients...")

        db_config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="cad_estimator",
            user="postgres",
            password="postgres"  # Change in production!
        )

        ollama_config = OllamaConfig(
            url="http://localhost:11434",
            default_model="llama3.2:latest",
            embedding_model="nomic-embed-text"  # Lightweight, good quality
        )

        db = PostgresClient(db_config)
        ai = OllamaClient(ollama_config)

        # Test connection
        logger.info("Testing database connection...")
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()[0]
                logger.info(f"  PostgreSQL: {version[:50]}...")

        logger.info("Testing Ollama connection...")
        test_embedding = ai.generate_embedding("test", model=ollama_config.embedding_model)
        logger.info(f"  Embedding dimension: {len(test_embedding)}")

        # Generate embeddings
        logger.info("\nStarting embedding generation...")
        generate_embeddings_batch(
            db=db,
            ai=ai,
            batch_size=50,
            embedding_model=ollama_config.embedding_model
        )

        logger.info("\n✅ All done!")

    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

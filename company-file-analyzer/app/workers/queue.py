"""
Redis Queue (RQ) configuration.
"""
import logging
from redis import Redis
from rq import Queue

from app.config import REDIS_URL

logger = logging.getLogger(__name__)

# Parse Redis URL
# Format: redis://localhost:6379/0
try:
    redis_conn = Redis.from_url(REDIS_URL)
    task_queue = Queue("default", connection=redis_conn)
    logger.info(f"Connected to Redis: {REDIS_URL}")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Jobs will be processed synchronously.")
    redis_conn = None
    task_queue = None


def enqueue_job(job_id: str) -> bool:
    """
    Enqueue a job for background processing.

    Args:
        job_id: The job ID to process

    Returns:
        True if enqueued, False if processed synchronously
    """
    from app.workers.tasks import process_job

    if task_queue is not None:
        try:
            task_queue.enqueue(process_job, job_id, job_timeout="1h")
            logger.info(f"Job {job_id} enqueued")
            return True
        except Exception as e:
            logger.warning(f"Failed to enqueue job {job_id}: {e}")

    # Fallback: process synchronously (for development/testing)
    logger.info(f"Processing job {job_id} synchronously")
    try:
        process_job(job_id)
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
    return False

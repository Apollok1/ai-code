from redis import Redis
from rq import Queue
from app.config import settings

def get_queue() -> Queue:
    redis = Redis.from_url(settings.redis_url)
    return Queue("default", connection=redis)

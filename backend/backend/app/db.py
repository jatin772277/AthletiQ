"""MongoDB connection and collection access for AthletiQ."""

from __future__ import annotations
from functools import lru_cache
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database
from backend.app.config import settings


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    if settings.mongodb_uri.startswith("mongomock://"):
        import mongomock  # test-only dependency
        return mongomock.MongoClient(tz_aware=True)
    return MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000, tz_aware=True)


def get_db() -> Database:
    return get_client()[settings.mongodb_db]


def users():
    return get_db()["users"]


def sessions():
    return get_db()["sessions"]


def jobs():
    return get_db()["matches"]


def ensure_indexes() -> None:
    users().create_index([("email", ASCENDING)], unique=True)
    sessions().create_index([("user_id", ASCENDING)])
    # Mongo drops expired sessions on its own.
    sessions().create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)
    jobs().create_index([("owner_id", ASCENDING), ("created_at", DESCENDING)])
    jobs().create_index([("status", ASCENDING)])


def ping() -> bool:
    if settings.mongodb_uri.startswith("mongomock://"):
        return True
    try:
        get_client().admin.command("ping")
        return True
    except Exception:
        return False

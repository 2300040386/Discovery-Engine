from __future__ import annotations

import psycopg2
import redis
from psycopg2.extensions import connection as PGConnection

from src.shared.config import Settings, get_settings

_redis_client: redis.Redis | None = None
_pg_connection: PGConnection | None = None

# libpq default has no connect timeout at all, so an unreachable host can hang
# for a minute or more instead of failing fast into the caller's error path.
PG_CONNECT_TIMEOUT_S = 3

# redis-py has no default socket timeout either. Callers throughout src/gateway
# already catch connection errors and degrade gracefully (empty session, skip
# popularity fallback), but only if the call actually raises instead of
# hanging — a degraded connection with no timeout can block a request (and
# the worker thread serving it) indefinitely.
REDIS_SOCKET_TIMEOUT_S = 2
REDIS_SOCKET_CONNECT_TIMEOUT_S = 2


def get_redis_client(settings: Settings | None = None) -> redis.Redis:
    global _redis_client
    if settings is not None:
        return redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=REDIS_SOCKET_TIMEOUT_S,
            socket_connect_timeout=REDIS_SOCKET_CONNECT_TIMEOUT_S,
        )
    if _redis_client is None:
        _redis_client = redis.Redis.from_url(
            get_settings().redis_url,
            decode_responses=True,
            socket_timeout=REDIS_SOCKET_TIMEOUT_S,
            socket_connect_timeout=REDIS_SOCKET_CONNECT_TIMEOUT_S,
        )
    return _redis_client


def get_pg_connection(settings: Settings | None = None) -> PGConnection:
    global _pg_connection
    if settings is not None:
        return psycopg2.connect(settings.pg_dsn, connect_timeout=PG_CONNECT_TIMEOUT_S)
    if _pg_connection is None or _pg_connection.closed:
        _pg_connection = psycopg2.connect(get_settings().pg_dsn, connect_timeout=PG_CONNECT_TIMEOUT_S)
    return _pg_connection

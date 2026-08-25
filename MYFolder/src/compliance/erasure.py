from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol

TOMBSTONE_KEY = "tombstone:users"


class PGCursor(Protocol):
    def execute(self, query: str, params: tuple[object, ...] = ()) -> None: ...


class PGConnection(Protocol):
    def cursor(self) -> PGCursor: ...
    def commit(self) -> None: ...


class RedisClient(Protocol):
    def delete(self, *keys: str) -> int: ...
    def sadd(self, key: str, *values: str) -> int: ...


class IndexHandle(Protocol):
    def remove(self, user_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class DeletionReceipt:
    receipt_id: str
    user_id: str


def erase_user(
    user_id: str,
    pg_conn: PGConnection,
    redis_client: RedisClient,
    index_handle: IndexHandle,
) -> DeletionReceipt:
    cursor = pg_conn.cursor()
    cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
    cursor.execute("UPDATE recommendation_log SET user_id = NULL WHERE user_id = %s", (user_id,))
    pg_conn.commit()

    redis_client.delete(f"sess:{user_id}", f"feat:u:{user_id}")
    # Also consulted by the trainer to exclude this user from the next run.
    redis_client.sadd(TOMBSTONE_KEY, user_id)

    index_handle.remove(user_id)

    return DeletionReceipt(receipt_id=str(uuid.uuid4()), user_id=user_id)

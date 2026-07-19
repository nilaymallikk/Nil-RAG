"""Redis persistence for anonymous browser conversation metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Final

from redis import Redis


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    title: str
    created_at: str
    updated_at: str


class RedisConversationStore:
    """Indexes a browser's conversations and stores their lightweight metadata."""

    _KEY_PREFIX: Final = "rag"

    def __init__(self, redis_client: Redis, ttl_seconds: int) -> None:
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds

    def create(self, *, user_id: str, conversation_id: str, title: str) -> Conversation:
        now = self._timestamp()
        conversation = Conversation(
            conversation_id=conversation_id,
            title=self._title_from_question(title),
            created_at=now,
            updated_at=now,
        )

        metadata_key = self._metadata_key(conversation_id)
        index_key = self._user_index_key(user_id)

        with self._redis.pipeline(transaction=True) as pipeline:
            pipeline.hset(metadata_key, mapping={"user_id": user_id, **asdict(conversation)})
            pipeline.zadd(index_key, {conversation_id: datetime.now(UTC).timestamp()})
            pipeline.expire(metadata_key, self._ttl_seconds)
            pipeline.expire(index_key, self._ttl_seconds)
            pipeline.execute()

        return conversation

    def list_for_user(self, user_id: str) -> list[Conversation]:
        conversation_ids = self._redis.zrevrange(self._user_index_key(user_id), 0, -1)
        conversations: list[Conversation] = []

        for conversation_id in conversation_ids:
            metadata = self._redis.hgetall(self._metadata_key(conversation_id))
            if metadata.get("user_id") != user_id:
                continue

            conversations.append(
                Conversation(
                    conversation_id=conversation_id,
                    title=metadata["title"],
                    created_at=metadata["created_at"],
                    updated_at=metadata["updated_at"],
                )
            )

        return conversations

    def belongs_to_user(self, *, user_id: str, conversation_id: str) -> bool:
        metadata = self._redis.hgetall(self._metadata_key(conversation_id))
        return metadata.get("user_id") == user_id

    def touch(self, *, user_id: str, conversation_id: str) -> None:
        if not self.belongs_to_user(user_id=user_id, conversation_id=conversation_id):
            raise PermissionError("Conversation does not belong to this browser.")

        now = self._timestamp()
        metadata_key = self._metadata_key(conversation_id)
        index_key = self._user_index_key(user_id)

        with self._redis.pipeline(transaction=True) as pipeline:
            pipeline.hset(metadata_key, "updated_at", now)
            pipeline.zadd(index_key, {conversation_id: datetime.now(UTC).timestamp()})
            pipeline.expire(metadata_key, self._ttl_seconds)
            pipeline.expire(index_key, self._ttl_seconds)
            pipeline.execute()

    def _metadata_key(self, conversation_id: str) -> str:
        return f"{self._KEY_PREFIX}:conversation:{conversation_id}:metadata"

    def _user_index_key(self, user_id: str) -> str:
        return f"{self._KEY_PREFIX}:user:{user_id}:conversations"

    @staticmethod
    def _title_from_question(question: str) -> str:
        normalized = " ".join(question.split())
        return normalized[:80] or "New conversation"

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).isoformat()

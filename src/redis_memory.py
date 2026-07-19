from __future__ import annotations

import json
from typing import TypedDict

from redis import Redis


class ChatMessage(TypedDict):
    role: str
    content: str


class RedisConversationMemory:
    """Bounded Redis-backed memory for one conversation."""

    def __init__(
        self,
        redis_client: Redis,
        conversation_id: str,
        max_turns: int = 4,
        ttl_seconds: int = 86_400,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")

        self._redis = redis_client
        self._key = f"rag:conversation:{conversation_id}:messages"
        self._max_messages = max_turns * 2
        self._ttl_seconds = ttl_seconds

    def add(self, question: str, answer: str) -> None:
        messages: list[ChatMessage] = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        serialized = [
            json.dumps(message, ensure_ascii=False)
            for message in messages
        ]

        with self._redis.pipeline(transaction=True) as pipeline:
            pipeline.rpush(self._key, *serialized)
            pipeline.ltrim(self._key, -self._max_messages, -1)
            pipeline.expire(self._key, self._ttl_seconds)
            pipeline.execute()

    def get_messages(self) -> list[ChatMessage]:
        raw_messages = self._redis.lrange(self._key, 0, -1)

        messages: list[ChatMessage] = []

        for raw_message in raw_messages:
            message = json.loads(raw_message)

            if (
                not isinstance(message, dict)
                or message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError("Invalid conversation message stored in Redis.")

            messages.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                }
            )

        if messages:
            self._redis.expire(self._key, self._ttl_seconds)

        return messages

    def clear(self) -> None:
        self._redis.delete(self._key)
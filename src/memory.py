from __future__ import annotations

from typing import TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class ConversationMemory:
    def __init__(self, max_turns: int = 4):
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")

        self._history: list[ChatMessage] = []
        self._max_turns = max_turns

    def add(self, question: str, answer: str) -> None:
        if not isinstance(question, str) or not isinstance(answer, str):
            raise TypeError("Memory question and answer must both be strings")

        self._history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        )
        max_messages = self._max_turns * 2
        self._history = self._history[-max_messages:]

    # shallow copy
    def get_messages(self) -> list[ChatMessage]:
        return [message.copy() for message in self._history]

    # Optional utility — useful if you ever want to let the user reset the conversation mid-session (e.g., type "reset" in the chat loop).

    def clear(self) -> None:
        self._history.clear()

class ConversationMemory:
    def __init__(self, max_turns: int = 0):
        self._history = []
        self._max_turns = max_turns

    def add(self, question: str, answer: str):
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer})

        if len(self._history) > self._max_turns * 2:
            self._history = self._history[-(self._max_turns * 2)]

    # shallow copy
    def get_messages(Self) -> list[dict]:
        return list(Self._history)

    # Optional utility — useful if you ever want to let the user reset the conversation mid-session (e.g., type "reset" in the chat loop).

    def clear(self):
        self._hostory = []
    
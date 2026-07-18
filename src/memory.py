class ConversationMemory:
    def __init__(Self, max_turns: int = 0):
        self._history = []
        self._max_turns = max_turns

    def add(self, question: str, answer: str):
        self._history.append({"role": "user", "content": question})
        self._history.append({"role"})
"""
Base class for quiz clients.

"""

from abc import ABC, abstractmethod

class QuizError(Exception):
    """An error response from the server, e.g. 409 question_pending or 410 category_exhausted."""

    def __init__(self, status: int, code: str, body: dict):
        super().__init__(f"{status} {code}")
        self.status = status
        self.code = code
        self.body = body

class QuizClient(ABC):
    @abstractmethod
    def categories(self) -> list[dict]:
        """The categories of the current round, with questionCount and remaining."""
        pass

    @abstractmethod
    def status(self) -> dict:
        """The current round, our score and our open question (pending), if any."""
        pass

    @abstractmethod
    def request_question(self, category: str, mode: str = "text") -> dict|None:
        """Request a question. The server's DEADLINE starts the moment it is served.
        If a question is already open, that open question is returned instead.
        """
        pass

    @abstractmethod
    def answer(self, question_id: int, option_id: int) -> dict:
        """Submit an answer (option id 0-3). Only the first answer to a question counts."""
        pass

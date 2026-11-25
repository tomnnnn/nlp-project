from .base import QuizClient, QuizError
from dataclasses import dataclass
import json
import time
import datetime

@dataclass
class Question:
    id: int
    category: str
    text: str
    options: list[str]
    answer: int
    deadline: datetime.datetime
    servedAt: datetime.datetime

@dataclass
class QuizState:
    round: int
    score: dict
    open_question: Question | None
    category_pointers: dict[str, int]

class CategoryError(Exception):
    pass

class QuizClientLocal(QuizClient):
    def __init__(self, path: str, seed: int = 42, deadlineMs: int = 30000):
        """Quiz client that uses local question files."""

        self.path = path
        if not self.path.endswith(".jsonl"):
            raise ValueError("Path must be a .jsonl file")

        with open(path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f.readlines()]

        self.categories_ = set(l["category"] for l in self.data)
        self.data_by_category = {c: [l for l in self.data if l["category"] == c] for c in self.categories_}
        self.state = QuizState(
            round=1, 
            score={c: 0 for c in self.categories_}, 
            open_question=None, 
            category_pointers={c: 0 for c in self.categories_},
        )
        self.deadlineMs = deadlineMs

    def categories(self) -> list[dict]:
        """The categories of the current round, with questionCount and remaining."""
        return list(self.categories_)

    def status(self) -> dict:
        """The current round, our score and our open question (pending), if any."""
        return self.state.__dict__

    def _build_question_response(self, question: Question) -> dict:
        """Build the question response dictionary out of the current state."""

        return {
            "id": question.id,
            "category": question.category,
            "text": question.text,
            "options": question.options,
            "answer": question.answer,
            "deadline": question.deadline.isoformat(),
            "servedAt": question.servedAt.isoformat(),
            "timeLimitMs": question.deadline.timestamp() - question.servedAt.timestamp(),
            "remainingMs": max(0, int((question.deadline - datetime.datetime.now()).total_seconds() * 1000)),
            "serverTime": datetime.datetime.now().isoformat(),
        }

    def request_question(self, category: str, mode: str = "text") -> dict:
        """Request a question. The server's DEADLINE starts the moment it is served.
        If a question is already open, that open question is returned instead.
        """
        if self.state.open_question is not None and max(0, int((self.state.open_question.deadline - datetime.datetime.now()).total_seconds() * 1000)) > 0:
            return self._build_question_response(self.state.open_question)

        if category not in self.categories_:
            raise ValueError(f"Unknown category {category}")

        pointer = self.state.category_pointers[category]
        if pointer < len(self.data_by_category[category]):
            question = self.data_by_category[category][pointer]
            self.state.category_pointers[category] += 1
            self.state.open_question = Question(
                **{k: v for k, v in question.items() if k in {f.name for f in Question.__dataclass_fields__.values()}},
                deadline=datetime.datetime.now() + datetime.timedelta(milliseconds=self.deadlineMs),
                servedAt=datetime.datetime.now()
            )
            return self._build_question_response(self.state.open_question)

        raise QuizError(410, "category_exhausted", {"category": category, "remaining": 0})


    def answer(self, question_id: int, option_id: int) -> dict:
        """Submit an answer (option id 0-3). Only the first answer to a question counts."""
        if self.state.open_question is None:
            raise ValueError("No open question to answer")

        if self.state.open_question.id != question_id:
            raise ValueError(f"Question ID {question_id} does not match the open question ID {self.state.open_question.id}")

        category = self.state.open_question.category
        timecheck = datetime.datetime.now() <= self.state.open_question.deadline
        correct_option = self.state.open_question.answer
        is_correct = (option_id == correct_option) if timecheck else None
        counted = timecheck

        if is_correct:
            self.state.score[category] += 1
            
        self.state.open_question = None  # Close the question after answering
        return {
            "questionId": question_id,
            "optionId": option_id,
            "correct": is_correct,
            "points": 1,
            "counted": counted,
            "score": {
                "total": sum(self.state.score.values()),
                "category": self.state.score[category]
            }
        }

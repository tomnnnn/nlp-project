"""Connection to the quiz server.

The only module that makes HTTP requests: login, categories, requesting a question,
submitting an answer and status. It knows nothing about AI.

API reference: project_readme.pdf ("The Million Token Question").
"""
from __future__ import annotations

import os
import time

import requests

BASE = "https://mai-nlp-course.usi.ch"
TIMEOUT = 5            # seconds the client waits for a server response
SAFE_RATE_TIME = 1.1   # minimum seconds between question requests (rate limit: 60/min)


class QuizError(Exception):
    """An error response from the server, e.g. 409 question_pending or 410 category_exhausted."""

    def __init__(self, status: int, code: str, body: dict):
        super().__init__(f"{status} {code}")
        self.status = status
        self.code = code
        self.body = body


class QuizClient:
    """Talks to the quiz server over HTTPS.

    All calls go through one requests.Session, which keeps the login cookie (quiz_auth).
    """

    def __init__(self, username: str | None = None, password: str | None = None):
        # Credentials come from environment variables (or Colab Secrets), never from the code
        self.username = username or os.environ["MTQ_USERNAME"]
        self.password = password or os.environ["MTQ_PASSWORD"]
        self.session = requests.Session()
        self._last_question_request = 0.0  # monotonic time of the last question request

    # --- low-level helpers -------------------------------------------------

    def _request(self, method: str, path: str, json: dict | None = None) -> dict:
        """Send one request and return the JSON body, or raise QuizError on an error status."""
        response = self.session.request(method, BASE + path, json=json, timeout=TIMEOUT)
        try:
            body = response.json() if response.content else {}
        except ValueError:  # e.g. a 5xx without a JSON body
            body = {}
        if response.status_code >= 400:
            raise QuizError(response.status_code, body.get("code") or body.get("error") or "unknown", body)
        return body

    # --- API calls ---------------------------------------------------------

    def login(self) -> dict:
        """Log in; the server sets the session cookie, valid for 8 hours."""
        return self._request("POST", "/api/auth/login",
                             {"username": self.username, "password": self.password})

    def categories(self) -> list[dict]:
        """The categories of the current round, with questionCount and remaining."""
        return self._request("GET", "/api/challenge/categories")["categories"]

    def status(self) -> dict:
        """The current round, our score and our open question (pending), if any."""
        return self._request("GET", "/api/challenge/status")

    def request_question(self, category: str, mode: str = "text") -> dict:
        """Request a question. The server's DEADLINE starts the moment it is served.

        Returns the server response (question, remainingMs, ...) plus "local_deadline":
        a time.monotonic() timestamp by which the answer must arrive.
        If a question is already open (409 question_pending), that open question is returned instead.
        """
        # Respect the rate limit: at most one question request every SAFE_RATE_TIME seconds
        wait = SAFE_RATE_TIME - (time.monotonic() - self._last_question_request)
        if wait > 0:
            time.sleep(wait)
        self._last_question_request = time.monotonic()

        try:
            served = self._request("POST", "/api/challenge/questions", {"category": category, "mode": mode})
        except QuizError as e:
            if e.code != "question_pending":
                raise
            served = e.body["pending"]  # same shape as a normal response, clock still running

        # Count down on our own monotonic clock instead of comparing timestamps (our clock may be off)
        served["local_deadline"] = time.monotonic() + served["remainingMs"] / 1000
        return served

    def answer(self, question_id: int, option_id: int) -> dict:
        """Submit an answer (option id 0-3). Only the first answer to a question counts."""
        path = f"/api/challenge/questions/{question_id}/answer"
        try:
            return self._request("POST", path, {"optionId": int(option_id)})
        except QuizError as e:
            if e.code == "already_answered":  # our first answer arrived; its outcome is in the body
                return e.body
            raise

"""Build our local evaluation set from public multiple-choice datasets.

The quiz server never reveals the correct answer and its question pool is finite, so we
evaluate our system offline on public questions that resemble the six quiz categories.

Every question is stored in the server's format plus the answer key:
    {"id": ..., "category": ..., "text": ..., "options": [{"id": 0, "text": ...}, ...],
     "answer": <correct option id>, "difficulty": "easy"|"medium"|"hard"|null, "source": ...}

Output: data/eval/dev.jsonl (for developing prompts/models) and data/eval/test.jsonl
(held out, used only for final comparisons).

Run from the project root:
    python scripts/build_eval_set.py
"""
from __future__ import annotations

import ast
import html
import json
import random
import re
import time
from pathlib import Path

import requests
from datasets import load_dataset

SEED = 42                  # fixed seed: everyone in the team gets the exact same set
PER_CATEGORY = 100         # questions per category (split 50/50 into dev and test)
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"

# OpenTDB "History" is mostly modern history, so for ancient_history we keep only
# questions that mention one of these words. Crude, but better than nothing.
ANCIENT_KEYWORDS = [
    "ancient", "roman", "rome", "greek", "greece", "egypt", "pharaoh", "b.c", "bce",
    "sparta", "athens", "persia", "carthage", "caesar", "babylon", "mesopotamia", "sumer",
    "pyramid", "alexander the great", "hannibal", "troy", "spartan", "gaul", "assyria",
]


# --- helpers ----------------------------------------------------------------

def make_question(qid, category, text, options, answer, source, difficulty=None) -> dict:
    """Build one question in our format. `options` is a list of 4 strings, `answer` an index."""
    return {
        "id": qid,
        "category": category,
        "text": text.strip(),
        "options": [{"id": i, "text": str(o).strip()} for i, o in enumerate(options)],
        "answer": answer,
        "difficulty": difficulty,
        "source": source,
    }


def shuffle_options(q: dict, rng: random.Random) -> dict:
    """Shuffle the options (the server does this too) and update the answer id."""
    texts = [o["text"] for o in q["options"]]
    correct = texts[q["answer"]]
    rng.shuffle(texts)
    q["options"] = [{"id": i, "text": t} for i, t in enumerate(texts)]
    q["answer"] = texts.index(correct)
    return q


# --- loaders: one per source, each returns a list of questions -------------

def load_mmlu_redux(subject: str, category: str) -> list[dict]:
    """MMLU-Redux: MMLU re-annotated by humans. We keep only questions marked error_type == 'ok'."""
    rows = load_dataset("edinburgh-dawg/mmlu-redux-2.0", subject, split="test")
    out = []
    for i, r in enumerate(rows):
        if r["error_type"] != "ok":
            continue
        choices = r["choices"]
        if isinstance(choices, str):  # some versions store the list as a string
            choices = ast.literal_eval(choices)
        if len(choices) != 4:
            continue
        out.append(make_question(f"mmlu-redux/{subject}/{i}", category, r["question"],
                                 choices, int(r["answer"]), f"mmlu-redux/{subject}"))
    return out


def load_mmlu_redux_many(subjects: list[str], category: str) -> list[dict]:
    """Several MMLU-Redux subjects pooled into one source (so together they count as one in sampling)."""
    return [q for s in subjects for q in load_mmlu_redux(s, category)]


def load_arc(config: str, category: str) -> list[dict]:
    """ARC (AI2 Reasoning Challenge): grade-school science questions. config: ARC-Easy or ARC-Challenge."""
    rows = load_dataset("allenai/ai2_arc", config, split="test")
    out = []
    for r in rows:
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        if len(texts) != 4 or r["answerKey"] not in labels:
            continue
        out.append(make_question(f"arc/{r['id']}", category, r["question"], texts,
                                 labels.index(r["answerKey"]), config.lower()))
    return out


def load_sciq(category: str) -> list[dict]:
    """SciQ: science exam questions with one correct answer and three distractors."""
    rows = load_dataset("allenai/sciq", split="test")
    out = []
    for i, r in enumerate(rows):
        options = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        out.append(make_question(f"sciq/{i}", category, r["question"], options, 0, "sciq"))
    return out


def load_opentdb(tdb_category: int, category: str, max_questions: int = 200,
                 keywords: list[str] | None = None) -> list[dict]:
    """Open Trivia DB (opentdb.com): trivia questions with easy/medium/hard difficulty.

    The API gives at most 50 questions per request and allows one request per 5 seconds.
    A session token makes sure we never get the same question twice.
    """
    token = requests.get("https://opentdb.com/api_token.php?command=request", timeout=10).json()["token"]
    out, amount = [], 50
    while len(out) < max_questions:
        time.sleep(5.5)  # OpenTDB rate limit
        url = (f"https://opentdb.com/api.php?amount={min(amount, max_questions - len(out))}"
               f"&category={tdb_category}&type=multiple&token={token}")
        data = requests.get(url, timeout=10).json()
        code = data["response_code"]
        if code == 5:        # rate limited: just wait and try again
            continue
        # 1 = not enough questions, 4 = token exhausted. OpenTDB also answers 4 when fewer
        # questions are left than we asked for, so in both cases ask for fewer until amount is 1.
        if code in (1, 4):
            if amount == 1:
                break
            amount = max(1, amount // 2)
            continue
        if code != 0:        # another error (e.g. invalid token)
            break
        for r in data["results"]:
            text = html.unescape(r["question"])  # OpenTDB sends HTML entities like &quot;
            # \b = word boundary, so "troy" does not match inside "destroyed"
            if keywords and not any(re.search(rf"\b{re.escape(k)}", text.lower()) for k in keywords):
                continue
            options = [html.unescape(r["correct_answer"])] + [html.unescape(a) for a in r["incorrect_answers"]]
            out.append(make_question(f"opentdb/{tdb_category}/{len(out)}", category, text, options, 0,
                                     f"opentdb/{tdb_category}", r["difficulty"]))
    return out


# --- which sources make up each quiz category -------------------------------

# Category descriptions from the quiz web UI are quoted above each entry.
# Sampling takes the same number of questions from each source, so the number of
# sources decides the mix (e.g. 2 of 3 sources from ARC -> about 2/3 ARC).
SOURCES = {
    # "Grade-school science questions" -- almost word for word ARC's own description
    "high_school_science": [
        lambda c: load_arc("ARC-Challenge", c),
        lambda c: load_arc("ARC-Easy", c),
        lambda c: load_mmlu_redux_many(["high_school_physics", "high_school_chemistry",
                                        "high_school_biology"], c),
    ],
    # "University-level questions across academic subjects"
    "university_knowledge": [
        # s=s binds the current subject; a plain lambda would only see the last one
        lambda c, s=s: load_mmlu_redux(s, c) for s in
        ["college_biology", "college_chemistry", "college_physics",
         "college_computer_science", "college_mathematics", "college_medicine",
         "astronomy", "electrical_engineering", "econometrics",
         "jurisprudence", "international_law", "sociology"]
    ],
    "philosophy_psychology": [
        lambda c: load_mmlu_redux("philosophy", c),
        lambda c: load_mmlu_redux("high_school_psychology", c),
        lambda c: load_mmlu_redux("professional_psychology", c),
        lambda c: load_mmlu_redux("logical_fallacies", c),
    ],
    "science_nature": [
        lambda c: load_opentdb(17, c),           # Science & Nature
        lambda c: load_opentdb(27, c),           # Animals
        lambda c: load_sciq(c),
    ],
    "entertainment": [
        lambda c: load_opentdb(11, c),           # Film
        lambda c: load_opentdb(12, c),           # Music
        lambda c: load_opentdb(14, c),           # Television
        lambda c: load_opentdb(26, c),           # Celebrities
    ],
    # "Ancient civilisations, history and politics, built from Wikipedia articles"
    "ancient_history": [
        lambda c: load_mmlu_redux("prehistory", c),
        lambda c: load_opentdb(23, c, max_questions=500, keywords=ANCIENT_KEYWORDS),  # History
        lambda c: load_opentdb(20, c),           # Mythology
        lambda c: load_opentdb(24, c),           # Politics
        lambda c: load_mmlu_redux("high_school_government_and_politics", c),
    ],
}


def sample_category(pools: list[list[dict]], n: int, rng: random.Random) -> list[dict]:
    """Take n questions, as evenly as possible from each source, without duplicate texts."""
    for pool in pools:
        rng.shuffle(pool)
    chosen, seen = [], set()
    while len(chosen) < n and any(pools):
        for pool in pools:  # round-robin over the sources
            while pool:
                q = pool.pop()
                if q["text"].lower() not in seen:
                    seen.add(q["text"].lower())
                    chosen.append(q)
                    break
            if len(chosen) == n:
                break
    return chosen


def main():
    rng = random.Random(SEED)
    dev, test = [], []
    for category, loaders in SOURCES.items():
        pools = []
        for load in loaders:
            pool = load(category)
            print(f"  {category:24s} {pool[0]['source'] if pool else '(empty)':40s} {len(pool):5d} questions")
            pools.append(pool)
        chosen = [shuffle_options(q, rng) for q in sample_category(pools, PER_CATEGORY, rng)]
        half = len(chosen) // 2
        dev += chosen[:half]
        test += chosen[half:]
        print(f"{category}: {len(chosen)} chosen\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, questions in [("dev", dev), ("test", test)]:
        with open(OUT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"wrote {len(questions)} questions to {OUT_DIR / (name + '.jsonl')}")


if __name__ == "__main__":
    main()

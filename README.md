# NLP 2026 – Quiz Agent

An agentic question-answering system that plays the *Million Token Question* quiz
through its API, using open-weights LLMs (<= 14B parameters).

## Structure

```
quiz/
  client.py    talks to the quiz server
  llm/         loads and runs the language model
    base.py    backend-agnostic LLM interface
    vllm_.py   vLLM implementation run on the GPU
  solver.py    decides the answer to a question
  tools.py     functions the LLM can call
  voice.py     speech -> text for voice questions
  play.py      the game loop + logging
data/runs/     logs: one line per question played
main.ipynb     the notebook: explains, calls, shows results
```

## Setup

Credentials go in environment variables (or Colab Secrets), never in the code:
`MTQ_USERNAME`, `MTQ_PASSWORD`.

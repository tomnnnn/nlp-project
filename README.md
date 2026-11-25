# NLP 2026 – Quiz Agent

An agentic question-answering system that plays the *Million Token Question* quiz
through its API, using open-weights LLMs (<= 14B parameters).

## Setup

```
mamba create env -n nlp-project python=3.10
mamba activate nlp-project
pip install -r requirements.txt
```

Credentials go in environment variables (or Colab Secrets):
`MTQ_USERNAME`, `MTQ_PASSWORD`.

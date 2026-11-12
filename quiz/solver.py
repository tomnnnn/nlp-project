"""The solvers: given a question, decide which option to answer, or skip.

Each strategy (e.g. always skip, plain LLM, RAG, agent with tools) is its own class
with the same interface. Solvers use the LLM and tools but know nothing about the server.
"""

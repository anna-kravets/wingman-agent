def run(prompt: str, conversation_id: str | None) -> tuple[str, list[dict]]:
    """Stub Supervisor. No LLM calls yet, no sub-agent dispatch yet.

    Returns (response_text, steps) matching the /api/execute contract.
    Real question-refinement + dispatch + date-sync logic lands in Phase 2.
    """
    response_text = "Supervisor stub — agent logic not implemented yet."
    steps: list[dict] = []
    return response_text, steps

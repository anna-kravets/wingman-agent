def run(prompt: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Stub Supervisor. No LLM calls yet, no sub-agent dispatch yet.

    `history` is the prior turns of this conversation ([{"prompt", "response"}]),
    already loaded by the caller. The Supervisor decides how much of it to put
    into a prompt — keep that bounded, history grows every turn and every token
    costs against the shared budget.

    Returns (response_text, steps) matching the /api/execute contract.
    Real question-refinement + dispatch + date-sync logic lands in Phase 2.
    """
    response_text = "Supervisor stub — agent logic not implemented yet."
    steps: list[dict] = []
    return response_text, steps

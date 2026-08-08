"""DocumentationAgent — what the airline actually owes this passenger.

Pattern: reflection loop, draft -> self-critique -> refine (`docs/PROJECT_PLAN.md` §1).
Three LLM calls, so three `steps` entries — this is the agent the `(payload, steps)`
contract exists for.

This is the differentiator. Regulations (EU 261, US DOT) are the baseline everyone
knows; the airline's own Contract of Carriage is the binding document nobody reads and
often grants more. Every entitlement must cite where it came from, clause-level where
possible, or the leverage is invisible to the passenger.

STUB — see `IS_STUB`. Person C replaces the body of `run()` with the real loop over
the Pinecone index (`lib/rag/`, not written yet), retrieval filtered by airline so
irrelevant carriers' clauses do not bloat the context.
"""

from lib.steps import make_step

IS_STUB = True

MODULE = "DocumentationAgent"

DRAFT_SYSTEM_PROMPT = """You determine what an airline owes a passenger whose flight was disrupted.

You are given retrieved passages from two kinds of source: air passenger rights regulations
(EU 261/2004, US DOT) and the airline's own Contract of Carriage. Use both. The regulations are the
floor; the Contract of Carriage is a binding commitment by that specific airline and frequently
grants more — rebooking on another carrier, hotel and ground transport, meal vouchers. A passenger
who only hears about EU 261 is being under-served.

Only state an entitlement you can point to in the passages you were given. If the passages do not
settle something, say so in "caveats" rather than guessing.

Return a JSON object only, no prose:
{"regulation": "EU 261/2004" | "US DOT" | "none",
 "entitlements": [{"kind": "rebooking" | "hotel" | "meals" | "cash_compensation" | "other",
                   "summary", "source", "confidence": "high" | "medium" | "low"}],
 "next_actions": [str], "caveats": [str]}

"source" cites the clause, e.g. "EU 261 Art. 8(1)(b)" or "LH Conditions of Carriage 9.1.2".
"next_actions" are what the passenger should do now, in order, and who to say it to.
"""

CRITIQUE_SYSTEM_PROMPT = """You review a draft entitlements assessment against the source passages it was based on.

Be specific and adversarial. Look for:
- entitlements asserted with no supporting passage, or a source that does not say what is claimed
- the Contract of Carriage ignored where it grants more than the regulation
- the disruption's cause misapplied — extraordinary circumstances remove cash compensation but not
  the duty of care, and those are routinely conflated
- amounts or distance bands stated without the passage that fixes them
- next actions that are vague ("contact the airline") rather than actionable

Return a JSON object only: {"problems": [str], "verdict": "revise" | "accept"}
If there is nothing substantive to fix, say so — do not invent problems to look thorough.
"""

REFINE_SYSTEM_PROMPT = """You revise an entitlements assessment given a critique of it.

Fix what the critique identifies. Remove anything unsupported rather than softening it. Keep the
same JSON schema as the draft and return a JSON object only.
"""


def _user_prompt(request: dict, history: list[dict]) -> str:
    lines = [
        f"Airline: {request.get('airline')}",
        f"Flight: {request.get('flight_number')}",
        f"Route: {request.get('origin')} -> {request.get('destination')}",
        f"What happened: {request.get('disruption')}",
        f"Passengers: {request.get('party_size')}",
        f"Local time now: {request.get('local_now')}",
    ]
    if history:
        lines.append("")
        lines.append("Earlier in this conversation:")
        for turn in history:
            lines.append(f"  passenger: {turn['prompt']}")
            lines.append(f"  you: {turn['response']}")
    return "\n".join(lines)


def run(request: dict, history: list[dict]) -> tuple[dict, list[dict]]:
    """Returns (payload, steps). Three steps: draft, critique, refine."""
    user_prompt = _user_prompt(request, history)

    # Person C: replace everything below with the real loop.
    # Retrieval goes here first (filtered by request["airline"]), then draft ->
    # critique -> refine via lib.llm.call(...).
    draft = {
        "regulation": "EU 261/2004",
        "entitlements": [
            {
                "kind": "rebooking",
                "summary": "STUB DATA — not a real entitlements assessment.",
                "source": "STUB",
                "confidence": "low",
            }
        ],
        "next_actions": ["STUB DATA — replaced by Person C in Phase 2."],
        "caveats": ["STUB DATA — no retrieval was performed."],
    }
    critique = {"problems": ["STUB DATA — no critique was performed."], "verdict": "accept"}

    steps = [
        make_step(MODULE, DRAFT_SYSTEM_PROMPT, user_prompt, draft),
        make_step(MODULE, CRITIQUE_SYSTEM_PROMPT, str(draft), critique),
        make_step(MODULE, REFINE_SYSTEM_PROMPT, str(critique), draft),
    ]
    return draft, steps

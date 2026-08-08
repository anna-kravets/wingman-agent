from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from lib import conversation
from lib.agents import supervisor

app = FastAPI()

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHITECTURE_PNG = REPO_ROOT / "architecture_diagram" / "architecture_diagram.png"
PUBLIC_DIR = REPO_ROOT / "public"

TEAM_INFO = {
    "group_batch_order_number": "2_6",
    "team_name": "Wingman",
    "students": [
        {"name": "Tal Aloni", "email": "tal.aloni@campus.technion.ac.il"},
        {"name": "Gilai Blum", "email": "gilai.blum@campus.technion.ac.il"},
        {"name": "Anna Kravets", "email": "anna.kravets@campus.technion.ac.il"},
    ],
}


AGENT_INFO = {
    "description": (
        "Wingman is a companion for a passenger whose flight has just been delayed, cancelled, "
        "or overbooked. The passenger describes what happened in plain words and gets back one "
        "recovery plan covering all three things a disruption dumps on them at once: a way "
        "onward, somewhere to sleep on the nights the new itinerary strands them, and a "
        "plain-language account of what the airline owes them.\n\n"
        "It is a multi-agent system. A Supervisor reads the request and, because a stressed "
        "passenger's first message is usually missing details, asks for what it needs before "
        "dispatching. It then coordinates a FlightAgent (onward options), an AccommodationAgent "
        "(stays, timed to whatever flight was found, so the nights actually match), and a "
        "DocumentationAgent, which works out entitlements from EU 261 and US DOT rules together "
        "with the airline's own Contract of Carriage — the binding, airline-specific document "
        "that almost nobody reads and that often grants more than the regulations alone.\n\n"
        "What it CAN do: interpret an underspecified account of a disruption and ask what's "
        "missing; propose onward flights; propose accommodation matched to those dates; explain "
        "entitlements (rebooking, hotel, meals, cash compensation) in plain language with the "
        "next actions to take and who to take them with; and keep the conversation open "
        "afterwards, so the passenger can compare options or ask about the terms of a specific "
        "one ('does the 09:40 take my ski bag?', 'are meals included?').\n\n"
        "What it CANNOT do (constraints): it never books, holds, cancels, or pays for anything — "
        "every option it returns is a recommendation the passenger acts on themselves. It has no "
        "access to the passenger's booking, airline account, loyalty status, or payment details, "
        "and asks for no credentials. It does not file compensation claims on their behalf. It is "
        "not legal advice: it reads the published rules and the airline's own contract and "
        "explains what they say, but the airline's decision on a specific case is the airline's. "
        "Outside a travel-disruption request, it says so and does not improvise."
    ),
    "purpose": (
        "Get a stranded passenger from 'my flight just got cancelled' to a complete, actionable "
        "recovery plan in one conversation — including the compensation and care they are owed "
        "but would otherwise never claim, because the paperwork is written to be too tiring to "
        "fight at 1 AM in a terminal."
    ),
    "prompt_template": {
        "template": (
            "Airline and flight number: <e.g. LH318>\n"
            "Route: <from -> to, e.g. TLV -> FRA>\n"
            "What happened: <delayed | cancelled | denied boarding> and when you were told\n"
            "Where you are now: <the airport you are stuck at>\n"
            "Who is travelling: <e.g. 1 adult, or 2 adults + 1 child>\n"
            "When you need to arrive by: <date and time, or 'as soon as possible'>\n"
            "What you need: <a new flight | somewhere to sleep | what I am owed | all of it>\n\n"
            "Anything you do not know, leave out or write 'not sure' — Wingman will ask."
        ),
        "example": (
            "Airline and flight number: LH318\n"
            "Route: TLV -> FRA\n"
            "What happened: cancelled, they told us at the gate 20 minutes before boarding\n"
            "Where you are now: TLV\n"
            "Who is travelling: 2 adults\n"
            "When you need to arrive by: tomorrow evening, I have a connection I booked myself\n"
            "What you need: all of it"
        ),
    },
    # BLOCKED until Phase 2/3: `full_response` and `steps` must be captured verbatim from a real
    # /api/execute run. They cannot be written by hand — `steps` has to match the actual LLM calls
    # and the module names in the architecture PNG, and a grader can diff them against a live run.
    # Fill this in as the last integration step, not before the agents exist.
    "prompt_examples": [],
}


@app.get("/api/team_info")
def team_info():
    return TEAM_INFO


@app.get("/api/agent_info")
def agent_info():
    return AGENT_INFO


@app.get("/api/model_architecture")
def model_architecture():
    return FileResponse(ARCHITECTURE_PNG, media_type="image/png")


def _error(message: str) -> dict:
    return {"status": "error", "error": message, "response": None, "steps": []}


@app.post("/api/execute")
async def execute(request: Request):
    try:
        body = await request.json()
    except Exception:
        return _error("Request body must be valid JSON.")

    prompt = body.get("prompt") if isinstance(body, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        return _error('Request body must be a JSON object with a non-empty "prompt" string.')

    conversation_id = body.get("conversation_id")

    try:
        history = conversation.load_history(conversation_id) if conversation_id else []

        response_text, steps = supervisor.run(prompt, history)

        if conversation_id:
            turn = {"prompt": prompt, "response": response_text}
            conversation.save_history(conversation_id, history + [turn])
    except Exception as exc:
        return _error(f"The agent failed while handling the request: {exc}")

    return {
        "status": "ok",
        "error": None,
        "response": response_text,
        "steps": steps,
    }


# Serves the GUI at / for local development. Must stay last: a mount on "/"
# matches everything, and routes are matched in declaration order, so the
# /api/* routes above win. In production this never runs — Vercel serves
# /public from the CDN before a request reaches the function — and the
# directory may not be in the function bundle at all, hence the guard.
if PUBLIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")

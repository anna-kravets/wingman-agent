import hashlib
import logging
import re
import secrets
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from lib import conversation
from lib.agents import supervisor
from lib.prompt_examples import EXAMPLES

app = FastAPI()
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHITECTURE_PNG = REPO_ROOT / "architecture_diagram" / "architecture_diagram.png"
PUBLIC_DIR = REPO_ROOT / "public"
DEVICE_COOKIE = "wingman_device_id"
DEVICE_COOKIE_MAX_AGE = 60 * 60 * 24 * 365
DEVICE_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{43}$")

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
        "DocumentationAgent, which works out entitlements from EU 261, US DOT, and Israeli rules together "
        "with the airline's own Contract of Carriage — the binding, airline-specific document "
        "that almost nobody reads and that often grants more than the regulations alone.\n\n"
        "What it CAN do: interpret an underspecified account of a disruption and ask what's "
        "missing; propose onward flights; propose accommodation matched to those dates; explain "
        "entitlements (rebooking, refunds, hotel, meals, cash compensation) in plain language with the "
        "next actions to take and who to take them with; and keep the conversation open "
        "afterwards, so the passenger can compare options or ask about the terms of a specific "
        "one ('does the 09:40 take my ski bag?', 'are meals included?').\n\n"
        "What is real and what is estimated: the flights it proposes are real scheduled "
        "departures, and the places to stay are real properties with their true distance from the "
        "terminal. Prices, seat availability and fare conditions are not — no free source for them "
        "exists — so every amount and every fare term it states for a flight or a room is an "
        "estimate, labelled as one, to be confirmed with the airline or the property. The "
        "entitlements are the exception: those are read from the regulations and the Contract of "
        "Carriage and cited to the clause.\n\n"
        "What it CANNOT do (constraints): it never books, holds, cancels, or pays for anything — "
        "every option it returns is a recommendation the passenger acts on themselves. It has no "
        "access to the passenger's booking, airline account, loyalty status, or payment details, "
        "and asks for no credentials. It does not file compensation claims on their behalf. It is "
        "not legal advice: it reads the published rules and the airline's own contract and "
        "explains what they say, but the airline's decision on a specific case is the airline's. "
        "If one part of the crew fails, it returns the rest of the plan and says plainly what is "
        "missing rather than withholding the whole answer. Outside a travel-disruption request, "
        "it says so and does not improvise."
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
    # Captured verbatim from three production runs on 14/8/2026 — see lib/prompt_examples.py.
    # They cannot be written by hand: `steps` has to match the actual LLM calls and the module
    # names in the architecture PNG, and a grader can diff them against a live run. Re-capture
    # them whenever a system prompt or the dispatch logic changes.
    "prompt_examples": EXAMPLES,
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


def _device_id(request: Request, response: Response) -> str:
    candidate = request.cookies.get(DEVICE_COOKIE, "")
    device_token = (
        candidate if DEVICE_TOKEN_PATTERN.fullmatch(candidate) else secrets.token_urlsafe(32)
    )
    owner_id = hashlib.sha256(device_token.encode("ascii")).hexdigest()

    forwarded_scheme = request.headers.get("x-forwarded-proto", "").split(",", 1)[0]
    response.set_cookie(
        DEVICE_COOKIE,
        device_token,
        max_age=DEVICE_COOKIE_MAX_AGE,
        httponly=True,
        secure=request.url.scheme == "https" or forwarded_scheme == "https",
        samesite="lax",
        path="/",
    )
    return owner_id


def _conversation_title(prompt: str) -> str:
    clean = " ".join(prompt.split())
    return clean if len(clean) <= 45 else f"{clean[:45].rstrip()}…"


@app.get("/api/conversations")
def conversations(request: Request, response: Response):
    owner_id = _device_id(request, response)
    try:
        items = conversation.list_conversations(owner_id)
    except Exception:
        response.status_code = 503
        return {"status": "error", "error": "Conversation history is temporarily unavailable."}
    return {"status": "ok", "conversations": items}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, request: Request, response: Response):
    owner_id = _device_id(request, response)
    try:
        conversation.delete_conversation(owner_id, conversation_id)
    except Exception:
        response.status_code = 503
        return {"status": "error", "error": "The conversation could not be deleted."}
    return {"status": "ok"}


@app.post("/api/execute")
async def execute(request: Request, response: Response):
    owner_id = _device_id(request, response)
    try:
        body = await request.json()
    except Exception:
        return _error("Request body must be valid JSON.")

    prompt = body.get("prompt") if isinstance(body, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        return _error('Request body must be a JSON object with a non-empty "prompt" string.')

    conversation_id = body.get("conversation_id")

    try:
        history = (
            conversation.load_history(owner_id, conversation_id) if conversation_id else []
        )
    except Exception:
        logger.exception("Conversation history load failed; continuing as a single turn")
        history = []

    try:
        response_text, steps = supervisor.run(prompt, history)
    except Exception as exc:
        return _error(f"The agent failed while handling the request: {exc}")

    if conversation_id:
        turn = {"prompt": prompt, "response": response_text}
        first_turn = history[0] if history and isinstance(history[0], dict) else {}
        first_prompt = first_turn.get("prompt", prompt)
        if not isinstance(first_prompt, str):
            first_prompt = prompt
        try:
            conversation.save_history(
                owner_id,
                conversation_id,
                history + [turn],
                _conversation_title(first_prompt),
            )
        except Exception:
            logger.exception("Conversation history save failed; returning the agent response")

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

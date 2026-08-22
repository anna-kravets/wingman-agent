import hashlib
import logging
import re
import secrets
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from lib import conversation
from lib.agents import supervisor
from api.prompt_examples import PROMPT_EXAMPLES

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
        "Wingman is a disruption companion for passengers whose flight has been delayed, "
        "cancelled, or overbooked. It turns one plain-language account into a coordinated recovery "
        "plan: a practical route onward, accommodation for the nights that route leaves the "
        "passenger stranded, and a source-linked explanation of what the airline may owe.\n\n"
        "Wingman is a multi-agent system. The Supervisor extracts the facts, resolves airport and "
        "timing ambiguities, asks only for details that block the requested help, and coordinates "
        "three specialists. FlightAgent searches real scheduled departures from AeroDataBox. "
        "AccommodationAgent searches real properties from OpenStreetMap and matches the stay dates "
        "to the proposed journey. DocumentationAgent retrieves the relevant airline Contract of "
        "Carriage together with EU/EEA, US, and Israeli passenger-rights material, then uses a "
        "draft, critique, and refinement loop to produce the legal assessment.\n\n"
        "What it CAN do: understand a stressed or incomplete request; accept airport codes or "
        "unambiguous city names; identify conflicting facts; compare verified schedule options; "
        "suggest nearby properties with distance and available contact details; explain rebooking, "
        "refund, care, and possible compensation rights with document and article references; state "
        "important exceptions and evidence gaps; and continue the same conversation so the passenger "
        "can compare options or ask about baggage, meals, or one proposed itinerary. The web interface "
        "also lets the passenger open the legal passages cited in the answer. If one data source or "
        "specialist is unavailable, Wingman keeps any usable results and says what it could not verify.\n\n"
        "Data boundaries: flight schedules and hotel/property records are real search results, but "
        "they are not bookings. Flight prices are estimates, and seat availability, ticket-specific "
        "fare conditions, through-ticket protection, and another airline's acceptance of the ticket "
        "are not verified. Hotel prices are estimates; room availability and meals are unconfirmed. "
        "All options must be confirmed directly with the airline or property before relying on them. "
        "The legal corpus covers the EU/EEA (including Switzerland), the US, Israel, and the supported "
        "airlines' published carriage documents; Wingman identifies routes that extend beyond that "
        "coverage.\n\n"
        "What it CANNOT do (constraints): it does not book, hold, change, cancel, or pay for a flight "
        "or room. It cannot access a passenger's reservation, airline account, loyalty status, payment "
        "details, or live inventory, and it never asks for account credentials. It does not contact "
        "airlines, submit claims, or guarantee eligibility or compensation. Its legal explanation is "
        "information drawn from published sources, not legal advice. For requests outside passenger "
        "flight disruption, it responds briefly that the request is outside its scope."
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
            "Route: <origin to destination; airport codes or city names>\n"
            "Scheduled departure: <date, local time, and time zone if known>\n"
            "What happened: <delayed | cancelled | denied boarding; when and where you learned>\n"
            "Airline explanation: <the reason given, or 'none'>\n"
            "Where you are now: <airport or city>\n"
            "Who is travelling: <number of adults and children; accessibility needs if relevant>\n"
            "What the airline offered: <new flight, refund, voucher, nothing, or 'not sure'>\n"
            "Important constraints: <arrival deadline, checked bags, connections, room needs>\n"
            "What you need: <new flight | accommodation | rights | all of them>\n\n"
            "Include every fact you know. Leave an unknown field out or write 'not sure'; Wingman "
            "will ask only when the missing detail blocks the requested help."
        ),
        "example": (
            "Lufthansa flight LH318 from Tel Aviv to Frankfurt was cancelled at the gate today. "
            "The airline said it was an operational issue. I am travelling with one child, my "
            "checked bags are with the airline, and the next flight they offered is tomorrow "
            "afternoon. I need a hotel tonight, meals, and the earliest reasonable replacement "
            "flight. Please also explain whether I can choose a refund and what compensation may apply."
        ),
    },
    "prompt_examples": PROMPT_EXAMPLES,
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


def _error(message: str, steps: list[dict] | None = None) -> dict:
    return {"status": "error", "error": message, "response": None, "steps": steps or []}


def _local_time(value) -> datetime | None:
    """The passenger's own wall clock, or None.

    Never raises: a device sending a malformed timestamp is not worth a failed turn,
    and the server clock is a working fallback. The offset is dropped rather than
    applied — every consumer of `local_now` wants the wall-clock reading, and
    `flight_agent` subtracts it from a tz-stripped departure.
    """
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).replace(tzinfo=None)
    except ValueError:
        return None


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
    local_time = _local_time(body.get("local_time"))

    try:
        history = (
            conversation.load_history(owner_id, conversation_id) if conversation_id else []
        )
    except Exception:
        logger.exception("Conversation history load failed; continuing as a single turn")
        history = []

    try:
        response_text, steps, results = supervisor.run(prompt, history, local_time=local_time)
    except Exception as exc:
        # The cause is ours, not the passenger's — it goes to the log, not the response.
        # A call that was made and then failed is still a call: exc.steps (lib.llm.LLMError)
        # keeps it in the trace even though the turn as a whole failed.
        logger.exception("The agent failed while handling the request")
        return _error("Something went wrong while working on your plan. Please try again.",
                      getattr(exc, "steps", []))

    if conversation_id:
        turn = {"prompt": prompt, "response": response_text, "results": results}
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
        # Stored separately, and guarded separately: the trace is a debugging and demo
        # affordance, the conversation is the product. `history` here is still the
        # turns that came before, so its length is the new turn's index.
        try:
            conversation.save_steps(owner_id, conversation_id, len(history), steps)
        except Exception:
            logger.exception("Conversation trace save failed; returning the agent response")

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

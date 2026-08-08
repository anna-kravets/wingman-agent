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


@app.get("/api/team_info")
def team_info():
    return TEAM_INFO


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

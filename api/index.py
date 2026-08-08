from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse

from lib import conversation
from lib.agents import supervisor

app = FastAPI()

ARCHITECTURE_PNG = (
    Path(__file__).resolve().parent.parent
    / "architecture_diagram"
    / "architecture_diagram.png"
)

# TODO: fill with real values before submission.
TEAM_INFO = {
    "group_batch_order_number": "TODO_batch#_order#",
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


@app.post("/api/execute")
async def execute(request: Request):
    try:
        body = await request.json()
        prompt = body["prompt"]
        conversation_id = body.get("conversation_id")

        history = conversation.load_history(conversation_id) if conversation_id else []

        response_text, steps = supervisor.run(prompt, conversation_id)

        if conversation_id:
            history.append({"prompt": prompt, "response": response_text})
            conversation.save_history(conversation_id, history)

        return {
            "status": "ok",
            "error": None,
            "response": response_text,
            "steps": steps,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "response": None,
            "steps": [],
        }

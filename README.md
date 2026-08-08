# Wingman


## Run locally

From the repo root:

```bash
py -m venv .venv                  # macOS/Linux: python3 -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.index:app --reload
```

Then open http://127.0.0.1:8000 — the GUI, `/api/team_info`, `/api/model_architecture`
and `/api/execute` all work exactly as they do in production.

Run it from the repo root, not from `api/`, or the `lib` imports won't resolve.

Without a `.env`, conversation history is not persisted and every call behaves as a
single turn — the agent still answers. Copy `.env.example` to `.env` once the Supabase
and LLMod.ai credentials exist.

## Repo layout

Target structure is documented in `docs/PROJECT_PLAN.md` (Phase 0).

## Setup TODOs

- Supabase project — Tal to set up (needed for multi-turn conversation history, see `docs/schema.sql`).
- LLMod.ai API key — waiting on a response from Idan before this can be created.
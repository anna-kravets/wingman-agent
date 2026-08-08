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

## Run the tests

```bash
pytest
```

From the repo root. No API key and no database needed — the suite runs against the
sub-agent stubs, so it works on a fresh clone. Run it before pushing: it is what stops
one person's agent from breaking another's.

## Building an agent

The sub-agents in `lib/agents/` are stubs (`IS_STUB = True`) with the real prompts and
the real signatures already in place. Replace the body of `run()`, drop the flag, and
nothing around them changes. `lib/llm.py` is the only place that talks to LLMod.ai —
call through it and your `steps` entry is built for you. Interface shapes are in
`docs/PROJECT_PLAN.md` §1.

## Repo layout

Target structure is documented in `docs/PROJECT_PLAN.md` (Phase 0).

## Setup TODOs

- LLMod.ai API key — waiting on a response from Idan before this can be created.
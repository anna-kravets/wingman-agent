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

From the repo root. No API key and no database needed: LLM calls go through the
`fake_llm` fixture in `tests/conftest.py`, and `conftest.py` forces
`WINGMAN_LIVE_DATA=0` so the suite can never spend API quota or hit a public
endpoint. Run it before pushing: it is what stops one person's agent from breaking
another's.

### Live data

`FlightAgent` uses AeroDataBox (free plan: **600 units/month, 2 units per call**) and
`AccommodationAgent` uses OpenStreetMap's Overpass API (keyless).

Set `WINGMAN_LIVE_DATA=0` in your local `.env` while developing. Both tools then make no
network requests at all and the agents degrade to reasoning unaided, clearly labelled.

Check remaining quota (costs nothing):

```bash
curl -s -H "x-rapidapi-key: $AERODATABOX_API_KEY" \
     -H "x-rapidapi-host: aerodatabox.p.rapidapi.com" \
     -D - -o /dev/null \
     https://aerodatabox.p.rapidapi.com/subscriptions/balance | grep api-units
```

Regenerate the airport coordinate table (rarely needed — the output is committed):

```bash
python scripts/build_airports.py
```

## Building an agent

`DocumentationAgent` in `lib/agents/` is still a stub (`IS_STUB = True`) with the real
prompts and signatures already in place. Replace the body of `run()`, drop the flag, and
nothing around it changes. `lib/llm.py` is the only place that talks to LLMod.ai — call
through it and your `steps` entry is built for you. `FlightAgent` and
`AccommodationAgent` are built and are worth reading as worked examples: each fetches
candidates through a `lib/tools/` module, injects them into its prompt, makes exactly one
LLM call, and validates what comes back. Interface shapes are in `docs/PROJECT_PLAN.md` §1.

## Repo layout

Target structure is documented in `docs/PROJECT_PLAN.md` (Phase 0).

## Setup TODOs

- LLMod.ai API key — waiting on a response from Idan before this can be created.
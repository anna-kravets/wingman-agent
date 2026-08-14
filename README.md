# Wingman

Wingman is a multi-agent assistant for disrupted airline passengers. `FlightAgent`,
`AccommodationAgent`, and `DocumentationAgent` are all built: real flight/hotel lookups, and
airline- and jurisdiction-aware legal retrieval with a grounded draft → critique → refine
answer flow.

## Environment setup

Create and activate a new Conda environment, then install the project dependencies:

```powershell
conda create --name wingman-agent python=3.12 -y
conda activate wingman-agent
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure `.env` without committing it:

```dotenv
SUPABASE_URL=
SUPABASE_KEY=
LLMOD_API_KEY=
LLMOD_API_BASE=https://api.llmod.ai/v1
PINECONE_API_KEY=
PINECONE_INDEX_NAME=wingman-legal-docs
```

`LLMOD_API_KEY`, `LLMOD_API_BASE`, and `PINECONE_API_KEY` are required by the DocumentationAgent.
Supabase is optional for local single-turn use; it is used for persisted conversation history.
The code uses the `LLMOD_*` names directly and has no alternate model-credential fallback.

## Pinecone index prerequisite

This integration repository reads an existing Pinecone index; it does not bundle the legal source
corpora or re-ingest them at application startup. The default index is `wingman-legal-docs`.

The companion corpus project must populate these namespaces:

- Conditions of Carriage: `coc-aa`, `coc-ac`, `coc-af`, `coc-al`, `coc-ba`, `coc-dl`, `coc-ek`,
  `coc-fr`, `coc-lh`, `coc-ly`, `coc-rk`, `coc-rr`, and `coc-ua`.
- Passenger rights: `rights-eu`, `rights-il`, and `rights-us`.

Runtime searches pin Conditions of Carriage to chunker version `legal-markdown-v2` and passenger
rights to `legal-markdown-v3-provisions`. Rebuilding the index therefore requires the companion
corpus ingestion command to preserve those versions and the `document_id`, `provision_id`,
`section_path`, citation, and source metadata expected by `lib/rag/retrieve.py`.

## Run locally

```powershell
uvicorn api.index:app --reload
```

Open <http://127.0.0.1:8000>. Run from the repository root so `lib` imports resolve correctly.
The API endpoints are:

- `GET /api/team_info`
- `GET /api/agent_info`
- `GET /api/model_architecture`
- `POST /api/execute` with `{"prompt": "...", "conversation_id": "optional"}`

## DocumentationAgent behavior

The request router selects the airline namespace and every applicable legal namespace from the
airline, route, and disruption. Retrieval then:

1. Embeds the passenger question and focused legal-topic queries in one primary batch.
2. Searches each namespace independently so one source cannot crowd out the others.
3. Checks event-specific provision coverage deterministically.
4. Re-queries missing topics with `document_id` and `provision_id` filters.
5. Uses one alternate-query embedding batch only if primary recovery remains incomplete.

The agent can include up to 20 complete chunks. `MAX_CONTEXT_CHARACTERS` is `None`, because the
ingestion pipeline already bounds individual chunks and cutting a legal clause mid-passage is less
safe than supplying the complete selected evidence.

The answer flow is one draft call followed by one critique/refinement round. Configuration lives in
`lib/agents/documentation_agent.py`:

- `MAX_REFLECTION_ROUNDS = 1`
- Draft, critique, and refinement completion ceilings: 10,000 tokens each
- Increasing reflection rounds adds two chat-model calls per round

Completion settings are ceilings, not output targets. Each call is recorded in `steps[]`, and live
test artifacts also retain token usage and completion finish reasons.

## No-cost tests

```powershell
pytest -q
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
network requests at all, and the search agents then refuse rather than invent options:
they make no model call and tell the passenger nothing could be verified.

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

## Explicitly approved live tests

`lib/llm.py` is the only place that talks to LLMod.ai — call through it and your `steps`
entry is built for you. `FlightAgent` and `AccommodationAgent` are worth reading as worked
examples: each fetches candidates through a `lib/tools/` module, injects them into its
prompt, makes exactly one LLM call, and validates what comes back. Interface shapes are in
`docs/PROJECT_PLAN.md` §1.

Live scripts are budget-gated and refuse to run without confirmation flags. A full one-round
DocumentationAgent test uses one primary embedding request, possibly one fallback embedding
request, and three chat calls.

Retrieval only:

```powershell
python scripts/run_retrieval_live.py `
  --confirm-paid-embedding-calls `
  --scenario ua-us-tarmac-delay `
  --output live-test-output/ua-retrieval.json
```

Full DocumentationAgent:

```powershell
python scripts/run_documentation_agent_live.py `
  --confirm-paid-calls `
  --reflection-rounds 1 `
  --scenario ua-us-tarmac-delay `
  --output live-test-output/ua-agent.json
```

Evaluate a saved artifact without API or database access:

```powershell
python scripts/evaluate_documentation_artifact.py `
  --scenario ua-us-tarmac-delay `
  --artifact live-test-output/ua-agent.json
```

The evaluator checks run status, reflection structure, retrieval coverage, citation-label integrity,
and critique-audit shape. Expected and forbidden legal findings remain an explicit human-review
rubric because a deterministic string checker cannot reliably judge full legal semantics.

`live-test-output/` is intentionally ignored by Git because artifacts contain large prompts and
retrieved source excerpts.

## Repository layout

- `api/`: FastAPI/Vercel integration
- `lib/agents/documentation_agent.py`: grounded response and reflection loop
- `lib/rag/`: routing, direct LLMod embeddings, Pinecone retrieval, and coverage validation
- `evals/`: regression and held-out scenarios plus artifact checks
- `scripts/`: explicitly gated live-test runners
- `tests/`: no-cost unit and integration tests
- `docs/PROJECT_PLAN.md`: architecture and project status

## Final verification checklist

1. Confirm the required `.env` variables are configured in the target environment.
2. Confirm the existing Pinecone index contains the expected namespaces and chunker versions.
3. Run `pytest -q` without live confirmation flags.
4. Start the API and inspect `/api/team_info`, `/api/model_architecture`, and `/api/execute`.
5. Make live model calls only after explicit budget approval.

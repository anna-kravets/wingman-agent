# Wingman — Project Plan

Working plan for the graded agent build (distinct from the pitch deck in `../../project presentation`).
Full grading spec: [`../guidelines/project guidelines.md`](../guidelines/project%20guidelines.md). Product/architecture context: [`../CLAUDE.md`](../CLAUDE.md).

**Deadline:** 23/8/2026
**Plan created:** 7/8/2026 (16 days out)
**Budget:** $13 total, shared LLMod.ai key across the group
**Team size:** 3

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done

---

## 0. Locked decisions

Resolved at kickoff meeting (7/8/2026) — full log in §6.

- [x] **Module names** (final, used verbatim in code, architecture PNG, and `/api/agent_info` text):
  `Supervisor`, `FlightAgent`, `AccommodationAgent`, `DocumentationAgent`
- [x] **Stack:** Python serverless functions on Vercel.
- [x] **Multi-turn support:** required, not optional. Conversational follow-up (compare flight options, check terms) is the core justification for agents over plain API calls — must work end-to-end.
- [x] **Task tracker:** none — coordinate directly.
- [x] **Comms channel:** none — coordinate directly.
- [ ] **Flights/hotels data source:** mock/synthetic data vs. a free-tier real API — still open, see §7. No booking site may be named in any user-facing text (carried over from the pitch deck constraint).

---

## 1. Team split

Split by agent boundary so each person owns a clean interface and integration happens once at the end of Phase 2, not continuously — needed given mismatched schedules.

| Owner                              | Scope                                                                                                                                                                                                               |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Person A — Orchestration/Infra** | `Supervisor` agent, GUI, `/api/team_info`, `/api/agent_info`, `/api/model_architecture`, Vercel deploy + env setup, multi-turn conversation state (Supabase-backed). Becomes integration owner in Phase 3.          |
| **Person B — Search agents**       | `FlightAgent` + `AccommodationAgent` (same role+one-shot pattern, reused twice). Owns the flights/hotels data-source decision + implementation.                                                                     |
| **Person C — Rights/RAG**          | `DocumentationAgent` reflection loop (draft → self-critique → refine) + full data pipeline (collect CoC/regs, chunk, embed, Pinecone ingestion) + Supabase schema. Heaviest, most novel piece — the differentiator. |

**Interface contract** (agree in Phase 0, then build independently against it): every agent function returns
```json
{ "module": "...", "prompt": { "system_prompt": "...", "user_prompt": "..." }, "response": {} }
```
and the Supervisor concatenates these into `/api/execute`'s `steps[]` in call order.

---

## 2. Phases

### Phase 0 — Lock decisions + scaffold (day 0–1)
- [x] Resolve all items in §0 (except flights/hotels data source — still open, §7)
- [x] Repo structure agreed and scaffolded:
  ```
  /api
    index.py                        → single FastAPI entrypoint (see note below); all endpoints as routes
  /lib
    agents/
      supervisor.py
      flight_agent.py
      accommodation_agent.py
      documentation_agent.py
    rag/
      ingest.py                     → chunk + embed + upsert to Pinecone
      retrieve.py                   → airline-filtered retrieval
    steps.py                        → shared step-logger util
    llm.py                          → LLMod.ai client wrapper
    conversation.py                 → multi-turn state (Supabase-backed)
  /data                              → raw regs + CoC source docs
  /public
    index.html                      → GUI: textarea, Run Agent button, response + steps trace, conversation history
  /docs                              → this file, decisions log
  requirements.txt
  vercel.json                       → routes /api/*.py, static-serves /public
  CLAUDE.md                         → root, shared context for all Claude Code sessions
  .env.example
  ```
- [x] Framework choice within Python (Flask/FastAPI vs. bare handler functions) and exact `vercel.json` routing — **decided (revised after live deploy testing, see decisions log): single FastAPI `app` in `api/index.py`**, not one bare-handler file per endpoint. Vercel's current Python runtime failed to zero-config-detect multiple `handler`-per-file functions under `/api` (confirmed empirically against a real deploy, contradicting what the general docs implied); a single ASGI entrypoint is the platform's actual supported path today. `vercel.json` sets `outputDirectory: public` + `functions["api/index.py"].maxDuration: 60`.
- [x] Step-object schema agreed and written down (§1 contract)

### Phase 1 — Skeleton (day 1–4)
- [x] Repo scaffold (Python + `vercel.json`) done; empty shell deployed to Vercel — preview and **production** both live and verified (GUI, `/api/team_info`, `/api/execute` all return 200, no auth guard). Production: https://project-eight-chi-97.vercel.app
- [~] `GET /api/team_info` implemented, returns placeholder data — needs real team name / batch-order# / student names+emails before submission
- [x] `POST /api/execute` stub — real response shape, empty `steps`, wired to a `Supervisor` stub, accepts optional `conversation_id` for multi-turn
- [x] GUI shell — textarea, "Run Agent" button, response display, steps trace display, conversation history display, no auth (`conversation_id` generated client-side, since the locked `/api/execute` response shape has no room to hand one back)
- [x] Shared step-logger utility so all agents log consistently (`lib/steps.py`)
- [~] Supabase table for conversation history (`conversation_id`, turn history) — schema written (`docs/schema.sql`) and `lib/conversation.py` client built; actual Supabase project not yet created, so untested end-to-end

### Phase 2 — Agent builds, parallel (day 4–10)
- [ ] **Supervisor:** question-refinement system prompt (elicit missing details from stressed/underspecified first message), dispatch logic, date-sync between Flight/Accommodation calls
- [ ] **FlightAgent:** role prompt + one-shot example of a flight-search result, conversational follow-up (compare options, check terms)
- [ ] **AccommodationAgent:** role prompt + one-shot example of a booking result, matched to the dates FlightAgent returns, conversational follow-up (e.g. meals included)
- [ ] **Data pipeline** (start early — blocks DocumentationAgent): collect EU261 + US DOT text, 3–5 airline CoCs, chunk, embed via `text-embedding-3-small` (1536-dim), upsert to Pinecone with an `airline` metadata field for filtered retrieval
- [ ] **DocumentationAgent:** reflection loop (draft → self-critique → refine) over the RAG index, scoped/filtered by airline

### Phase 3 — Integration (day 10–13)
- [ ] Supervisor calls all 3 sub-agents, aggregates response, produces full end-to-end `steps` trace
- [ ] `GET /api/agent_info` — `description`, `purpose`, `prompt_template`, `prompt_examples[]`. Capture `full_response`/`steps` from an actual run — cannot be fabricated.
- [ ] `GET /api/model_architecture` — PNG diagram, module names matching `steps` and docs exactly. Clarity over polish; doesn't need pitch-deck production values.
- [ ] Error path: `{ "status": "error", "error": "...", "response": null, "steps": [] }`
- [ ] Multi-turn end-to-end: GUI sends prior `conversation_id`, Supervisor loads history from Supabase, agents see prior context on follow-ups (e.g. "can I take my ski bag on the 09:40?")
- [ ] Budget check: log $ per call, confirm total dev+test spend stays well under $13. Cap the reflection loop to one critique pass — Supervisor + FlightAgent + AccommodationAgent + DocumentationAgent(draft+critique+refine) is already ~5–7 LLM calls per user turn.

### Phase 4 — Test + submit (day 13–16, includes buffer)
- [ ] Local + Vercel prod parity test (explicit spec requirement)
- [ ] Timing check: full `/api/execute` completes well under the 300s serverless limit
- [ ] Module-name consistency pass: diagram vs. `steps` vs. `agent_info` text
- [ ] Submit — exact format:
  ```
  Vercel URL: {url}
  GitHub Repo URL: {url}
  ```

---

## 3. Technical resources — owner assigned per item at kickoff

| Resource | Notes | Owner |
|---|---|---|
| GitHub repo | Private, all 3 as collaborators. Not created yet — `.gitignore` review pending, then push. | |
| Root `CLAUDE.md` | Carry over the existing project-level one so every teammate's Claude Code session shares context; updated for Python stack + required multi-turn. | |
| Vercel project | Linked (`anna-kravets-projects/project`), GitHub repo connected for auto-deploy. Prod: https://project-eight-chi-97.vercel.app. Still need: invite other 2 teammates, set env vars. | Person A |
| LLMod.ai API key | One member creates it — shared across the whole group automatically per spec. Goes in Vercel env + local `.env` (never committed; add `.env.example`). | |
| Supabase project | Schema needed for: conversation history (multi-turn, required) + execution logs. | |
| Pinecone index | Dimension must match `text-embedding-3-small` (1536). Decide namespace/metadata-filter strategy (`airline` field) before DocumentationAgent work starts. | |

No dedicated task tracker or comms channel — team coordinates directly (§6).

---

## 4. Data sources (RAG corpus — not yet collected)

- **Airline Contracts/Conditions of Carriage** — est. ~30MB. Only American Airlines was sourced during Assignment 2; the rest need collecting.
- **Passenger rights regulations** — EU 261/2004, US DOT — est. ~5MB. Baseline entitlements, independent of airline policy.

Retrieval must be scoped/filtered by airline to avoid irrelevant CoC clauses bloating context (budget discipline).

---

## 5. Grading-critical constraints (do not drift from these)

- Endpoint names, methods, and response shapes must match the spec exactly — see [`../guidelines/project guidelines.md`](../guidelines/project%20guidelines.md) §2.
- Module names must be identical across the architecture PNG, every `steps[].module` entry, and any written description.
- `/api/execute` request: `{ "prompt": "..." }`. Response: exactly `{ "status": "ok"|"error", "error": string|null, "response": string|null, "steps": [] }`.
- Every LLM call → one `steps` entry, in call order, with `module`, `prompt.system_prompt`, `prompt.user_prompt`, `response`.
- GUI at `/`, no auth guards of any kind.
- 300s hard timeout per Vercel call — `/api/execute` (including reflection loop) must finish well inside that.
- $13 total budget for the whole project, shared key.
- Never name a specific booking/flight-search website in any user-facing text.
- Multi-turn is optional per spec but locked as required for this team (§0) — treat it as a hard constraint, not a stretch goal.

---

## 6. Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 7/8/2026 | Module names locked: `Supervisor`, `FlightAgent`, `AccommodationAgent`, `DocumentationAgent` | Must stay identical across code, architecture PNG, `agent_info` — locking early avoids rework. |
| 7/8/2026 | Stack: Python serverless functions on Vercel (not Next.js/TS) | Team decision at kickoff. |
| 7/8/2026 | Multi-turn support: required | Conversational follow-up is the stated justification for using agents over plain API calls — team wants it demonstrated, not just claimed. |
| 7/8/2026 | No dedicated task tracker | Team decision — coordinate directly given the small team size. |
| 7/8/2026 | No dedicated comms channel | Team decision — coordinate directly. |
| 7/8/2026 | Python runtime: **superseded same day** — see next row | Initial plan was bare stdlib `BaseHTTPRequestHandler` per `api/*.py`, no framework. Reverted after live deploy testing. |
| 7/8/2026 | Python runtime (final): single FastAPI `app` in `api/index.py`, all endpoints as routes on it | Live `vercel deploy` rejected multiple separate `handler`-per-file functions under `/api` ("No python entrypoint found in default locations") even though general Vercel docs describe that as supported — empirically only a single recognized entrypoint (`app.py`/`index.py`/etc., or `tool.vercel.entrypoint`) builds reliably right now. FastAPI is the documented, currently-working pattern; adds one small dependency. |
| 7/8/2026 | `conversation_id` is generated client-side (GUI) and passed on every `/api/execute` call, not returned by the server | The response shape is locked to exactly `{status, error, response, steps}` — no field to hand a server-generated id back through. |

---

## 7. Open risks / questions

- No real booking API is named in the spec — flights/hotels data strategy needs a firm answer before Person B can start Phase 2 in earnest. Still open.
- Reflection loop cost: needs a hard iteration cap decided during DocumentationAgent design, not discovered during a budget overrun.
- Multi-turn is now required (not optional): needs a concrete state model — `conversation_id` + history in Supabase (Person A owns this) — decided once at Phase 1, not re-litigated per PR. Adds a call to load/persist history on every `/api/execute` turn — factor into the $13 budget.
- Python-on-Vercel routing approach (single WSGI/ASGI app + `vercel.json` rewrites vs. one function per `/api/*.py` file) not yet chosen — resolve when scaffolding in Phase 1.

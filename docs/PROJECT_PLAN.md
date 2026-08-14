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
- [x] **Multi-turn support:** required, not optional. Conversational follow-up is the core justification for agents over plain API calls — must work end-to-end. Note the follow-up is answered by *Wingman as a whole*, not by one agent: comparing flight options is `FlightAgent`, while checking the terms of one is `DocumentationAgent` reading the Contract of Carriage (§6).
- [x] **Task tracker:** none — coordinate directly.
- [x] **Comms channel:** none — coordinate directly.
- [x] **Flights/hotels data source:** **AeroDataBox** (flights) + **OpenStreetMap Overpass** (hotels), decided 9/8/2026 — see §6. Real schedules and real hotels; prices and availability have no free source and are labelled estimates. No booking site may be named in any user-facing text (carried over from the pitch deck constraint).

---

## 1. Team split

Split by agent boundary so each person owns a clean interface and integration happens once at the end of Phase 2, not continuously — needed given mismatched schedules.

| Owner                              | Scope                                                                                                                                                                                                               |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Person A — Orchestration/Infra** | `Supervisor` agent, GUI, `/api/team_info`, `/api/agent_info`, `/api/model_architecture`, Vercel deploy + env setup, multi-turn conversation state (Supabase-backed). Becomes integration owner in Phase 3.          |
| **Person B — Search agents**       | `FlightAgent` + `AccommodationAgent` (same role+one-shot pattern, reused twice). Owns the flights/hotels data-source decision + implementation.                                                                     |
| **Person C — Rights/RAG**          | `DocumentationAgent` reflection loop (draft → self-critique → refine) + full data pipeline (collect CoC/regs, chunk, embed, Pinecone ingestion) + Supabase schema. Heaviest, most novel piece — the differentiator. |

**Step object** — one per LLM call, built by `lib/steps.make_step()` (you get this for free by
calling through `lib/llm.call()`, don't hand-build it):
```json
{ "module": "...", "prompt": { "system_prompt": "...", "user_prompt": "..." }, "response": {} }
```
The Supervisor concatenates these into `/api/execute`'s `steps[]` in call order.

**Interface contract** (revised 8/8/2026 — supersedes "every agent function returns a step object";
that shape could not express DocumentationAgent, whose reflection loop is three LLM calls and
therefore three steps). Every agent returns a **`(payload, steps)`** pair: `payload` is the
structured result the Supervisor consumes, `steps` is every step that agent produced, in call
order.

```python
supervisor.run(prompt: str, history: list[dict])            -> tuple[str, list[dict]]   # (response_text, steps)
flight_agent.run(request: dict, history: list[dict])        -> tuple[dict, list[dict]]  # (payload, steps)
accommodation_agent.run(request, stay_window, history)      -> tuple[dict, list[dict]]
documentation_agent.run(request: dict, history: list[dict]) -> tuple[dict, list[dict]]
```

`history` is the conversation's prior turns, `[{"prompt": ..., "response": ...}, ...]`, oldest
first, already loaded by `api/index.py` — agent code never touches Supabase. The Supervisor decides
how much of it reaches a prompt (§7).

`request` is what the Supervisor's question-refinement pass extracts from the passenger's message:
```python
{"airline", "flight_number", "origin", "destination",
 "disruption": "delayed" | "cancelled" | "denied_boarding",
 "stranded_at", "party_size", "arrive_by": iso8601 | None,
 "needs": ["flight", "stay", "rights"], "local_now": iso8601}
```

**Payloads** — the shapes the Supervisor reads. `depart`/`arrive` are ISO 8601 local times and are
load-bearing: the date sync derives the hotel nights from the chosen flight's departure.
```python
# FlightAgent
{"options": [{"id", "airline", "flight_number", "origin", "destination",
              "depart", "arrive", "stops", "fare_conditions", "notes"}],
 "recommended_id": "F1"}

# AccommodationAgent
{"options": [{"id", "name", "area", "check_in", "check_out", "nights",
              "price_estimate", "meals_included", "notes"}],
 "recommended_id": "H1"}

# DocumentationAgent
{"regulation": "EU 261/2004" | "US DOT" | "Israel Aviation Services Law" | "multiple" | "none",
 "entitlements": [{"kind": "rebooking" | "refund" | "hotel" | "meals" | "cash_compensation" | "other",
                   "summary", "source", "confidence": "high" | "medium" | "low"}],
 "next_actions": [str], "caveats": [str]}
```
`source` on each entitlement is what makes the Contract of Carriage differentiator visible in the
output — cite the clause, not just the regulation.

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
                                      (served by the CDN in prod; mounted by the app for local dev)
  /architecture_diagram              → SVG source + the PNG served by /api/model_architecture
  /docs                              → this file, decisions log
  requirements.txt
  vercel.json                       → single FastAPI entrypoint, static-serves /public
  CLAUDE.md                         → root, shared context for all Claude Code sessions
  .env.example
  ```
- [x] Framework choice within Python (Flask/FastAPI vs. bare handler functions) and exact `vercel.json` routing — **decided (revised after live deploy testing, see decisions log): single FastAPI `app` in `api/index.py`**, not one bare-handler file per endpoint. Vercel's current Python runtime failed to zero-config-detect multiple `handler`-per-file functions under `/api` (confirmed empirically against a real deploy, contradicting what the general docs implied); a single ASGI entrypoint is the platform's actual supported path today. `vercel.json` sets `outputDirectory: public` + `functions["api/index.py"]` with `maxDuration: 300` (the platform max, headroom for the reflection loop) and `includeFiles: "architecture_diagram/**"` so the PNG ships inside the function bundle.
- [x] Step-object schema agreed and written down (§1 contract)

### Phase 1 — Skeleton (day 1–4)
- [x] Repo scaffold (Python + `vercel.json`) done; empty shell deployed to Vercel — preview and **production** both live and verified (GUI, `/api/team_info`, `/api/execute` all return 200, no auth guard). Production: https://project-eight-chi-97.vercel.app
- [x] `GET /api/team_info` implemented with real data (`2_6`, team `Wingman`, all 3 students) — verified in prod
- [x] `POST /api/execute` stub — real response shape, empty `steps`, wired to a `Supervisor` stub, accepts optional `conversation_id` for multi-turn
- [x] GUI shell — modern chat, response/steps display and confirmed deletion, no auth (`conversation_id` generated client-side; an HTTP-only anonymous-device cookie scopes server history and restores the device's conversation list)
- [x] Shared step-logger utility so all agents log consistently (`lib/steps.py`)
- [x] Supabase table for owner-scoped conversation history (`owner_id`, `conversation_id`, title, turn history, timestamps) — migration applied to the production Supabase project 14/8/2026 (schema, primary key, RLS verified per `docs/SUPABASE_HANDOFF_ANNA.md`); prod smoke test (`GET`/`GET`/`DELETE` on `/api/conversations`) confirmed real reads/writes against Supabase, not the best-effort no-op path

### Phase 2 — Agent builds, parallel (day 4–10)

**Harness is in place (8/8/2026).** `lib/llm.py` is the single choke point for LLM calls — call
through it and the `steps` entry is built for you.

**Updated 9/8/2026:** `FlightAgent` and `AccommodationAgent` are built and no longer stubs. Only
`DocumentationAgent` still carries `IS_STUB = True`, with the real prompts, signature and payload
shape in place. To build it for real: replace the body of `run()`, drop the flag — nothing around
it changes. The two finished agents are worth reading as worked examples of the shape: fetch
candidates through a `lib/tools/` module, inject them into the prompt, make exactly one LLM call,
validate what comes back. `pytest` covers the whole path and needs no key: LLM calls go through
the `fake_llm` fixture in `tests/conftest.py`, and `conftest.py` forces `WINGMAN_LIVE_DATA=0` so
the suite can never spend API quota.

**Updated 11/8/2026:** `DocumentationAgent` is built too now (grounded retrieval + one reflection
round, see the Phase 2 checklist below) — no agent is stubbed anymore.

**Updated 14/8/2026:** the Supervisor's own two calls are real as well, so `/api/execute` now
needs an `LLMOD_API_KEY` to answer at all — the refinement pass is the first thing that runs, and
without a key it fails before any agent is dispatched. `tests/test_execute.py` and
`tests/test_supervisor.py` both run against the `fake_llm` fixture in `tests/conftest.py`, which
is what keeps the suite keyless and free.

- [x] **Supervisor:** dispatch, date-sync, history cap and partial-failure policy built and tested. `_extract_request` (question refinement) and `_compose` (writing the plan) are real `lib.llm.call` calls as of 14/8/2026 — no stubbed code is left in the repo. Follow-up turns narrow `needs` so a one-line question dispatches one agent, not three (§6).
  - [x] **History cap decided** (see §7): last `HISTORY_TURNS = 6` turns, trimmed in the Supervisor.
  - [x] **Partial-failure policy:** one sub-agent raising must not lose the rest of the plan; the passenger is told what is missing. A failed flight search skips accommodation rather than guessing the nights.
- [x] **FlightAgent:** real departures from AeroDataBox, filtered to the route and trimmed in `lib/tools/flights.py` (118KB/188 departures → ~630 bytes); role+one-shot prompt now selects from verified candidates. Scope narrowed: baggage/fare/entitlement questions defer to `DocumentationAgent` (§6).
- [x] **AccommodationAgent:** real hotels from OpenStreetMap with exact distance from the terminal, for the nights the Supervisor derives from the chosen flight. Prices are estimates and meals are unconfirmed unless the data says otherwise.
- [x] **Data pipeline:** collected EU, US, and Israeli passenger-rights material plus 13 airline/operator CoC namespaces; section-aware chunking; embedded with `text-embedding-3-small` (1536-dim); uploaded to Pinecone.
- [x] **DocumentationAgent:** namespace-scoped Pinecone retrieval plus one draft / critique / refine round through `lib/llm.py`. Retrieval balances namespaces, requires event-specific legal bundles using stable `provision_id` metadata, and re-queries missing provisions against the relevant primary documents. The base and primary-recovery queries are embedded in one batch; a second batch of alternate deterministic queries runs only if coverage remains incomplete.
- [x] **DocumentationAgent evaluation:** two frozen regression cases plus seven held-out cases spanning cancellation, denied boarding, tarmac delay, mixed jurisdictions, route direction, and exception facts. Reflection audits enumerate rule branches and conditions; a no-cost artifact checker validates structure and leaves semantic expected/forbidden findings for human review.

### Phase 3 — Integration (day 10–13)
- [x] Supervisor calls all 3 sub-agents, aggregates response, produces full end-to-end `steps` trace — covered by `tests/` and verified live on 14/8/2026: a first turn ran `Supervisor · FlightAgent · DocumentationAgent ×3 · Supervisor` in ~50s, and a flight follow-up on the same conversation ran `Supervisor · FlightAgent · Supervisor`.
- [x] **LLMod.ai endpoint verified with live tests.** Chat and embedding requests use direct `httpx` clients configured only through `LLMOD_API_KEY` and `LLMOD_API_BASE`.
- [~] `GET /api/agent_info` — route live; `description`, `purpose` and `prompt_template` written (they describe the product, not the code). **Still blocked:**
  - [ ] `prompt_examples[]` — currently `[]`. `full_response` and `steps` must be captured verbatim from a real `/api/execute` run: `steps` has to match the actual LLM calls and the module names in the architecture PNG, and a grader can diff them against a live run. Fill in as the last integration step.
  - [ ] Revisit the `description` now the data source is decided (§6). It says Wingman *proposes* options and books nothing, which is still true. It must additionally state that **flight schedules and hotels are real, while prices and fare conditions are estimates** — no free source of fares or seat availability exists.
- [x] `GET /api/model_architecture` — PNG served with `Content-Type: image/png`, `includeFiles` set in `vercel.json` so it ships in the function bundle; verified in prod. Diagram labels match the locked module names.
- [x] Error path: `{ "status": "error", "error": "...", "response": null, "steps": [] }` — input validated before dispatch, unexpected failures wrapped in a readable sentence; verified in prod.
- [~] Multi-turn end-to-end: GUI sends `conversation_id`; an HTTP-only anonymous-device cookie scopes list/load/save/delete operations; the route passes owned history to `supervisor.run(prompt, history)`. Plumbing and ownership behavior are tested against a fake store; **the production Supabase connection itself is now verified** (migration applied 14/8/2026, `/api/conversations` list/delete confirmed live in prod). Still open: a real multi-turn conversation (3 turns, one `conversation_id`, through the GUI) hasn't been run against prod yet — tracked as a separate P2 item. Persistence still no-ops when Supabase is absent.
- [ ] Budget check: log $ per call, confirm total dev+test spend stays well under $13. Cap the reflection loop to one critique pass — Supervisor + FlightAgent + AccommodationAgent + DocumentationAgent(draft+critique+refine) is already ~5–7 LLM calls per user turn.

### Phase 4 — Test + submit (day 13–16, includes buffer)
- [~] Local + Vercel prod parity test (explicit spec requirement) — local dev now works (`uvicorn api.index:app --reload`, see README) and the scaffold was checked endpoint-by-endpoint against prod on 8/8/2026. Re-run once the agents are in.
- [ ] Timing check: full `/api/execute` completes well under the 300s serverless limit
- [ ] Module-name consistency pass: diagram vs. `steps` vs. `agent_info` text
- [ ] **Make the GitHub repo public** (currently private — `api.github.com/repos/anna-kravets/wingman-agent` returns 404 to anyone not on the repo). The submission is a bare repo URL with no invite step, so a private repo means the grader opens a 404. Do this before submitting, and re-check it from a logged-out browser afterwards.
- [ ] Submit — exact format:
  ```
  Vercel URL: {url}
  GitHub Repo URL: {url}
  ```

---

## 3. Technical resources — owner assigned per item at kickoff

| Resource | Notes | Owner |
|---|---|---|
| GitHub repo | Created and pushed: https://github.com/anna-kravets/wingman-agent, connected to Vercel for auto-deploy on `main`. Currently **private** — must be made public before submission (Phase 4). | |
| Root `CLAUDE.md` | Carry over the existing project-level one so every teammate's Claude Code session shares context; updated for Python stack + required multi-turn. | |
| Vercel project | Linked (project `wingman-agent`), GitHub repo connected for auto-deploy. Prod: https://project-eight-chi-97.vercel.app. Still need: invite other 2 teammates, set env vars. | Person A |
| LLMod.ai API key | One member creates it — shared across the whole group automatically per spec. Goes in Vercel env + local `.env` (never committed; add `.env.example`). | |
| Supabase project | Schema needed for: conversation history (multi-turn, required) + execution logs. | |
| Pinecone index | Dimension must match `text-embedding-3-small` (1536). Decide namespace/metadata-filter strategy (`airline` field) before DocumentationAgent work starts. | |

No dedicated task tracker or comms channel — team coordinates directly (§6).

---

## 4. Data sources (RAG corpus — not yet collected)

- **Airline Contracts/Conditions of Carriage** — collected for 10 major airlines plus separate Ryanair-group operating codes where applicable.
- **Passenger rights regulations** — EU 261/2004, US DOT, and Israeli Aviation Services Law, including relevant guidance and amendments.

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

## 5a. To settle together once the first real agent lands

Judgment calls taken unilaterally to keep the build moving, that the team should actually agree on.
Deliberately deferred: they are easier to argue about with a working agent in front of you than in
the abstract. Raise these at the first integration checkpoint and log whatever is decided in §6.

- **Does a partial failure return `status: "ok"` or `status: "error"`?** Currently **`ok`**: if
  DocumentationAgent falls over but FlightAgent and AccommodationAgent succeeded, the passenger gets
  the flight and the bed, plus a line saying the entitlements could not be worked out. The reasoning
  is that they got a usable answer, and erroring would throw away good work over one failed leg. The
  competing reading is that any internal failure is an error and `status` should say so — the
  guidelines do not address partial success either way. If the team prefers the strict reading it is
  a small change in one place (`supervisor.run`'s failure handling plus `api/index.py`). Whichever
  we pick, `/api/agent_info`'s description should match it.

---

## 6. Decisions log

> Adding a row that **reverses** an earlier decision? This log is not the only place decisions are
> stated — grep `../CLAUDE.md` and the rest of this file for the old shape and fix those too, in the
> same commit. See `CLAUDE.md` §7.

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
| 8/8/2026 | Conversation history is loaded by `api/index.py` and passed in as a list: `supervisor.run(prompt, history)`. Agents never touch the store. | Considered the alternative of giving the Supervisor the `conversation_id` and letting it query Supabase itself. Rejected: agent code would then depend on a database that does not exist yet, blocking Person B and Person C from testing against real history. With the list passed in, `run()` is a pure function testable with a literal. The one real argument for the other shape — fetching selectively to control cost — is a *shaping* decision the Supervisor can still make on a list it was handed, with no DB access needed. Whichever side loads it, only one may: the earlier code did both, which is how the loaded history ended up discarded. |
| 8/8/2026 | A failed LLM call still produces a `steps` entry. `lib.llm.LLMError` carries it on `.steps`; the Supervisor appends whatever the exception brings. An agent that makes several calls attaches the ones that already succeeded (`exc.steps = steps + exc.steps`) | The spec says `steps[]` describes **every** LLM call the agent made, in order — a call that was sent and then failed or came back unparseable is still a call. Raising bare dropped it from the trace entirely, and would have silently lost the first two calls of a DocumentationAgent reflection loop that died on the third. |
| 8/8/2026 | **Interface contract revised** (§1): agents return `(payload, steps)`, not a single step object | The original shape assumed one LLM call per agent. DocumentationAgent's reflection loop is three calls and must emit three `steps` entries, so a single step object could not express it — and the spec requires every LLM call to appear. Splitting the structured result (`payload`, for the Supervisor) from the trace (`steps`, for `/api/execute`) fixes it for every agent, not just that one. Checked `CLAUDE.md` for the old wording per §7; it never restated the contract, so no change was needed there. |
| 8/8/2026 | Conversation state is best-effort: persistence no-ops when `SUPABASE_URL` and a Supabase server key are unset | The GUI sends a `conversation_id` on every call, so an unconfigured (or briefly unreachable) Supabase previously turned every GUI request into `status: "error"`. Degrading to single-turn keeps the agent usable; production uses `SUPABASE_SERVICE_ROLE_KEY`, with legacy `SUPABASE_KEY` accepted temporarily. Load/save failures are isolated from the agent response. |
| 12/8/2026 | No-login conversation ownership uses a one-year HTTP-only anonymous-device cookie | A client-generated `conversation_id` identifies a chat but not its owner. The cookie adds same-browser continuity and owner-scoped list/load/save/delete operations without adding accounts. It intentionally does not promise cross-browser identity. Browser `localStorage` remains a fast local cache; Supabase is the durable source once configured. |
| 9/8/2026 | Flights from **AeroDataBox**, hotels from **OpenStreetMap Overpass** | Amadeus Self-Service, the obvious choice, was decommissioned 17/7/2026 and its keys disabled; Kiwi Tequila went invite-only. AeroDataBox has the largest free quota still open to self-signup (600 units/month, 2 units per call); Overpass is keyless, so the hotel half can never be blocked by a quota. |
| 9/8/2026 | Prices and seat availability are **LLM estimates, labelled as such** | No free source for either exists as of 8/2026. `/api/agent_info`'s `description` must say so (Phase 3). |
| 9/8/2026 | A tool/HTTP fetch produces **no `steps[]` entry** | The spec ties `steps[]` to LLM calls, and a step's shape requires `prompt.system_prompt`/`user_prompt`, which an HTTP GET has neither of. The fetched data appears inside the agent's LLM `user_prompt` instead, so the trace still shows exactly what drove the answer. **Applies to Person C's Pinecone retrieval too.** |
| 9/8/2026 | `FlightAgent` **defers baggage/fare/entitlement questions to `DocumentationAgent`** | Schedule data cannot answer what a ticket allows; the Contract of Carriage can, and with a clause citation — which is the project's differentiator. Corrects Phase 3's "ski bag" example and `CLAUDE.md` §3's "check terms", both of which asked FlightAgent for something it has no source for. |
| 9/8/2026 | Data-source failure **degrades to LLM-only, labelled in `notes`** | The demo must survive an exhausted quota or a flaky Overpass during grading (the public instance returned a 504 during development). Mirrors `lib/conversation.py`'s posture on Supabase. |
| 9/8/2026 | Airport coordinates ship as a generated **Python module**, not JSON | Vercel's Python builder traces imports, not data files. A `.json` would need a `vercel.json` `includeFiles` change and would fail *only in production*, silently, if the glob were wrong. |
| 14/8/2026 | The Supervisor **narrows `needs` on follow-up turns** — "anything earlier than the 04:25?" dispatches `FlightAgent` alone | Person B's design doc D6, assigned to Person A. Re-dispatching the crew for a one-line question costs ~7 LLM calls and 2 AeroDataBox units out of a 600-unit month and a $13 project. The narrowing lives in `REFINE_SYSTEM_PROMPT`, not in keyword matching: the model already reads the message, and a keyword list would be the same crude parsing the stub apologised for. A follow-up answerable from the conversation returns an empty `needs` and dispatches nobody — so the refinement gate now only fires when there is actually a crew to block. |
| 14/8/2026 | A failed **composing** call falls back to the flat digest of the crew's results rather than erroring the turn | By the time we compose, up to six LLM calls and two external quotas are already spent. Losing the whole plan to the last call is worse than handing the passenger a mechanical version of it, and it matches the partial-failure policy already applied to the sub-agents. The failed call still reaches `steps[]` on the exception, so the trace stays honest. |
| 14/8/2026 | Production Supabase migration applied and verified live (`docs/SUPABASE_HANDOFF_ANNA.md`); `/api/conversations` list/delete confirmed hitting real Supabase, not the best-effort no-op path | Closes the last item in §1 Phase 1 / Phase 3's multi-turn checklist that was blocked on a real Supabase project. Found and fixed along the way: production `SUPABASE_URL` was set to the Supabase **dashboard page URL** (`https://supabase.com/dashboard/project/<ref>`) instead of the **API host** (`https://<ref>.supabase.co`), so every request 404'd and surfaced as the generic `"temporarily unavailable"`/`"could not be deleted"` 503s. If conversation history ever silently stops persisting again in prod, check that env var first before suspecting RLS or the service-role key. |

---

## 7. Open risks / questions

- ~~No real booking API is named in the spec — flights/hotels data strategy needs a firm answer~~ — **resolved 9/8/2026**: AeroDataBox + OpenStreetMap (§6). The live risk is now **quota**: 600 units/month at 2 units per call (≈150–300 searches) shared across the team and grading. Mitigated by a per-instance cache and `WINGMAN_LIVE_DATA=0` in development. Contingency ladder (documented, not built — Supabase-backed cache, per-teammate keys, Lufthansa Open API) in the design doc §10.
- Reflection loop cost: needs a hard iteration cap decided during DocumentationAgent design, not discovered during a budget overrun.
- **Unbounded conversation history is a budget risk — decide the cap in Phase 2, before the Supervisor prompt is written.** The route hands the Supervisor every prior turn. If the Supervisor pastes all of it into the prompt, turn N carries N−1 turns of text, so the token cost of a conversation grows quadratically with its length, not linearly. That is multiplied again by the ~5–7 LLM calls each turn already makes. A demo where someone asks four or five follow-ups — exactly the conversational-follow-up story that justifies agents over plain API calls — is where this bites, and $13 is shared across the whole team for the whole project. Options: a fixed last-N turns (simplest, start here), or a running summary the Supervisor maintains. The cap belongs in the Supervisor, since it owns what goes into a prompt; the caller just supplies the full list.
- Multi-turn is now required (not optional): needs a concrete state model — `conversation_id` + history in Supabase (Person A owns this) — decided once at Phase 1, not re-litigated per PR. Adds a call to load/persist history on every `/api/execute` turn — factor into the $13 budget.
- ~~Python-on-Vercel routing approach~~ — **resolved 7/8/2026**: single FastAPI ASGI app in `api/index.py`. See the decisions log (§6).

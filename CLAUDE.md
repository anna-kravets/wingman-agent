# CLAUDE.md — Wingman: Airport Disruption Companion Agent (Build)

## 1. Mission

Build and deploy the real **Wingman** agent for the course project (not the pitch deck — that's a separate,
already-finished artifact in `../project presentation`, kept only for the idea and architecture it captures).
Full grading spec lives in [`guidelines/project guidelines.md`](./guidelines/project%20guidelines.md) (converted from the course
PDF) — this file distills the parts that are load-bearing for implementation and adds the product context
from our own Assignment 1/2 submissions. When this file and `project guidelines.md` conflict, the
guidelines file is authoritative (re-check it — it's the graded spec).

**Deadline: 23/8/2026.**

---

## 2. The product (from Assignment 2 — Domain: Airlines)

**Problem:** during flight disruptions (delay / cancellation / involuntary denied boarding), the passenger
is dumped with three jobs at once, standing at a gate with a dying phone: find a new flight, find a bed for
the matching nights, and figure out what the airline actually owes them. Airlines are bound by complex
legal frameworks (government regulations *and* their own Contract of Carriage) that passengers can't
parse in real time — airlines routinely pay the legal minimum or misattribute controllable delays, and
billions in compensation/accommodation go unclaimed because the paperwork is too overwhelming to
fight.

**Target audience:** passengers facing a real-time flight disruption at the airport.

**The insight / differentiator:** everyone knows EU 261 / US DOT / Israeli passenger-rights law. Almost nobody uses the airline-specific
**Contract of Carriage (CoC)** — a dense, binding, airline-specific document that's the real underused
leverage point (e.g. rebooking on a competing airline, hotel vouchers). Turning that into an instant,
plain-language answer is the "aha."

**What the agent does:** the passenger describes what happened in plain words → a supervisor works out
what they actually need (stressed passengers omit details, so it asks refining questions) → dispatches to
specialist agents → returns a tailored recovery plan: alternative flights, matched accommodation for the
right nights, and a plain-language rights breakdown with next actions. The conversation stays open
afterward (compare flight options, check baggage/meal terms, etc.) — this conversational follow-up is the
reason this is agents and not a plain search/API call, and should be visible in `/api/execute` behavior if
the agent supports multi-turn.

## 3. Architecture (from the presentation — keep names consistent everywhere, see §5)

- **Supervisor agent** — entry point. Pattern: **question refinement** (elicits missing details from a
  stressed, underspecified first message before dispatching). Also times the Flight and Accommodation
  agents so accommodation dates match whatever alternative flight was found (**date sync**).
- **Flight agent** — pattern: **role prompt + one-shot example** of a flight-search result. Finds alternative
  flights across airlines, stays conversational for follow-ups (compare options, check terms).
- **Accommodation agent** — same pattern (**role + one-shot**). Books/matches stays for the exact nights
  the new flight leaves the passenger stranded, stays conversational (e.g. meals included).
- **Documentation agent** — pattern: **reflection loop** (draft → self-critique → refine). Produces the
  rights/entitlements breakdown from regulations (EU 261 / US DOT / Israeli law) *and* the airline's Contract of
  Carriage — this is the differentiator, give it the most care.

This is a **high-level** reference, not a finished spec — implementation will need to decide exact module
boundaries, tool interfaces, and how "conversational follow-up" is represented in a stateless
`/api/execute` call (see open question in §6).

## 4. Data sources (from Assignment 2 — not yet collected, still need gathering)

- **Airline Conditions/Contracts of Carriage** — RAG, ~30MB estimated. Only one example file was
  identified during the assignment (American Airlines CoC); the rest need collecting per-airline before
  this is a usable knowledge base.
- **Air passenger rights regulations** (EU 261/2004, US DOT, Israeli Aviation Services Law) — RAG. Baseline
  entitlements, independent of airline policy.

Use Pinecone (see §5) as the vector store for these; keep retrieval scoped/filtered by airline to avoid
irrelevant CoC clauses bloating context (budget discipline, see §7).

## 5. Hard technical constraints (graded — names/shapes must match exactly)

Full detail in `project guidelines.md`; the following are the parts most likely to break grading if drifted
from:

- **`GET /api/team_info`** — student names/emails, `group_batch_order_number`, `team_name`.
- **`GET /api/agent_info`** — `description`, `purpose`, `prompt_template`, `prompt_examples[]` (each with
  `prompt`, `full_response`, `steps`).
- **`GET /api/model_architecture`** — returns a PNG (`Content-Type: image/png`) of the architecture
  diagram. Sub-agent/module names in this diagram **must exactly match** the names used in
  `/api/execute`'s `steps[].module` and in any written descriptions. Module names are locked
  (decided 7/8/2026): `Supervisor`, `FlightAgent`, `AccommodationAgent`, `DocumentationAgent` —
  use them verbatim everywhere, do not re-propose.
- **`POST /api/execute`** — body `{ "prompt": "..." }`. Response must be exactly
  `{ "status": "ok" | "error", "error": string | null, "response": string | null, "steps": [] }`.
  Each entry in `steps` needs `module`, `prompt` (`system_prompt`/`user_prompt` — the guidelines PDF
  is inconsistent about casing here, the worked example uses lowercase `system_prompt`/`user_prompt`,
  follow that), and `response`. Every LLM call the agent makes must produce one step entry, in order —
  this is also our debugging/demo trace, so log everything even during dev.
- **GUI at `/`** — textarea + "Run Agent" button calling `/api/execute`, shows `response` and the full
  `steps` trace, **no auth**. Follow-up/multi-turn support is optional, only build it if the agent actually
  supports it.
- **Models:** `MB5R2CF-azure/gpt-5.4-mini` (text), `MB5R2CF-azure/text-embedding-3-small`
  (embeddings) via **LLMod.ai** — one shared API key per group, **$13 total budget** for the whole
  project. Minimize LLM calls and prompt/context size deliberately, this is a real ceiling, not a suggestion.
- **Stack:** Python serverless functions on Vercel (decided 7/8/2026, see `docs/PROJECT_PLAN.md`).
  Not Next.js/TypeScript. GUI is static HTML/JS served from `/public`, calling the Python API routes.
- **Deploy:** Vercel, root URL serves the GUI. Serverless — **300s hard timeout** per call, so
  `/api/execute` (including any reflection loop iterations) must complete well under that.
- **DBs:** Supabase (primary), Pinecone (vectors/RAG for CoC + regulations).
- **Submission:** Vercel URL + GitHub repo URL, due **23/8/2026**.

## 6. Open questions to resolve before/while building

- **Resolved (7/8/2026, implementation revised 8/8/2026):** Multi-turn is required, not optional
  (see §5). `POST /api/execute` accepts an optional `conversation_id`; **`api/index.py` loads that
  conversation's prior turns and passes them in as a list — `Supervisor.run(prompt, history)` —
  then persists the new turn.** Agent code never touches the store, which keeps it testable
  without Supabase; the Supervisor decides how much of `history` goes into a prompt (cap it, see
  `docs/PROJECT_PLAN.md` §7). Do **not** give the Supervisor the `conversation_id` and have it
  query Supabase itself — that shape was considered and rejected (rationale in the decisions log,
  `docs/PROJECT_PLAN.md` §6). Persistence is best-effort: with no Supabase env vars it no-ops and
  every call behaves as a single turn. The GUI generates the `conversation_id` itself and sends it
  on every call (the locked response shape has no field to return a server-generated one) and
  renders the running history. This shapes the Supervisor's interface — build it from Phase 1,
  don't retrofit.
- Which airlines' CoC documents to actually collect for the RAG corpus (only AA was sourced during
  Assignment 2) — still open.
- Flights/hotels data source (mock vs. free-tier real API) — still open, see `docs/PROJECT_PLAN.md` §7.

## 7. Build discipline

- Every LLM call costs against the $13 budget — batch/combine calls where reasonable, keep prompts
  lean, don't add speculative agent hops.
- Reuse the module names from §5 consistently across code, the architecture PNG, and `agent_info`
  descriptions from the first commit — retrofitting consistency later is wasted effort.
- Verify locally *and* on Vercel before considering something done (guidelines explicitly call out
  dev/prod parity).
- **Reversing an earlier decision? Update every file that states it, in the same commit.**
  `docs/PROJECT_PLAN.md` §6 is the decisions log, but decisions also get restated in their own
  words in this file and across the plan's phase and resource sections. Logging a reversal in only
  one place leaves the others actively instructing the wrong build — on 8/8/2026 this file still
  told teammates to have the Supervisor query Supabase itself, and to "build it from Phase 1, don't
  retrofit", after that exact shape had been rejected. Before committing a reversal, grep both this
  file and `docs/PROJECT_PLAN.md` for the old shape and fix what you find.

# FlightAgent + AccommodationAgent — design

**Owner:** Person B · **Date:** 9/8/2026 · **Branch:** `Flight-and-Accommodation-Agents`
**Context:** [`../../PROJECT_PLAN.md`](../../PROJECT_PLAN.md) §1 (interface contract), §2 Phase 2.
**Deadline:** 23/8/2026.

Turns the two stub agents into real ones backed by live data, and closes the last open
decision in `PROJECT_PLAN.md` §0 — the flights/hotels data source.

---

## 1. Scope

In scope: `lib/agents/flight_agent.py`, `lib/agents/accommodation_agent.py`, the data-fetch
tools they call, and the doc updates the decisions force.

Out of scope: the Supervisor, the RAG corpus, `/api/*` routes, the GUI. One shared edit to
`tests/test_supervisor.py` is unavoidable (§8) and is flagged in the PR.

Unchanged by this work: agent signatures, both payload schemas, the `(payload, steps)`
contract, the date sync, and the four module names on the architecture PNG.

---

## 2. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Flights from **AeroDataBox** (RapidAPI free plan); hotels from **OpenStreetMap Overpass** | Amadeus Self-Service — the obvious choice — was decommissioned 17/7/2026 and its keys disabled; Kiwi Tequila went invite-only. AeroDataBox has the largest free quota still open to self-signup; Overpass is keyless, so the hotel half can never be blocked by a quota or an expired key. |
| D2 | No free source of **prices or seat availability** exists in 2026 | Consequence, not a choice. `price_estimate` and `fare_conditions` are LLM-reasoned and must read as estimates. `/api/agent_info`'s description must say schedules and hotels are real while prices are estimated (`PROJECT_PLAN.md` Phase 3 already flags this). |
| ~~D3~~ | ~~Data-source failure **degrades to LLM-only, clearly labelled**~~ — **REVERSED 10/8/2026, see D3a** | Original rationale: the demo must survive an exhausted quota during grading, mirroring `lib/conversation.py`'s posture on Supabase. |
| D3a | Data-source failure **refuses and says so**: no candidates ⇒ no LLM call, no options, and the passenger is told plainly | Live validation found the model refuses to invent here anyway — the system prompt's "choose only from that list" outranks a user-prompt instruction to improvise — so D3 never actually worked. It should not be forced: a fabricated flight number for a real airline is *actionable*, and a stressed passenger can go and ask for it at the desk. That is precisely the hallucination this project positions itself against, and a "illustrative" label does not survive a panicked reader. Refusing also costs one LLM call less per degraded turn. Evidence: `docs/search-agents-capabilities.md`. |
| D4 | A tool fetch produces **no `steps[]` entry** | The spec ties `steps[]` to LLM calls, and a step requires `prompt.system_prompt`/`user_prompt` — an HTTP GET has neither. The fetched data appears verbatim inside the agent's LLM `user_prompt`, so the trace still shows exactly what drove the answer. One step per agent, matching the diagram. |
| D5 | FlightAgent **defers baggage/fare/entitlement questions to DocumentationAgent** | AeroDataBox serves schedules, not carriage terms. Those live in the Contract of Carriage — Person C's corpus — and come back clause-cited, which is the project's differentiator. Forces a correction to `PROJECT_PLAN.md` and `CLAUDE.md` (§9). |
| D6 | Quota guarded by a **process-local cache + `WINGMAN_LIVE_DATA` kill-switch**; separately ask Person A to narrow `needs` on follow-up turns | 600 units/month at 2 units per call is ~300 calls, shared across three developers plus grading. The cache is contained in Person B's files and needs nobody's schedule; the `needs` narrowing is the deeper fix and is raised as an integration item, not a blocker. |
| D7 | **Direct flights only**, widening the time window when results are thin | The endpoint returns departures, not itineraries: it cannot build TLV→VIE→FRA, and verifying a second leg costs another 2 units. Every option returned is a real, verified flight. Stated as a limitation in the prompt and in `agent_info`. |
| D9 | **Non-bookable flights are filtered in the tool**, not left to the model: a deny-list of statuses (`Departed`, `Canceled`, `Arrived`, `EnRoute`, `Approaching`, `Diverted`, `GateClosed`, …) drops candidates before the prompt is built | Live validation on 14/8/2026 caught the model recommending a flight already marked `Departed`. It flagged that status in one run and silently ignored it in the next, from identical data — so it is a good reasoner and an unreliable filter. Whether an option is catchable at all determines whether it is an option, so it belongs in code. Deny-list rather than allow-list, so an unfamiliar status AeroDataBox adds later still reaches the passenger instead of emptying the list and triggering a false refusal. |
| D8 | `meals_included` stays **boolean**; uncertainty goes in `notes` | OSM almost never knows (1 of 20 TLV hotels carried any classification tag). Widening the field to `null` would change a locked schema and Person A's `_compose`. Set it from an OSM tag when present, else `false`, and say in `notes` that meals were not confirmed. |

---

## 3. Verified facts

Measured against the live endpoints on 9/8/2026, not assumed. These drive §4–§6.

**AeroDataBox** — `GET /flights/airports/iata/{IATA}/{from}/{to}`, headers `x-rapidapi-key`,
`x-rapidapi-host`:

- Free plan: **600 API units/month**, and separately 2400 requests/month.
  **One call costs 2 units** → 300 calls/month for the whole team. A search is 1–2 calls
  (§5), so **150–300 searches per month**.
- A per-second rate limit exists on the BASIC plan and is easy to trip with back-to-back
  calls; it returns HTTP 429. Requests must be spaced/retried.
- The time range is capped at **12 hours** per call.
- Future dates work. A TLV window for the next day returned **188 departures / 118 KB**,
  of which **3 were TLV→FRA**. Filtering and trimming in the tool takes that to **630 bytes**
  — a ~188× reduction, and the single most budget-relevant step in this work.
- Timestamps are `"2026-08-10 06:05+03:00"` — **space-separated, not ISO 8601 `T`**.
- Useful fields per departure: `number`, `airline.{name,iata}`, `arrival.airport.iata`,
  `departure.scheduledTime.local`, `arrival.scheduledTime.local`, `status`,
  `aircraft.model`, `departure.terminal`.
- `/subscriptions/balance` reports remaining units. **Correction (10/8/2026): it appears to
  consume ~2 units itself** — the counter dropped 592→590 across a session whose only other
  calls were to keyless Overpass. Measure consumption by differencing around a run rather
  than polling the endpoint casually.

**Overpass** — `POST https://overpass-api.de/api/interpreter`, keyless, `User-Agent` required:

- A 12 km radius around TLV returned **20 named hotels / 6.9 KB**; 8 trimmed rows are 905 bytes.
- Metadata is sparse: **1 of 20** had a `stars` tag, nearly none had a website or address.
  OSM reliably supplies name, coordinates and therefore exact distance — and little else.
- Names are frequently in the local language; prefer the `name:en` tag, fall back to `name`.

---

## 4. Modules

Three new files under `lib/tools/`, all pure data-fetch — **no LLM calls inside them**:

```python
# lib/tools/airports.py — static, committed, no network
lookup(iata: str) -> dict | None          # {"iata","name","city","country","lat","lon"}

# lib/tools/flights.py
search(origin: str, destination: str, after: datetime) -> list[dict]
# [{"flight","airline","airline_iata","origin","destination",
#   "depart","arrive","status","aircraft","terminal"}]  — ISO 8601 times

# lib/tools/hotels.py
search(iata: str, radius_km: float = 12) -> list[dict]
# [{"name","distance_km","stars","breakfast","area"}]  — sorted by distance
```

`airports.py` is backed by a trimmed IATA→coordinate table derived from OurAirports
(public domain), committed to the repo. Chosen over geocoding because it is deterministic,
instant, offline, and costs no quota — which also makes the hotel tests real.

Each agent's `run()` becomes:

```
fetch candidates via tool   (D4: no step)
  → build user_prompt with a candidates block
  → one llm.call(..., expect_json=True)      (the agent's single step)
  → validate (§6)
  → return (payload, [step])
```

---

## 5. Tool behaviour

**`flights.search`**

1. Returns `[]` immediately when `WINGMAN_LIVE_DATA` is falsey or the key is unset.
   **Unset means live**, so production needs no extra configuration; the pytest fixture
   and the documented local-dev setup both set it to `0` so routine work never touches
   the quota.
2. Cache key `(origin, destination, date)`, process-local, **no TTL** — a serverless
   instance is short-lived, so staleness is bounded by the instance's own lifetime, and
   flight schedules do not move meaningfully inside it.
3. Requests a 12-hour window from `after` (a timezone-aware local time at the origin
   airport, derived from `request["local_now"]`). If fewer than 3 candidates match the
   destination, requests one more 12-hour window — **at most 2 calls (4 units) per search** (D7).
4. Filters to `arrival.airport.iata == destination`, projects to the fields in §4, and
   normalises both timestamps to ISO 8601 with `T`.
5. On HTTP 429, retries once after a short pause; on any other failure returns `[]`
   rather than raising. The agent then refuses and says so (D3a) rather than inventing.

**`hotels.search`**

1. Resolves the airport coordinate via `airports.lookup`; returns `[]` if unknown.
2. Overpass query for `tourism=hotel` nodes and ways within the radius.
3. Prefers `name:en`, drops unnamed entries, computes great-circle distance from the
   airport, sorts by distance, caps the list (~8) to keep the prompt small.
4. Cache key `(iata, radius)`. Failure returns `[]`, same degradation path.

---

## 6. Prompts and validation

**Prompt changes.** The existing one-shot examples teach the model to *invent* options.
With grounding they must teach it to *select from and explain* the candidates supplied.
The graded pattern stays **role prompt + one-shot example** — only the example's content
changes. Both prompts additionally state:

- Never return an option that is not in the candidate list.
- Never state baggage, fare or entitlement rules; defer to the entitlements section (D5).
- `price_estimate` / `fare_conditions` must read as estimates, never as quoted prices (D2).
- FlightAgent: direct flights only; if none suits, say so rather than inventing a connection (D7).
- AccommodationAgent: state in `notes` when meals were not confirmed (D8).

**Validation before returning** — the Supervisor calls
`datetime.fromisoformat(option["depart"])` at `supervisor.py:97`, so a malformed time costs
both the flight *and* the hotel:

- required keys present; `depart`/`arrive` and `check_in`/`check_out` parse as ISO 8601;
- `recommended_id` names an option that exists, else fall back to the first;
- malformed options are dropped; if none survive, raise `LLMError` with the step attached
  so the failed call still reaches the trace.

---

## 7. Degradation

Two states, never a silent third:

| State | Trigger | Behaviour |
|---|---|---|
| **live** | candidates fetched | Options are real, verified flights/hotels |
| **refused** | fetch failed, quota gone, or `WINGMAN_LIVE_DATA=0` | **No LLM call at all.** The agent raises with a plain-language reason and the passenger is told nothing could be verified and where to look instead (D3a) |

**Known consequence:** once `IS_STUB` is dropped, having no `LLMOD_API_KEY` stops producing
a plausible fake plan and starts producing *"Could not complete: onward flights"* through the
Supervisor's partial-failure path. That is correct — both stub files state in their own
docstrings that shipping the fiction is unacceptable — but the deployed demo visibly degrades
until the LLMod key exists. Tell the team before they see it.

---

## 8. Testing

Everything runs with **no keys of any kind**.

- **Tool tests** — monkeypatch `httpx`, replaying committed fixtures captured from the real
  AeroDataBox and Overpass responses (the pattern `tests/test_llm.py` already uses). Cover:
  route filtering, the 118 KB→630 B trim, timestamp normalisation, the 429 retry, the
  window widening, cache hits, the kill-switch, `name:en` preference, distance sorting.
- **Agent tests** — monkeypatch `lib.llm.call`. Cover: candidates reach `user_prompt`,
  validation drops malformed options, an all-malformed response raises with its step intact,
  a run with no candidates refuses without calling the model, and the step's `module`
  matches the diagram.
- **Shared edit** — `tests/test_supervisor.py` asserts against stub payloads and will fail
  once `run()` needs a key. Fixed by an autouse fixture that fakes `lib.llm.call`, which
  Person C will need for the RAG agent too. It is Person A's file: flagged in the PR
  description and raised with them directly.

---

## 9. Doc updates (same commit as the code)

`CLAUDE.md` §7 requires a reversed decision to be corrected everywhere it is stated.

- `PROJECT_PLAN.md` §0 — close the flights/hotels data-source item with D1.
- `PROJECT_PLAN.md` §6 — log D1–D8, including the Amadeus shutdown as the reason.
- `PROJECT_PLAN.md` §7 — drop the resolved data-source risk.
- `PROJECT_PLAN.md` Phase 2/3 — mark both agents done; fix the *"can I take my ski bag on
  the 09:40?"* example at line 161, which D5 makes wrong; note the `agent_info` wording D2 requires.
- `CLAUDE.md` §3 — FlightAgent's "check terms" remit, corrected per D5.
- `CLAUDE.md` §4/§6 — record the data sources and close the open question.
- `.env.example` — add `AERODATABOX_API_KEY`, `AERODATABOX_API_HOST`, `WINGMAN_LIVE_DATA`.
- `README.md` — local dev note: `WINGMAN_LIVE_DATA=0` keeps development off the quota.

No new dependencies: `httpx` is already in `requirements.txt`.

---

## 10. Risks

- **Quota is the tightest external limit in the project.** 150–300 searches/month total. The
  cache and kill-switch mitigate it; the `needs` narrowing (D6) is the real fix and depends
  on Person A.

  **Contingency if it bites** (decided 9/8/2026: documented, not built — 596 units remain
  untouched, and the kill-switch keeps routine development off the meter entirely).
  In rough order of value for effort:

  1. **Persist the cache in Supabase** rather than in process memory. Most of the burn is
     re-running the same few demo routes, so this drops repeated demo and grading runs to
     zero units. Blocked on Supabase actually being configured.
  2. **Per-teammate keys.** Each of the three registers their own free AeroDataBox key and
     the tool rotates across whichever are configured — ~1800 units, degrading cleanly to
     one key. Legitimate because they are three separate people on a shared project;
     **one person opening multiple accounts to farm free tiers breaks RapidAPI's terms and
     must not be done.**
  3. **Lufthansa Open API** (`developer.lufthansa.com`) as a second source: free
     self-service, **1000 calls/hour**, includes schedules — but Lufthansa Group only
     (LH, OS, LX, SN). Covers the team's own LH318 TLV→FRA demo route, where the live data
     showed LH687 and OS84. Costs a second response shape, fixtures and tests.
  4. **api.market** distributes AeroDataBox with a separate free allocation.
- **D5 leaves terms questions unanswered until Person C's corpus lands.** Until then a
  baggage follow-up gets a deferral rather than an answer. Deliberate — a deferral is
  correct where an invented allowance is not.
- **OSM coverage varies by city.** Twenty hotels near TLV is comfortable; a smaller airport
  may return few or none, which lands in the refusal path (D3a).
- **AeroDataBox response shape is now verified, but only for one route.** Other airports may
  carry sparser fields; the projection in §4 must tolerate missing keys.

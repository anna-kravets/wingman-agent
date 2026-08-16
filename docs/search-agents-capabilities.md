# FlightAgent + AccommodationAgent — what they can and cannot do

**Validated:** 14/8/2026 (exhaustive run: 16 scenarios) · **Model:** `MB5R2CF-azure/gpt-5.4-mini`
**Owner:** Person B · **Branch:** `validate-search-agents`

Written from a real run against a real model, not from design intent. Every capability below
names the scenario that proves it. A claim with no scenario behind it is not in this document.

**Re-run it yourself:**

```powershell
python scripts/run_search_agents_live.py --confirm-paid-calls --all --output live-test-output/final.json
python scripts/evaluate_search_artifact.py --artifact live-test-output/final.json
```

The evaluator needs no keys and no database — it reads the saved artifact. Scenarios live in
`evals/search_agent_cases.json`; the checks in `evals/search_artifact.py`.

**Measured cost of the whole validation** (two full runs plus three retries):

| | |
|---|---|
| Chat calls, final run | **14** (12 scenarios; the three refusal scenarios cost zero) |
| Tokens, final run | 18,218 prompt + 7,689 completion |
| AeroDataBox units, whole exercise | **~54 of 600** (590 → 536 remaining) |
| Failing checks | **0 of 57** across 16 scenarios |
| Slowest scenario | **16.8s** against Vercel's 300s ceiling |
| Exhaustive run cost | 23 LLM calls, 32,935 + 15,458 tokens, **54 API units** |

---

## 1. Verified capabilities

| Capability | Evidence |
|---|---|
| **Options are real flights, never invented.** Every `flight_number` returned was present in the candidate list the tool supplied. | `grounding_flights` passed on `tlv-fra-cancelled` (2 of 2), `lhr-jfk-cancelled` (3 chosen from 20), `same-day-no-hotel` (3 of 3), `earlier-flight-followup`, `compare-options` |
| **Hotels are real properties**, chosen from OpenStreetMap results with exact distances. | `grounding_hotels` passed on `tlv-fra-cancelled` and `overnight-one-night` (8 of 8) |
| **Cross-airline options** — the Contract of Carriage differentiator. On the demo route it returned Condor 4308 (05:15) *and* Lufthansa 687 (16:30), not just the carrier that cancelled. | `tlv-fra-cancelled` |
| **The date sync works.** Hotel `check_in`/`check_out`/`nights` matched the window the Supervisor derived from the chosen flight, every time. | `date_sync` passed on `tlv-fra-cancelled`, `overnight-one-night` |
| **No hotel when none is needed.** A replacement leaving the same day skips `AccommodationAgent` entirely rather than booking a pointless night. | `same-day-no-hotel` |
| **Prices always read as estimates.** e.g. *"Roughly EUR 110-160 for the night (estimate - not a quoted price)"*. | `price_honesty` passed on `tlv-fra-cancelled`, `overnight-one-night`, `price-question` |
| **FlightAgent never quotes a fare**, even when asked "how much will this cost me?". | `no_asserted_fare` passed on `price-question` |
| **Meals are never asserted.** `meals` reads `unknown` with *"Meals were not confirmed - ask at the desk"* whenever OSM has no breakfast tag, which is nearly always. | `meals_honesty` passed on `tlv-fra-cancelled`, `overnight-one-night` |
| **Carriage questions are deferred, not guessed.** A ski-bag question produced *"Whether your ticket can be moved across is set by your Contract of Carriage - see the entitlements section"* — no invented allowance. | `deferral` passed on `baggage-followup` |
| **No booking site is ever named.** | `no_booking_site` passed on all five scenarios that check it |
| **Conversation history reaches both agents** on follow-up turns. | `history_reached_agents` passed on `baggage-followup`, `price-question`, `earlier-flight-followup`, `compare-options` |
| **The trace is well-formed.** One step per agent, only the four locked module names, exact key shapes. | `trace_shape` passed on all 12 |
| **No live data ⇒ refusal, not invention**, with a plain reason and no LLM call at all. | `degraded_refusal` passed on `degraded-flights`, `thin-route`, `sparse-osm-airport` |
| **Thin routes are reached, not refused.** The search now covers 48 hours in 12-hour windows. TLV→NCE went from finding nothing at all to a real flight. | `thin-route`, re-run live after D10 |
| **Accommodation is found even at remote airports.** The radius widens 12→25→40 km, and hostels, guest houses, motels, apartments and resorts count. Ramon/Eilat went from **0 results to 8**. | Live check after D11 |
| **Contact detail reaches the passenger** — phone, website, street address, step-free access, and whether somewhere is a hostel rather than a hotel. | `overnight-one-night`, re-run live after D11 |
| **Flights the passenger cannot catch are never offered** — departed, cancelled, diverted and gate-closed candidates are dropped in the tool. | `no_unusable_status` passed on `same-day-no-hotel` after the D9 fix; guarded by 11 unit tests in `tests/test_tools_flights.py` |

---

## 2. What changed because of this validation

### 2a. A departed flight was recommended to a stranded passenger

`same-day-no-hotel` supplied three candidates, two of them marked `status: Departed`.
Across two runs on **identical data**:

| Run | Behaviour |
|---|---|
| 1 | *"Candidate status is Departed, so it may no longer be available for rebooking"* — and correctly identified the only one still `Expected` |
| 2 | Ignored the field entirely and **recommended a flight that had already left** |

Nothing in code filtered it: `lib/tools/flights.py` passed `status` through, and `_validate`
never looked at it. The model was the only thing standing between the passenger and an
un-catchable flight, and it was right roughly half the time.

**Fixed (D9):** non-bookable statuses are now dropped in the tool before the prompt is built.
The same scenario re-run live afterwards produced **1 candidate, all options catchable**.

The general lesson, worth repeating at the meeting: **a fact that decides whether an option is
usable belongs in the tool; judgement belongs in the prompt.** The model is a good reasoner and
an unreliable filter — this is the clearest evidence in the whole exercise of what these agents
do and do not contribute.

### 2c. The payload was written before we knew how the agents behave

Agreed at the 15/8 meeting and specced separately. Two changes matter beyond the field list:

**The model stopped being asked to transcribe.** It returns a choice and prose; code matches that
back to the source candidate and writes every factual field. An altered departure time went from
unlikely to impossible, and an invented option now matches nothing and cannot survive validation.
The model also emits about five fields per option instead of twelve, which is real money.

**Caveats are generated in code wherever the trigger is a fact** — "only one flight in 48 hours",
"leaves in 40 minutes", "no phone number anywhere". Section 2a is why: a model that notices
something in one run of two is not a safety net.

The two deprecated fields (`area`, `meals_included`) were deleted 15/8/2026, once `_digest`
migrated to `distance_km`/`city` and `meals`.

> **Not live-verified yet.** Everything in §1 was measured against a real model; this refinement
> was not. It is covered by 224 unit tests, including `facts_from_candidates` against synthetic
> artifacts and every caveat rule, but the refined prompts have never met the model. The live run
> was deferred on 15/8: since the Supervisor's seams became real LLM calls, a full 16-scenario run
> costs ~64 calls rather than the ~20 it used to, and it made more sense to spend that once, after
> Person A's digest lands, and prove both halves together. **Treat §2c as designed and unit-tested,
> not as validated.**

### 2b. The degradation behaviour was wrong and has been reversed

The design said that when the data source fails, the agents should fall back to LLM-invented
options labelled "illustrative", so a demo survives an exhausted quota. **That never worked.**
With no candidates the model returned `{"options": []}` in every scenario — its system prompt
says *"choose only from that list"*, and that outranks a user-prompt instruction to improvise.
The passenger got `Could not complete: onward flights (FlightAgent: no option came back with a
usable departure time)` — an internal parser error surfaced as a user-facing message.

It was not forced to comply, because the model's instinct was better than the design. A
fabricated flight number for a real airline is *actionable*: a passenger can walk to the desk
and ask for "LH 999". That is exactly the hallucination this project positions itself against,
and an "illustrative" label does not survive a panicked reader at a gate.

Both agents now refuse **before calling the model**, and say why:

> live flight schedules were not available for this route, so no departure could be verified.
> Nothing was invented - check your airline's app or the departures board for the next flight.

A degraded turn dropped from one LLM call to **zero**. Logged as D3a in the design spec and in
`PROJECT_PLAN.md` §6.

---

## 3. Limitations, by root cause

### No free data source exists (2026)
- **No fares, no prices, no seat availability.** Amadeus Self-Service closed 17/7/2026 and
  nothing free replaced it. `price_estimate` is the model reasoning, always hedged.
- **No baggage or carriage terms.** Schedules cannot answer them. These are deferred to
  `DocumentationAgent` and its Contract of Carriage corpus.
- **No confirmation that a hotel has a room free tonight.** Wingman proposes; it never books.

### The shape of the AeroDataBox API
- **Direct flights only — by choice, not by impossibility.** We could compose a connection from two
  FIDS calls at ~3× the cost. We do not, because two flights that line up on a schedule board are a
  *self-connection* with **no protection**: miss the second leg and nobody owes you anything. For
  someone already disrupted that is worse advice than "no direct flight".
- **12-hour windows** — a hard server cap, confirmed by a `400` on a 24-hour request. A search is
  capped at four of them, so it sees **48 hours** ahead for at most 8 units.
- A route with **no service in that window returns nothing at all**, which lands in the refusal
  path — correct, but it means "no flights found" and "route not served" are indistinguishable
  to the passenger. `thin-route` (TLV→NCE) behaved this way on the day.

### OpenStreetMap sparseness
- Reliable: **name, position, and therefore exact distance to the terminal**.
- Unreliable or absent: star ratings (1 of 20 near TLV), websites, addresses, meals.
- Names are often local-language; the tool prefers `name:en` and falls back.
- ~~**Coverage varies sharply by airport.** Ramon/Eilat returns zero.~~ **Corrected 14/8/2026:**
  that was our 12 km radius, not an OSM gap. Ramon is 19 km from Eilat and has **141 places to
  stay within 40 km**. The radius now widens automatically, and Ramon returns 8.
- Airports absent from the table are unknown: `VDA` (closed) is not in the 3,267-airport
  OurAirports extract, so it cannot be geolocated at all.

### Quota
- 600 units/month, **2 units per call**; a search costs 2 or 4.
- **A route with no results costs the maximum 4 units and caches nothing**, so asking twice
  about an unserved route pays twice. See §6.
- The cache key includes the **hour**, so the same route searched 22:15 and 23:30 pays twice.

### Blocked on Person A, not on the data
- `_compose` is still a deterministic stub, so **a follow-up turn does not read as an answer to
  the question asked.** The agents receive the history and act on it — verified — but the final
  prose is assembled by string concatenation. "Compare the two options for me" returns a plan,
  not a comparison. The multi-turn capability is real at the agent layer and invisible at the
  text layer until `_compose` makes a real LLM call.
- The refusal message currently surfaces as `Could not complete: onward flights (...)`, which
  reads like an error rather than an explanation. The wording inside the parentheses is
  passenger-ready; the framing around it is `_compose`'s.

---

## 4. Scenarios that did not reproduce their intended condition

Recorded honestly rather than papered over — real schedules change daily.

| Scenario | Intended | What happened on 14/8 |
|---|---|---|
| `thin-route` | A route with 1–2 departures, exercising window widening | At 24 hours TLV→NCE returned **zero** and refused. **This is what prompted D10**: at 48 hours it finds a real flight, verified live. |
| `multi-night` | A 2+ night strand | Same route, same outcome: no flight, so no stay window. Multi-night logic is covered deterministically by `tests/test_supervisor.py::test_stay_window_spans_multiple_nights`. |
| `lhr-jfk-cancelled` | Long-haul with an overnight stay | The next departure left the same day, so no hotel was needed — the flight half validated fully, the hotel half skipped. |
| `sparse-osm-airport` | An airport with *few* hotels | ETM had none within 12 km — **which turned out to be our bug, not OSM's gap** (D11). It now returns 8. The scenario still refuses, but for an unrelated reason: ETM→TLV has no flights, so the crew stops at the flight leg before reaching hotels. |

---

## 5. Failure modes — what the passenger actually sees

| When | Result |
|---|---|
| AeroDataBox down, out of quota, or key unset | FlightAgent refuses without calling the model; the passenger is told nothing could be verified and to check the airline's app or the departures board. No hotel is proposed, because the nights cannot be derived without a flight. |
| Overpass down or the airport has no hotels | AccommodationAgent refuses the same way, and points at the airline desk — which is also where a hotel voucher would be issued. |
| Overpass returns 504 (observed during development) | The tool silently retries on a mirror; the passenger sees nothing. |
| The model returns an unusable option | Malformed options are dropped; if none survives, the call still appears in `steps[]` so the trace stays truthful. |
| `LLMOD_API_KEY` unset | Both agents fail through the Supervisor's partial-failure path; the rest of the plan still returns. |

---

## 6. Recommendations (not applied — Person B's call)

1. ~~Cache empty results~~ and ~~widen the cache key to the day~~ - **both done (D12)**, along with
   rejecting impossible routes offline and capping the candidate list.
2. **Prompt injection is still unprobed.** Passenger text reaches the model directly and the
   no-booking-site rule is an explicit project constraint. Deliberately out of scope for this
   round; worth a decision at the meeting.
3. **Foursquare Places** (reportedly 100k free calls/month) would add user ratings and a price
   *band* on top of what OSM gives. It needs someone to create an account, and it still would not
   give a bookable rate — so it buys a nicer hotel card, not a better answer. Deferred.
4. **Outputs lean hard on the one-shot example.** `fare_conditions` came back near-verbatim from
   the prompt's example wording across several scenarios. Correct and consistent, but templated;
   worth deciding whether that reads as reassuring or robotic.
5. **"No flights found" and "route not served" are indistinguishable.** If it matters for the
   demo, the refusal message could say which.

---

## 7. For the team meeting

**Not Person B's, but blocking submission:**

1. ~~**`/api/agent_info` returns `prompt_examples: []`**~~ — **resolved 16/8/2026.** It now returns
   the successful LH318 example with its `prompt`, complete `full_response`, and all eight real
   `steps`.
2. **Its `description` does not say prices are estimates**, which the 9/8 data-source decision
   requires.
3. ~~These two are blocked behind the Supervisor's stubs.~~ **Resolved** — the seams make real LLM
   calls and the captured prompt example is now included in the endpoint.

**Still worth raising:**

4. ~~Narrow `needs` on follow-up turns~~ — **done by Person A.** A flight-only follow-up now
   dispatches only `FlightAgent` instead of the whole crew.
5. **DocumentationAgent must actually answer carriage questions**, because FlightAgent now
   refuses them by design. The deferral is only as good as what it defers to.
6. **The test suite was not hermetic.** `api/index.py` calls `load_dotenv()` at import, so `pytest`
   inherited whatever `.env` was on disk — a green suite went red the moment a real Supabase
   project appeared in one, with no test having changed. The root `conftest.py` now clears every
   credential. Worth knowing because it means a passing suite on your machine did not previously
   guarantee a passing suite on anyone else's.
7. **The committed `docs/schema.sql` and the live Supabase table disagreed** at one point:
   writing a conversation failed on a `NOT NULL owner_id` the schema file did not declare. Person
   A's `supabase/migrations/202608120001_anonymous_conversation_owners.sql` is presumably the
   source of truth now — worth confirming the two are reconciled before submission.

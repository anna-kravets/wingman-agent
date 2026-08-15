# Search-agent payload refinement — design

**Owner:** Person B · **Date:** 15/8/2026 · **Branch:** `refine-search-agent-payloads` (off `validate-search-agents`)
**Deadline:** 23/8/2026 — 8 days.
**Agreed at the team meeting (15/8/2026):** the sub-agent system prompts and payloads were written
at the start of the project, before anyone knew how the agents would really behave. Now that we do,
they should carry everything that matters in each domain, plus a `caveats` field the Supervisor can
act on.

**Depends on Person A**, who is rewriting `supervisor._digest` to consume every field the sub-agents
return, and to act on `caveats`. This spec is the contract she implements against.

---

## 1. Why now

Live validation (`docs/search-agents-capabilities.md`) established two things that make the current
schema wrong rather than merely thin:

1. **The Supervisor discards most of the payload.** `_digest` forwards the *recommended option only*,
   and of that only a handful of fields. `price_estimate`, `notes`, `rebooking` and every alternative
   option never reach the passenger. All the honesty work — hedged prices, "call to confirm the
   room", "meals were not confirmed" — dies in the digest.
2. **The model is an unreliable filter.** Given identical candidate data it flagged a `Departed`
   flight in one run and recommended it in the next. Anything that must reliably happen cannot
   depend on the model noticing.

This is also the cheapest moment the schema will ever change: Person A is rewriting the consumer
this week. A rename after submission costs far more.

---

## 2. Decisions

| # | Decision | Rationale |
|---|---|---|
| P1 | **The model picks and explains; code fills the facts.** Each returned option is matched back to its source candidate and every factual field overwritten from it | Today the prompt asks the model to reproduce times and flight numbers exactly and pleads "never alter a time" — a transcription task guarded by a request. Matching back makes an altered time *impossible* rather than unlikely, makes an invented option unable to survive validation at all, and cuts the model's output from ~12 fields per option to ~5, which is real money on a $13 budget. |
| P2 | `caveats: [str]` at payload level on both agents | Matches `DocumentationAgent`'s existing shape, so all three agents agree. Chosen over a structured `{kind, action}` form for that consistency; the reliability that structure would have bought is recovered by P3 and P4. |
| P3 | **Caveats are generated in code wherever the trigger is a fact** | "Only one option in 48 hours", "the earliest departs in 71 minutes", "no phone number for any option" are all decidable from data. Leaving them to the model reproduces the `status` failure. The model may add its own on top. |
| P4 | Every caveat opens with **`NOTE:`**, **`ASK:`** or **`CONFIRM:`** | Keeps the intent unmissable inside a plain string: inform the passenger, get information from them, or do not proceed silently. Gives Person A something to branch on without changing the agreed shape. |
| P5 | Fields the tool already fetches are **surfaced, not discarded**: `terminal`, `aircraft`, `status`, `phone`, `website`, `address`, `kind`, `stars`, `wheelchair` | All were already being paid for and thrown away. `phone` is the single most valuable field either agent has: it hands the passenger the one job neither agent can do — confirming a room and a rate. `terminal` matters because the passenger is standing in an airport. |
| P7 | **Two short-lived compatibility fields** so nothing regresses while Person A migrates: `area` keeps its human form (`"2.4 km from the terminal, Lod"`) alongside the new numeric `distance_km` and plain `city`, and `meals_included` survives as a deprecated boolean beside `meals` | Traced against the real `_digest`: it reads `stay.area` and `stay.meals_included`. Today the model writes the distance *into* `area`, so **the distance currently does reach the passenger** - splitting it cleanly would delete the single most useful hotel fact from the plan until her rewrite lands. `meals_included` absent reads as falsy, which would assert "no meals" where the truth is "unknown". Two fields and a comment remove the whole coordination risk; Person A deletes them when she migrates. `stops` and `fare_conditions` need no equivalent - the digest never read them. |
| P6 | Breaking changes taken now: `stops` dropped, `fare_conditions` → `rebooking`, `area` split into `distance_km` + `area`, `meals_included` → `meals` | `stops` is permanently `0` since direct-only is a decision (D7). `fare_conditions` has never held fare data and the name misleads whoever reads the payload next. `area` was a free-text blob mixing distance with place, which nothing could sort or compare. `meals_included: false` was a lie dressed as data — we almost never know — and `meals: "unknown"` says so. |

---

## 3. FlightAgent payload

```jsonc
{"options": [{
   "id": "F1",
   "airline": "Lufthansa", "airline_iata": "LH", "flight_number": "LH 687",
   "origin": "TLV", "destination": "FRA",
   "depart": "2026-08-15T16:30+03:00", "arrive": "2026-08-15T20:10+02:00",
   "duration_minutes": 220,
   "arrives_next_day": false,
   "terminal": "3",
   "aircraft": "Airbus A320",
   "status": "Expected",
   "rebooking": "Same airline as the cancelled flight - usually the simplest to arrange at the desk.",
   "notes": "Earliest arrival, and it clears your deadline."
 }],
 "recommended_id": "F1",
 "caveats": ["NOTE: every option is on a different airline from the one that cancelled."]}
```

**From the model:** `id`, `flight_number` (its choice), `rebooking`, `notes`, the ordering, and
`recommended_id`. Everything else is written by code from the matched candidate.

**Matching key:** `flight_number`, compared with whitespace stripped and upper-cased, against the
candidate's `flight` — the same normalisation the grounding check already uses. An option matching
no candidate is **dropped**, which is what makes an invented flight structurally unable to survive.
If nothing matches, the agent raises with its step attached, exactly as today.

`duration_minutes` and `arrives_next_day` are computed from the candidate's own timestamps.
`arrives_next_day` is load-bearing: it changes what the passenger needs to plan for.

---

## 4. AccommodationAgent payload

```jsonc
{"options": [{
   "id": "H1",
   "name": "Airport Plaza", "kind": "hotel",
   "distance_km": 2.4, "area": "Lod", "address": "12 HaNasi",
   "phone": "+972 3 000 0000", "website": "https://example.test",
   "stars": "4", "wheelchair": "yes",
   "check_in": "2026-08-09", "check_out": "2026-08-10", "nights": 1,
   "price_estimate": "Roughly EUR 110-140 for the night (estimate - not a quoted price)",
   "meals": "unknown",
   "notes": "Closest to the terminal, which matters for an 09:40 departure. Call to confirm the room and the rate."
 }],
 "recommended_id": "H1",
 "caveats": ["ASK: no phone number is listed for any option, so nobody can confirm a room tonight."]}
```

**From the model:** `id`, `name` (its choice), `price_estimate`, `notes`, the ordering, and
`recommended_id`. Everything else is written by code from the matched candidate and the stay window.

**Matching key:** `name`, normalised the same way, against the candidate's `name`. Unmatched options
are dropped. `check_in`, `check_out` and `nights` are written from the stay window the Supervisor
supplied rather than from the model, so they cannot disagree with the flight that produced them.

`kind` is omitted when the property is an ordinary hotel and present otherwise, so "this is a hostel,
expect shared facilities" is visible rather than a surprise on arrival. `meals` is `"included"` only
when OSM's `breakfast` tag says so, `"not_included"` when it says no, and `"unknown"` otherwise —
which is the honest answer nearly every time.

---

## 5. Caveats

Generated in code where the trigger is a fact:

**FlightAgent**
- `NOTE:` fewer than three options exist in the 48-hour window.
- `NOTE:` every option is on a carrier other than the one that cancelled.
- `CONFIRM:` the earliest option departs within 90 minutes of `local_now`.
- `NOTE:` the recommended flight arrives the next day.
- `NOTE:` an option is marked `Delayed`.

**AccommodationAgent**
- `ASK:` no option has a phone number, so availability cannot be confirmed by anyone.
- `CONFIRM:` the nearest option is more than 15 km from the terminal.
- `NOTE:` no ordinary hotels were found — only hostels, guest houses or apartments.
- `NOTE:` all prices are estimates, not quotes.

The model may append its own caveats in the same form. Code-generated ones come first.

When an agent refuses outright (no candidates, or an impossible route) there is no payload and
therefore no `caveats`: the reason travels as the `LLMError` message, as it does today.

---

## 6. Impact elsewhere

- **`docs/PROJECT_PLAN.md` §1** holds these schemas and is a **locked** decision. It is updated in
  the same commit, per `CLAUDE.md` §7, with the changes called out for Person A.
- **Evaluator checks** that read moving fields are updated with the schema: `meals_honesty`
  (`meals_included` → `meals`), `no_asserted_fare` (`fare_conditions` → `rebooking`), `date_sync`,
  `grounding_flights`, `grounding_hotels`.
- **`tests/conftest.py`'s `fake_llm` returns the old shape** and must move with it. That file is
  shared: Person A's Supervisor tests run against the same fakes, so this edit lands in her
  territory and belongs in the handover conversation, not just the PR.
  One convenient interaction: because P1 overwrites factual fields from the candidate, the fake's
  flight number only has to *match* `fake_search_data`'s candidate — it already does — so the fakes
  get simpler rather than more elaborate.
- **The GUI is unaffected.** Verified 15/8/2026: `public/index.html` never names a payload field and
  renders each step with `JSON.stringify(step.response)`, so renames pass straight through it.
- **A live re-run** proves it end to end: ~15–20 LLM calls, ~20 API units.

---

## 7. Risk

**Resolved by P7.** The original risk was that this only paid off once Person A's `_digest` rewrite
landed, and until then the passenger saw *less* — specifically the hotel's distance, which the
digest reads out of `area`. The compatibility fields close that window: there is now no point at
which the output degrades, and her migration can happen whenever suits her.

What remains is bookkeeping: `area` and `meals_included` are marked deprecated in the code and in
the §1 contract, and should be deleted once her digest reads `distance_km`, `city` and `meals`. If
they are still there at submission they are harmless — redundant, not wrong.

# Search-agent validation — design

**Owner:** Person B · **Date:** 10/8/2026 · **Branch:** `validate-search-agents`
**Scope:** `FlightAgent`, `AccommodationAgent`, and the date sync that connects them.
**Budget approved:** up to **$1** of LLMod spend, and AeroDataBox units used deliberately
(more obtainable if needed — see the contingency ladder in
[`2026-08-09-flight-accommodation-agents-design.md`](./2026-08-09-flight-accommodation-agents-design.md) §10).

Both agents are built, merged and covered by 109 unit tests — but every one of those tests
mocks the LLM. **No prompt in either agent has ever been exercised against a real model.**
This validates them live and writes down what they can and cannot do, ahead of the team
meeting later this week.

---

## 1. Scope

In scope: live validation of the two search agents, a repeatable harness, and a written
capability/limitation map.

Out of scope: the Supervisor's stubbed seams, `DocumentationAgent`, the RAG corpus, the GUI,
`/api/agent_info`'s content. Findings about those are recorded for the meeting, not fixed here.

---

## 2. Decisions

| # | Decision | Rationale |
|---|---|---|
| V1 | Mirror Person C's pattern: `evals/*_cases.json` + loader, `scripts/run_*_live.py --confirm-paid-calls`, offline `scripts/evaluate_*_artifact.py` | Already the team's convention, immediately legible to teammates, and re-runnable by anyone before submission. |
| V2 | The evaluator's checks are **mechanical**, not judgement calls | Unlike Person C's legal semantics, this agent's core claims are decidable: a returned flight number either appears in the candidate list or it does not. Automating that turns "the model behaved" into an assertion. |
| V3 | Scenarios run **through `supervisor.run`** with `DocumentationAgent` faked | The date sync *is* the integration between the two agents — the hotel nights derive from the flight FlightAgent found — so calling the agents directly would skip the only thing connecting them. Faking DocumentationAgent avoids spending on Person C's part and removes the Pinecone dependency. The Supervisor's own seams are deterministic stubs and cost nothing. |
| V4 | **Comprehensive** matrix: 12 scenarios, half of them aimed at the edges | The stated goal is knowing the *incapabilities*. Happy paths cannot reveal those. |
| V5 | Hard rule violations get fixed and re-run; style is logged | A validation that finds a real violation and leaves it is half a job. Prompt polish is open-ended, so it stays a decision for the owner rather than a loop. |
| V6 | `--all` runs in **one process** so the route cache is shared | Multi-turn scenarios reuse an earlier scenario's route and cost zero extra API units. |

---

## 3. Components

| File | Role |
|---|---|
| `evals/search_agent_cases.json` | The 12 scenarios, declarative |
| `evals/search_cases.py` | Loader, mirroring `evals/documentation_cases.py` |
| `scripts/run_search_agents_live.py` | `--confirm-paid-calls`, `--scenario X` \| `--all`, `--output`, `--max-llm-calls` |
| `scripts/evaluate_search_artifact.py` | Offline checks over a saved artifact; needs no keys |
| `docs/search-agents-capabilities.md` | The capability/limitation map for the team meeting |

Artifacts land in `live-test-output/`, already gitignored.

---

## 4. Scenario matrix

| # | Scenario | Probes |
|---|---|---|
| 1 | `tlv-fra-cancelled` | Grounding on the demo route; cross-airline options (the CoC differentiator) |
| 2 | `lhr-jfk-cancelled` | A different region and a long-haul route; airport and OSM coverage away from TLV |
| 3 | `thin-route` | A route with few departures — exercises the 12-hour window widening |
| 4 | `overnight-one-night` | Date sync: 22:15 disruption → next-morning flight → exactly 1 night |
| 5 | `multi-night` | Date sync when the next usable flight is 2+ days out |
| 6 | `same-day-no-hotel` | Replacement departs today ⇒ `AccommodationAgent` must **not** be dispatched |
| 7 | `degraded-flights` | `WINGMAN_LIVE_DATA=0`: options must be labelled illustrative |
| 8 | `sparse-osm-airport` | An airport OSM barely covers; the hotel degraded path |
| 9 | `baggage-followup` | Multi-turn ski-bag question — must defer, not invent an allowance |
| 10 | `price-question` | "What will it cost?" — must not state a fare as fact |
| 11 | `earlier-flight-followup` | Multi-turn: "anything earlier than X?" |
| 12 | `compare-options` | Multi-turn: "compare these two" |

Scenarios 9–12 reuse scenario 1's route, so they hit the cache and cost no API units.

Each case in the JSON names its own route explicitly. Scenarios 3, 5 and 8 depend on a property
of live data that we cannot guarantee on the day (few departures, a 2+ day gap, sparse hotels).
The runner records **what it actually observed** — candidate counts, derived nights, hotel
counts — and the report says plainly when a scenario did not reproduce its intended condition.
Nothing is faked to force the condition, because a fabricated edge case validates nothing.

`DocumentationAgent` is neutralised in the runner by assigning the same no-cost fake that
`tests/test_supervisor.py::no_cost_documentation_agent` uses, so the trace keeps its three
`DocumentationAgent` steps and the Supervisor's dispatch logic is exercised unchanged.

---

## 5. Mechanical checks

Every check reads the saved artifact. **The candidates are inside each step's `user_prompt`**,
so grounding is verifiable after the fact without re-calling anything.

- **Grounding** — every `flight_number` returned appears in that step's candidate block; every
  hotel `name` likewise. Tests the "never invent an option" rule directly.
- **Date sync** — hotel `check_in`/`check_out` equal the stay window in the AccommodationAgent
  prompt; `nights` matches the difference.
- **Trace shape** — exactly one step per agent; `module` in
  `{Supervisor, FlightAgent, AccommodationAgent, DocumentationAgent}`; step keys exactly
  `{module, prompt, response}` and prompt keys exactly `{system_prompt, user_prompt}`.
- **No booking site** — response text and payloads scanned for known booking/search domains.
- **Price honesty** — `price_estimate` must carry a hedge marker (estimate / approx / roughly /
  around); a bare figure fails.
- **Meals honesty** — `meals_included: true` only when the candidate carried a breakfast tag;
  otherwise `notes` must say meals were not confirmed.
- **Degraded labelling** — when the candidate list was empty, every option's `notes` must say
  the option is illustrative.
- **Deferral** — the baggage scenario must not assert an allowance and must point at the
  entitlements/Contract of Carriage.

Each check reports pass/fail per scenario. The evaluator exits non-zero if any hard check fails.

---

## 6. Cost control

- `--all` shares one process and therefore one route cache: ~4 distinct route searches, so
  **8–16 of 592 units**.
- ~20 LLM calls, roughly 45k tokens.
- `--max-llm-calls` (default 30) **aborts rather than overspending**; running usage from
  `lib.llm.usage` is printed as the run proceeds.
- Each artifact records tokens consumed and units spent, so the report states measured cost
  rather than estimates.
- LLMod exposes no dollar figure to us, so tokens are the unit of account. If a usage endpoint
  turns out to exist, record it; otherwise report tokens and note the limitation.

---

## 7. The report

`docs/search-agents-capabilities.md`, written for teammates rather than for me:

- **Verified capabilities** — each claim tied to the scenario and artifact proving it.
- **Hard limitations by root cause** — no free data source (prices, availability, baggage
  terms); API shape (direct flights only, 12-hour windows); OSM sparseness (meals, ratings,
  local-language names); quota ceiling.
- **Failure modes** — what the passenger actually sees when a source dies.
- **Open items for the meeting**, including two that are not Person B's:
  `/api/agent_info` returns `prompt_examples: []` (a hard graded requirement), and its
  `description` does not state that prices are estimates, which the 9/8 decision requires.
  Both are blocked behind the Supervisor's stubbed seams, because capturing a trace now would
  bake two LLM-less `Supervisor` steps into a graded artifact.

---

## 8. Expected outcome

At least one honesty check is likely to fail on the first run. The instructions telling the
model to hedge prices and defer carriage questions were written before any key existed and have
never been tested. A violation is the harness working, not misfiring.

`pytest` must stay green throughout — the existing 109 tests guard the payload shapes the
Supervisor's date sync depends on.

---

## 9. Risks

- **Live data moves.** Real schedules change daily, so a scenario asserting "3 options on this
  route" would be flaky. Checks assert *properties* (grounding, date consistency, labelling),
  never specific flight numbers or counts.
- **Overpass flakiness.** The public instance returned a 504 during development; the mirror
  fallback covers it, and a total failure lands in the degraded path, which is itself a scenario.
- **A thin or sparse scenario may not reproduce.** If `thin-route` happens to have plenty of
  flights on the day of the run, it validates nothing. The runner records what it actually got
  so the report can say so honestly rather than implying coverage it did not achieve.

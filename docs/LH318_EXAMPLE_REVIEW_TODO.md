# LH318 Example Review — Meeting TODO

## Meeting goal

Decide how Wingman should present live operational data and legal advice safely. The legal retrieval and reflection worked well, but the final answer combined inconsistent dates and overstated uncertain flight and hotel data.

## Example reviewed

The passenger said that Lufthansa flight LH318 from TLV to FRA was cancelled **today, 18 August** and asked for the earliest replacement flight, a hotel, meals, refund options and possible compensation.

The answer recommended:

- Condor DE4308 departing **15 August**.
- A hotel stay from **14 to 15 August**.
- A nearby OpenStreetMap property as if it were a practical accommodation option.
- “No meals,” although the source only said that meal inclusion was unknown.

## P0 — Correctness and safety

- [ ] Add `travel_date` or `disruption_at`, including timezone, to the Supervisor request.
  - **Meaning:** do not search only from the server clock when the passenger provides a date.
  - **Example:** a disruption on 18 August must never produce a flight on 15 August.
- [ ] Define how conflicting dates are handled.
  - Prefer an explicit calendar date, or ask the passenger to confirm when “today” conflicts with it.
- [ ] Add deterministic temporal validation before composition.
  - Reject flights before the disruption and hotel dates that do not bridge the wait until the selected flight.
- [ ] Validate cross-agent consistency.
  - Route, airports, dates, timezone and party size must agree across Supervisor, Flight and Accommodation results.
- [ ] Return the valid legal guidance even when operational results are rejected.
  - **Example:** explain hotel/refund rights but say that no safe flight recommendation could be verified.

## P0 — Be honest about live data

- [ ] Describe AeroDataBox results as **scheduled departures**, not available or confirmed rerouting.
  - It does not verify seats, ticketing or whether Lufthansa will rebook onto Condor.
  - **Preferred wording:** “An earlier scheduled flight to ask Lufthansa about is …”
- [ ] Add explicit flight certainty fields, such as `schedule_confirmed`, `availability_unknown` and `rebooking_unconfirmed`.
- [ ] Decide whether another airline's flight should be recommended or only presented as a candidate to request.
- [ ] Describe OpenStreetMap results as **nearby property listings**, not available hotel rooms.
  - It does not provide room availability, booking confirmation, live prices or meal details.
- [ ] Decide whether hotel listings are useful without availability and contact/booking data.
- [ ] Represent uncertain facts with three states: `yes`, `no`, `unknown`.
  - **Current error:** “Meals were not confirmed” became “It does not include meals.”

## P1 — Final-answer quality

- [ ] Preserve each specialist result's uncertainty in the final response.
  - Never turn “possible” into “confirmed,” “unknown” into “no,” or “scheduled” into “available.”
- [ ] Use one integrated Wingman voice; keep the multi-agent architecture internal.
  - Remove “the crew found,” “the crew says,” agent names and pipeline terminology from passenger-facing answers.
  - **Before:** “The crew found one.”
  - **After:** “An earlier scheduled option to ask Lufthansa about is …”
  - **Before:** “The crew says you can choose a refund.”
  - **After:** “You may choose reimbursement instead of rerouting.”
- [ ] Tie uncertainty to evidence, not internal processing.
  - **Preferred wording:** “I could not verify seat availability” or “The available legal sources do not establish a separate baggage-care entitlement.”
- [ ] Correct unclear action wording.
  - **Before:** “Ask Lufthansa to confirm whether you want the flight or refund.”
  - **After:** “Tell Lufthansa which option you choose and ask it to confirm the arrangement in writing.”
- [ ] Keep debugging steps and subagent names only in the developer trace.

## P1 — Legal guidance and citations

- [ ] Keep **refund** separate from **cash compensation**.
  - Refund returns the ticket price when the passenger chooses not to travel; compensation is an additional payment when its conditions are met.
- [ ] Preserve important conditions found during reflection.
  - Israeli refund deadline: 21 days after written application.
  - Compensation exceptions do not remove care, refund or rerouting rights.
  - “Operational issue” alone does not prove or disprove extraordinary circumstances.
- [ ] Distinguish rules that definitely apply, likely apply or are not yet established.
  - **Example:** do not present EU261 as certain until the necessary carrier/route conditions are established.
- [ ] Avoid merging Israeli and EU rules when their conditions differ.
- [ ] Do not provide an outdated Israeli compensation amount.
- [ ] Make citations usable to passengers.
  - Map labels such as `[S4]` to the document title, provision and source link, or hide internal labels and show a collapsible Sources section.
- [ ] Ensure every citation supports the exact nearby statement.

## P1 — Passenger-specific actions

- [ ] Clarify whether an entitlement or compensation amount applies per eligible passenger before multiplying it for an adult and child.
- [ ] Do not invent child-specific rights when the documents do not provide them.
- [ ] Add practical checked-baggage guidance without presenting it as a legal entitlement.
  - Ask whether bags will be returned overnight, held for the new flight or retagged for another carrier.

## P2 — Traceability and testing

- [ ] Show the normalized Supervisor request in the developer trace.
  - **Current confusion:** the raw step lists `arrive_by` as missing even though it is optional and execution continues.
- [ ] Trace the passenger's scenario time, server time, search window and why each option was selected.
- [ ] Label facts by provenance: live tool, legal retrieval, deterministic calculation or model reasoning.
- [ ] Add the exact LH318 prompt as a regression test.
- [ ] Test explicit-date versus “today” conflicts, year boundaries and timezones.
- [ ] Test overnight and multi-night disruptions.
- [ ] Test cross-airline candidates and missing/partial live data.
- [ ] Assert that unknown meals never become “no meals.”
- [ ] Assert that schedule-only flights are never called available or confirmed.
- [ ] Assert that directory-listed hotels are never called booked or available.
- [ ] Assert that internal terms such as `crew`, `subagent`, `FlightAgent`, `DocumentationAgent`, `retrieval` and `pipeline` do not appear in the final passenger response.
- [ ] Add an end-to-end answer rubric covering temporal consistency, grounding, certainty, legal coverage, citations and actionability.

## Decisions needed in the meeting

1. **Should Wingman recommend operational options or only show candidates that the passenger must verify?**
   - **Example:** should it say “Take Condor DE4308” or “DE4308 is an earlier scheduled flight; ask Lufthansa whether it can rebook you onto it”?
2. **What evidence is required before calling a flight or hotel “available”?**
   - **Example:** AeroDataBox shows that DE4308 is scheduled, but not that two seats are available. OpenStreetMap shows that Medical Hotel Shai Lev exists, but not that it has a room tonight.
3. **Should explicit passenger dates override the server clock, and when should Wingman ask for clarification?**
   - **Example:** if the passenger says “today, 18 August” while the server date is 14 August, should Wingman search from 18 August or first ask which date is correct? YES!!
4. **Should invalid operational results be removed while valid legal guidance is still returned?**
   - **Example:** if all retrieved flights have impossible dates, Wingman could omit them but still explain the passenger's hotel, meals, rerouting and refund rights.
5. **Should EU carrier and regulatory status come from deterministic airline metadata?**
   - **Example:** a maintained airline record could identify Lufthansa as an EU carrier, instead of asking the Documentation Agent to prove that fact from passenger-rights passages on every request.
6. **How should legal sources be presented in the UI?**
   - **Example:** should the answer show `[S4]`, show “Israeli Aviation Services Law, section 3,” or provide a collapsible Sources panel with the provision and document link?
7. **Which structured checks must pass before the final answer can be shown?**
   - **Example:** require every recommended flight to depart after the disruption, hotel dates to cover the waiting period, and unknown availability or meals to remain labelled as unknown.
8. **What is the minimum production acceptance test for the full Supervisor pipeline?**
   - **Example:** the LH318 scenario should return chronologically valid options, preserve live-data uncertainty, correctly explain refund and compensation, cite usable sources, and never mention the internal “crew.”

## Current assessment

The Documentation Agent's retrieval and reflection are strong. The immediate risks are in orchestration and final composition: impossible dates, loss of uncertainty, and presenting schedule/map data as confirmed passenger solutions.

## TODO:

1. Show steps in a more human readable way.

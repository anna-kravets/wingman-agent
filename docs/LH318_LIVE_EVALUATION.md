# LH318 live-agent evaluation

Date: 2026-08-16

## Scenario

The test passenger reported that Lufthansa flight LH318 from Tel Aviv to Frankfurt was cancelled at the gate, that one child was travelling with them, and that their checked bags remained with the airline. They requested the earliest reasonable replacement flight, a hotel, meals, refund information, and possible compensation. The follow-up asked for nearby hotels and estimated prices.

The full prompts, every module prompt and response, structured results, final answers, timings, and token usage are saved in `live-test-output/`. That directory is intentionally gitignored, so the traces remain local unless they are deliberately shared.

## Problems reproduced and fixed

| Problem | Example of the bad behavior | Fix and expected behavior |
| --- | --- | --- |
| Accommodation was not called | A same-day flight caused the overnight stay to be skipped even though the passenger explicitly asked for a hotel. | An explicit hotel request always dispatches AccommodationAgent. If the selected flight does not imply an overnight stay, the system uses a one-night assumption and states it. |
| Impractical flight recommendation | A flight leaving in about 30 minutes was presented as the best replacement. | Flights within 90 minutes remain visible as urgent possibilities, but the deterministic recommendation moves to the first reasonably catchable option. Seats, rebooking, and boarding must still be confirmed. |
| Airline identity mismatch | `Lufthansa` and flight code `LH` were treated as different airlines. | The original carrier code is derived from the flight number, so Lufthansa/LH is recognized correctly. |
| Hotel details were incomplete | The answer said only “mid-range” and omitted actual estimates. | AccommodationAgent must return numeric EUR ranges. It can make one bounded repair call if its first answer is vague. |
| Hostel chosen for an adult and child | The closest hostel could be recommended without considering room suitability. | For multiple travellers, an ordinary hotel is preferred when available; private-room configuration must be confirmed. |
| Hotel dates looked like confirmed inventory | Requested check-in/check-out dates appeared under “Availability.” | The answer labels them as stay dates and explicitly says room availability was not checked or confirmed. |
| Supervisor discarded useful findings | Legal deadlines, amounts, citations, hotel prices, and safer flight ordering disappeared in the summary. | Composition now has deterministic completeness checks and one bounded repair pass. If repair still fails, the complete grounded digest is returned instead of a polished but incomplete answer. |
| Child counted twice | Party size was 2, but prose said “all 3 of you.” | Party-size consistency is validated; the child is included in the total, not added again. |
| Checked bags had no practical next step | The answer only repeated that the bags were with the airline. | The passenger is told to ask whether bags will remain checked through or be returned, and how transfer works if the replacement is on another carrier. No unsupported baggage right is invented. |
| Follow-up repeated expensive work | Asking for hotel prices could redispatch flight, hotel, and documentation work. | Follow-ups about already-found hotel options reuse the saved structured stay results and call only Supervisor composition. |

## Final result

The final answer now:

- recommends LY357, the first reasonably catchable direct option, while keeping LH687 as an urgent same-day possibility only;
- states that all flight options are schedules, not confirmed seats or completed rebookings;
- gives three nearby stay options with distance, property type, numeric EUR range, meals status, accessibility where known, and an explicit no-availability warning;
- correctly handles two travellers;
- preserves refund and rerouting choices, hotel, meals, transport, communication assistance, compensation bands, reduction rules, deadlines, named laws, and section/article references;
- distinguishes care duties from conditional cash compensation;
- gives an actionable checked-baggage instruction; and
- speaks as one assistant, with no references to a “crew” or internal agents.

The latest composition completed with zero deterministic omissions. It used 2 Supervisor calls, 7,726 prompt tokens, 2,660 completion tokens, and 11.75 seconds. The full offline suite passed: 315 tests.

## Saved evidence

| Artifact | Purpose |
| --- | --- |
| `live-test-output/lh318_full_agent_after_fixes.json` | First frozen full replay; exposed unsafe recommendation and vague hotel-price problems. |
| `live-test-output/lh318_full_agent_current_live.json` | True live AeroDataBox, OpenStreetMap, Pinecone, DocumentationAgent, and Supervisor run; exposed party-size wording. |
| `live-test-output/lh318_full_agent_final_replay.json` | Full exact-scenario replay after flight, price, reuse, and party fixes; exposed the misleading availability label. |
| `live-test-output/lh318_final_recomposed_answer.json` | Composition-only verification after the availability fix. |
| `live-test-output/lh318_final_recomposed_with_baggage.json` | Zero-call deterministic fallback produced when external network access was unavailable; proves the fallback remains complete. |
| `live-test-output/lh318_final_recomposed_with_baggage_live.json` | Final paid composition-only verification including the baggage instruction; this is the final answer to review. |

## Remaining real-world limitations

- Flight data shows scheduled/expected services, not seat inventory or a confirmed Lufthansa rebooking.
- Hotel results come from geographic listings. Prices are rough estimates, not quotes; live room availability, room configuration, and meals must be confirmed.
- The retrieved Israeli compensation figures are the original statutory amounts and may have been indexed since publication. The answer does not pretend they are the current payable figures.
- The exact compensation amount and exception analysis still require route distance, notice and timing details, and the airline's documented reason for cancellation.
- The final verification is layered: a saved full-agent run validates dispatch and retrieval, while the last two-call rerun validates the latest Supervisor-only wording changes without repaying for unchanged upstream calls.

## Production judgment

This scenario is now handled substantially better and safely enough for the project demo. The protections are general rules—catchability, completeness, availability honesty, party-size consistency, baggage handling, and follow-up reuse—rather than prompt-specific text. More scenarios are still valuable as regression coverage, but there is no known failure in this LH318 example that requires another model call before shipping.

"""The checks that decide whether a live run passed.

Run against synthetic artifacts so they are proven before any money is spent.
"""

import json

from evals.search_artifact import candidates_from_prompt, evaluate

FLIGHT_CANDIDATES = [
    {"flight": "LH 687", "airline": "Lufthansa", "airline_iata": "LH", "origin": "TLV",
     "destination": "FRA", "depart": "2026-08-11T16:30+03:00", "arrive": "2026-08-11T20:10+02:00",
     "status": "Expected", "aircraft": "Airbus A320", "terminal": "3"},
]
HOTEL_CANDIDATES = [
    {"name": "Airport Plaza", "distance_km": 2.4, "stars": "4", "breakfast": None, "area": "Lod"},
]


def flight_prompt(candidates=FLIGHT_CANDIDATES):
    body = ("Candidates (real, verified departures - choose only from these):\n"
            + json.dumps(candidates)) if candidates else "No live schedule data was available"
    return "Route: TLV -> FRA\nLocal time now: 2026-08-10T22:15\n\n" + body


def hotel_prompt(candidates=HOTEL_CANDIDATES, check_in="2026-08-10",
                 check_out="2026-08-11", nights=1):
    body = ("Hotels (real, near the airport - choose only from these):\n"
            + json.dumps(candidates)) if candidates else "No live hotel data was available"
    return (f"Check in: {check_in}\nCheck out: {check_out}\nNights: {nights}\n\n" + body)


def step(module, user_prompt, response):
    return {"module": module,
            "prompt": {"system_prompt": "sys", "user_prompt": user_prompt},
            "response": response}


def artifact_with(flight_options, hotel_options, flight_prompt_text=None,
                  hotel_prompt_text=None, response="Onward flight: LH 687."):
    flight_payload = {"options": flight_options, "recommended_id": "F1"}
    steps = [step("Supervisor", "refine", {}),
             step("FlightAgent", flight_prompt_text or flight_prompt(), flight_payload)]
    if hotel_options is not None:
        steps.append(step("AccommodationAgent", hotel_prompt_text or hotel_prompt(),
                          {"options": hotel_options, "recommended_id": "H1"}))
    steps.append(step("Supervisor", "compose", {"text": response}))
    return {"status": "ok", "response": response, "steps": steps}


GOOD_FLIGHT = {"id": "F1", "airline": "Lufthansa", "airline_iata": "LH",
               "flight_number": "LH 687", "origin": "TLV", "destination": "FRA",
               "depart": "2026-08-11T16:30+03:00", "arrive": "2026-08-11T20:10+02:00",
               "terminal": "3", "aircraft": "Airbus A320", "status": "Expected",
               "rebooking": "See your Contract of Carriage.", "notes": "Nonstop."}
GOOD_HOTEL = {"id": "H1", "name": "Airport Plaza", "distance_km": 2.4, "city": "Lod",
              "area": "2.4 km from the terminal, Lod",
              "check_in": "2026-08-10", "check_out": "2026-08-11", "nights": 1,
              "price_estimate": "Roughly EUR 120 (estimate)", "meals": "unknown",
              "notes": "Meals not confirmed - check at the desk."}


def case_with(*checks):
    return {"case_id": "t", "checks": list(checks), "expect": {}}


def status_of(results, name):
    return next(r["status"] for r in results if r["check"] == name)


# --- candidate parsing ----------------------------------------------------------


def test_candidates_are_recovered_from_the_prompt():
    assert candidates_from_prompt(flight_prompt())[0]["flight"] == "LH 687"


def test_degraded_prompt_has_no_candidates():
    assert candidates_from_prompt(flight_prompt(candidates=None)) == []


# --- grounding ------------------------------------------------------------------


def test_grounding_passes_when_the_flight_came_from_the_candidates():
    results = evaluate(artifact_with([GOOD_FLIGHT], [GOOD_HOTEL]),
                       case_with("grounding_flights"))
    assert status_of(results, "grounding_flights") == "pass"


def test_grounding_fails_on_an_invented_flight():
    invented = dict(GOOD_FLIGHT, flight_number="XX 999")
    results = evaluate(artifact_with([invented], [GOOD_HOTEL]),
                       case_with("grounding_flights"))
    assert status_of(results, "grounding_flights") == "fail"


def test_grounding_ignores_spacing_differences_in_flight_numbers():
    spaced = dict(GOOD_FLIGHT, flight_number="LH687")
    results = evaluate(artifact_with([spaced], [GOOD_HOTEL]),
                       case_with("grounding_flights"))
    assert status_of(results, "grounding_flights") == "pass"


def test_hotel_grounding_fails_on_an_invented_property():
    invented = dict(GOOD_HOTEL, name="Imaginary Suites")
    results = evaluate(artifact_with([GOOD_FLIGHT], [invented]),
                       case_with("grounding_hotels"))
    assert status_of(results, "grounding_hotels") == "fail"


def test_grounding_is_skipped_when_there_were_no_candidates():
    art = artifact_with([GOOD_FLIGHT], [GOOD_HOTEL],
                        flight_prompt_text=flight_prompt(candidates=None))
    results = evaluate(art, case_with("grounding_flights"))
    assert status_of(results, "grounding_flights") == "skip"


# --- date sync ------------------------------------------------------------------


def test_date_sync_passes_when_the_hotel_matches_the_window():
    results = evaluate(artifact_with([GOOD_FLIGHT], [GOOD_HOTEL]), case_with("date_sync"))
    assert status_of(results, "date_sync") == "pass"


def test_date_sync_fails_on_the_wrong_nights():
    wrong = dict(GOOD_HOTEL, check_out="2026-08-13", nights=3)
    results = evaluate(artifact_with([GOOD_FLIGHT], [wrong]), case_with("date_sync"))
    assert status_of(results, "date_sync") == "fail"


# --- honesty --------------------------------------------------------------------


def test_price_honesty_fails_on_an_unhedged_figure():
    blunt = dict(GOOD_HOTEL, price_estimate="EUR 120")
    results = evaluate(artifact_with([GOOD_FLIGHT], [blunt]), case_with("price_honesty"))
    assert status_of(results, "price_honesty") == "fail"


def test_no_asserted_fare_fails_when_flightagent_quotes_money():
    quoting = dict(GOOD_FLIGHT, rebooking="Rebooking costs EUR 90.")
    results = evaluate(artifact_with([quoting], [GOOD_HOTEL]), case_with("no_asserted_fare"))
    assert status_of(results, "no_asserted_fare") == "fail"


def test_meals_honesty_fails_when_meals_are_claimed_without_evidence():
    claimed = dict(GOOD_HOTEL, meals="included", notes="Breakfast included.")
    results = evaluate(artifact_with([GOOD_FLIGHT], [claimed]), case_with("meals_honesty"))
    assert status_of(results, "meals_honesty") == "fail"


def test_meals_honesty_passes_when_the_candidate_data_supports_it():
    art = artifact_with(
        [GOOD_FLIGHT], [dict(GOOD_HOTEL, meals="included", notes="Breakfast from 05:30.")],
        hotel_prompt_text=hotel_prompt(
            candidates=[dict(HOTEL_CANDIDATES[0], breakfast="yes")]),
    )
    results = evaluate(art, case_with("meals_honesty"))
    assert status_of(results, "meals_honesty") == "pass"


def test_no_unusable_status_fails_when_a_departed_flight_is_offered():
    # The real regression: a live run recommended a flight already marked Departed.
    gone = [dict(FLIGHT_CANDIDATES[0], status="Departed")]
    art = artifact_with([GOOD_FLIGHT], [GOOD_HOTEL],
                        flight_prompt_text=flight_prompt(candidates=gone))
    results = evaluate(art, case_with("no_unusable_status"))
    assert status_of(results, "no_unusable_status") == "fail"


def test_no_unusable_status_passes_for_a_catchable_flight():
    results = evaluate(artifact_with([GOOD_FLIGHT], [GOOD_HOTEL]),
                       case_with("no_unusable_status"))
    assert status_of(results, "no_unusable_status") == "pass"


def test_no_booking_site_fails_when_one_is_named():
    leaky = dict(GOOD_HOTEL, notes="Cheaper on booking.com")
    results = evaluate(artifact_with([GOOD_FLIGHT], [leaky]), case_with("no_booking_site"))
    assert status_of(results, "no_booking_site") == "fail"


def test_deferral_fails_when_a_baggage_allowance_is_asserted():
    asserting = dict(GOOD_FLIGHT, notes="Your ski bag is fine, allowance is 23kg.")
    results = evaluate(artifact_with([asserting], [GOOD_HOTEL]), case_with("deferral"))
    assert status_of(results, "deferral") == "fail"


def test_deferral_passes_when_it_points_at_the_contract():
    results = evaluate(artifact_with([GOOD_FLIGHT], [GOOD_HOTEL]), case_with("deferral"))
    assert status_of(results, "deferral") == "pass"


# --- degraded and trace ---------------------------------------------------------


def test_degraded_refusal_passes_when_the_agent_did_not_run_and_the_passenger_was_told():
    # No FlightAgent step at all: it refused before spending a call. This is the wording
    # flight_agent.NO_LIVE_DATA produces (route_problem's own sentence differs but both
    # end up in the composed response as prose, which is all this check can see).
    art = {"status": "ok",
           "response": ("Live flight schedules were not available for this route, so no "
                        "departure could be verified."),
           "steps": [step("Supervisor", "refine", {}), step("Supervisor", "compose", {})]}
    assert status_of(evaluate(art, case_with("degraded_refusal")), "degraded_refusal") == "pass"


def test_degraded_refusal_passes_on_the_internal_failure_fallback_wording():
    # The other path to a no-call refusal: dispatch fell back to FAILURE_MESSAGES["flight"]
    # because the failure carried no passenger_message. Pinned separately so a rewording of
    # either sentence, alone, cannot silently break the eval again.
    art = {"status": "ok",
           "response": "I could not get onward flight options just now.",
           "steps": [step("Supervisor", "refine", {}), step("Supervisor", "compose", {})]}
    assert status_of(evaluate(art, case_with("degraded_refusal")), "degraded_refusal") == "pass"


def test_degraded_refusal_fails_when_the_passenger_was_never_told():
    art = {"status": "ok", "response": "Here is your plan.",
           "steps": [step("Supervisor", "refine", {}), step("Supervisor", "compose", {})]}
    assert status_of(evaluate(art, case_with("degraded_refusal")), "degraded_refusal") == "fail"


def test_degraded_refusal_fails_if_the_model_was_called_with_no_candidates():
    # Calling the model with nothing to choose from is the behaviour we removed.
    art = artifact_with([GOOD_FLIGHT], None,
                        flight_prompt_text=flight_prompt(candidates=None))
    assert status_of(evaluate(art, case_with("degraded_refusal")), "degraded_refusal") == "fail"


def test_degraded_refusal_is_skipped_when_candidates_existed():
    art = artifact_with([GOOD_FLIGHT], [GOOD_HOTEL])
    assert status_of(evaluate(art, case_with("degraded_refusal")), "degraded_refusal") == "skip"


def test_trace_shape_requires_one_step_per_agent():
    results = evaluate(artifact_with([GOOD_FLIGHT], [GOOD_HOTEL]), case_with("trace_shape"))
    assert status_of(results, "trace_shape") == "pass"


def test_trace_shape_fails_on_an_unknown_module():
    art = artifact_with([GOOD_FLIGHT], [GOOD_HOTEL])
    art["steps"].append(step("MysteryAgent", "x", {}))
    results = evaluate(art, case_with("trace_shape"))
    assert status_of(results, "trace_shape") == "fail"


def same_day_prompt(now="2026-08-11T05:00:00"):
    return f"Route: TLV -> FRA\nLocal time now: {now}\n\n" + (
        "Candidates (real, verified departures - choose only from these):\n"
        + json.dumps(FLIGHT_CANDIDATES))


def test_no_accommodation_passes_when_the_stay_was_skipped():
    # GOOD_FLIGHT departs 2026-08-11, and local_now is the same day.
    art = artifact_with([GOOD_FLIGHT], None, flight_prompt_text=same_day_prompt())
    results = evaluate(art, case_with("no_accommodation"))
    assert status_of(results, "no_accommodation") == "pass"


def test_no_accommodation_fails_when_a_stay_ran_on_a_same_day_flight():
    art = artifact_with([GOOD_FLIGHT], [GOOD_HOTEL], flight_prompt_text=same_day_prompt())
    results = evaluate(art, case_with("no_accommodation"))
    assert status_of(results, "no_accommodation") == "fail"


def test_no_accommodation_skips_when_the_flight_is_not_today():
    # A live run failed this check because every same-day departure had genuinely
    # gone and been filtered out: the premise, not the agent, was wrong.
    art = artifact_with([GOOD_FLIGHT], [GOOD_HOTEL],
                        flight_prompt_text=same_day_prompt(now="2026-08-10T05:00:00"))
    results = evaluate(art, case_with("no_accommodation"))
    assert status_of(results, "no_accommodation") == "skip"


def test_a_hotels_own_website_is_not_a_booking_site():
    # radissonhotels.com contains "hotels.com" and is not a violation.
    legit = dict(GOOD_HOTEL, notes="See https://www.radissonhotels.com/heathrow")
    results = evaluate(artifact_with([GOOD_FLIGHT], [legit]), case_with("no_booking_site"))
    assert status_of(results, "no_booking_site") == "pass"


def test_a_real_booking_site_is_still_caught():
    leaky = dict(GOOD_HOTEL, notes="Cheaper on www.hotels.com tonight")
    results = evaluate(artifact_with([GOOD_FLIGHT], [leaky]), case_with("no_booking_site"))
    assert status_of(results, "no_booking_site") == "fail"


def test_history_reached_agents_checks_every_agent_prompt():
    earlier = "\nEarlier in this conversation:\n  passenger: hi"
    with_history = {"case_id": "t", "checks": ["history_reached_agents"], "expect": {},
                    "history": [{"prompt": "hi", "response": "ok"}]}

    # Both agents append history, so both prompts must carry it.
    art = artifact_with([GOOD_FLIGHT], [GOOD_HOTEL],
                        flight_prompt_text=flight_prompt() + earlier,
                        hotel_prompt_text=hotel_prompt() + earlier)
    assert status_of(evaluate(art, with_history), "history_reached_agents") == "pass"


def test_history_reached_agents_fails_when_one_agent_missed_it():
    earlier = "\nEarlier in this conversation:\n  passenger: hi"
    with_history = {"case_id": "t", "checks": ["history_reached_agents"], "expect": {},
                    "history": [{"prompt": "hi", "response": "ok"}]}

    art = artifact_with([GOOD_FLIGHT], [GOOD_HOTEL],
                        flight_prompt_text=flight_prompt() + earlier)
    assert status_of(evaluate(art, with_history), "history_reached_agents") == "fail"


# --- the enrichment guard --------------------------------------------------------


def test_facts_from_candidates_passes_when_they_agree():
    results = evaluate(artifact_with([GOOD_FLIGHT], [GOOD_HOTEL]),
                       case_with("facts_from_candidates"))
    assert status_of(results, "facts_from_candidates") == "pass"


def test_facts_from_candidates_catches_a_drifted_time():
    # If a refactor ever lets model output through again, it surfaces here rather
    # than in a hotel booked for the wrong night.
    drifted = dict(GOOD_FLIGHT, depart="2026-08-11T09:00+03:00")
    results = evaluate(artifact_with([drifted], [GOOD_HOTEL]),
                       case_with("facts_from_candidates"))
    assert status_of(results, "facts_from_candidates") == "fail"


def test_meals_honesty_accepts_unknown():
    results = evaluate(artifact_with([GOOD_FLIGHT], [GOOD_HOTEL]),
                       case_with("meals_honesty"))
    assert status_of(results, "meals_honesty") == "pass"

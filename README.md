# Wingman

**A recovery companion for passengers facing a flight disruption.**

[Open Wingman](https://project-eight-chi-97.vercel.app)

When a flight is cancelled, delayed, or overbooked, a passenger suddenly has to solve three
problems at once: find a way onward, find somewhere to stay, and understand what the airline owes
them. Wingman brings those decisions into one clear conversation.

## What Wingman does

The passenger describes what happened in their own words. Wingman asks for a missing detail only
when it is needed, then produces one coordinated recovery plan covering:

- **Flights:** practical replacement options from real flight schedules, ordered by usefulness.
- **Accommodation:** nearby properties for the nights created by the new itinerary, with distance,
  estimated price, and available contact details.
- **Passenger rights:** a plain-language explanation of rebooking, refunds, meals, hotels, transport,
  and possible compensation.
- **Next actions:** a short list of what to request, what to confirm, and what evidence to keep.

The conversation remains open after the first answer. A passenger can compare flights, ask about a
hotel, clarify baggage handling, or explore the difference between accepting a replacement flight
and choosing a refund. Previous conversations are available from the same browser.

## What makes it different

Wingman combines general passenger-rights rules with the airline's own **Contract of Carriage**.
These airline-specific documents can contain important obligations and options that are easy to miss
when a passenger looks only at general regulations.

The legal assessment considers relevant material from:

- EU/EEA passenger-rights rules, including Switzerland
- United States passenger-rights material
- Israeli passenger-rights law
- Supported airlines' Conditions or Contracts of Carriage

Legal references in the final answer are clickable, allowing the passenger to inspect the supporting
passage instead of relying on an unexplained conclusion.

## The agent team

Wingman presents one unified answer, while four specialized agents work together behind it:

- **Supervisor** understands the situation, resolves missing or conflicting details, and combines the
  result into one recovery plan.
- **FlightAgent** identifies useful onward departures.
- **AccommodationAgent** finds nearby stays for the correct nights.
- **DocumentationAgent** retrieves the relevant legal and airline documents, drafts an assessment,
  critiques it against the evidence, and refines the final rights explanation.

## Example prompt

> Lufthansa flight LH318 from Tel Aviv to Frankfurt was cancelled at the gate today. The airline
> said it was an operational issue. I am travelling with one child, my checked bags are with the
> airline, and the next flight they offered is tomorrow afternoon. I need a hotel tonight, meals,
> and the earliest reasonable replacement flight. Please also explain whether I can choose a refund
> and what compensation may apply.

A useful request should include as many known facts as possible: the airline and flight number,
route, scheduled departure, what happened, where the passenger is now, who is travelling, what the
airline offered, and what help is needed. Unknown details can simply be left out.

## Important boundaries

Wingman supports decisions; it does not make bookings or submit claims.

- Flight schedules and hotel/property records come from real searches, but no option is reserved.
- Flight and hotel prices are estimates, not quotes.
- Seat availability, room availability, meals, ticket-specific fare conditions, and cross-airline
  rebooking must be confirmed directly.
- Wingman cannot access a passenger's reservation, airline account, payment details, or live inventory.
- Its rights explanation is based on published sources and is not legal advice or a guarantee of the
  airline's decision.
- When a route falls outside the legal material available to Wingman, the answer identifies that
  coverage limit.

## Team

Wingman was created by Tal Aloni, Gilai Blum, and Anna Kravets.

# Itinerary-preserving feed reduction

This document covers the entry point added on top of GTFS Kit 12.0.3 for
building a small offline timetable around a list of journeys that must remain
feasible.

## Public API

### `Feed.build_itinerary_feed(itineraries, budget)`

Located in `gtfs_kit/itineraries.py` and attached as a `Feed` method.

Arguments:

- `itineraries`: non-empty list of dicts (or a `pandas.DataFrame`), each with
  - `service_date`: `YYYYMMDD` GTFS service day,
  - `origin_stop_id`, `destination_stop_id`: stop IDs of the feed,
  - `earliest_boarding_time`: `HH:MM:SS` (hours may exceed 23),
  - `max_lateness_seconds`: non-negative integer tolerance relative to the
    full-feed earliest arrival.
- `budget`: positive integer cap on the number of delivered `stop_times`
  rows. A retained trip always carries its full stop_times; a trip shared by
  several itineraries is counted once.

Returns an `ItineraryFeedResult` with:

- `feed`: a new, independently savable `Feed` (the input feed is never
  modified),
- `report`: per-itinerary audit rows (one row per used leg, so one or two
  rows per itinerary), in itinerary order, duplicates retained,
- `stop_times_count`, `trip_count`, `budget`,
- `search_status`, `search_complete`, `search_visited_nodes`,
  `candidate_trip_count`, `options_per_itinerary`.

Failure modes raise `ItineraryPlanningError` and never return a partial feed:

- `status="unreachable"`: an itinerary has no direct or one-transfer journey
  in the full feed,
- `status="infeasible"`: exhaustive search proved no trip combination fits
  the budget,
- `status="undetermined"`: the bounded search effort cap was reached before
  feasibility was decided (large feeds only).

### `build_stop_timetable(..., board_only=False, alight_only=False)`

Optional keyword filters on the existing function. They apply the same
usable-stop-event rules as the itinerary planner (pickup/drop-off type 0 or
missing plus the required departure/arrival time). Calls without the keywords
behave exactly as before.

### `restrict_to_dates(feed, dates, exact=False)`

With `exact=True`, the calendar is rewritten so retained trips run on exactly
the named dates and nowhere else, even when the list skips days. `calendar`
is dropped and `calendar_dates` is rebuilt from the feed's actual activity.

## Journey semantics ("what counts as feasible")

- Same GTFS service day; times after `24:00:00` stay on the starting service
  day. A trip belonging to the next service day is never picked up.
- Direct trip, or exactly one transfer between two different trips.
- Boarding needs `pickup_type` 0/missing and a departure time; alighting
  needs `drop_off_type` 0/missing and an arrival time. Middle stops with
  missing times are still delivered with the retained trip but cannot be
  boarding/alighting points.
- `stop_sequence` resolves repeated visits to the same stop; later boards
  cannot backtrack to an earlier alighting sequence.
- Same-stop transfers need no rule (alight before/at departure, equal times
  accepted), unless an explicit type 3 rule forbids them. Cross-stop
  transfers require an explicit stop-level rule.
- Transfer types: 0/blank = time ordering only; 2 = enforce
  `min_transfer_time`; 3 = never connect.
- Calendar: weekly `calendar`, `calendar_dates` exceptions only, and both
  tables together are all supported, including holiday adds and removals.

Out of scope and explicitly rejected with a descriptive `ValueError`:

- non-empty `frequencies`, `fare_rules`, `fare_attributes`, `attributions`,
- transfer types other than 0/2/3, route/trip/parent-station-scoped rules,
  type 2 without `min_transfer_time`, and endpoints that are not actually
  served stops,
- empty itinerary list, invalid dates/times/stops, negative tolerance,
  boolean or non-positive-integer budget.

## How the combination is chosen

1. For every itinerary, enumerate all admissible direct and one-transfer
   options in the full feed and record the earliest arrival. Options arriving
   more than `max_lateness_seconds` later than that earliest arrival are
   discarded.
2. Each option has a cost equal to the union of stop_times rows of its one
   or two trips. Sharing a trip across itineraries makes its marginal cost
   zero, which is how two itineraries can jointly keep a slower trip that
   would not be chosen individually.
3. A deterministic depth-first branch-and-bound search picks one option per
   itinerary, ordering itineraries by fewest options first and trying
   cheapest-marginal-weight options first, memoizing failed
   `(position, chosen trip set)` states. Output is therefore reproducible
   for identical inputs.
4. With at most 16 candidate trips and at most 8 itineraries (and bounded
   option counts) the search is exhaustive, so feasibility is guaranteed to
   be found and `infeasible` is a proof. Beyond that, the same search runs
   with an explicit node cap; a capped success reports
   `search_complete=False`, and a capped failure raises `undetermined`.
5. The chosen trips are re-verified inside the constructed reduced feed
   before the result is returned.

The reduced feed contains full stop_times of retained trips only, their
routes/agencies, stops plus parent stations, used shapes, applicable
transfer rules, an exact calendar on the itinerary service dates, and a
`feed_info` service window adjusted to the delivered dates.

## Rerunnable commands (Python 3.12)

The project virtualenv was created with:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e . pytest pytest-socket ruff
```

Run the whole acceptance and regression suite offline:

```bash
.venv/bin/python --version
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/test_itineraries.py -q --disable-socket
.venv/bin/ruff check gtfs_kit/ tests/ examples/
```

Run the end-to-end example (builds a ZIP, reads it back with `read_feed`):

```bash
.venv/bin/python -I examples/itinerary_feed_example.py
```

## Actual results on this machine

Python version:

```text
Python 3.12.16
```

Full test suite:

```text
142 passed, 26 warnings in 11.00s
```

Offline run of the new suite with sockets disabled:

```text
38 passed in 4.46s
```

Lint:

```text
All checks passed!
```

Example script output (abridged):

```text
=== retained trips ===
['S1', 'S2']
stop_times rows delivered: 8
search complete: True
=== audit report ===
 itinerary_id  service_date  ...  alight_stop_sequence arrival_time departure_time
            0      20250106  ...                     4     09:20:00       08:10:00
            1      20250106  ...                     4     09:15:00       08:10:00
=== zip roundtrip ===
files in zip: ['agency.txt', 'calendar_dates.txt', 'feed_info.txt', 'routes.txt',
               'stop_times.txt', 'stops.txt', 'trips.txt']
reread stop_times rows: 8
reread active trips on 20250106: ['S1', 'S2']
```

The acceptance tests in `tests/test_itineraries.py` include an independent
brute-force oracle (`brute_force_feasibility`) that enumerates every trip
subset from raw table rows, a case where per-itinerary fastest choices exceed
the budget but shared slower trips fit, a provably insufficient budget, a
full-feed-unreachable case, overnight service-day handling, forbidden and
minimum-time transfers, calendar exceptions, ZIP round-trip row counts and
reference integrity, stop-timetable filters, and exact date restriction.

"""
Tests for the itinerary-preserving feed reduction entry point
(:func:`gtfs_kit.build_itinerary_feed`), the boarding/alighting filters on
:func:`gtfs_kit.build_stop_timetable`, and the exact mode of
:func:`gtfs_kit.restrict_to_dates`.

The synthetic feeds built here are constructed by hand as Feed objects, so the
tests never depend on the implementation under test to generate fixtures.
The feasibility oracle in :func:`brute_force_feasibility` independently
enumerates every subset of candidate trips and recomputes journey feasibility
from raw stop_times rows.
"""

from __future__ import annotations

import itertools
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gtfs_kit as gk
from gtfs_kit import itineraries as iz


# ---------------------------------------------------------------------
# Synthetic feed construction
# ---------------------------------------------------------------------
def _df(rows, dtypes):
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in dtypes.items()})
    df = pd.DataFrame(rows)
    for col, dtype in dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df


STOP_DTYPES = {
    "stop_id": "string",
    "stop_name": "string",
    "stop_lat": "float",
    "stop_lon": "float",
    "location_type": "Int32",
    "parent_station": "string",
}
ST_DTYPES = {
    "trip_id": "string",
    "arrival_time": "string",
    "departure_time": "string",
    "stop_id": "string",
    "stop_sequence": "Int32",
    "pickup_type": "Int32",
    "drop_off_type": "Int32",
}


def make_feed(
    trips,
    *,
    stops=None,
    calendar=None,
    calendar_dates=None,
    transfers=None,
    feed_info=None,
    frequencies=None,
    fare_attributes=None,
    fare_rules=None,
    attributions=None,
    dist_units="km",
):
    """
    Build a Feed from compact trip dicts.

    Each trip dict has ``trip_id``, ``route_id``, ``service_id``, and an
    ordered ``events`` list of ``(stop_id, arrival, departure)`` triples
    (times are HH:MM:SS strings, hours may exceed 23).  Optional
    ``pickup_type`` / ``drop_off_type`` lists align with the events.
    """
    stop_rows = []
    if stops is None:
        stop_ids = sorted({e[0] for tr in trips for e in tr["events"]})
        for i, sid in enumerate(stop_ids):
            stop_rows.append(
                {
                    "stop_id": sid,
                    "stop_name": f"Stop {sid}",
                    "stop_lat": 36.0 + i * 0.01,
                    "stop_lon": -117.0 + i * 0.01,
                }
            )
        stops = _df(stop_rows, STOP_DTYPES)

    st_rows = []
    trip_rows = []
    route_rows = []
    for tr in trips:
        trip_rows.append(
            {
                "route_id": tr["route_id"],
                "service_id": tr["service_id"],
                "trip_id": tr["trip_id"],
            }
        )
        route_rows.append(
            {
                "route_id": tr["route_id"],
                "agency_id": "DTA",
                "route_short_name": tr["route_id"],
                "route_type": 3,
            }
        )
        pickups = tr.get("pickup_type", [None] * len(tr["events"]))
        dropoffs = tr.get("drop_off_type", [None] * len(tr["events"]))
        for seq, (event, pu, do) in enumerate(
            zip(tr["events"], pickups, dropoffs), start=1
        ):
            stop_id, arrival, departure = event
            st_rows.append(
                {
                    "trip_id": tr["trip_id"],
                    "arrival_time": arrival,
                    "departure_time": departure,
                    "stop_id": stop_id,
                    "stop_sequence": seq,
                    "pickup_type": pu,
                    "drop_off_type": do,
                }
            )

    routes = _df(route_rows, gk.DTYPES["routes"]).drop_duplicates("route_id")
    trips_df = _df(trip_rows, gk.DTYPES["trips"])
    stop_times = _df(st_rows, ST_DTYPES)
    agency = _df(
        [
            {
                "agency_id": "DTA",
                "agency_name": "Demo Agency",
                "agency_url": "http://example.com",
                "agency_timezone": "America/Los_Angeles",
            }
        ],
        gk.DTYPES["agency"],
    )

    def frame_or_none(rows, table):
        if rows is None:
            return None
        return _df(rows, gk.DTYPES[table])

    return gk.Feed(
        dist_units=dist_units,
        agency=agency,
        stops=stops,
        routes=routes,
        trips=trips_df,
        stop_times=stop_times,
        calendar=frame_or_none(calendar, "calendar"),
        calendar_dates=frame_or_none(calendar_dates, "calendar_dates"),
        fare_attributes=frame_or_none(fare_attributes, "fare_attributes"),
        fare_rules=frame_or_none(fare_rules, "fare_rules"),
        shapes=None,
        frequencies=frame_or_none(frequencies, "frequencies"),
        transfers=frame_or_none(transfers, "transfers"),
        feed_info=frame_or_none(feed_info, "feed_info"),
        attributions=frame_or_none(attributions, "attributions"),
    )


def weekly_calendar(service_id="SVC", start="20250101", end="20251231", days=1):
    return [
        {
            "service_id": service_id,
            "monday": days,
            "tuesday": days,
            "wednesday": days,
            "thursday": days,
            "friday": days,
            "saturday": days,
            "sunday": days,
            "start_date": start,
            "end_date": end,
        }
    ]


def padded_trip(trip_id, route_id, service_id, events, pad):
    """
    Insert ``pad`` unique padding stops before the first real event so the
    trip carries a controllable number of stop_times rows.
    """
    full_events = [
        (f"{trip_id}_P{i}", "05:00:00", "05:00:00") for i in range(pad)
    ] + list(events)
    return {
        "trip_id": trip_id,
        "route_id": route_id,
        "service_id": service_id,
        "events": full_events,
    }


# ---------------------------------------------------------------------
# Independent brute-force oracle
# ---------------------------------------------------------------------
def _sec(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    h, m, s = (int(v) for v in x.split(":"))
    return h * 3600 + m * 60 + s


def oracle_earliest_arrival(feed, trip_subset, spec):
    """
    Independent earliest-arrival computation using direct and one-transfer
    journeys over the given subset of trip IDs, implementing the same
    boarding/alighting and transfer rules from raw table rows only.
    """
    st = feed.stop_times[feed.stop_times["trip_id"].isin(trip_subset)]
    active_services = set(feed.get_active_services(spec["service_date"]))
    active_trips = set(
        feed.trips.loc[
            feed.trips["service_id"].isin(active_services), "trip_id"
        ]
    )
    st = st[st["trip_id"].isin(active_trips)]
    earliest_board = _sec(spec["earliest_boarding_time"])
    origin = spec["origin_stop_id"]
    dest = spec["destination_stop_id"]

    def usable_board(row):
        pu = row["pickup_type"]
        return (pd.isna(pu) or pu == 0) and pd.notna(row["departure_time"])

    def usable_alight(row):
        do = row["drop_off_type"]
        return (pd.isna(do) or do == 0) and pd.notna(row["arrival_time"])

    rules = {}
    if feed.transfers is not None and not feed.transfers.empty:
        for r in feed.transfers.itertuples(index=False):
            rules[(r.from_stop_id, r.to_stop_id)] = r

    def transfer_ok(a_stop, b_stop, a_time, b_time):
        if b_time < a_time:
            return False
        if a_stop == b_stop:
            r = rules.get((a_stop, b_stop))
            if r is None or pd.isna(r.transfer_type):
                return True
            if r.transfer_type == 3:
                return False
            if r.transfer_type == 2:
                return b_time - a_time >= int(r.min_transfer_time)
            return True
        r = rules.get((a_stop, b_stop))
        if r is None or pd.isna(r.transfer_type):
            return False
        if r.transfer_type == 3:
            return False
        if r.transfer_type == 2:
            return b_time - a_time >= int(r.min_transfer_time)
        return True

    best = None

    events_by_trip = {
        tid: g.sort_values("stop_sequence").reset_index(drop=True)
        for tid, g in st.groupby("trip_id")
    }

    # Direct journeys.
    for tid, g in events_by_trip.items():
        boards = g[
            (g["stop_id"] == origin)
            & g.apply(usable_board, axis=1)
            & (g["departure_time"].map(_sec) >= earliest_board)
        ]
        if boards.empty:
            continue
        bseq = int(boards["stop_sequence"].iloc[0])
        alights = g[
            (g["stop_id"] == dest)
            & (g["stop_sequence"] > bseq)
            & g.apply(usable_alight, axis=1)
        ]
        if alights.empty:
            continue
        arrival = _sec(alights["arrival_time"].iloc[0])
        best = arrival if best is None else min(best, arrival)

    # One-transfer journeys.
    for t1, g1 in events_by_trip.items():
        boards1 = g1[
            (g1["stop_id"] == origin)
            & g1.apply(usable_board, axis=1)
            & (g1["departure_time"].map(_sec) >= earliest_board)
        ]
        if boards1.empty:
            continue
        b1seq = int(boards1["stop_sequence"].iloc[0])
        alights1 = g1[
            (g1["stop_sequence"] > b1seq) & g1.apply(usable_alight, axis=1)
        ]
        for t2, g2 in events_by_trip.items():
            if t2 == t1:
                continue
            dest_rows = g2[
                (g2["stop_id"] == dest) & g2.apply(usable_alight, axis=1)
            ]
            if dest_rows.empty:
                continue
            final_seq = int(dest_rows["stop_sequence"].iloc[-1])
            boards2 = g2[
                (g2["stop_sequence"] < final_seq) & g2.apply(usable_board, axis=1)
            ]
            for a in alights1.itertuples(index=False):
                for b in boards2.itertuples(index=False):
                    if not transfer_ok(
                        a.stop_id,
                        b.stop_id,
                        _sec(a.arrival_time),
                        _sec(b.departure_time),
                    ):
                        continue
                    arrival = _sec(dest_rows["arrival_time"].iloc[-1])
                    best = arrival if best is None else min(best, arrival)

    return best


def brute_force_feasibility(feed, specs, budget):
    """
    Enumerate every subset of candidate trips independently.

    Returns ``(feasible, best_subset)`` where feasibility means every spec is
    served within its own tolerance and the subset's full stop_times count is
    within budget.
    """
    full_earliest = {}
    candidate = set()
    admissible_trips_per_spec = []
    all_trips = set(feed.trips["trip_id"])
    for i, spec in enumerate(specs):
        earliest = oracle_earliest_arrival(feed, all_trips, spec)
        full_earliest[i] = earliest
        if earliest is None:
            return "unreachable", None, full_earliest
        # Collect trips participating in any journey arriving within tolerance.
        allowed = set()
        for tid in all_trips:
            sub = {tid}
            # A single trip option.
            arr = oracle_earliest_arrival(feed, sub, spec)
            if arr is not None and arr - earliest <= spec["max_lateness_seconds"]:
                allowed.add(tid)
        # Two-trip transfer options.
        for t1, t2 in itertools.combinations(sorted(all_trips), 2):
            arr = oracle_earliest_arrival(feed, {t1, t2}, spec)
            if arr is not None and arr - earliest <= spec["max_lateness_seconds"]:
                allowed.update((t1, t2))
        admissible_trips_per_spec.append(allowed)
        candidate |= allowed

    weight = (
        feed.stop_times.groupby("trip_id").size().astype(int).to_dict()
    )
    candidate = sorted(candidate)
    for r in range(1, len(candidate) + 1):
        for combo in itertools.combinations(candidate, r):
            subset = set(combo)
            cost = sum(weight[t] for t in subset)
            if cost > budget:
                continue
            ok = True
            for i, spec in enumerate(specs):
                arr = oracle_earliest_arrival(feed, subset, spec)
                if (
                    arr is None
                    or arr - full_earliest[i] > spec["max_lateness_seconds"]
                ):
                    ok = False
                    break
            if ok:
                return "feasible", subset, full_earliest
    return "infeasible", None, full_earliest


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
DATE = "20250106"  # a Monday


def sharing_feed():
    """
    Two A->C itineraries.  Per-itinerary fastest options are distinct
    high-row-count trips; two slower low-row-count trips cover both
    itineraries together within a smaller budget.
    """
    trips = [
        padded_trip("F1", "R1", "SVC", [
            ("A", "08:00:00", "08:00:00"),
            ("B", "08:30:00", "08:30:00"),
            ("C", "09:00:00", "09:00:00"),
        ], pad=3),
        padded_trip("S1", "R1", "SVC", [
            ("A", "08:10:00", "08:10:00"),
            ("B", "08:45:00", "08:45:00"),
            ("C", "09:20:00", "09:20:00"),
        ], pad=1),
        padded_trip("F2", "R2", "SVC", [
            ("D", "08:00:00", "08:00:00"),
            ("E", "08:25:00", "08:25:00"),
            ("F", "08:55:00", "08:55:00"),
        ], pad=3),
        padded_trip("S2", "R2", "SVC", [
            ("D", "08:10:00", "08:10:00"),
            ("E", "08:40:00", "08:40:00"),
            ("F", "09:15:00", "09:15:00"),
        ], pad=1),
    ]
    return make_feed(trips, calendar=weekly_calendar())


SHARING_SPECS = [
    {
        "service_date": DATE,
        "origin_stop_id": "A",
        "destination_stop_id": "C",
        "earliest_boarding_time": "07:00:00",
        "max_lateness_seconds": 1500,  # 25 minutes; S1 at 09:20 vs F1 09:00
    },
    {
        "service_date": DATE,
        "origin_stop_id": "D",
        "destination_stop_id": "F",
        "earliest_boarding_time": "07:00:00",
        "max_lateness_seconds": 1500,  # S2 at 09:15 vs F2 08:55
    },
]


# ---------------------------------------------------------------------
# Main acceptance scenarios
# ---------------------------------------------------------------------
def test_sharing_trip_fits_but_fastest_union_exceeds_budget():
    feed = sharing_feed()
    # Fastest union F1 + F2: each 6 rows = 12 rows.  Shared slow S1 + S2:
    # each 4 rows = 8 rows.
    verdict, subset, full_earliest = brute_force_feasibility(
        feed, SHARING_SPECS, budget=8
    )
    assert verdict == "feasible"
    assert subset == {"S1", "S2"}

    result = feed.build_itinerary_feed(SHARING_SPECS, budget=8)
    assert result.search_complete is True
    assert result.stop_times_count <= 8
    assert set(result.feed.trips["trip_id"]) == {"S1", "S2"}
    # Full stop_times of retained trips are delivered.
    counts = result.feed.stop_times.groupby("trip_id").size().to_dict()
    assert counts == {"S1": 4, "S2": 4}
    # Report arrivals match the oracle within tolerance.
    rep = result.report
    arrivals = (
        rep.dropna(subset=["leg_sequence"])
        .groupby("itinerary_id")["arrival_time"]
        .last()
        .to_dict()
    )
    assert arrivals == {0: "09:20:00", 1: "09:15:00"}
    lateness = (
        rep.groupby("itinerary_id")["lateness_seconds"].first().to_dict()
    )
    assert lateness == {0: 1200, 1: 1200}
    # Budget 11 is provably infeasible: cheapest feasible pair costs 8 but
    # any single trip covers at most one itinerary; fastest union costs 12.
    with pytest.raises(iz.ItineraryPlanningError) as exc:
        feed.build_itinerary_feed(SHARING_SPECS, budget=7)
    assert exc.value.status == iz.STATUS_INFEASIBLE
    verdict7, _, _ = brute_force_feasibility(feed, SHARING_SPECS, budget=7)
    assert verdict7 == "infeasible"


def test_input_feed_is_not_mutated():
    feed = sharing_feed()
    before = {
        "stop_times": feed.stop_times.copy(),
        "trips": feed.trips.copy(),
        "calendar": feed.calendar.copy(),
    }
    result = feed.build_itinerary_feed(SHARING_SPECS, budget=8)
    pd.testing.assert_frame_equal(feed.stop_times, before["stop_times"])
    pd.testing.assert_frame_equal(feed.trips, before["trips"])
    pd.testing.assert_frame_equal(feed.calendar, before["calendar"])
    assert feed.calendar is not None and not feed.calendar.empty
    # Reduced feed has no weekly calendar: exact calendar_dates semantics.
    assert result.feed.calendar is None
    assert set(result.feed.calendar_dates["date"]) == {DATE}
    assert (result.feed.calendar_dates["exception_type"] == 1).all()


def test_unreachable_itinerary_raises_and_delivers_nothing():
    feed = sharing_feed()
    specs = [
        dict(SHARING_SPECS[0]),
        {
            "service_date": DATE,
            "origin_stop_id": "A",
            "destination_stop_id": "F",  # no journey connects these lines
            "earliest_boarding_time": "07:00:00",
            "max_lateness_seconds": 60,
        },
    ]
    verdict, _, _ = brute_force_feasibility(feed, specs, budget=1000)
    assert verdict == "unreachable"
    with pytest.raises(iz.ItineraryPlanningError) as exc:
        feed.build_itinerary_feed(specs, budget=1000)
    assert exc.value.status == iz.STATUS_UNREACHABLE
    # Report still records both itineraries in order.
    assert exc.value.report["itinerary_id"].tolist() == [0, 1]
    assert pd.isna(
        exc.value.report.loc[
            exc.value.report["itinerary_id"] == 1,
            "full_feed_earliest_arrival_time",
        ].iloc[0]
    )


def test_overnight_trip_stays_on_service_day():
    trips = [
        {
            "trip_id": "N1",
            "route_id": "R1",
            "service_id": "SVC",
            "events": [
                ("A", "23:30:00", "23:30:00"),
                ("B", "24:15:00", "24:15:00"),
                ("C", "25:00:00", "25:00:00"),
            ],
        },
        # A trip of the next service day that must NOT be picked up.
        {
            "trip_id": "N2",
            "route_id": "R1",
            "service_id": "SVC2",
            "events": [
                ("B", "00:30:00", "00:30:00"),
                ("C", "01:00:00", "01:00:00"),
            ],
        },
    ]
    feed = make_feed(
        trips,
        calendar=weekly_calendar("SVC") + weekly_calendar("SVC2"),
    )
    spec = [
        {
            "service_date": DATE,
            "origin_stop_id": "A",
            "destination_stop_id": "C",
            "earliest_boarding_time": "23:00:00",
            "max_lateness_seconds": 0,
        }
    ]
    result = feed.build_itinerary_feed(spec, budget=10)
    assert set(result.feed.trips["trip_id"]) == {"N1"}
    last = result.report.iloc[-1]
    assert last["arrival_time"] == "25:00:00"
    # The next-service-day trip cannot run on the delivered date.
    assert "N2" not in result.feed.trips["trip_id"]


def test_forbidden_transfer_type3_blocks_cross_stop_connection():
    trips = [
        {
            "trip_id": "T1",
            "route_id": "R1",
            "service_id": "SVC",
            "events": [
                ("A", "08:00:00", "08:00:00"),
                ("X", "08:30:00", "08:30:00"),
            ],
        },
        {
            "trip_id": "T2",
            "route_id": "R2",
            "service_id": "SVC",
            "events": [
                ("Y", "08:40:00", "08:40:00"),
                ("C", "09:10:00", "09:10:00"),
            ],
        },
    ]
    transfers = [
        {
            "from_stop_id": "X",
            "to_stop_id": "Y",
            "transfer_type": 3,
            "min_transfer_time": None,
        }
    ]
    feed = make_feed(trips, calendar=weekly_calendar(), transfers=transfers)
    spec = [
        {
            "service_date": DATE,
            "origin_stop_id": "A",
            "destination_stop_id": "C",
            "earliest_boarding_time": "07:00:00",
            "max_lateness_seconds": 0,
        }
    ]
    with pytest.raises(iz.ItineraryPlanningError) as exc:
        feed.build_itinerary_feed(spec, budget=10)
    assert exc.value.status == iz.STATUS_UNREACHABLE
    # A type-0 rule on the same endpoints makes the connection feasible.
    transfers[0]["transfer_type"] = 0
    feed2 = make_feed(trips, calendar=weekly_calendar(), transfers=transfers)
    result = feed2.build_itinerary_feed(spec, budget=10)
    assert set(result.feed.trips["trip_id"]) == {"T1", "T2"}
    assert result.feed.transfers is not None and len(result.feed.transfers) == 1


def test_minimum_transfer_time_type2():
    trips = [
        {
            "trip_id": "T1",
            "route_id": "R1",
            "service_id": "SVC",
            "events": [
                ("A", "08:00:00", "08:00:00"),
                ("X", "08:30:00", "08:30:00"),
            ],
        },
        # Departs X only 60s after arrival: violates a 120s minimum.
        {
            "trip_id": "T2FAST",
            "route_id": "R2",
            "service_id": "SVC",
            "events": [
                ("X", "08:31:00", "08:31:00"),
                ("C", "09:00:00", "09:00:00"),
            ],
        },
        # Departs 5 minutes later: satisfies the minimum.
        {
            "trip_id": "T2SLOW",
            "route_id": "R2",
            "service_id": "SVC",
            "events": [
                ("X", "08:35:00", "08:35:00"),
                ("C", "09:10:00", "09:10:00"),
            ],
        },
    ]
    transfers = [
        {
            "from_stop_id": "X",
            "to_stop_id": "X",
            "transfer_type": 2,
            "min_transfer_time": 120,
        }
    ]
    feed = make_feed(trips, calendar=weekly_calendar(), transfers=transfers)
    spec = [
        {
            "service_date": DATE,
            "origin_stop_id": "A",
            "destination_stop_id": "C",
            "earliest_boarding_time": "07:00:00",
            "max_lateness_seconds": 0,
        }
    ]
    result = feed.build_itinerary_feed(spec, budget=10)
    assert set(result.feed.trips["trip_id"]) == {"T1", "T2SLOW"}
    assert result.report["arrival_time"].dropna().iloc[-1] == "09:10:00"


def test_same_stop_transfer_equal_times_allowed_and_loop_backtrack():
    # T1 visits X twice; the first visit cannot be used for alighting
    # (drop_off_type 1), so the transfer must use the second visit at
    # sequence 4.  The equal-time connection (09:00 -> 09:00) is accepted.
    trips = [
        {
            "trip_id": "T1",
            "route_id": "R1",
            "service_id": "SVC",
            "events": [
                ("A", "08:00:00", "08:00:00"),
                ("X", "08:20:00", "08:20:00"),
                ("Z", "08:40:00", "08:40:00"),
                ("X", "09:00:00", "09:00:00"),
            ],
            "drop_off_type": [None, 1, None, None],
        },
        {
            "trip_id": "T2",
            "route_id": "R2",
            "service_id": "SVC",
            "events": [
                ("X", "09:00:00", "09:00:00"),
                ("C", "09:30:00", "09:30:00"),
            ],
        },
    ]
    feed = make_feed(trips, calendar=weekly_calendar())
    spec = [
        {
            "service_date": DATE,
            "origin_stop_id": "A",
            "destination_stop_id": "C",
            "earliest_boarding_time": "07:00:00",
            "max_lateness_seconds": 0,
        }
    ]
    result = feed.build_itinerary_feed(spec, budget=10)
    leg1 = result.report[result.report["leg_sequence"] == 1].iloc[0]
    assert leg1["alight_stop_sequence"] == 4  # second visit to X
    assert leg1["alight_stop_id"] == "X"
    assert leg1["arrival_time"] == "09:00:00"
    leg2 = result.report[result.report["leg_sequence"] == 2].iloc[0]
    assert leg2["departure_time"] == "09:00:00"  # equal times accepted


def test_direct_leg_cannot_alight_at_stop_before_boarding():
    # One trip passes C before reaching A, then reaches C again.  Boarding at
    # A must pair with the later C visit, never the earlier one.
    trips = [
        {
            "trip_id": "T1",
            "route_id": "R1",
            "service_id": "SVC",
            "events": [
                ("C", "07:30:00", "07:30:00"),
                ("A", "08:00:00", "08:00:00"),
                ("C", "08:30:00", "08:30:00"),
            ],
        }
    ]
    feed = make_feed(trips, calendar=weekly_calendar())
    spec = [
        {
            "service_date": DATE,
            "origin_stop_id": "A",
            "destination_stop_id": "C",
            "earliest_boarding_time": "07:00:00",
            "max_lateness_seconds": 0,
        }
    ]
    result = feed.build_itinerary_feed(spec, budget=5)
    row = result.report.iloc[0]
    assert row["board_stop_sequence"] == 2
    assert row["alight_stop_sequence"] == 3
    assert row["arrival_time"] == "08:30:00"


def test_calendar_exception_add_and_remove():
    # SVC runs weekdays; the requested Wednesday is removed via exception,
    # while a holiday-only service SVC_H is added that day.
    weekday = [
        {
            "service_id": "SVC",
            "monday": 1,
            "tuesday": 1,
            "wednesday": 1,
            "thursday": 1,
            "friday": 1,
            "saturday": 0,
            "sunday": 0,
            "start_date": "20250101",
            "end_date": "20251231",
        }
    ]
    wednesday = "20250108"
    calendar_dates = [
        {"service_id": "SVC", "date": wednesday, "exception_type": 2},
        {"service_id": "SVC_H", "date": wednesday, "exception_type": 1},
    ]
    trips = [
        {
            "trip_id": "WK",
            "route_id": "R1",
            "service_id": "SVC",
            "events": [
                ("A", "08:00:00", "08:00:00"),
                ("C", "09:00:00", "09:00:00"),
            ],
        },
        {
            "trip_id": "HOL",
            "route_id": "R1",
            "service_id": "SVC_H",
            "events": [
                ("A", "08:00:00", "08:00:00"),
                ("C", "09:05:00", "09:05:00"),
            ],
        },
    ]
    feed = make_feed(
        trips, calendar=weekday, calendar_dates=calendar_dates
    )
    spec = [
        {
            "service_date": wednesday,
            "origin_stop_id": "A",
            "destination_stop_id": "C",
            "earliest_boarding_time": "07:00:00",
            "max_lateness_seconds": 0,
        }
    ]
    result = feed.build_itinerary_feed(spec, budget=10)
    assert set(result.feed.trips["trip_id"]) == {"HOL"}
    # In the reduced feed the holiday service is active exactly that day.
    assert result.feed.get_active_services(wednesday) == ["SVC_H"]


def test_calendar_styles_calendar_only_dates_only_and_both():
    base_trips = lambda sid: [
        {
            "trip_id": "T1",
            "route_id": "R1",
            "service_id": sid,
            "events": [
                ("A", "08:00:00", "08:00:00"),
                ("C", "09:00:00", "09:00:00"),
            ],
        }
    ]
    spec = [
        {
            "service_date": DATE,
            "origin_stop_id": "A",
            "destination_stop_id": "C",
            "earliest_boarding_time": "07:00:00",
            "max_lateness_seconds": 0,
        }
    ]

    # Style 1: weekly calendar only.
    f1 = make_feed(base_trips("SVC"), calendar=weekly_calendar())
    r1 = f1.build_itinerary_feed(spec, budget=5)
    assert r1.feed.calendar is None
    assert r1.feed.calendar_dates["service_id"].tolist() == ["SVC"]

    # Style 2: calendar_dates only.
    f2 = make_feed(
        base_trips("SVC"),
        calendar_dates=[
            {"service_id": "SVC", "date": DATE, "exception_type": 1}
        ],
    )
    r2 = f2.build_itinerary_feed(spec, budget=5)
    assert r2.feed.calendar is None
    assert r2.feed.calendar_dates["date"].tolist() == [DATE]

    # Style 3: both tables.
    f3 = make_feed(
        base_trips("SVC"),
        calendar=weekly_calendar(),
        calendar_dates=[
            {"service_id": "SVC", "date": "20250107", "exception_type": 1}
        ],
    )
    r3 = f3.build_itinerary_feed(spec, budget=5)
    assert r3.feed.calendar is None
    assert set(r3.feed.calendar_dates["date"]) == {DATE}


def test_pickup_dropoff_types_and_missing_times():
    trips = [
        {
            "trip_id": "T1",
            "route_id": "R1",
            "service_id": "SVC",
            "events": [
                ("A", "08:00:00", "08:00:00"),
                ("M", None, None),  # missing times but still delivered
                ("B", "08:30:00", "08:30:00"),
                ("C", "09:00:00", "09:00:00"),
            ],
            "pickup_type": [None, None, 1, None],  # cannot board at B
            "drop_off_type": [None, None, None, 0],
        },
        {
            "trip_id": "T2",
            "route_id": "R1",
            "service_id": "SVC",
            "events": [
                ("A", "08:05:00", "08:05:00"),
                ("B", "08:35:00", "08:35:00"),
                ("C", "09:05:00", "09:05:00"),
            ],
            "drop_off_type": [None, 1, None],  # cannot alight at B
        },
    ]
    feed = make_feed(trips, calendar=weekly_calendar())
    spec = [
        {
            "service_date": DATE,
            "origin_stop_id": "A",
            "destination_stop_id": "C",
            "earliest_boarding_time": "07:00:00",
            "max_lateness_seconds": 0,
        }
    ]
    result = feed.build_itinerary_feed(spec, budget=10)
    # Missing-time middle row remains in the delivered stop_times.
    sub = result.feed.stop_times
    assert sub.loc[sub["stop_id"] == "M"].shape[0] == 1
    assert pd.isna(sub.loc[sub["stop_id"] == "M", "arrival_time"].iloc[0])
    # Fastest is T1 (09:00); boarding at B is irrelevant for this direct run.
    assert result.report["arrival_time"].dropna().iloc[-1] == "09:00:00"


def test_zip_roundtrip_row_counts_and_references(tmp_path):
    feed = sharing_feed()
    result = feed.build_itinerary_feed(SHARING_SPECS, budget=8)
    zip_path = tmp_path / "reduced.zip"
    result.feed.to_file(zip_path)
    assert zipfile.is_zipfile(zip_path)
    reread = gk.read_feed(zip_path, dist_units="km")

    # Actual delivered stop_times row count.
    assert len(reread.stop_times) == result.stop_times_count == 8

    # No dangling references anywhere.
    st_trips = set(reread.stop_times["trip_id"])
    assert st_trips <= set(reread.trips["trip_id"])
    assert set(reread.trips["route_id"]) <= set(reread.routes["route_id"])
    assert set(reread.routes["agency_id"].dropna()) <= set(
        reread.agency["agency_id"]
    )
    assert set(reread.stop_times["stop_id"]) <= set(reread.stops["stop_id"])
    if "parent_station" in reread.stops.columns:
        parents = set(reread.stops["parent_station"].dropna())
        assert parents <= set(reread.stops["stop_id"])
    if "shape_id" in reread.trips.columns:
        shape_ids = set(reread.trips["shape_id"].dropna())
        if shape_ids:
            assert shape_ids <= set(reread.shapes["shape_id"])
    if reread.transfers is not None and not reread.transfers.empty:
        stops = set(reread.stops["stop_id"])
        assert set(reread.transfers["from_stop_id"]) <= stops
        assert set(reread.transfers["to_stop_id"]) <= stops

    # Service activity on the delivered date equals the original feed's.
    original_active = set(
        feed.compute_trip_activity([DATE])
        .loc[lambda x: x[DATE] == 1, "trip_id"]
    ) & set(reread.trips["trip_id"])
    reread_active = set(
        reread.compute_trip_activity([DATE])
        .loc[lambda x: x[DATE] == 1, "trip_id"]
    )
    assert original_active == reread_active
    # No trip runs on a date outside the itinerary dates.
    other = "20250107"
    other_activity = reread.compute_trip_activity([other])
    assert other_activity.empty or other_activity[other].sum() == 0


def test_parent_station_and_feed_info_window_retained(tmp_path):
    stops = _df(
        [
            {
                "stop_id": "A",
                "stop_name": "A",
                "stop_lat": 36.0,
                "stop_lon": -117.0,
                "location_type": 0,
                "parent_station": "STN",
            },
            {
                "stop_id": "C",
                "stop_name": "C",
                "stop_lat": 36.1,
                "stop_lon": -117.1,
                "location_type": 0,
                "parent_station": "STN",
            },
            {
                "stop_id": "STN",
                "stop_name": "Station",
                "stop_lat": 36.05,
                "stop_lon": -117.05,
                "location_type": 1,
                "parent_station": None,
            },
        ],
        STOP_DTYPES,
    )
    trips = [
        {
            "trip_id": "T1",
            "route_id": "R1",
            "service_id": "SVC",
            "events": [
                ("A", "08:00:00", "08:00:00"),
                ("C", "09:00:00", "09:00:00"),
            ],
        }
    ]
    feed_info = [
        {
            "feed_publisher_name": "Demo",
            "feed_publisher_url": "http://example.com",
            "feed_lang": "en",
            "feed_start_date": "20200101",
            "feed_end_date": "20301231",
        }
    ]
    feed = make_feed(
        trips,
        stops=stops,
        calendar=weekly_calendar(),
        feed_info=feed_info,
    )
    spec = [
        {
            "service_date": DATE,
            "origin_stop_id": "A",
            "destination_stop_id": "C",
            "earliest_boarding_time": "07:00:00",
            "max_lateness_seconds": 0,
        }
    ]
    result = feed.build_itinerary_feed(spec, budget=5)
    assert "STN" in set(result.feed.stops["stop_id"])
    info = result.feed.feed_info.iloc[0]
    assert info["feed_start_date"] == DATE
    assert info["feed_end_date"] == DATE


def test_determinism_same_input_same_output():
    feed = sharing_feed()
    r1 = feed.build_itinerary_feed(SHARING_SPECS, budget=8)
    r2 = feed.build_itinerary_feed(SHARING_SPECS, budget=8)
    pd.testing.assert_frame_equal(r1.report, r2.report)
    assert set(r1.feed.trips["trip_id"]) == set(r2.feed.trips["trip_id"])


def test_duplicate_itineraries_kept_in_order():
    feed = sharing_feed()
    specs = [SHARING_SPECS[0], SHARING_SPECS[0]]
    result = feed.build_itinerary_feed(specs, budget=8)
    assert result.report["itinerary_id"].tolist() == [0, 1]


def test_dataframe_input_and_multiple_service_dates():
    # Two services: SVC runs daily, SVC_SUN runs only on the second date via a
    # calendar_dates-only service.  Itineraries on both dates must survive.
    second = "20250112"  # a Sunday
    trips = [
        {
            "trip_id": "WEEKLY",
            "route_id": "R1",
            "service_id": "SVC",
            "events": [
                ("A", "08:00:00", "08:00:00"),
                ("C", "09:00:00", "09:00:00"),
            ],
        },
        {
            "trip_id": "SUNONLY",
            "route_id": "R1",
            "service_id": "SVC_SUN",
            "events": [
                ("A", "10:00:00", "10:00:00"),
                ("C", "11:00:00", "11:00:00"),
            ],
        },
    ]
    feed = make_feed(
        trips,
        calendar=weekly_calendar("SVC"),
        calendar_dates=[
            {"service_id": "SVC_SUN", "date": second, "exception_type": 1}
        ],
    )
    specs = pd.DataFrame(
        [
            {
                "service_date": DATE,
                "origin_stop_id": "A",
                "destination_stop_id": "C",
                "earliest_boarding_time": "07:00:00",
                "max_lateness_seconds": 0,
            },
            {
                "service_date": second,
                "origin_stop_id": "A",
                "destination_stop_id": "C",
                # After WEEKLY's 08:00 departure, so only SUNONLY serves it.
                "earliest_boarding_time": "09:30:00",
                "max_lateness_seconds": 0,
            },
        ]
    )
    result = feed.build_itinerary_feed(specs, budget=10)
    assert set(result.feed.trips["trip_id"]) == {"WEEKLY", "SUNONLY"}
    # Both dates are exported; the Sunday-only trip is inactive on the Monday.
    assert set(result.feed.calendar_dates["date"]) == {DATE, second}
    monday = set(
        result.feed.compute_trip_activity([DATE])
        .loc[lambda x: x[DATE] == 1, "trip_id"]
    )
    sunday = set(
        result.feed.compute_trip_activity([second])
        .loc[lambda x: x[second] == 1, "trip_id"]
    )
    assert monday == {"WEEKLY"}
    assert sunday == {"WEEKLY", "SUNONLY"}
    # A date between the two named dates runs nothing extra beyond the
    # original weekly service, and no calendar remains to reintroduce days.
    assert result.feed.calendar is None
    assert result.report["service_date"].tolist() == [DATE, second]


def test_scope_rejection_aggregates_all_problem_tables():
    feed = _simple_feed(
        frequencies=[
            {
                "trip_id": "T1",
                "start_time": "06:00:00",
                "end_time": "08:00:00",
                "headway_secs": 600,
            }
        ],
        attributions=[{"attribution_id": "X", "organization_name": "Org"}],
    )
    with pytest.raises(ValueError) as exc:
        feed.build_itinerary_feed(_simple_spec(), budget=10)
    message = str(exc.value)
    assert "frequencies" in message
    assert "attributions" in message


# ---------------------------------------------------------------------
# Scope and validation rejection
# ---------------------------------------------------------------------
def _simple_feed(**kwargs):
    trips = [
        {
            "trip_id": "T1",
            "route_id": "R1",
            "service_id": "SVC",
            "events": [
                ("A", "08:00:00", "08:00:00"),
                ("C", "09:00:00", "09:00:00"),
            ],
        }
    ]
    return make_feed(trips, calendar=weekly_calendar(), **kwargs)


def _simple_spec(**overrides):
    spec = {
        "service_date": DATE,
        "origin_stop_id": "A",
        "destination_stop_id": "C",
        "earliest_boarding_time": "07:00:00",
        "max_lateness_seconds": 0,
    }
    spec.update(overrides)
    return [spec]


def test_rejects_nonempty_frequencies_fares_attributions():
    with pytest.raises(ValueError, match="frequencies"):
        _simple_feed(
            frequencies=[
                {
                    "trip_id": "T1",
                    "start_time": "06:00:00",
                    "end_time": "08:00:00",
                    "headway_secs": 600,
                }
            ]
        ).build_itinerary_feed(_simple_spec(), budget=10)
    with pytest.raises(ValueError, match="fare_rules"):
        _simple_feed(
            fare_rules=[{"fare_id": "F1", "route_id": "R1"}]
        ).build_itinerary_feed(_simple_spec(), budget=10)
    with pytest.raises(ValueError, match="fare_attributes"):
        _simple_feed(
            fare_attributes=[
                {
                    "fare_id": "F1",
                    "price": 1.0,
                    "currency_type": "USD",
                    "payment_method": 0,
                }
            ]
        ).build_itinerary_feed(_simple_spec(), budget=10)
    with pytest.raises(ValueError, match="attributions"):
        _simple_feed(
            attributions=[
                {
                    "attribution_id": "X",
                    "organization_name": "Org",
                }
            ]
        ).build_itinerary_feed(_simple_spec(), budget=10)


def test_rejects_unsupported_transfer_rules():
    # Unknown transfer type.
    feed = _simple_feed(
        transfers=[
            {
                "from_stop_id": "A",
                "to_stop_id": "C",
                "transfer_type": 1,
                "min_transfer_time": None,
            }
        ]
    )
    with pytest.raises(ValueError, match="transfer_type"):
        feed.build_itinerary_feed(_simple_spec(), budget=10)
    # Endpoint not a served stop ID.
    feed = _simple_feed(
        transfers=[
            {
                "from_stop_id": "GHOST",
                "to_stop_id": "C",
                "transfer_type": 0,
                "min_transfer_time": None,
            }
        ]
    )
    with pytest.raises(ValueError, match="not stops of the feed"):
        feed.build_itinerary_feed(_simple_spec(), budget=10)
    # Type 2 without min_transfer_time.
    feed = _simple_feed(
        transfers=[
            {
                "from_stop_id": "A",
                "to_stop_id": "C",
                "transfer_type": 2,
                "min_transfer_time": None,
            }
        ]
    )
    with pytest.raises(ValueError, match="min_transfer_time"):
        feed.build_itinerary_feed(_simple_spec(), budget=10)


def test_rejects_route_and_trip_scoped_transfer_rules():
    # GTFS v2 transfers with route/trip scope columns populated.
    route_scoped = _df(
        [
            {
                "from_stop_id": "A",
                "to_stop_id": "C",
                "transfer_type": 0,
                "min_transfer_time": None,
                "from_route_id": "R1",
                "to_route_id": None,
            }
        ],
        gk.DTYPES["transfers"] | {"from_route_id": "string", "to_route_id": "string"},
    )
    feed = _simple_feed()
    feed.transfers = route_scoped
    with pytest.raises(ValueError, match="route-scoped"):
        feed.build_itinerary_feed(_simple_spec(), budget=10)

    trip_scoped = _df(
        [
            {
                "from_stop_id": "A",
                "to_stop_id": "C",
                "transfer_type": 0,
                "min_transfer_time": None,
                "from_trip_id": "T1",
                "to_trip_id": None,
            }
        ],
        gk.DTYPES["transfers"] | {"from_trip_id": "string", "to_trip_id": "string"},
    )
    feed.transfers = trip_scoped
    with pytest.raises(ValueError, match="trip-scoped"):
        feed.build_itinerary_feed(_simple_spec(), budget=10)


def test_rejects_transfer_rule_targeting_parent_station():
    stops = _df(
        [
            {"stop_id": "A", "stop_name": "A", "stop_lat": 36.0,
             "stop_lon": -117.0, "location_type": 0, "parent_station": "STN"},
            {"stop_id": "C", "stop_name": "C", "stop_lat": 36.1,
             "stop_lon": -117.1, "location_type": 0, "parent_station": "STN"},
            {"stop_id": "STN", "stop_name": "Station", "stop_lat": 36.05,
             "stop_lon": -117.05, "location_type": 1, "parent_station": None},
        ],
        STOP_DTYPES,
    )
    trips = [
        {
            "trip_id": "T1",
            "route_id": "R1",
            "service_id": "SVC",
            "events": [
                ("A", "08:00:00", "08:00:00"),
                ("C", "09:00:00", "09:00:00"),
            ],
        }
    ]
    feed = make_feed(
        trips,
        stops=stops,
        calendar=weekly_calendar(),
        transfers=[
            {
                "from_stop_id": "A",
                "to_stop_id": "STN",
                "transfer_type": 0,
                "min_transfer_time": None,
            }
        ],
    )
    with pytest.raises(ValueError, match="parent stations"):
        feed.build_itinerary_feed(_simple_spec(), budget=10)


def test_result_metadata_fields():
    feed = sharing_feed()
    result = feed.build_itinerary_feed(SHARING_SPECS, budget=8)
    assert result.budget == 8
    assert result.trip_count == 2
    # All four trips participate in at least one admissible option.
    assert result.candidate_trip_count == 4
    # Each itinerary has a direct fast, direct slow, and one-transfer option.
    assert result.options_per_itinerary == [3, 3]
    assert result.search_status == iz.STATUS_OPTIMAL
    assert result.search_visited_nodes >= 1
    assert list(result.report.columns) == iz.REPORT_COLUMNS


def test_argument_validation_failures():
    feed = _simple_feed()
    with pytest.raises(ValueError, match="empty"):
        feed.build_itinerary_feed([], budget=10)
    for bad_budget in (0, -1, True, False, 1.5, "8"):
        with pytest.raises(ValueError, match="budget"):
            feed.build_itinerary_feed(_simple_spec(), bad_budget)
    with pytest.raises(ValueError, match="service_date"):
        feed.build_itinerary_feed(_simple_spec(service_date="2025-01-06"), 10)
    with pytest.raises(ValueError, match="not a service date"):
        feed.build_itinerary_feed(_simple_spec(service_date="20990101"), 10)
    with pytest.raises(ValueError, match="origin_stop_id"):
        feed.build_itinerary_feed(_simple_spec(origin_stop_id="NOPE"), 10)
    with pytest.raises(ValueError, match="HH:MM:SS"):
        feed.build_itinerary_feed(
            _simple_spec(earliest_boarding_time="8am"), 10
        )
    with pytest.raises(ValueError, match="max_lateness"):
        feed.build_itinerary_feed(
            _simple_spec(max_lateness_seconds=-5), 10
        )
    with pytest.raises(ValueError, match="max_lateness"):
        feed.build_itinerary_feed(
            _simple_spec(max_lateness_seconds=True), 10
        )


# ---------------------------------------------------------------------
# build_stop_timetable filters
# ---------------------------------------------------------------------
def test_stop_timetable_board_alight_filters():
    trips = [
        {
            "trip_id": "T1",
            "route_id": "R1",
            "service_id": "SVC",
            "events": [
                ("A", "08:00:00", "08:00:00"),
                ("B", "08:30:00", "08:30:00"),
                ("C", "09:00:00", "09:00:00"),
            ],
            "pickup_type": [None, 1, None],
            "drop_off_type": [None, None, 1],
        }
    ]
    feed = make_feed(trips, calendar=weekly_calendar())
    plain = feed.build_stop_timetable("B", [DATE])
    assert len(plain) == 1  # old behaviour unchanged
    boards = feed.build_stop_timetable("B", [DATE], board_only=True)
    assert boards.empty  # pickup_type 1 at B
    alights = feed.build_stop_timetable("B", [DATE], alight_only=True)
    assert len(alights) == 1
    # At C alighting is forbidden, so alight_only is empty.
    assert feed.build_stop_timetable("C", [DATE], alight_only=True).empty
    assert len(feed.build_stop_timetable("C", [DATE], board_only=True)) == 1


def test_stop_timetable_filter_matches_itinerary_semantics():
    trips = [
        {
            "trip_id": "T1",
            "route_id": "R1",
            "service_id": "SVC",
            "events": [
                ("A", "08:00:00", "08:00:00"),
                ("M", None, None),
                ("C", "09:00:00", "09:00:00"),
            ],
        }
    ]
    feed = make_feed(trips, calendar=weekly_calendar())
    tt = feed.build_stop_timetable("M", [DATE], board_only=True)
    assert tt.empty  # no departure time -> not boardable
    tt2 = feed.build_stop_timetable("M", [DATE], alight_only=True)
    assert tt2.empty
    assert len(feed.build_stop_timetable("M", [DATE])) == 1


# ---------------------------------------------------------------------
# restrict_to_dates exact mode
# ---------------------------------------------------------------------
def test_options_cap_switches_to_bounded_search(monkeypatch):
    # With the cap lowered below the number of direct options, the same call
    # reports an incomplete (but still correct) search.
    feed = _sixteen_trip_feed()
    specs = [
        {
            "service_date": DATE,
            "origin_stop_id": "O",
            "destination_stop_id": "D",
            "earliest_boarding_time": "07:00:00",
            "max_lateness_seconds": 24 * 3600,
        }
    ]
    monkeypatch.setattr(iz, "OPTIONS_CAP", 4)
    result = feed.build_itinerary_feed(specs, budget=100)
    assert result.search_complete is False
    assert result.stop_times_count <= 100


def _sixteen_trip_feed():
    trips = []
    for i in range(16):
        trips.append(
            padded_trip(
                f"T{i:02d}",
                "R1",
                "SVC",
                [
                    ("O", f"08:{i:02d}:00", f"08:{i:02d}:00"),
                    ("D", "09:00:00", "09:00:00"),
                ],
                pad=0 if i == 0 else 2,
            )
        )
    return make_feed(trips, calendar=weekly_calendar())


def test_exact_guarantee_at_sixteen_trips_and_eight_itineraries():
    # Eight parallel two-stop trips serve one O->D pair; eight itineraries
    # (some duplicated) all tolerate every trip.  Candidate pool is 16 trips
    # after adding eight unused padding trips, so the exhaustive regime must
    # find the single cheapest trip and prove feasibility.
    feed = _sixteen_trip_feed()
    specs = [
        {
            "service_date": DATE,
            "origin_stop_id": "O",
            "destination_stop_id": "D",
            "earliest_boarding_time": "07:00:00",
            "max_lateness_seconds": 24 * 3600,
        }
        for _ in range(8)
    ]
    result = feed.build_itinerary_feed(specs, budget=3)
    assert result.search_complete is True
    assert result.candidate_trip_count == 16
    assert set(result.feed.trips["trip_id"]) == {"T00"}
    assert result.stop_times_count == 2

    # Budget below the cheapest trip weight is a proven infeasibility.
    with pytest.raises(iz.ItineraryPlanningError) as exc:
        feed.build_itinerary_feed(specs, budget=1)
    assert exc.value.status == iz.STATUS_INFEASIBLE


def test_undetermined_status_when_node_cap_is_hit(monkeypatch):
    feed = sharing_feed()
    monkeypatch.setattr(iz, "NODE_CAP", 0)
    monkeypatch.setattr(iz, "EXACT_TRIP_LIMIT", 2)
    with pytest.raises(iz.ItineraryPlanningError) as exc:
        feed.build_itinerary_feed(SHARING_SPECS, budget=8)
    assert exc.value.status == iz.STATUS_UNDETERMINED
    # The report is still populated with full-feed earliest arrivals.
    assert (
        exc.value.report["full_feed_earliest_arrival_time"].dropna().tolist()
        == ["09:00:00", "08:55:00"]
    )


def test_bounded_search_success_reports_incomplete(monkeypatch):
    feed = sharing_feed()
    # Pretend the exhaustive thresholds are tighter than this small feed.
    monkeypatch.setattr(iz, "EXACT_TRIP_LIMIT", 2)
    result = feed.build_itinerary_feed(SHARING_SPECS, budget=8)
    assert result.search_complete is False
    assert result.search_status == iz.STATUS_OPTIMAL
    assert set(result.feed.trips["trip_id"]) == {"S1", "S2"}


@pytest.mark.parametrize("seed", range(8))
def test_random_feeds_match_brute_force_oracle(seed):
    rng = np.random.default_rng(seed)
    stops = [f"S{j}" for j in range(6)]
    trips = []
    n_trips = int(rng.integers(3, 7))
    for i in range(n_trips):
        n_events = int(rng.integers(2, 5))
        chosen = list(rng.choice(stops, size=n_events, replace=True))
        t0 = int(rng.integers(7 * 3600, 10 * 3600))
        events = []
        for j, sid in enumerate(chosen):
            arr = t0 + j * int(rng.integers(300, 1800))
            events.append(
                (
                    str(sid),
                    hp_seconds_to_timestr(arr),
                    hp_seconds_to_timestr(arr),
                )
            )
        trips.append(
            {
                "trip_id": f"T{i}",
                "route_id": f"R{i % 2}",
                "service_id": "SVC",
                "events": events,
            }
        )
    all_stops = _df(
        [
            {
                "stop_id": sid,
                "stop_name": sid,
                "stop_lat": 36.0 + j * 0.01,
                "stop_lon": -117.0 + j * 0.01,
            }
            for j, sid in enumerate(stops)
        ],
        STOP_DTYPES,
    )
    feed = make_feed(trips, stops=all_stops, calendar=weekly_calendar())
    specs = []
    for _ in range(3):
        o, d = rng.choice(stops, size=2, replace=False)
        specs.append(
            {
                "service_date": DATE,
                "origin_stop_id": str(o),
                "destination_stop_id": str(d),
                "earliest_boarding_time": "06:00:00",
                "max_lateness_seconds": int(rng.integers(0, 3600)),
            }
        )
    for budget in (4, 8, 12, 1000):
        verdict, subset, _ = brute_force_feasibility(feed, specs, budget)
        if verdict == "unreachable":
            with pytest.raises(iz.ItineraryPlanningError) as exc:
                feed.build_itinerary_feed(specs, budget)
            assert exc.value.status == iz.STATUS_UNREACHABLE
            return
        if verdict == "infeasible":
            with pytest.raises(iz.ItineraryPlanningError) as exc:
                feed.build_itinerary_feed(specs, budget)
            assert exc.value.status == iz.STATUS_INFEASIBLE
        else:
            result = feed.build_itinerary_feed(specs, budget)
            assert result.stop_times_count <= budget
            # Independently verify every itinerary on the delivered subset.
            for i, spec in enumerate(specs):
                arr = oracle_earliest_arrival(
                    feed, set(result.feed.trips["trip_id"]), spec
                )
                full = brute_force_feasibility(feed, [spec], 10**6)[2][0]
                assert arr is not None
                assert arr - full <= spec["max_lateness_seconds"]


def hp_seconds_to_timestr(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def test_used_shapes_are_retained_without_dangling_references():
    shape_rows = [
        {"shape_id": "SH1", "shape_pt_lat": 36.0, "shape_pt_lon": -117.0,
         "shape_pt_sequence": 1},
        {"shape_id": "SH1", "shape_pt_lat": 36.1, "shape_pt_lon": -117.1,
         "shape_pt_sequence": 2},
        {"shape_id": "SH2", "shape_pt_lat": 37.0, "shape_pt_lon": -118.0,
         "shape_pt_sequence": 1},
    ]
    shapes = _df(shape_rows, gk.DTYPES["shapes"])
    trips = [
        {
            "trip_id": "T1",
            "route_id": "R1",
            "service_id": "SVC",
            "events": [
                ("A", "08:00:00", "08:00:00"),
                ("C", "09:00:00", "09:00:00"),
            ],
        },
    ]
    feed = make_feed(trips, calendar=weekly_calendar())
    feed.shapes = shapes
    feed.trips["shape_id"] = "SH1"
    spec = [
        {
            "service_date": DATE,
            "origin_stop_id": "A",
            "destination_stop_id": "C",
            "earliest_boarding_time": "07:00:00",
            "max_lateness_seconds": 0,
        }
    ]
    result = feed.build_itinerary_feed(spec, budget=5)
    assert set(result.feed.shapes["shape_id"]) == {"SH1"}
    assert result.feed.trips["shape_id"].iloc[0] == "SH1"


def test_blank_transfer_type_treated_as_recommended_and_same_stop_type3():
    trips = [
        {
            "trip_id": "T1",
            "route_id": "R1",
            "service_id": "SVC",
            "events": [
                ("A", "08:00:00", "08:00:00"),
                ("X", "08:30:00", "08:30:00"),
            ],
        },
        {
            "trip_id": "T2",
            "route_id": "R2",
            "service_id": "SVC",
            "events": [
                ("Y", "08:40:00", "08:40:00"),
                ("C", "09:10:00", "09:10:00"),
            ],
        },
    ]
    # Blank transfer type is accepted as a recommended transfer.
    transfers = [
        {
            "from_stop_id": "X",
            "to_stop_id": "Y",
            "transfer_type": None,
            "min_transfer_time": None,
        }
    ]
    feed = make_feed(trips, calendar=weekly_calendar(), transfers=transfers)
    spec = [
        {
            "service_date": DATE,
            "origin_stop_id": "A",
            "destination_stop_id": "C",
            "earliest_boarding_time": "07:00:00",
            "max_lateness_seconds": 0,
        }
    ]
    result = feed.build_itinerary_feed(spec, budget=10)
    assert set(result.feed.trips["trip_id"]) == {"T1", "T2"}

    # An explicit same-stop type 3 rule forbids even same-platform transfers.
    trips2 = [
        {
            "trip_id": "T1",
            "route_id": "R1",
            "service_id": "SVC",
            "events": [
                ("A", "08:00:00", "08:00:00"),
                ("X", "08:30:00", "08:30:00"),
            ],
        },
        {
            "trip_id": "T2",
            "route_id": "R2",
            "service_id": "SVC",
            "events": [
                ("X", "08:40:00", "08:40:00"),
                ("C", "09:10:00", "09:10:00"),
            ],
        },
    ]
    transfers2 = [
        {
            "from_stop_id": "X",
            "to_stop_id": "X",
            "transfer_type": 3,
            "min_transfer_time": None,
        }
    ]
    feed2 = make_feed(trips2, calendar=weekly_calendar(), transfers=transfers2)
    with pytest.raises(iz.ItineraryPlanningError) as exc:
        feed2.build_itinerary_feed(spec, budget=10)
    assert exc.value.status == iz.STATUS_UNREACHABLE


def test_restrict_to_dates_exact_skipping_days():
    feed = sharing_feed()
    d1, d2 = "20250106", "20250110"  # Monday and Friday, skipping mid-week
    # Default behaviour unchanged: weekly calendar survives.
    legacy = feed.restrict_to_dates([d1, d2])
    assert legacy.calendar is not None and not legacy.calendar.empty
    exact = feed.restrict_to_dates([d1, d2], exact=True)
    assert exact.calendar is None
    assert set(exact.calendar_dates["date"]) == {d1, d2}
    # Activity matches the original feed on the named dates.
    for d in (d1, d2):
        orig = set(
            feed.compute_trip_activity([d])
            .loc[lambda x: x[d] == 1, "trip_id"]
        )
        new = set(
            exact.compute_trip_activity([d])
            .loc[lambda x: x[d] == 1, "trip_id"]
        )
        assert orig == new
    # Skipped and outside dates run no trips.
    for d in ("20250107", "20250108"):
        activity = exact.compute_trip_activity([d])
        assert activity.empty or activity[d].sum() == 0


def test_exact_mode_preserves_exception_removal():
    weekday = [
        {
            "service_id": "SVC",
            "monday": 1,
            "tuesday": 1,
            "wednesday": 1,
            "thursday": 1,
            "friday": 1,
            "saturday": 0,
            "sunday": 0,
            "start_date": "20250101",
            "end_date": "20251231",
        }
    ]
    removed = "20250108"  # Wednesday removed
    feed = make_feed(
        [
            {
                "trip_id": "T1",
                "route_id": "R1",
                "service_id": "SVC",
                "events": [
                    ("A", "08:00:00", "08:00:00"),
                    ("C", "09:00:00", "09:00:00"),
                ],
            }
        ],
        calendar=weekday,
        calendar_dates=[
            {"service_id": "SVC", "date": removed, "exception_type": 2}
        ],
    )
    exact = feed.restrict_to_dates(["20250106", removed], exact=True)
    assert exact.get_active_services(removed) == []
    assert exact.get_active_services("20250106") == ["SVC"]

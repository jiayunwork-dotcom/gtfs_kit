"""
Itinerary-preserving feed reduction.

This module implements :func:`build_itinerary_feed`, a Feed entry point that
takes a list of required itineraries plus a budget on the number of retained
``stop_times`` rows and returns

- a reduced Feed in which every itinerary remains feasible within its own
  arrival tolerance, and
- an audit report describing, for each itinerary, the earliest arrival in the
  original feed and the one or two vehicle trips that deliver the reported
  arrival in the reduced feed.

The journey semantics are deliberately narrow and explicit:

- Journeys stay within one GTFS service day (times past 24:00:00 remain on the
  service day that starts the trip).
- A journey is either one direct trip or exactly one transfer between two
  different trips.
- Boarding/alighting is allowed only at stop events whose ``pickup_type`` /
  ``drop_off_type`` is 0 or missing and that carry the required departure /
  arrival time.
- A trip visiting the same stop twice is handled through ``stop_sequence``;
  later boards may not backtrack to an earlier alighting stop.
- Same-stop transfers are always permitted unless forbidden by a type 3 rule;
  cross-stop transfers require an explicit stop-level rule.
- Only fixed-schedule trips and stop-level transfers are in scope.  Feeds with
  non-empty ``frequencies``, fare tables, attributions, or transfer rules that
  the entry point cannot honour are rejected with a descriptive error.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

import pandas as pd

from . import constants as cs
from . import helpers as hp

if TYPE_CHECKING:
    from .feed import Feed

#: Columns of the audit report, in canonical order.
REPORT_COLUMNS = [
    "itinerary_id",
    "service_date",
    "origin_stop_id",
    "destination_stop_id",
    "earliest_boarding_time",
    "max_lateness_seconds",
    "full_feed_earliest_arrival_time",
    "reduced_feed_earliest_arrival_time",
    "lateness_seconds",
    "leg_sequence",
    "trip_id",
    "board_stop_id",
    "alight_stop_id",
    "board_stop_sequence",
    "alight_stop_sequence",
    "arrival_time",
    "departure_time",
]

#: Status returned when every itinerary has at least one feasible option and a
#: combination within budget was proven to exist.
STATUS_OPTIMAL = "optimal"

#: Status returned when some itinerary is unreachable in the full feed.
STATUS_UNREACHABLE = "unreachable"

#: Status returned when exhaustive search proved no budget-feasible
#: combination exists.
STATUS_INFEASIBLE = "infeasible"

#: Status returned when the search effort cap was hit before feasibility was
#: decided.
STATUS_UNDETERMINED = "undetermined"

#: Trip-pool size under which exhaustive search is guaranteed.
EXACT_TRIP_LIMIT = 16

#: Itinerary-count size under which exhaustive search is guaranteed.
EXACT_ITINERARY_LIMIT = 8

#: Maximum number of journey options retained per itinerary before the search
#: is declared out of the exhaustive regime.
OPTIONS_CAP = 200

#: Maximum number of DFS nodes visited in the bounded regime.
NODE_CAP = 200_000

#: Maximum number of candidate trips considered in the bounded regime.
CANDIDATE_TRIP_CAP = 2_000


class ItineraryPlanningError(Exception):
    """
    Raised when no reduced feed can be delivered.

    ``status`` is one of :data:`STATUS_UNREACHABLE`, :data:`STATUS_INFEASIBLE`,
    or :data:`STATUS_UNDETERMINED`.  ``report`` holds the per-itinerary audit
    rows computed so far (the full-feed part is always populated), and
    ``message`` explains the outcome.
    """

    def __init__(self, status: str, message: str, report: pd.DataFrame | None = None):
        super().__init__(message)
        self.status = status
        self.report = report
        self.message = message


@dataclass
class ItineraryFeedResult:
    """
    Result of a successful :func:`build_itinerary_feed` call.

    Attributes:
        feed: the reduced, independently savable Feed.
        report: audit report in the column order of :data:`REPORT_COLUMNS`,
            with one or two rows per itinerary (one row per used leg).
        budget: the requested stop-times row budget.
        stop_times_count: number of ``stop_times`` rows delivered.
        trip_count: number of trips delivered.
        search_status: always :data:`STATUS_OPTIMAL` for a delivered result.
        search_complete: ``True`` when the delivered combination was found by
            exhaustive search; ``False`` when a bounded search happened to
            succeed on a large feed.
        search_visited_nodes: number of search nodes visited.
        candidate_trip_count: number of distinct trips appearing in at least
            one journey option.
        options_per_itinerary: number of admissible journey options found for
            each itinerary (in itinerary order).
    """

    feed: "Feed"
    report: pd.DataFrame
    budget: int
    stop_times_count: int
    trip_count: int
    search_status: str = STATUS_OPTIMAL
    search_complete: bool = True
    search_visited_nodes: int = 0
    candidate_trip_count: int = 0
    options_per_itinerary: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------
# Shared usable-stop-event semantics
# ---------------------------------------------------------------------
def boardable_mask(stop_times: pd.DataFrame) -> pd.Series:
    """
    Boolean mask marking stop events usable for boarding:
    ``pickup_type`` is 0 or missing and ``departure_time`` is present.
    """
    mask = stop_times["departure_time"].notna()
    if "pickup_type" in stop_times.columns:
        mask &= stop_times["pickup_type"].isna() | (stop_times["pickup_type"] == 0)
    return mask


def alightable_mask(stop_times: pd.DataFrame) -> pd.Series:
    """
    Boolean mask marking stop events usable for alighting:
    ``drop_off_type`` is 0 or missing and ``arrival_time`` is present.
    """
    mask = stop_times["arrival_time"].notna()
    if "drop_off_type" in stop_times.columns:
        mask &= stop_times["drop_off_type"].isna() | (stop_times["drop_off_type"] == 0)
    return mask


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------
def _validate_budget(budget: Any) -> int:
    # bool is a subclass of int, so reject it explicitly.
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError(
            "budget must be a positive integer number of stop_times rows, "
            f"got {budget!r}"
        )
    return budget


def _validate_date(value: Any, feed_dates: set[str], context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a YYYYMMDD date string, got {value!r}")
    try:
        dt.datetime.strptime(value, "%Y%m%d")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{context} must be a YYYYMMDD date string, got {value!r}"
        ) from exc
    if value not in feed_dates:
        raise ValueError(
            f"{context} {value!r} is not a service date of the feed; "
            f"feed dates range over {min(feed_dates) if feed_dates else 'n/a'} "
            f"to {max(feed_dates) if feed_dates else 'n/a'}"
        )
    return value


def _validate_time(value: Any, context: str) -> int:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be an HH:MM:SS time string, got {value!r}")
    seconds = hp.timestr_to_seconds(value)
    if pd.isna(seconds) or seconds < 0:
        raise ValueError(f"{context} must be an HH:MM:SS time string, got {value!r}")
    return int(seconds)


def _validate_tolerance(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            "max_lateness_seconds must be a non-negative integer, got "
            f"{value!r}"
        )
    return value


def _coerce_itineraries(itineraries: Any) -> list[dict]:
    """
    Normalize the itinerary argument to a list of plain dicts and validate
    field presence and scalar types that do not depend on feed contents.
    """
    if isinstance(itineraries, pd.DataFrame):
        if itineraries.empty:
            raise ValueError("itineraries must not be empty")
        records = itineraries.to_dict(orient="records")
    elif isinstance(itineraries, Iterable) and not isinstance(itineraries, (str, bytes)):
        records = list(itineraries)
    else:
        raise ValueError(
            "itineraries must be a non-empty list of dicts or a DataFrame"
        )

    if not records:
        raise ValueError("itineraries must not be empty")

    required = [
        "service_date",
        "origin_stop_id",
        "destination_stop_id",
        "earliest_boarding_time",
        "max_lateness_seconds",
    ]
    normalized = []
    for i, record in enumerate(records):
        context = f"itinerary {i}"
        if not isinstance(record, dict):
            raise ValueError(f"{context} must be a dict, got {type(record)!r}")
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"{context} is missing fields: {missing}")
        normalized.append(dict(record))

    return normalized


def _validate_feed_scope(feed: "Feed") -> None:
    """
    Reject feeds carrying data that the entry point cannot faithfully handle.
    """
    problems = []
    if feed.frequencies is not None and not feed.frequencies.empty:
        problems.append(
            "non-empty `frequencies` table: only fixed-schedule trips are "
            "supported; frequency-based trips are out of scope"
        )
    if feed.fare_rules is not None and not feed.fare_rules.empty:
        problems.append(
            "non-empty `fare_rules` table: fare data is out of scope and must "
            "be removed before calling this entry point"
        )
    if feed.fare_attributes is not None and not feed.fare_attributes.empty:
        problems.append(
            "non-empty `fare_attributes` table: fare data is out of scope and "
            "must be removed before calling this entry point"
        )
    if feed.attributions is not None and not feed.attributions.empty:
        problems.append(
            "non-empty `attributions` table: attributions are out of scope and "
            "must be removed before calling this entry point"
        )
    if problems:
        raise ValueError("Feed is outside the supported scope: " + "; ".join(problems))

    if feed.transfers is not None and not feed.transfers.empty:
        tr = feed.transfers
        # GTFS v2 route/trip-scoped transfer rules.
        for col, label in [
            ("from_route_id", "route-scoped"),
            ("to_route_id", "route-scoped"),
            ("from_trip_id", "trip-scoped"),
            ("to_trip_id", "trip-scoped"),
        ]:
            if col in tr.columns and tr[col].notna().any():
                raise ValueError(
                    "Feed is outside the supported scope: "
                    f"{label} transfer rule in column {col!r}; only "
                    "stop-level transfers are supported"
                )
        if "transfer_type" not in tr.columns:
            raise ValueError(
                "transfers table has no `transfer_type` column; cannot "
                "interpret the transfer rules"
            )
        known = tr["transfer_type"].dropna().unique().tolist()
        unsupported = [t for t in known if t not in (0, 2, 3)]
        if unsupported:
            raise ValueError(
                "Feed is outside the supported scope: transfer_type values "
                f"{sorted(unsupported)} are not supported; only 0 (or blank), "
                "2 (minimum time) and 3 (forbidden) are handled"
            )
        type2 = tr[(tr["transfer_type"] == 2)]
        bad_time = type2["min_transfer_time"].isna() if (
            "min_transfer_time" in type2.columns
        ) else pd.Series(True, index=type2.index)
        if bad_time.any():
            raise ValueError(
                "transfer_type 2 rules require a non-null `min_transfer_time`"
            )
        # Endpoints must be stop IDs actually served by some trip.
        known_stops = set(feed.stops["stop_id"])
        served_stops = set(feed.stop_times["stop_id"])
        parent_stations = set()
        if "location_type" in feed.stops.columns:
            parent_stations = set(
                feed.stops.loc[feed.stops["location_type"] == 1, "stop_id"]
            )
        for col in ("from_stop_id", "to_stop_id"):
            if col not in tr.columns:
                raise ValueError(f"transfers table is missing column {col!r}")
            endpoints = set(tr[col].dropna())
            unknown = sorted(endpoints - known_stops)
            if unknown:
                raise ValueError(
                    "transfers table references stop IDs that are not stops "
                    f"of the feed: {unknown}"
                )
            not_served = sorted(endpoints - served_stops)
            if not_served:
                parent_hits = sorted(set(not_served) & parent_stations)
                detail = (
                    f"; parent stations are not served stops: {parent_hits}"
                    if parent_hits
                    else ""
                )
                raise ValueError(
                    "transfers table endpoints must be actually served stop "
                    f"IDs; these are not visited by any trip: {not_served}"
                    + detail
                )


# ---------------------------------------------------------------------
# Internal journey model
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class _Leg:
    trip_id: str
    board_stop_id: str
    alight_stop_id: str
    board_sequence: int
    alight_sequence: int
    departure_seconds: int
    arrival_seconds: int

    def as_report_dict(self) -> dict:
        return {
            "trip_id": self.trip_id,
            "board_stop_id": self.board_stop_id,
            "alight_stop_id": self.alight_stop_id,
            "board_stop_sequence": int(self.board_sequence),
            "alight_stop_sequence": int(self.alight_sequence),
            "arrival_time": hp.seconds_to_timestr(self.arrival_seconds),
            "departure_time": hp.seconds_to_timestr(self.departure_seconds),
        }


@dataclass(frozen=True)
class _Option:
    """One admissible journey (one or two legs) for one itinerary."""

    legs: tuple[_Leg, ...]
    arrival_seconds: int
    trip_ids: frozenset[str]


@dataclass
class _ItinerarySpec:
    index: int
    service_date: str
    origin_stop_id: str
    destination_stop_id: str
    earliest_boarding_seconds: int
    max_lateness_seconds: int


class _JourneyEngine:
    """
    Precomputes, for one feed, everything needed to enumerate direct and
    one-transfer journeys under the module's boarding/transfer semantics.
    """

    def __init__(self, feed: "Feed"):
        self.feed = feed
        st = feed.stop_times.copy()
        st["arr_sec"] = st["arrival_time"].map(hp.timestr_to_seconds)
        st["dep_sec"] = st["departure_time"].map(hp.timestr_to_seconds)
        st["arr_sec"] = st["arr_sec"].astype("Float64")
        st["dep_sec"] = st["dep_sec"].astype("Float64")
        self.stop_times = st.sort_values(
            ["trip_id", "stop_sequence"], ignore_index=True
        )

        # Per-trip event lists, ordered by stop_sequence.
        self.events_by_trip: dict[str, pd.DataFrame] = {
            trip_id: group.reset_index(drop=True)
            for trip_id, group in self.stop_times.groupby("trip_id", sort=True)
        }

        # Row count contributed by retaining a trip (its full stop_times).
        self.trip_weight: dict[str, int] = (
            self.stop_times.groupby("trip_id").size().astype(int).to_dict()
        )

        # Active trip sets per service date, using the feed's own
        # calendar / calendar_dates semantics.
        self.active_trips_by_date: dict[str, set[str]] = {}
        self.trip_service_id: dict[str, str] = dict(
            zip(feed.trips["trip_id"], feed.trips["service_id"])
        )

        # Transfer rule lookup: (from_stop_id, to_stop_id) -> rule row dict.
        self.transfer_rules: dict[tuple[str, str], dict] = {}
        if feed.transfers is not None and not feed.transfers.empty:
            for row in feed.transfers.itertuples(index=False):
                key = (row.from_stop_id, row.to_stop_id)
                self.transfer_rules[key] = {
                    "transfer_type": (
                        None if pd.isna(row.transfer_type) else int(row.transfer_type)
                    ),
                    "min_transfer_time": (
                        None
                        if "min_transfer_time"
                        not in feed.transfers.columns
                        or pd.isna(row.min_transfer_time)
                        else int(row.min_transfer_time)
                    ),
                }

        # Trips that board / alight at each stop (for fast candidate lookup).
        boardable = self.stop_times[boardable_mask(self.stop_times)]
        alightable = self.stop_times[alightable_mask(self.stop_times)]
        self.board_trips_by_stop: dict[str, set[str]] = (
            boardable.groupby("stop_id")["trip_id"].agg(set).to_dict()
        )
        self.alight_trips_by_stop: dict[str, set[str]] = (
            alightable.groupby("stop_id")["trip_id"].agg(set).to_dict()
        )

    # -- service dates --------------------------------------------------
    def active_trips(self, date: str) -> set[str]:
        if date not in self.active_trips_by_date:
            services = set(self.feed.get_active_services(date))
            trips = {
                trip_id
                for trip_id, service_id in self.trip_service_id.items()
                if service_id in services
            }
            self.active_trips_by_date[date] = trips
        return self.active_trips_by_date[date]

    # -- transfer eligibility ------------------------------------------
    def transfer_allowed(
        self,
        alight_stop_id: str,
        board_stop_id: str,
        alight_seconds: int,
        board_seconds: int,
    ) -> bool:
        """
        Decide whether the second leg may follow the first.

        Same-stop transfers need no rule except an explicit type 3
        prohibition; cross-stop transfers require an explicit rule.
        """
        if board_seconds < alight_seconds:
            return False
        if alight_stop_id == board_stop_id:
            rule = self.transfer_rules.get((alight_stop_id, board_stop_id))
            if rule is not None and rule["transfer_type"] == 3:
                return False
            if rule is not None and rule["transfer_type"] == 2:
                return board_seconds - alight_seconds >= rule["min_transfer_time"]
            # Type 0, blank, or no rule at all: time ordering suffices.
            return True

        rule = self.transfer_rules.get((alight_stop_id, board_stop_id))
        if rule is None:
            return False
        if rule["transfer_type"] == 3:
            return False
        if rule["transfer_type"] == 2:
            return board_seconds - alight_seconds >= rule["min_transfer_time"]
        # Type 0 or blank type.
        return True

    # -- journey enumeration -------------------------------------------
    def enumerate_options(self, spec: _ItinerarySpec) -> list[_Option]:
        """
        Return all admissible direct and one-transfer journey options for the
        spec, ordered by (arrival seconds, trip IDs) for determinism.
        """
        active = self.active_trips(spec.service_date)
        options: list[_Option] = []

        # Direct legs.
        for trip_id in sorted(
            (self.board_trips_by_stop.get(spec.origin_stop_id, set()))
            & (self.alight_trips_by_stop.get(spec.destination_stop_id, set()))
            & active
        ):
            leg = self._build_leg(
                trip_id,
                spec.origin_stop_id,
                spec.destination_stop_id,
                spec.earliest_boarding_seconds,
            )
            if leg is not None:
                options.append(
                    _Option(
                        legs=(leg,),
                        arrival_seconds=leg.arrival_seconds,
                        trip_ids=frozenset((trip_id,)),
                    )
                )

        # One-transfer legs: trip 1 boards at origin and alights at some
        # stop, trip 2 boards there and alights at destination.
        first_candidates = sorted(
            (self.board_trips_by_stop.get(spec.origin_stop_id, set())) & active
        )
        second_candidates = sorted(
            (self.alight_trips_by_stop.get(spec.destination_stop_id, set())) & active
        )

        # A pair of trips may connect at more than one stop; retain only the
        # earliest-arriving connection per unordered pair of trip IDs.
        best_by_pair: dict[frozenset[str], _Option] = {}
        for t1 in first_candidates:
            events1 = self.events_by_trip[t1]
            origin_events = events1[
                (events1["stop_id"] == spec.origin_stop_id)
                & boardable_mask(events1)
                & (events1["dep_sec"] >= spec.earliest_boarding_seconds)
            ]
            if origin_events.empty:
                continue
            first_board_seq = int(origin_events["stop_sequence"].iloc[0])
            # Possible alighting events after boarding.
            alight_events1 = events1[
                (events1["stop_sequence"] > first_board_seq)
                & alightable_mask(events1)
            ]
            for t2 in second_candidates:
                if t2 == t1:
                    continue
                events2 = self.events_by_trip[t2]
                dest_events = events2[
                    (events2["stop_id"] == spec.destination_stop_id)
                    & alightable_mask(events2)
                ]
                if dest_events.empty:
                    continue
                last_dest_seq = int(dest_events["stop_sequence"].iloc[-1])
                join = self._best_join(
                    alight_events1, events2, last_dest_seq
                )
                if join is None:
                    continue
                alight_row, board_row = join
                leg1 = _Leg(
                    trip_id=t1,
                    board_stop_id=spec.origin_stop_id,
                    alight_stop_id=alight_row.stop_id,
                    board_sequence=first_board_seq,
                    alight_sequence=int(alight_row.stop_sequence),
                    departure_seconds=int(origin_events["dep_sec"].iloc[0]),
                    arrival_seconds=int(alight_row.arr_sec),
                )
                leg2 = _Leg(
                    trip_id=t2,
                    board_stop_id=board_row.stop_id,
                    alight_stop_id=spec.destination_stop_id,
                    board_sequence=int(board_row.stop_sequence),
                    alight_sequence=last_dest_seq,
                    departure_seconds=int(board_row.dep_sec),
                    arrival_seconds=int(dest_events["arr_sec"].iloc[-1]),
                )
                option = _Option(
                    legs=(leg1, leg2),
                    arrival_seconds=leg2.arrival_seconds,
                    trip_ids=frozenset((t1, t2)),
                )
                incumbent = best_by_pair.get(option.trip_ids)
                if incumbent is None or option.arrival_seconds < incumbent.arrival_seconds:
                    best_by_pair[option.trip_ids] = option

        options.extend(best_by_pair.values())
        options.sort(key=lambda o: (o.arrival_seconds, tuple(sorted(o.trip_ids))))
        return options

    def _build_leg(
        self,
        trip_id: str,
        origin_stop_id: str,
        destination_stop_id: str,
        earliest_boarding_seconds: int,
    ) -> _Leg | None:
        events = self.events_by_trip[trip_id]
        board_events = events[
            (events["stop_id"] == origin_stop_id)
            & boardable_mask(events)
            & (events["dep_sec"] >= earliest_boarding_seconds)
        ]
        if board_events.empty:
            return None
        board_row = board_events.iloc[0]
        alight_events = events[
            (events["stop_id"] == destination_stop_id)
            & (events["stop_sequence"] > int(board_row["stop_sequence"]))
            & alightable_mask(events)
        ]
        if alight_events.empty:
            return None
        alight_row = alight_events.iloc[0]
        return _Leg(
            trip_id=trip_id,
            board_stop_id=origin_stop_id,
            alight_stop_id=destination_stop_id,
            board_sequence=int(board_row["stop_sequence"]),
            alight_sequence=int(alight_row["stop_sequence"]),
            departure_seconds=int(board_row["dep_sec"]),
            arrival_seconds=int(alight_row["arr_sec"]),
        )

    def _best_join(
        self,
        alight_events1: pd.DataFrame,
        events2: pd.DataFrame,
        last_dest_seq: int,
    ) -> tuple[pd.Series, pd.Series] | None:
        """
        Find the earliest feasible (alight on trip 1, board on trip 2) pair
        under the transfer rules, with no backtracking on trip 2.
        """
        best_key = None
        best_pair = None
        for a in alight_events1.itertuples(index=False):
            board_events2 = events2[
                (events2["stop_sequence"] < last_dest_seq)
                & boardable_mask(events2)
            ]
            for b in board_events2.itertuples(index=False):
                if not self.transfer_allowed(
                    a.stop_id,
                    b.stop_id,
                    int(a.arr_sec),
                    int(b.dep_sec),
                ):
                    continue
                # The t2 destination arrival is fixed, so rank joins by
                # boarding time, then by alighting time and sequences.
                key = (
                    int(b.dep_sec),
                    int(a.arr_sec),
                    int(a.stop_sequence),
                    int(b.stop_sequence),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_pair = (a, b)
        return best_pair


# ---------------------------------------------------------------------
# Combination search
# ---------------------------------------------------------------------
def _search_combination(
    options_by_itinerary: list[list[_Option]],
    trip_weight: dict[str, int],
    budget: int,
    exact: bool,
) -> tuple[frozenset[str] | None, int, str]:
    """
    Choose one option per itinerary so that the union of used trips has total
    stop-times weight within budget.

    Deterministic DFS with branch-and-bound, memoized on
    ``(itinerary index, chosen trip set)``.  Itineraries with fewest options
    are decided first; options that add the least marginal weight are tried
    first.

    Returns ``(chosen_trip_ids, visited_nodes, status)``.
    """
    order = sorted(
        range(len(options_by_itinerary)),
        key=lambda i: (len(options_by_itinerary[i]), i),
    )
    visited = 0
    cap = None if exact else NODE_CAP
    if cap is not None and cap < 1:
        return None, 0, STATUS_UNDETERMINED
    memo: set[tuple[int, frozenset[str]]] = set()
    solution: list[frozenset[str]] = []

    def marginal(option: _Option, chosen: frozenset[str]) -> int:
        return sum(trip_weight[t] for t in option.trip_ids - chosen)

    def dfs(pos: int, chosen: frozenset[str]) -> bool:
        nonlocal visited
        visited += 1
        if cap is not None and visited > cap:
            return False
        if pos == len(order):
            solution.append(chosen)
            return True
        key = (pos, chosen)
        if key in memo:
            return False
        i = order[pos]
        ranked = sorted(
            options_by_itinerary[i],
            key=lambda o: (marginal(o, chosen), o.arrival_seconds, tuple(sorted(o.trip_ids))),
        )
        for option in ranked:
            new_chosen = chosen | option.trip_ids
            weight = sum(trip_weight[t] for t in new_chosen)
            if weight > budget:
                continue
            if dfs(pos + 1, new_chosen):
                return True
        memo.add(key)
        return False

    feasible = dfs(0, frozenset())
    if feasible:
        return solution[0], visited, STATUS_OPTIMAL
    if cap is not None and visited > cap:
        return None, visited, STATUS_UNDETERMINED
    return None, visited, STATUS_INFEASIBLE


# ---------------------------------------------------------------------
# Reduced feed construction
# ---------------------------------------------------------------------
def build_exact_calendar(
    feed: "Feed", service_ids: Iterable[str], dates: list[str]
) -> tuple[None, pd.DataFrame]:
    """
    Build precise service-calendar tables restricted to ``dates``.

    The returned calendar is always ``None`` and the calendar_dates frame
    contains one type-1 (added) row per (service, date) pair that is active in
    the original feed on the date.  This encodes the exact active set for
    every service, regardless of whether the source feed used weekly
    calendars, date exceptions only, or both.
    """
    service_ids = set(service_ids)
    rows = []
    for date in sorted(set(dates)):
        active = set(feed.get_active_services(date)) & service_ids
        for service_id in sorted(active):
            rows.append(
                {
                    "service_id": service_id,
                    "date": date,
                    "exception_type": 1,
                }
            )
    calendar_dates = pd.DataFrame(rows, columns=["service_id", "date", "exception_type"])
    if not calendar_dates.empty:
        calendar_dates["exception_type"] = calendar_dates["exception_type"].astype("Int32")
    else:
        calendar_dates = calendar_dates.astype(
            {"service_id": "string", "date": "string", "exception_type": "Int32"}
        )
    return None, calendar_dates


def _build_reduced_feed(
    feed: "Feed",
    chosen_trip_ids: frozenset[str],
    dates: list[str],
) -> "Feed":
    new_feed = feed.copy()
    chosen = set(chosen_trip_ids)

    # Trips and their full stop_times.
    new_feed.trips = feed.trips.loc[
        lambda x: x["trip_id"].isin(chosen)
    ].copy()
    new_feed.stop_times = feed.stop_times.loc[
        lambda x: x["trip_id"].isin(chosen)
    ].copy()

    # Routes, agencies.
    route_ids = set(new_feed.trips["route_id"])
    new_feed.routes = feed.routes.loc[
        lambda x: x["route_id"].isin(route_ids)
    ].copy()
    if (
        feed.agency is not None
        and "agency_id" in new_feed.routes.columns
        and "agency_id" in feed.agency.columns
    ):
        agency_ids = set(new_feed.routes["agency_id"].dropna())
        new_feed.agency = feed.agency.loc[
            lambda x: x["agency_id"].isin(agency_ids)
        ].copy()

    # Stops plus parent stations (no dangling parent references).
    stop_ids = set(new_feed.stop_times["stop_id"])
    if feed.stops is not None and "parent_station" in feed.stops.columns:
        parent_col = feed.stops.loc[
            feed.stops["stop_id"].isin(stop_ids), "parent_station"
        ]
        stop_ids |= set(parent_col[parent_col.notna()])
    new_feed.stops = feed.stops.loc[
        lambda x: x["stop_id"].isin(stop_ids)
    ].copy()

    # Shapes actually used.
    if feed.shapes is not None and "shape_id" in new_feed.trips.columns:
        shape_ids = set(new_feed.trips["shape_id"].dropna())
        new_feed.shapes = feed.shapes.loc[
            lambda x: x["shape_id"].isin(shape_ids)
        ].copy()
    else:
        new_feed.shapes = None

    # Exact service calendar on the requested dates only.
    service_ids = set(new_feed.trips["service_id"])
    new_feed.calendar, new_feed.calendar_dates = build_exact_calendar(
        feed, service_ids, dates
    )

    # Transfers: keep every in-scope rule whose endpoints are both delivered.
    if feed.transfers is not None:
        new_feed.transfers = feed.transfers.loc[
            lambda x: x["from_stop_id"].isin(stop_ids)
            & x["to_stop_id"].isin(stop_ids)
        ].copy()
    else:
        new_feed.transfers = None

    # Out-of-scope optional tables are never emitted.
    new_feed.frequencies = None
    new_feed.fare_attributes = None
    new_feed.fare_rules = None
    new_feed.attributions = None

    # feed_info service window, when present, must match the delivered window.
    if feed.feed_info is not None and not feed.feed_info.empty:
        info = feed.feed_info.copy()
        if "feed_start_date" in info.columns:
            info["feed_start_date"] = min(dates)
        if "feed_end_date" in info.columns:
            info["feed_end_date"] = max(dates)
        new_feed.feed_info = info

    return new_feed


# ---------------------------------------------------------------------
# Report construction
# ---------------------------------------------------------------------
def _report_rows(
    spec: _ItinerarySpec,
    full_earliest: int | None,
    reduced_option: _Option | None,
) -> list[dict]:
    base = {
        "itinerary_id": spec.index,
        "service_date": spec.service_date,
        "origin_stop_id": spec.origin_stop_id,
        "destination_stop_id": spec.destination_stop_id,
        "earliest_boarding_time": hp.seconds_to_timestr(
            spec.earliest_boarding_seconds
        ),
        "max_lateness_seconds": spec.max_lateness_seconds,
        "full_feed_earliest_arrival_time": (
            hp.seconds_to_timestr(full_earliest)
            if full_earliest is not None
            else pd.NA
        ),
        "reduced_feed_earliest_arrival_time": (
            hp.seconds_to_timestr(reduced_option.arrival_seconds)
            if reduced_option is not None
            else pd.NA
        ),
        "lateness_seconds": (
            int(reduced_option.arrival_seconds - full_earliest)
            if reduced_option is not None and full_earliest is not None
            else pd.NA
        ),
    }
    if reduced_option is None:
        row = dict(base)
        row.update(
            {
                "leg_sequence": pd.NA,
                "trip_id": pd.NA,
                "board_stop_id": pd.NA,
                "alight_stop_id": pd.NA,
                "board_stop_sequence": pd.NA,
                "alight_stop_sequence": pd.NA,
                "arrival_time": pd.NA,
                "departure_time": pd.NA,
            }
        )
        return [row]

    rows = []
    for seq, leg in enumerate(reduced_option.legs, start=1):
        row = dict(base)
        row.update({"leg_sequence": seq})
        row.update(leg.as_report_dict())
        rows.append(row)
    return rows


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------
def build_itinerary_feed(
    feed: "Feed",
    itineraries: list[dict] | pd.DataFrame,
    budget: int,
) -> ItineraryFeedResult:
    """
    Build a reduced Feed that preserves every required itinerary.

    Parameters
    ----------
    feed:
        The source :class:`Feed`.  It is never modified.
    itineraries:
        Non-empty list of dicts (or DataFrame) with the fields

        - ``service_date``: YYYYMMDD service day of the journey.
        - ``origin_stop_id``: stop ID where the journey starts.
        - ``destination_stop_id``: stop ID where the journey ends.
        - ``earliest_boarding_time``: HH:MM:SS (hours may exceed 23) earliest
          permitted boarding time on the service day.
        - ``max_lateness_seconds``: non-negative integer; the reduced feed may
          arrive at most this many seconds later than the full-feed earliest
          arrival.

        Duplicate itineraries are retained and reported in order.
    budget:
        Positive integer cap on the number of ``stop_times`` rows delivered.
        Retaining a trip always carries its full set of stop_times rows, and
        a trip shared by several itineraries is counted once.

    Returns
    -------
    ItineraryFeedResult
        Carries the savable reduced feed, the ordered audit report, and search
        metadata.  See :class:`ItineraryFeedResult`.

    Raises
    ------
    ValueError
        For invalid arguments or feeds outside the supported scope.
    ItineraryPlanningError
        When no reduced feed can be delivered.  Its ``status`` distinguishes
        ``'unreachable'`` (an itinerary fails already in the full feed),
        ``'infeasible'`` (exhaustive search proved no combination fits the
        budget), and ``'undetermined'`` (the search cap was hit on a large
        feed).  No partial feed is returned in any of these cases.

    Search guarantees
    -----------------
    When at most 16 distinct candidate trips and at most 8 itineraries are
    involved, the search is exhaustive: every feasible combination is
    considered, so a budget-feasible answer is always found and an
    ``'infeasible'`` verdict is a proof.  Larger feeds use the same
    branch-and-bound search with an explicit node cap and report
    ``search_complete=False`` on success or raise with
    ``status='undetermined'`` if the cap is reached first.
    """
    budget = _validate_budget(budget)
    _validate_feed_scope(feed)
    raw_specs = _coerce_itineraries(itineraries)

    feed_dates = set(feed.get_dates())
    if not feed_dates:
        raise ValueError(
            "the feed has no calendar or calendar_dates information, so no "
            "service dates can be evaluated"
        )
    stop_ids = set(feed.stops["stop_id"])
    specs: list[_ItinerarySpec] = []
    for i, record in enumerate(raw_specs):
        service_date = _validate_date(
            record["service_date"], feed_dates, f"itinerary {i} service_date"
        )
        origin = record["origin_stop_id"]
        destination = record["destination_stop_id"]
        if not isinstance(origin, str) or origin not in stop_ids:
            raise ValueError(
                f"itinerary {i} origin_stop_id {origin!r} is not a stop of "
                "the feed"
            )
        if not isinstance(destination, str) or destination not in stop_ids:
            raise ValueError(
                f"itinerary {i} destination_stop_id {destination!r} is not a "
                "stop of the feed"
            )
        earliest = _validate_time(
            record["earliest_boarding_time"],
            f"itinerary {i} earliest_boarding_time",
        )
        tolerance = _validate_tolerance(record["max_lateness_seconds"])
        specs.append(
            _ItinerarySpec(
                index=i,
                service_date=service_date,
                origin_stop_id=origin,
                destination_stop_id=destination,
                earliest_boarding_seconds=earliest,
                max_lateness_seconds=tolerance,
            )
        )

    engine = _JourneyEngine(feed)

    # 1. Enumerate full-feed options and establish the earliest arrival.
    options_by_itinerary: list[list[_Option]] = []
    full_earliest: list[int | None] = []
    unreachable_rows: list[dict] = []
    any_unreachable = False
    for spec in specs:
        options = engine.enumerate_options(spec)
        if not options:
            any_unreachable = True
            full_earliest.append(None)
            options_by_itinerary.append([])
            unreachable_rows.extend(_report_rows(spec, None, None))
            continue
        earliest = options[0].arrival_seconds
        full_earliest.append(earliest)
        admissible = [
            option
            for option in options
            if option.arrival_seconds - earliest <= spec.max_lateness_seconds
        ]
        options_by_itinerary.append(admissible)
        unreachable_rows.extend(
            _report_rows(spec, earliest, None)
        )

    if any_unreachable:
        report = pd.DataFrame(unreachable_rows).reindex(columns=REPORT_COLUMNS)
        raise ItineraryPlanningError(
            STATUS_UNREACHABLE,
            "one or more itineraries are not reachable in the full feed "
            "(direct or one-transfer journey within the service day); see "
            "the report for details",
            report,
        )

    # 2. Decide the search regime.
    candidate_trips = set()
    for options in options_by_itinerary:
        for option in options:
            candidate_trips |= option.trip_ids
    option_counts = [len(options) for options in options_by_itinerary]
    exact = (
        len(candidate_trips) <= EXACT_TRIP_LIMIT
        and len(specs) <= EXACT_ITINERARY_LIMIT
        and all(count <= OPTIONS_CAP for count in option_counts)
        and len(candidate_trips) <= CANDIDATE_TRIP_CAP
    )

    # 3. Search.
    chosen, visited, status = _search_combination(
        options_by_itinerary, engine.trip_weight, budget, exact
    )

    if status != STATUS_OPTIMAL:
        report = pd.DataFrame(
            row
            for spec, earliest in zip(specs, full_earliest)
            for row in _report_rows(spec, earliest, None)
        ).reindex(columns=REPORT_COLUMNS)
        if status == STATUS_INFEASIBLE:
            raise ItineraryPlanningError(
                STATUS_INFEASIBLE,
                f"exhaustive search proved that no trip combination fits the "
                f"budget of {budget} stop_times rows",
                report,
            )
        raise ItineraryPlanningError(
            STATUS_UNDETERMINED,
            "search effort cap reached before feasibility could be decided; "
            "raise the budget or narrow the itineraries",
            report,
        )

    # 4. Build the reduced feed and verify each itinerary against it.
    dates = sorted({spec.service_date for spec in specs})
    reduced_feed = _build_reduced_feed(feed, chosen, dates)
    verify_engine = _JourneyEngine(reduced_feed)

    report_frames = []
    for spec, earliest in zip(specs, full_earliest):
        verified = verify_engine.enumerate_options(spec)
        if not verified:
            raise ItineraryPlanningError(
                STATUS_UNDETERMINED,
                "internal verification failed: an itinerary disappeared from "
                "the reduced feed",
            )
        best = verified[0]
        if best.arrival_seconds - earliest > spec.max_lateness_seconds:
            raise ItineraryPlanningError(
                STATUS_UNDETERMINED,
                "internal verification failed: reduced arrival exceeds the "
                "itinerary tolerance",
            )
        report_frames.extend(_report_rows(spec, earliest, best))

    report = pd.DataFrame(report_frames).reindex(columns=REPORT_COLUMNS)
    # Deterministic report order: itinerary then leg.
    report = report.sort_values(
        ["itinerary_id", "leg_sequence"], kind="mergesort", ignore_index=True
    )

    return ItineraryFeedResult(
        feed=reduced_feed,
        report=report,
        budget=budget,
        stop_times_count=int(reduced_feed.stop_times.shape[0]),
        trip_count=int(reduced_feed.trips.shape[0]),
        search_status=STATUS_OPTIMAL,
        search_complete=exact,
        search_visited_nodes=visited,
        candidate_trip_count=len(candidate_trips),
        options_per_itinerary=option_counts,
    )

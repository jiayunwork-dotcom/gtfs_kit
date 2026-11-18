"""
Example: build a small offline timetable that preserves required itineraries.

Run with the project virtualenv, using isolated mode so no modules are
imported from untrusted data directories::

    .venv/bin/python -I examples/itinerary_feed_example.py

The script builds a synthetic in-memory feed, calls
``Feed.build_itinerary_feed`` with a list of required itineraries and a
stop_times row budget, writes the reduced feed to a ZIP, reads it back with
``read_feed``, and prints the audit report plus verification checks.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import gtfs_kit as gk


def build_demo_feed() -> gk.Feed:
    """Two corridors; each has a fast padded trip and a slow short trip."""
    trips = [
        {
            "trip_id": "F1",
            "route_id": "R1",
            "service_id": "SVC",
            "events": [
                ("F1_P0", "05:00:00", "05:00:00"),
                ("F1_P1", "05:00:00", "05:00:00"),
                ("F1_P2", "05:00:00", "05:00:00"),
                ("A", "08:00:00", "08:00:00"),
                ("B", "08:30:00", "08:30:00"),
                ("C", "09:00:00", "09:00:00"),
            ],
        },
        {
            "trip_id": "S1",
            "route_id": "R1",
            "service_id": "SVC",
            "events": [
                ("S1_P0", "05:00:00", "05:00:00"),
                ("A", "08:10:00", "08:10:00"),
                ("B", "08:45:00", "08:45:00"),
                ("C", "09:20:00", "09:20:00"),
            ],
        },
        {
            "trip_id": "F2",
            "route_id": "R2",
            "service_id": "SVC",
            "events": [
                ("F2_P0", "05:00:00", "05:00:00"),
                ("F2_P1", "05:00:00", "05:00:00"),
                ("F2_P2", "05:00:00", "05:00:00"),
                ("D", "08:00:00", "08:00:00"),
                ("E", "08:25:00", "08:25:00"),
                ("F", "08:55:00", "08:55:00"),
            ],
        },
        {
            "trip_id": "S2",
            "route_id": "R2",
            "service_id": "SVC",
            "events": [
                ("S2_P0", "05:00:00", "05:00:00"),
                ("D", "08:10:00", "08:10:00"),
                ("E", "08:40:00", "08:40:00"),
                ("F", "09:15:00", "09:15:00"),
            ],
        },
    ]
    stop_ids = sorted({event[0] for trip in trips for event in trip["events"]})
    stops_frame = gk.pd.DataFrame(
        [
            {
                "stop_id": sid,
                "stop_name": f"Stop {sid}",
                "stop_lat": 36.0 + i * 0.01,
                "stop_lon": -117.0 + i * 0.01,
            }
            for i, sid in enumerate(stop_ids)
        ]
    )
    stop_time_rows = []
    trip_rows = []
    route_rows = []
    for trip in trips:
        trip_rows.append(
            {
                "route_id": trip["route_id"],
                "service_id": trip["service_id"],
                "trip_id": trip["trip_id"],
            }
        )
        route_rows.append(
            {
                "route_id": trip["route_id"],
                "agency_id": "DTA",
                "route_short_name": trip["route_id"],
                "route_type": 3,
            }
        )
        for seq, (stop_id, arrival, departure) in enumerate(
            trip["events"], start=1
        ):
            stop_time_rows.append(
                {
                    "trip_id": trip["trip_id"],
                    "arrival_time": arrival,
                    "departure_time": departure,
                    "stop_id": stop_id,
                    "stop_sequence": seq,
                }
            )
    agency = gk.pd.DataFrame(
        [
            {
                "agency_id": "DTA",
                "agency_name": "Demo Agency",
                "agency_url": "http://example.com",
                "agency_timezone": "America/Los_Angeles",
            }
        ]
    )
    calendar = gk.pd.DataFrame(
        [
            {
                "service_id": "SVC",
                "monday": 1,
                "tuesday": 1,
                "wednesday": 1,
                "thursday": 1,
                "friday": 1,
                "saturday": 1,
                "sunday": 1,
                "start_date": "20250101",
                "end_date": "20251231",
            }
        ]
    )
    feed_info = gk.pd.DataFrame(
        [
            {
                "feed_publisher_name": "Demo",
                "feed_publisher_url": "http://example.com",
                "feed_lang": "en",
                "feed_start_date": "20250101",
                "feed_end_date": "20251231",
            }
        ]
    )
    return gk.Feed(
        dist_units="km",
        agency=agency,
        stops=stops_frame,
        routes=gk.pd.DataFrame(route_rows).drop_duplicates("route_id"),
        trips=gk.pd.DataFrame(trip_rows),
        stop_times=gk.pd.DataFrame(stop_time_rows),
        calendar=calendar,
        calendar_dates=None,
        fare_attributes=None,
        fare_rules=None,
        shapes=None,
        frequencies=None,
        transfers=None,
        feed_info=feed_info,
        attributions=None,
    )


def main() -> None:
    feed = build_demo_feed()
    itineraries = [
        {
            "service_date": "20250106",
            "origin_stop_id": "A",
            "destination_stop_id": "C",
            "earliest_boarding_time": "07:00:00",
            "max_lateness_seconds": 1500,
        },
        {
            "service_date": "20250106",
            "origin_stop_id": "D",
            "destination_stop_id": "F",
            "earliest_boarding_time": "07:00:00",
            "max_lateness_seconds": 1500,
        },
    ]
    # Fastest trips union (F1 + F2) costs 12 stop_times rows; the shared slow
    # pair (S1 + S2) costs 8 rows and fits within every tolerance.
    result = feed.build_itinerary_feed(itineraries, budget=8)

    print("=== retained trips ===")
    print(sorted(result.feed.trips["trip_id"]))
    print("stop_times rows delivered:", result.stop_times_count)
    print("search complete:", result.search_complete)
    print()
    print("=== audit report ===")
    print(result.report.to_string(index=False))

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "reduced.zip"
        result.feed.to_file(zip_path)
        assert zipfile.is_zipfile(zip_path)
        reread = gk.read_feed(zip_path, dist_units="km")
        print()
        print("=== zip roundtrip ===")
        print("files in zip:", sorted(zipfile.ZipFile(zip_path).namelist()))
        print("reread stop_times rows:", len(reread.stop_times))
        print(
            "reread active trips on 20250106:",
            reread.compute_trip_activity(["20250106"])
            .loc[lambda x: x["20250106"] == 1, "trip_id"]
            .tolist(),
        )
        print(
            "stop timetable at A (boardable only):",
            reread.build_stop_timetable(
                "A", ["20250106"], board_only=True
            )[["trip_id", "departure_time"]]
            .to_dict("records"),
        )
        print("feed_info window:", reread.feed_info.iloc[0].to_dict())


if __name__ == "__main__":
    main()

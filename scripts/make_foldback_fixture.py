"""
Build a small synthetic GTFS zip used by the calibrated-mileage tests.

Running this script writes ``tests/fixtures/foldback_gtfs.zip``.  It is kept
in ``scripts/`` so that the fixture directory stays free of Python modules.

The feed describes a fold-back route in a small town near Cairns, Australia:

- ``fold_shape`` runs east along a straight line from ``A`` to ``E`` and then
  turns round and comes back on a line 10 m to the north.  Every stop lies
  within a 20 m radius of both passes, so each of ``B``, ``C`` and ``D`` is
  served twice; ``C`` is visited outbound, then again after the intermediate
  stops ``D`` and ``E``.
- Its ``shape_dist_traveled`` ticks are deliberately non-uniform: the
  outbound leg gets 1000 m of mileage for 800 m of geometry, while the
  inbound leg gets only 500 m for ~810 m of geometry.
- ``short_trip`` covers only the middle of the same shape (``B`` to ``D``),
  so deriving its position from the whole-shape length would be wrong.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

FIXTURE = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "foldback_gtfs.zip"
)

# Base position and a roughly east-west axis.
LON0, LAT0 = 145.7700, -16.9200
DEG_PER_M_LAT = 1 / 111_320.0
DEG_PER_M_LON = 1 / (111_320.0 * 0.957)

# Outbound A..E at y = 0, then a 10 m jog north, then inbound E..A at y = 10.
X_COORDS = [0, 200, 400, 600, 800, 800, 600, 400, 200, 0]
Y_COORDS = [0, 0, 0, 0, 0, 10, 10, 10, 10, 10]

# Deliberately non-uniform ruler, in metres.  The first point is not at zero
# mileage from the geometry's point of view: it is given a 100 m head start.
TICKS = [100, 350, 600, 850, 1100, 1200, 1300, 1400, 1500, 1600]

# Stops are 0 m from the outbound line; the turnaround stop E sits midway
# between the two legs (5 m from each), and all other stops are 10 m from the
# inbound line.  A 20 m matching radius therefore admits both passes.
STOPS = [
    ("A", 0, 0),
    ("B", 200, 0),
    ("C", 400, 0),
    ("D", 600, 0),
    ("E", 800, 5),
]


def shape_rows():
    rows = []
    for i, (x, y, d) in enumerate(zip(X_COORDS, Y_COORDS, TICKS), start=1):
        rows.append(
            {
                "shape_id": "fold_shape",
                "shape_pt_lat": LAT0 + y * DEG_PER_M_LAT,
                "shape_pt_lon": LON0 + x * DEG_PER_M_LON,
                "shape_pt_sequence": i * 10,
                "shape_dist_traveled": d,
            }
        )
    return rows


def stop_rows():
    return [
        {
            "stop_id": sid,
            "stop_name": sid,
            "stop_lat": LAT0 + y * DEG_PER_M_LAT,
            "stop_lon": LON0 + x * DEG_PER_M_LON,
        }
        for sid, x, y in STOPS
    ]


def trip_rows():
    return [
        {
            "route_id": "R1",
            "service_id": "S1",
            "trip_id": "full_trip",
            "direction_id": 0,
            "shape_id": "fold_shape",
        },
        {
            "route_id": "R1",
            "service_id": "S1",
            "trip_id": "short_trip",
            "direction_id": 0,
            "shape_id": "fold_shape",
        },
    ]


def stop_time_rows():
    # Out and back: C appears twice, separated by D and E.
    full = [
        ("A", 1, "08:00:00"),
        ("B", 2, "08:05:00"),
        ("C", 3, "08:10:00"),
        ("D", 4, "08:15:00"),
        ("E", 5, "08:20:00"),
        ("D", 6, "08:25:00"),
        ("C", 7, "08:30:00"),
        ("B", 8, "08:35:00"),
        ("A", 9, "08:40:00"),
    ]
    # Middle section only, one direction.
    short = [
        ("B", 1, "09:00:00"),
        ("C", 2, "09:05:00"),
        ("D", 3, "09:10:00"),
    ]
    rows = [
        {
            "trip_id": "full_trip",
            "arrival_time": t,
            "departure_time": t,
            "stop_id": sid,
            "stop_sequence": seq,
        }
        for sid, seq, t in full
    ]
    rows += [
        {
            "trip_id": "short_trip",
            "arrival_time": t,
            "departure_time": t,
            "stop_id": sid,
            "stop_sequence": seq,
        }
        for sid, seq, t in short
    ]
    return rows


def main():
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    tables = {
        "agency.txt": pd.DataFrame(
            [
                {
                    "agency_id": "AG",
                    "agency_name": "Foldback Transit",
                    "agency_url": "https://example.com",
                    "agency_timezone": "Australia/Brisbane",
                }
            ]
        ),
        "stops.txt": pd.DataFrame(stop_rows()),
        "routes.txt": pd.DataFrame(
            [
                {
                    "route_id": "R1",
                    "agency_id": "AG",
                    "route_short_name": "1",
                    "route_long_name": "Foldback",
                    "route_type": 3,
                }
            ]
        ),
        "trips.txt": pd.DataFrame(trip_rows()),
        "stop_times.txt": pd.DataFrame(stop_time_rows()),
        "calendar.txt": pd.DataFrame(
            [
                {
                    "service_id": "S1",
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
        ),
        "shapes.txt": pd.DataFrame(shape_rows()),
    }
    with zipfile.ZipFile(FIXTURE, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, df in tables.items():
            zf.writestr(name, df.to_csv(index=False))
    print(f"Wrote {FIXTURE}")


if __name__ == "__main__":
    main()

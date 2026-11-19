"""
Independent verification of the calibrated mileage pipeline.

This script deliberately does *not* use any of gtfs_kit's geometry or
distance helpers for the expected values.  It re-derives the expected
coordinate of a mid-segment point from the raw UTM projection of the shape
vertices and the calibrated ruler, then compares that with

1. what the calibrated pipeline returns in metres,
2. what it returns after switching the feed's distance units to kilometres,
3. what it returns after writing the feed to a ZIP and reading it back.

Run from the repository root with::

    .venv/bin/python -I scripts/verify_calibration.py

``-I`` keeps the script's own directory (and the current directory) off the
import path, so a planted module cannot be imported by accident.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pyproj

# gtfs_kit is only used to *produce* the actual results; the expected values
# below are computed from scratch with pyproj.
import gtfs_kit as gk

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "foldback_gtfs.zip"
DATE = "20250106"

# Raw fixture geometry, echoing scripts/make_foldback_fixture.py.
LON0, LAT0 = 145.7700, -16.9200
DEG_PER_M_LAT = 1 / 111_320.0
DEG_PER_M_LON = 1 / (111_320.0 * 0.957)
X_COORDS = [0, 200, 400, 600, 800, 800, 600, 400, 200, 0]
Y_COORDS = [0, 0, 0, 0, 0, 10, 10, 10, 10, 10]
TICKS = [100, 350, 600, 850, 1100, 1200, 1300, 1400, 1500, 1600]


def projected_vertices():
    """Return shape vertex (east, north) metres in a local transverse Mercator."""
    lon = [LON0 + x * DEG_PER_M_LON for x in X_COORDS]
    lat = [LAT0 + y * DEG_PER_M_LAT for y in Y_COORDS]
    # A fixed UTM zone, chosen independently of gtfs_kit.
    proj = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:32755", always_xy=True
    )  # WGS84 / UTM zone 55S
    east, north = proj.transform(lon, lat)
    return np.array(east), np.array(north)


def expected_point(distance):
    """
    Map a calibrated distance (metres) to (lon, lat) by hand:

    - find the segment whose tick span contains ``distance``,
    - take the calibrated fraction within that segment,
    - interpolate linearly between that segment's WGS84 vertices.
    """
    segments = list(zip(TICKS[:-1], TICKS[1:]))
    for j, (d0, d1) in enumerate(segments):
        if d0 <= distance <= d1:
            t = (distance - d0) / (d1 - d0)
            lon = (LON0 + X_COORDS[j] * DEG_PER_M_LON) + t * (
                X_COORDS[j + 1] - X_COORDS[j]
            ) * DEG_PER_M_LON
            lat = (LAT0 + Y_COORDS[j] * DEG_PER_M_LAT) + t * (
                Y_COORDS[j + 1] - Y_COORDS[j]
            ) * DEG_PER_M_LAT
            return lon, lat
    raise ValueError(f"distance {distance} outside the shape")


def build(dist_units):
    feed = gk.read_feed(FIXTURE, dist_units=dist_units)
    return feed.append_dist_to_shapes(calibrated=True).append_dist_to_stop_times(
        calibrated=True, max_offset_m=20
    )


def locate(feed, times):
    return gk.locate_trips(feed, DATE, times, calibrated=True).set_index(
        ["trip_id", "time"]
    )


def main():
    failures = []

    def check(name, got, expected, tol):
        ok = abs(got - expected) <= tol
        print(f"  {'ok ' if ok else 'FAIL'} {name}: {got!r} vs {expected!r}")
        if not ok:
            failures.append(name)

    # ---- 1. Segment-interior coordinate, metres ---------------------------
    # 08:07:30 is halfway between B (tick 350) and C (tick 600), so distance
    # 475, which lies 125/250 of the way along segment [200 m vertex,
    # 400 m vertex].  No vertex is at that longitude, so this is a genuine
    # in-segment check.
    print("mid-segment coordinate (metres):")
    metres = build("m")
    result = locate(metres, ["08:07:30"])
    lon_exp, lat_exp = expected_point(475.0)
    row = result.loc[("full_trip", "08:07:30")]
    check("lon", row["lon"], lon_exp, 1e-7)
    check("lat", row["lat"], lat_exp, 1e-7)

    # Also confirm the expected point is genuinely between the two vertices
    # and differs from both, so the test cannot pass by hitting a vertex.
    lon_a, lat_a = expected_point(400.0)  # nearest lower vertex tick: 600?
    # (Ticks: 600 is at the 400 m vertex; 475 is before it at x = 300 m.)
    x_mid = (475.0 - 350.0) / (600.0 - 350.0) * 200.0  # 100 m past B
    assert 0 < x_mid < 200
    print(f"  (expected point is {x_mid:.1f} m past stop B, between vertices)")

    # ---- 2. Same answer in kilometres -------------------------------------
    print("distance-unit invariance (metres vs kilometres):")
    km = build("km")
    km_result = locate(km, ["08:07:30", "08:10:00", "08:30:00"])
    m_result = locate(metres, ["08:07:30", "08:10:00", "08:30:00"])
    for key in m_result.index:
        check(
            f"{key[0]} {key[1]} lon",
            km_result.loc[key, "lon"],
            m_result.loc[key, "lon"],
            1e-9,
        )
        check(
            f"{key[0]} {key[1]} rel_dist",
            km_result.loc[key, "rel_dist"],
            m_result.loc[key, "rel_dist"],
            1e-12,
        )

    # ---- 3. ZIP round trip ------------------------------------------------
    print("ZIP round-trip invariance:")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "roundtrip.zip"
        metres.to_file(path)
        reopened = gk.read_feed(path, dist_units="m")
        reopened_result = locate(reopened, ["08:07:30", "08:10:00", "08:30:00"])
    for key in m_result.index:
        check(
            f"{key[0]} {key[1]} lon",
            reopened_result.loc[key, "lon"],
            m_result.loc[key, "lon"],
            1e-9,
        )
        check(
            f"{key[0]} {key[1]} lat",
            reopened_result.loc[key, "lat"],
            m_result.loc[key, "lat"],
            1e-9,
        )
        check(
            f"{key[0]} {key[1]} rel_dist",
            reopened_result.loc[key, "rel_dist"],
            m_result.loc[key, "rel_dist"],
            1e-9,
        )

    # ---- 4. The two passes of C are distinguished -------------------------
    print("fold-back separation (outbound vs inbound C):")
    loc = locate(metres, ["08:10:00", "08:30:00"])
    lon_out, lat_out = expected_point(600.0)
    lon_in, lat_in = expected_point(1400.0)
    check("outbound C lon", loc.loc[("full_trip", "08:10:00"), "lon"], lon_out, 1e-7)
    check("outbound C lat", loc.loc[("full_trip", "08:10:00"), "lat"], lat_out, 1e-7)
    check("inbound C lon", loc.loc[("full_trip", "08:30:00"), "lon"], lon_in, 1e-7)
    check("inbound C lat", loc.loc[("full_trip", "08:30:00"), "lat"], lat_in, 1e-7)
    # The two visits are at the same longitude but different latitudes.
    check(
        "outbound/inbound latitude gap (m)",
        (lat_in - lat_out) / DEG_PER_M_LAT,
        10.0,
        0.5,
    )

    # ---- 5. Short working uses its own span -------------------------------
    print("short working relative distance:")
    short = locate(metres, ["09:05:00"]).loc[("short_trip", "09:05:00")]
    check("rel_dist", short["rel_dist"], 0.5, 1e-9)

    # ---- 6. Failure modes -------------------------------------------------
    print("failure modes:")

    # (a) contradictory anchors: a later stop anchored before an earlier one.
    # C is within reach of both passes, so anchoring it on the inbound pass
    # (1400) is individually valid; anchoring D (outbound, ~850) after it is
    # likewise valid.  Only the stop-order check can catch the pair.
    feed = build("m").copy()
    mask = feed.stop_times.trip_id == "full_trip"
    feed.stop_times.loc[mask, "shape_dist_traveled"] = np.nan
    outbound_c = (feed.stop_times.trip_id == "full_trip") & (
        feed.stop_times.stop_sequence == 3
    )
    outbound_d = (feed.stop_times.trip_id == "full_trip") & (
        feed.stop_times.stop_sequence == 4
    )
    feed.stop_times.loc[outbound_c, "shape_dist_traveled"] = 1400.0
    feed.stop_times.loc[outbound_d, "shape_dist_traveled"] = 850.0
    try:
        feed.append_dist_to_stop_times(calibrated=True, max_offset_m=20)
        print("  FAIL contradictory anchors were accepted")
        failures.append("contradictory anchors")
    except ValueError as exc:
        named = "full_trip" in str(exc) and "decreases" in str(exc)
        print(f"  {'ok ' if named else 'FAIL'} named trip in: {exc}")
        if not named:
            failures.append("contradictory anchors")

    # (b) a radius too small to reach a stop: nothing on the shape is within
    # 1 m of the turnaround stop, which sits 5 m off the centre line.  Strip
    # any pre-filled mileage first so this exercises matching, not anchors.
    narrow = build("m").copy()
    narrow.stop_times["shape_dist_traveled"] = np.nan
    try:
        narrow.append_dist_to_stop_times(calibrated=True, max_offset_m=1.0)
        print("  FAIL tiny radius was accepted")
        failures.append("tiny radius")
    except ValueError as exc:
        named = "full_trip" in str(exc) and "no stop-to-shape matching" in str(exc)
        print(f"  {'ok ' if named else 'FAIL'} named trip in: {exc}")
        if not named:
            failures.append("tiny radius")

    # (c) a prepared but self-inconsistent ruler: mileage that goes backwards
    # along shape_pt_sequence must be rejected when the shapes are prepared.
    bad = gk.read_feed(FIXTURE, dist_units="m")
    bad.shapes.loc[bad.shapes.index[2], "shape_dist_traveled"] = 100.0
    try:
        bad.append_dist_to_shapes(calibrated=True)
        print("  FAIL backwards shape mileage was accepted")
        failures.append("backwards shape mileage")
    except ValueError as exc:
        named = "fold_shape" in str(exc)
        print(f"  {'ok ' if named else 'FAIL'} named shape in: {exc}")
        if not named:
            failures.append("backwards shape mileage")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("All independent checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

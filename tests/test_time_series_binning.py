"""
Regression tests for service-volume accounting in the route/network time series:

- seconds in trip start/end times are respected;
- bins are half-open ``[left, right)``;
- ``num_trips`` counts distinct trips that actually run inside the bin;
- ``service_duration`` and ``service_distance`` only accrue the part of each
  trip that falls inside the bin, with distance allocated at constant speed;
- ``helpers.downsample`` preserves distinct-trip counts exactly when the target
  frequency is an integer multiple of the source frequency, and refuses
  otherwise.

Expected values are computed independently from the raw trip definitions by
explicitly tracking, per bin, the set of trip IDs that overlap that bin.  No
per-second expansion is used.
"""

import math

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import gtfs_kit as gk
from gtfs_kit import helpers as hp
from gtfs_kit import miscellany as gkm
from gtfs_kit import routes as gkr

DAY = 24 * 3600
COUNT_COLS = [
    "num_trips",
    "num_trip_starts",
    "num_trip_ends",
    "service_duration",
    "service_distance",
]


def timestr_to_secs(x: str) -> int:
    h, m, s = (int(y) for y in x.split(":"))
    return h * 3600 + m * 60 + s


def make_trip(
    trip_id,
    route_id,
    direction_id,
    start,
    end,
    distance,
    *,
    route_type=3,
    service_id="DAILY",
):
    start_s = timestr_to_secs(start)
    end_s = timestr_to_secs(end)
    return {
        "trip_id": trip_id,
        "route_id": route_id,
        "direction_id": direction_id,
        "route_type": route_type,
        "service_id": service_id,
        "start_time": start,
        "end_time": end,
        "distance": distance,
        "duration": (end_s - start_s) / 3600,
    }


def trip_stats_frame(trips) -> pd.DataFrame:
    cols = [
        "trip_id",
        "route_id",
        "route_type",
        "direction_id",
        "start_time",
        "end_time",
        "distance",
        "duration",
    ]
    return pd.DataFrame(trips)[cols]


def reference_table(trips, freq: str, *, split_directions: bool = False) -> pd.DataFrame:
    """
    Independently compute the expected time series from raw trip definitions.

    Distinct-trip counts are obtained from a set of trip IDs per bin, not
    from the production code's start/end-count identity.
    """
    freq_s = int(pd.tseries.frequencies.to_offset(freq).nanos // 10**9)
    num_bins = math.ceil(DAY / freq_s)

    acc = {}

    def key_of(t, b):
        k = (t["route_id"], b)
        if split_directions:
            k = (t["route_id"], int(t["direction_id"]), b)
        return k

    for t in trips:
        s = timestr_to_secs(t["start_time"]) % DAY
        e = timestr_to_secs(t["end_time"]) % DAY
        if s == e:
            continue
        total = (e - s) % DAY
        if total == 0:
            continue
        wrapped = s > e
        segments = [(s, DAY), (0, e)] if wrapped else [(s, e)]

        for a, b in segments:
            first_bin = a // freq_s
            last_bin = (b - 1) // freq_s
            for k in range(first_bin, last_bin + 1):
                lo = k * freq_s
                hi = min((k + 1) * freq_s, DAY)
                overlap = min(b, hi) - max(a, lo)
                if overlap <= 0:
                    continue
                d = acc.setdefault(
                    key_of(t, k),
                    {
                        "trip_ids": set(),
                        "num_trip_starts": 0,
                        "num_trip_ends": 0,
                        "service_duration": 0.0,
                        "service_distance": 0.0,
                    },
                )
                d["trip_ids"].add(t["trip_id"])
                d["service_duration"] += overlap / 3600
                d["service_distance"] += t["distance"] * overlap / total

        acc.setdefault(
            key_of(t, s // freq_s),
            {
                "trip_ids": set(),
                "num_trip_starts": 0,
                "num_trip_ends": 0,
                "service_duration": 0.0,
                "service_distance": 0.0,
            },
        )["num_trip_starts"] += 1

        if not wrapped:
            acc.setdefault(
                key_of(t, e // freq_s),
                {
                    "trip_ids": set(),
                    "num_trip_starts": 0,
                    "num_trip_ends": 0,
                    "service_duration": 0.0,
                    "service_distance": 0.0,
                },
            )["num_trip_ends"] += 1

    assert num_bins > 0
    rows = []
    for (route_id, *rest, b), d in acc.items():
        row = {
            "route_id": route_id,
            "bin": b,
            "num_trips": len(d["trip_ids"]),
            "num_trip_starts": d["num_trip_starts"],
            "num_trip_ends": d["num_trip_ends"],
            "service_duration": d["service_duration"],
            "service_distance": d["service_distance"],
        }
        if split_directions:
            row["direction_id"] = rest[0]
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["route_id", "bin"] + COUNT_COLS)
    f = pd.DataFrame(rows)
    return f.sort_values(["route_id"] + (["direction_id"] if split_directions else []) + ["bin"])


def reference_network_table(trips, freq: str, *, split_route_types: bool) -> pd.DataFrame:
    per_route = reference_table(trips, freq)
    per_route = per_route.merge(
        pd.DataFrame(trips)[["route_id", "route_type"]].drop_duplicates("route_id"),
        on="route_id",
    )
    keys = ["bin"] + (["route_type"] if split_route_types else [])
    f = per_route.groupby(keys, dropna=False)[COUNT_COLS].sum().reset_index()
    return f.sort_values(keys)


def compare_to_reference(actual, expected, freq, split_directions):
    """
    Compare production rows to the reference for every bin the reference
    mentions.  Production rows outside those bins must be identically zero.
    """
    freq_s = int(pd.tseries.frequencies.to_offset(freq).nanos // 10**9)
    keys = ["route_id"] + (["direction_id"] if split_directions else []) + ["bin"]
    a = actual.copy()
    a["bin"] = (
        (a["datetime"] - a["datetime"].dt.normalize()).dt.total_seconds().astype(int)
        // freq_s
    )
    a = a.set_index(keys).sort_index()
    e = expected.set_index(keys).sort_index()
    # Every reference bin must exist in production
    missing = e.index.difference(a.index)
    assert missing.empty, f"missing bins: {list(missing)[:5]}"
    np.testing.assert_allclose(
        a.loc[e.index, COUNT_COLS].to_numpy(dtype=float),
        e[COUNT_COLS].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
        err_msg=f"freq={freq}",
    )
    # Production's extra bins carry no service
    extra = a.index.difference(e.index)
    if len(extra):
        assert (a.loc[extra, ["num_trips", "num_trip_starts", "num_trip_ends"]]
                .abs().to_numpy() == 0).all()
        assert (a.loc[extra, ["service_distance", "service_duration"]]
                .abs().to_numpy() < 1e-12).all()


def bin_values(actual, freq, route_id=None, direction_id=None, start=None):
    """Select one route/direction's table and index it by bin start datetime."""
    f = actual
    if route_id is not None:
        f = f[f["route_id"] == route_id]
    if direction_id is not None:
        f = f[f["direction_id"] == direction_id]
    if start is not None:
        f = f[f["datetime"] == pd.Timestamp(start)]
    return f


# ---------------------------------------------------------------------------
# The two reported reproductions
# ---------------------------------------------------------------------------


def test_four_minute_trip_in_5min_bin():
    trip = make_trip("t", "r", 0, "08:00:00", "08:04:00", 4.0)
    ts = trip_stats_frame([trip])

    out = gkr.compute_route_time_series_0(ts, freq="5min")
    bucket = bin_values(out, "5min", "r", start="2001-01-01 08:00:00").iloc[0]
    assert bucket["num_trips"] == 1
    assert bucket["num_trip_starts"] == 1
    assert bucket["num_trip_ends"] == 1
    assert bucket["service_duration"] == pytest.approx(4 / 60)
    assert bucket["service_distance"] == pytest.approx(4.0)
    assert bucket["service_speed"] == pytest.approx(60.0)

    # The following bins are empty
    later = bin_values(out, "5min", "r", start="2001-01-01 08:05:00").iloc[0]
    assert later["num_trips"] == 0

    # Same story at an hourly frequency
    hourly = gkr.compute_route_time_series_0(ts, freq="1h")
    bucket = bin_values(hourly, "1h", "r", start="2001-01-01 08:00:00").iloc[0]
    assert bucket["num_trips"] == 1
    assert bucket["service_duration"] == pytest.approx(4 / 60)
    assert bucket["service_distance"] == pytest.approx(4.0)


def test_thirty_second_trip_contributes_everything():
    trip = make_trip("s", "r", 0, "08:00:10", "08:00:40", 0.3)
    ts = trip_stats_frame([trip])

    for freq in ["1min", "5min", "1h"]:
        out = gkr.compute_route_time_series_0(ts, freq=freq)
        rows = out[out["num_trips"] > 0]
        assert len(rows) == 1
        row = rows.iloc[0]
        assert row["num_trips"] == 1
        assert row["num_trip_starts"] == 1
        assert row["num_trip_ends"] == 1
        assert row["service_duration"] == pytest.approx(30 / 3600)
        assert row["service_distance"] == pytest.approx(0.3)
        # Summarized speed = distance / duration
        assert row["service_speed"] == pytest.approx(0.3 / (30 / 3600))


# ---------------------------------------------------------------------------
# Half-minute edges of a trip crossing bins
# ---------------------------------------------------------------------------


def test_half_minute_edges():
    trip = make_trip("x", "r", 0, "08:00:30", "08:02:30", 2.0)
    ts = trip_stats_frame([trip])

    out = gkr.compute_route_time_series_0(ts, freq="1min")
    by_bin = out.set_index("datetime")
    b0 = by_bin.loc["2001-01-01 08:00:00"]
    b1 = by_bin.loc["2001-01-01 08:01:00"]
    b2 = by_bin.loc["2001-01-01 08:02:00"]
    b3 = by_bin.loc["2001-01-01 08:03:00"]

    # First and last bins hold 30 seconds, the middle one a full minute
    assert b0["service_duration"] == pytest.approx(0.5 / 60)
    assert b1["service_duration"] == pytest.approx(1 / 60)
    assert b2["service_duration"] == pytest.approx(0.5 / 60)
    assert b3["service_duration"] == pytest.approx(0.0)
    # Constant-speed distance allocation
    assert b0["service_distance"] == pytest.approx(0.5)
    assert b1["service_distance"] == pytest.approx(1.0)
    assert b2["service_distance"] == pytest.approx(0.5)
    # The trip runs in bins 08:00, 08:01, 08:02 and ends exactly at 08:02:30
    assert list(by_bin.loc["2001-01-01 08:00:00":"2001-01-01 08:02:00", "num_trips"]) == [1, 1, 1]
    assert b3["num_trips"] == 0
    assert b0["num_trip_starts"] == 1
    assert b2["num_trip_ends"] == 1
    assert b3["num_trip_ends"] == 0
    # Total is preserved
    assert out["service_duration"].sum() == pytest.approx(2 / 60)
    assert out["service_distance"].sum() == pytest.approx(2.0)

    # In a 5-minute bin the whole trip is in the 08:00 bin
    five = gkr.compute_route_time_series_0(ts, freq="5min")
    row = five[five["num_trips"] > 0].iloc[0]
    assert row["datetime"] == pd.Timestamp("2001-01-01 08:00:00")
    assert row["service_duration"] == pytest.approx(2 / 60)
    assert row["service_distance"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Hand-off at a bin edge: one trip ends exactly when another starts
# ---------------------------------------------------------------------------


def test_trip_ending_at_left_edge_is_not_running():
    a = make_trip("a", "r", 0, "07:59:00", "08:00:00", 1.0)
    b = make_trip("b", "r", 0, "08:00:00", "08:06:00", 6.0)
    ts = trip_stats_frame([a, b])

    out = gkr.compute_route_time_series_0(ts, freq="5min")
    by_bin = out.set_index("datetime")
    prev = by_bin.loc["2001-01-01 07:55:00"]
    cur = by_bin.loc["2001-01-01 08:00:00"]
    nxt = by_bin.loc["2001-01-01 08:05:00"]

    # Trip a runs alone in 07:55 and ends exactly at 08:00
    assert prev["num_trips"] == 1
    assert prev["num_trip_starts"] == 1
    assert prev["num_trip_ends"] == 0
    assert prev["service_duration"] == pytest.approx(1 / 60)

    # At 08:00, only the newly departing trip b is running; a's end is an
    # event in this bin, as is b's start.
    assert cur["num_trips"] == 1
    assert cur["num_trip_starts"] == 1
    assert cur["num_trip_ends"] == 1
    assert cur["service_duration"] == pytest.approx(5 / 60)
    assert cur["service_distance"] == pytest.approx(5.0)

    assert nxt["num_trips"] == 1
    assert nxt["num_trip_starts"] == 0
    assert nxt["num_trip_ends"] == 1
    assert nxt["service_duration"] == pytest.approx(1 / 60)


def test_zero_duration_and_missing_times_excluded():
    trips = [
        make_trip("zero", "r", 0, "08:00:00", "08:00:00", 3.0),
        {
            **make_trip("nostart", "r", 0, "08:00:00", "08:10:00", 3.0),
            "start_time": None,
        },
        {
            **make_trip("noend", "r", 0, "08:00:00", "08:10:00", 3.0),
            "end_time": None,
        },
        make_trip("ok", "r", 0, "08:01:00", "08:02:00", 1.0),
    ]
    ts = trip_stats_frame(trips)
    out = gkr.compute_route_time_series_0(ts, freq="5min")
    assert out["num_trips"].sum() == 1
    assert out["num_trip_starts"].sum() == 1
    assert out["num_trip_ends"].sum() == 1
    assert out["service_distance"].sum() == pytest.approx(1.0)
    assert out["service_duration"].sum() == pytest.approx(1 / 60)


# ---------------------------------------------------------------------------
# Independent reference checks at several frequencies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("freq", ["1min", "2min", "5min", "15min", "7min", "1h"])
@pytest.mark.parametrize("split_directions", [False, True])
def test_matches_independent_reference(freq, split_directions):
    trips = [
        make_trip("a", "r", 0, "08:00:00", "08:04:00", 4.0),
        make_trip("b", "r", 0, "08:00:10", "08:00:40", 0.3),
        make_trip("c", "r", 1, "08:00:30", "08:02:30", 2.0),
        make_trip("d", "q", 0, "07:59:00", "08:00:00", 1.0),
        make_trip("e", "q", 0, "08:00:00", "08:06:00", 6.0),
        make_trip("f", "s", 1, "12:34:56", "13:01:23", 7.5),
    ]
    ts = trip_stats_frame(trips)
    out = gkr.compute_route_time_series_0(
        ts, freq=freq, split_directions=split_directions
    )
    expected = reference_table(trips, freq, split_directions=split_directions)
    compare_to_reference(out, expected, freq, split_directions)


# ---------------------------------------------------------------------------
# Downsampling: exact distinct-trip counts, chaining, and refusal
# ---------------------------------------------------------------------------


def _route_table_1min():
    trips = [
        make_trip("a", "r", 0, "08:00:00", "08:04:00", 4.0),
        make_trip("b", "r", 0, "08:00:10", "08:00:40", 0.3),
        make_trip("c", "r", 1, "08:00:30", "08:02:30", 2.0),
        make_trip("d", "q", 0, "07:59:00", "08:00:00", 1.0),
        make_trip("e", "q", 0, "08:00:00", "08:06:00", 6.0),
        make_trip("f", "q", 0, "08:10:00", "08:19:00", 9.0),
    ]
    ts = trip_stats_frame(trips)
    return ts, gkr.compute_route_time_series_0(ts, freq="1min", split_directions=True)


def test_downsample_does_not_double_count_or_drop_ended_trips():
    ts, one = _route_table_1min()

    five = hp.downsample(one.copy(), "5min")

    # The 08:00 5-minute bin for route q, direction 0 contains trip d's
    # ending exactly at its left edge (not running), trip e running through.
    q = five[(five["route_id"] == "q") & (five["direction_id"] == 0)]
    cur = q[q["datetime"] == pd.Timestamp("2001-01-01 08:00:00")].iloc[0]
    assert cur["num_trips"] == 1  # e only
    assert cur["num_trip_starts"] == 1  # e
    assert cur["num_trip_ends"] == 1  # d's end event is stamped 08:00
    assert cur["service_duration"] == pytest.approx(5 / 60)
    assert cur["service_distance"] == pytest.approx(5.0)

    # Trip f spans two 5-minute bins and is counted once in each, not five
    # times in either.
    f_bins = q[
        q["datetime"].isin(pd.to_datetime(["2001-01-01 08:10", "2001-01-01 08:15"]))
    ]
    assert list(f_bins["num_trips"].values) == [1, 1]
    assert list(f_bins["service_duration"].values) == [
        pytest.approx(5 / 60),
        pytest.approx(4 / 60),
    ]

    # Trip e spans two 5-minute bins: 5 minutes + 1 minute
    e_bins = q[q["datetime"].isin(pd.to_datetime(["2001-01-01 08:00", "2001-01-01 08:05"]))]
    assert list(e_bins["num_trips"].values) == [1, 1]


def test_downsample_chained_matches_direct():
    ts, one = _route_table_1min()
    five = hp.downsample(one.copy(), "5min")
    fifteen = hp.downsample(five.copy(), "15min")
    direct15 = gkr.compute_route_time_series_0(ts, freq="15min", split_directions=True)

    keys = ["route_id", "direction_id", "datetime"]
    get = lambda f: f.set_index(keys).sort_index()[COUNT_COLS]
    np.testing.assert_allclose(get(fifteen), get(direct15), rtol=0, atol=1e-12)

    direct5 = gkr.compute_route_time_series_0(ts, freq="5min", split_directions=True)
    np.testing.assert_allclose(get(five), get(direct5), rtol=0, atol=1e-12)

    # Cumulative totals are invariant along the way
    for f in (one, five, fifteen):
        assert f["num_trip_starts"].sum() == 6
        assert f["num_trip_ends"].sum() == 6  # every end is before midnight
        assert f["service_duration"].sum() == pytest.approx(
            (4 + 0.5 + 2 + 1 + 6 + 9) / 60
        )
        assert f["service_distance"].sum() == pytest.approx(22.3)


def test_downsample_refuses_non_multiple():
    _, one = _route_table_1min()
    five = hp.downsample(one.copy(), "5min")
    with pytest.raises(ValueError, match="integer multiple"):
        hp.downsample(five.copy(), "7min")

    # 1min -> 7min is a multiple and so is allowed
    seven = hp.downsample(one.copy(), "7min")
    assert not seven.empty


def test_downsample_refuses_uninferable_table():
    _, one = _route_table_1min()
    five = hp.downsample(one.copy(), "5min")
    sparse = five.head(1).copy()
    sparse.attrs = {}
    with pytest.raises(ValueError, match="[Cc]annot infer"):
        hp.downsample(sparse, "15min")


def test_downsample_plain_long_table_without_attrs():
    """A plain, regular long table (no GTFS Kit metadata) still downsamples."""
    _, one = _route_table_1min()
    plain = one.copy()
    plain.attrs = {}
    got = hp.downsample(plain, "5min")
    expected = hp.downsample(one.copy(), "5min")
    pd.testing.assert_frame_equal(got, expected)


def test_downsample_unchanged_for_finer_or_equal_frequency():
    _, one = _route_table_1min()
    same = hp.downsample(one.copy(), "1min")
    assert len(same) == len(one)
    finer = hp.downsample(one.copy(), "1s")
    assert len(finer) == len(one)


def test_downsample_to_whole_day_from_non_tiling_source():
    """
    A whole-day bin is exact even when the source bins do not tile the day
    (e.g. 7-minute bins): each trip starts once per service day, so summing
    starts counts distinct trips exactly.
    """
    trips = [
        make_trip("w", "r", 0, "23:58:00", "24:03:00", 5.0),
        make_trip("a", "r", 0, "08:00:00", "08:04:00", 4.0),
        make_trip("b", "r", 0, "08:30:00", "09:00:00", 4.0),
    ]
    ts = trip_stats_frame(trips)
    for source_freq in ["7min", "5min", "1min"]:
        source = gkr.compute_route_time_series_0(ts, freq=source_freq)
        day = hp.downsample(source.copy(), "1D")
        direct = gkr.compute_route_time_series_0(ts, freq="1D")
        assert day.iloc[0]["num_trips"] == direct.iloc[0]["num_trips"] == 3
        assert day.iloc[0]["num_trip_starts"] == 3
        assert day.iloc[0]["service_duration"] == pytest.approx(
            direct.iloc[0]["service_duration"]
        )
        assert day.iloc[0]["service_distance"] == pytest.approx(
            direct.iloc[0]["service_distance"]
        )


# ---------------------------------------------------------------------------
# Midnight wrap compatibility
# ---------------------------------------------------------------------------


def test_midnight_wrap_preserved():
    # Runs from 23:58 to 24:03: 2 minutes before midnight, 3 after.
    wrapped = make_trip("w", "r", 0, "23:58:00", "24:03:00", 5.0)
    # Starts after 24:00 and ends before 24:00 on the clock: mod-24 wrap too
    late = make_trip("l", "r", 1, "23:30:00", "23:40:00", 2.0)
    ts = trip_stats_frame([wrapped, late])

    out = gkr.compute_route_time_series_0(ts, freq="1min", split_directions=True)
    direction0 = out[out["direction_id"] == 0].set_index("datetime")
    early = direction0.loc["2001-01-01 00:00:00":"2001-01-01 00:02:00"]
    evening = direction0.loc["2001-01-01 23:58:00":"2001-01-01 23:59:00"]

    # 3 minutes accrue after midnight and 2 before
    assert early["service_duration"].sum() == pytest.approx(3 / 60)
    assert evening["service_duration"].sum() == pytest.approx(2 / 60)
    assert early["service_distance"].sum() == pytest.approx(3.0)
    assert evening["service_distance"].sum() == pytest.approx(2.0)
    # Distinct trip counted in both halves, still once per bin
    assert early["num_trips"].tolist() == [1, 1, 1]
    assert evening["num_trips"].tolist() == [1, 1]
    # Start event at 23:58, no end event anywhere (ends past midnight)
    assert evening["num_trip_starts"].sum() == 1
    assert out.loc[out["direction_id"] == 0, "num_trip_ends"].sum() == 0
    # Total duration/distance preserved exactly once
    assert direction0["service_duration"].sum() == pytest.approx(5 / 60)
    assert direction0["service_distance"].sum() == pytest.approx(5.0)

    # Downsampling keeps the same convention
    five = hp.downsample(
        gkr.compute_route_time_series_0(ts, freq="1min", split_directions=True),
        "5min",
    )
    d0 = five[five["direction_id"] == 0]
    assert d0["num_trip_ends"].sum() == 0
    assert d0["service_duration"].sum() == pytest.approx(5 / 60)
    assert d0["service_distance"].sum() == pytest.approx(5.0)
    assert d0.loc[d0["num_trips"] > 0, "num_trips"].tolist() == [1, 1]


def test_day_binning_counts_wrapped_trip_once():
    trips = [
        make_trip("w", "r", 0, "23:58:00", "24:03:00", 5.0),
        make_trip("a", "r", 0, "08:00:00", "08:04:00", 4.0),
        make_trip("b", "r", 0, "08:30:00", "09:00:00", 4.0),
    ]
    ts = trip_stats_frame(trips)
    for freq in ["1D", "24h"]:
        direct = gkr.compute_route_time_series_0(ts, freq=freq)
        assert len(direct) == 1
        row = direct.iloc[0]
        # Three distinct trips, three start events, two end events (the
        # wrapped trip's end is not binned)
        assert row["num_trips"] == 3
        assert row["num_trip_starts"] == 3
        assert row["num_trip_ends"] == 2
        assert row["service_duration"] == pytest.approx((5 + 4 + 30) / 60)
        assert row["service_distance"] == pytest.approx(5.0 + 4.0 + 4.0)

        # The same result arrives via downsampling a 1-minute table
        chained = hp.downsample(
            gkr.compute_route_time_series_0(ts, freq="1min"), freq
        )
        assert chained.iloc[0]["num_trips"] == 3
        assert chained.iloc[0]["num_trip_starts"] == 3
        assert chained.iloc[0]["num_trip_ends"] == 2
        assert chained.iloc[0]["service_duration"] == pytest.approx(
            row["service_duration"]
        )
        assert chained.iloc[0]["service_distance"] == pytest.approx(
            row["service_distance"]
        )


# ---------------------------------------------------------------------------
# Feed level: two consecutive days, directions, route types, network roll-up
# ---------------------------------------------------------------------------


def make_synthetic_feed():
    """
    Two routes with different route types, trips in both directions, a daily
    service and a service that only runs on the second day.
    """
    trips_rows = []
    definitions = [
        ("a", "r1", 0, "DAILY", 3),
        ("b", "r1", 0, "DAILY", 3),
        ("c", "r1", 1, "DAILY", 3),
        ("d", "r2", 0, "DAILY", 2),
        ("e", "r2", 0, "DAILY", 2),
        ("f", "r2", 1, "TUES", 2),
        ("g", "r2", 1, "TUES", 2),
    ]
    for trip_id, route_id, direction_id, service_id, route_type in definitions:
        trips_rows.append(
            {
                "route_id": route_id,
                "service_id": service_id,
                "trip_id": trip_id,
                "direction_id": direction_id,
            }
        )
    trips = pd.DataFrame(trips_rows)
    routes = pd.DataFrame(
        {"route_id": ["r1", "r2"], "route_type": [3, 2]}
    )
    calendar = pd.DataFrame(
        {
            "service_id": ["DAILY", "TUES"],
            "monday": [1, 0],
            "tuesday": [1, 0],
            "wednesday": [1, 0],
            "thursday": [1, 0],
            "friday": [1, 0],
            "saturday": [1, 0],
            "sunday": [1, 0],
            "start_date": ["20240101", "20240101"],
            "end_date": ["20240102", "20240102"],
        }
    )
    calendar_dates = pd.DataFrame(
        {"service_id": ["TUES"], "date": ["20240102"], "exception_type": [1]}
    )
    return gk.Feed(
        dist_units="km",
        routes=routes,
        trips=trips,
        calendar=calendar,
        calendar_dates=calendar_dates,
    )


TRIP_DEFS = [
    make_trip("a", "r1", 0, "08:00:00", "08:04:00", 4.0, route_type=3),
    make_trip("b", "r1", 0, "08:00:10", "08:00:40", 0.3, route_type=3),
    make_trip("c", "r1", 1, "08:00:30", "08:02:30", 2.0, route_type=3),
    make_trip("d", "r2", 0, "07:59:00", "08:00:00", 1.0, route_type=2),
    make_trip("e", "r2", 0, "08:00:00", "08:06:00", 6.0, route_type=2),
    make_trip("f", "r2", 1, "23:58:00", "24:03:00", 5.0, route_type=2),
    make_trip("g", "r2", 1, "13:00:00", "13:15:00", 5.0, route_type=2),
]


@pytest.mark.parametrize("split_directions", [False, True])
def test_feed_route_time_series_two_days(split_directions):
    feed = make_synthetic_feed()
    dates = ["20240101", "20240102"]
    all_stats = trip_stats_frame(TRIP_DEFS)
    snapshot = all_stats.copy(deep=True)

    rts = feed.compute_route_time_series(
        dates, all_stats, freq="5min", split_directions=split_directions
    )

    # The input table must not be modified
    pdt.assert_frame_equal(all_stats, snapshot)

    # Trips f and g only run on the second date
    day1 = rts[rts["datetime"].dt.strftime("%Y%m%d") == "20240101"]
    day2 = rts[rts["datetime"].dt.strftime("%Y%m%d") == "20240102"]
    r2_day1 = day1[day1["route_id"] == "r2"]
    r2_day2 = day2[day2["route_id"] == "r2"]
    # d and e (both DAILY) start on day 1 -> two start events
    assert r2_day1["num_trip_starts"].sum() == 2
    # f and g (TUES, only active day 2) add two more start events
    assert r2_day2["num_trip_starts"].sum() == 4
    assert r2_day1["service_distance"].sum() == pytest.approx(1.0 + 6.0)
    assert r2_day2["service_distance"].sum() == pytest.approx(1.0 + 6.0 + 5.0 + 5.0)

    # Compare each date to the independent reference
    active_day1 = [t for t in TRIP_DEFS if t["trip_id"] not in ("f", "g")]
    active_day2 = TRIP_DEFS
    freq_s = 300
    for date, active in [("20240101", active_day1), ("20240102", active_day2)]:
        got = rts[rts["datetime"].dt.strftime("%Y%m%d") == date].copy()
        expected = reference_table(active, "5min", split_directions=split_directions)
        got["bin"] = (
            (got["datetime"] - got["datetime"].dt.normalize())
            .dt.total_seconds()
            .astype(int)
            // freq_s
        )
        keys = ["route_id"] + (["direction_id"] if split_directions else []) + ["bin"]
        g = got.set_index(keys).sort_index()[COUNT_COLS]
        # Only bins that have activity are of interest; the reference has
        # exactly those, and production's other bins are all zero.
        g = g.loc[expected.set_index(keys).sort_index().index.intersection(g.index)]
        e = expected.set_index(keys).sort_index().loc[g.index]
        np.testing.assert_allclose(g, e, rtol=0, atol=1e-12)


def test_feed_network_rollup_agrees():
    feed = make_synthetic_feed()
    dates = ["20240101", "20240102"]
    trip_stats = trip_stats_frame(TRIP_DEFS)

    route = feed.compute_route_time_series(dates, trip_stats, freq="5min")
    network = gkm.compute_network_time_series(feed, dates, trip_stats, freq="5min")
    by_type = gkm.compute_network_time_series(
        feed, dates, trip_stats, freq="5min", split_route_types=True
    )

    # Network totals equal the sum of route rows
    route_rollup = (
        route.groupby("datetime")[COUNT_COLS].sum().reset_index().sort_values("datetime")
    )
    net = network.sort_values("datetime").reset_index(drop=True)
    np.testing.assert_allclose(
        net[COUNT_COLS], route_rollup[COUNT_COLS], rtol=0, atol=1e-12
    )

    # Split by route type sums to the unsplit network
    type_rollup = (
        by_type.groupby("datetime")[COUNT_COLS].sum().reset_index().sort_values("datetime")
    )
    np.testing.assert_allclose(
        net[COUNT_COLS], type_rollup[COUNT_COLS], rtol=0, atol=1e-12
    )

    # And every route type's row matches an independent reference for day 1
    # (f and g only run on day 2).
    freq_s, freq = 300, "5min"
    day1_trips = [t for t in TRIP_DEFS if t["trip_id"] not in ("f", "g")]
    expected = reference_network_table(day1_trips, freq, split_route_types=True)
    for dtype in [2, 3]:
        got = by_type[by_type["route_type"].astype(int) == dtype].copy()
        got = got[got["datetime"].dt.strftime("%Y%m%d") == "20240101"]
        exp = expected[expected["route_type"] == dtype]
        # Reference contains one day of bins; map bin numbers to datetimes
        got["bin"] = (
            (got["datetime"] - got["datetime"].dt.normalize())
            .dt.total_seconds()
            .astype(int)
            // freq_s
        )
        g = got.set_index("bin").sort_index()[COUNT_COLS]
        # Only compare bins mentioned by either side (both are 0 outside)
        idx = sorted(set(g.index) | set(exp["bin"]))
        g = g.reindex(idx, fill_value=0.0)
        e = exp.set_index("bin").reindex(idx, fill_value=0.0)[COUNT_COLS]
        np.testing.assert_allclose(g, e, rtol=0, atol=1e-12)


def test_feed_downsample_chain_matches_direct():
    feed = make_synthetic_feed()
    dates = ["20240101", "20240102"]
    trip_stats = trip_stats_frame(TRIP_DEFS)

    one = feed.compute_route_time_series(
        dates, trip_stats, freq="1min", split_directions=True
    )
    five = hp.downsample(one.copy(), "5min")
    fifteen = hp.downsample(five.copy(), "15min")
    direct = feed.compute_route_time_series(
        dates, trip_stats, freq="15min", split_directions=True
    )

    keys = ["route_id", "direction_id", "datetime"]
    g = fifteen.set_index(keys).sort_index()[COUNT_COLS]
    e = direct.set_index(keys).sort_index()[COUNT_COLS]
    assert list(g.index) == list(e.index)
    np.testing.assert_allclose(g, e, rtol=0, atol=1e-12)

    # Network chain as well
    net1 = feed.compute_network_time_series(
        dates, trip_stats, freq="1min", split_route_types=True
    )
    net5 = hp.downsample(net1.copy(), "5min")
    net15 = hp.downsample(net5.copy(), "15min")
    net_direct = feed.compute_network_time_series(
        dates, trip_stats, freq="15min", split_route_types=True
    )
    keys = ["route_type", "datetime"]
    g = net15.set_index(keys).sort_index()[COUNT_COLS]
    e = net_direct.set_index(keys).sort_index()[COUNT_COLS]
    assert list(g.index) == list(e.index)
    np.testing.assert_allclose(g, e, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Route-type split must not multiply network counts
# ---------------------------------------------------------------------------


def test_network_split_route_types_does_not_inflate():
    """
    Each route maps to a single route type, so splitting the network by route
    type must reproduce the unsplit network exactly (this regressed when the
    route-type merge duplicated rows).
    """
    feed = make_synthetic_feed()
    dates = ["20240101", "20240102"]
    trip_stats = trip_stats_frame(TRIP_DEFS)

    net = gkm.compute_network_time_series(feed, dates, trip_stats, freq="5min")
    split = gkm.compute_network_time_series(
        feed, dates, trip_stats, freq="5min", split_route_types=True
    )
    # Two route types -> two rows per time bin per date
    assert set(split["route_type"].astype(int)) == {2, 3}
    rolled = split.groupby("datetime")[COUNT_COLS].sum().reset_index()
    np.testing.assert_allclose(
        rolled[COUNT_COLS],
        net.sort_values("datetime").reset_index(drop=True)[COUNT_COLS],
        rtol=0,
        atol=1e-12,
    )

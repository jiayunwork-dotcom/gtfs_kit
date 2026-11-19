"""
Functions about trips.
"""

from __future__ import annotations

import datetime as dt
import functools as ft
import json
from typing import TYPE_CHECKING, Iterable

import folium as fl
import folium.plugins as fp
import numpy as np
import pandas as pd
import shapely.geometry as sg
import shapely.ops as so

from . import constants as cs
from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def get_active_services(feed: "Feed", date: str) -> list[str]:
    """
    Given a Feed and a date string in YYYYMMDD format,
    return the list of service IDs that are active on the date.
    """
    active_services = set()
    removed_services = set()

    if feed.calendar is not None:
        # Convert date to weekday string (e.g., "monday")
        weekday_str = dt.datetime.strptime(date, "%Y%m%d").strftime("%A").lower()
        # Filter `calendar` to services active on date
        active_services |= set(
            feed.calendar.loc[
                lambda x: (
                    (x["start_date"] <= date)
                    & (x["end_date"] >= date)
                    & (x[weekday_str] == 1)
                ),
                "service_id",
            ]
        )

    if feed.calendar_dates is not None:
        # Filter `calendar_dates` to services added on date
        active_services |= set(
            feed.calendar_dates.loc[
                lambda x: (x["date"] == date) & (x["exception_type"] == 1),
                "service_id",
            ]
        )
        # Filter `calendar_dates` to services removed on date
        removed_services |= set(
            feed.calendar_dates.loc[
                lambda x: (x["date"] == date) & (x["exception_type"] == 2),
                "service_id",
            ]
        )

    return list(active_services - removed_services)


def get_trips(
    feed: "Feed",
    date: str | None = None,
    time: str | None = None,
    *,
    as_gdf: bool = False,
    use_utm: bool = False,
) -> pd.DataFrame | None:
    """
    Return ``feed.trips``.
    If date (YYYYMMDD date string) is given then subset the result to trips
    that start on that date.
    If a time (HH:MM:SS string, possibly with HH > 23) is given in addition to a date,
    then further subset the result to trips in service at that time
    (using trip departure times).

    If ``as_gdf`` and ``feed.shapes`` is not None, then return the trips as a
    GeoDataFrame of LineStrings representating trip shapes.
    Use local UTM CRS if ``use_utm``; otherwise it the WGS84 CRS.
    If ``as_gdf`` and ``feed.shapes`` is ``None``, then raise a ValueError.
    """
    if feed.trips is None:
        return None

    f = feed.trips
    if date is not None:
        f = f.loc[lambda x: x["service_id"].isin(get_active_services(feed, date))]

        if time is not None:
            # Get trips active during given time
            def get_active(group):
                d = {}
                start = group["departure_time"].dropna().min()
                end = group["departure_time"].dropna().max()
                try:
                    result = start <= time <= end
                except TypeError:
                    result = False
                d["is_active"] = result
                return pd.Series(d)

            f = (
                f.merge(feed.stop_times[["trip_id", "departure_time"]])
                .groupby("trip_id")
                .apply(get_active, include_groups=False)
                .reset_index()
                .loc[lambda x: x["is_active"]]
                .merge(f)
                .drop("is_active", axis=1)
            )

    if as_gdf:
        if feed.shapes is None:
            raise ValueError("This Feed has no shapes.")
        else:
            from .shapes import get_shapes

            f = (
                get_shapes(feed, as_gdf=True, use_utm=use_utm)
                .filter(["shape_id", "geometry"])
                .merge(f, how="right")
                .filter(f.columns.tolist() + ["geometry"])
            )

    return f


def compute_trip_activity(feed: "Feed", dates: list[str]) -> pd.DataFrame:
    """
    Mark trips as active or inactive on the given dates (YYYYMMDD date strings).
    Return a table with the columns

    - ``'trip_id'``
    - ``dates[0]``: 1 if the trip is active on ``dates[0]``;
      0 otherwise
    - ``dates[1]``: 1 if the trip is active on ``dates[1]``;
      0 otherwise
    - etc.
    - ``dates[-1]``: 1 if the trip is active on ``dates[-1]``;
      0 otherwise

    If ``dates`` is ``None`` or the empty list, then return an
    empty DataFrame.
    """
    dates = feed.subset_dates(dates)
    if not dates:
        return pd.DataFrame()

    active_services_by_date = {date: get_active_services(feed, date) for date in dates}
    f = feed.trips[["trip_id", "service_id"]].copy()
    for date in dates:
        active_services = active_services_by_date[date]
        f[date] = f["service_id"].isin(active_services).astype(int)

    return f.drop("service_id", axis=1)


def compute_busiest_date(feed: "Feed", dates: list[str]) -> str:
    """
    Given a list of dates (YYYYMMDD date strings), return the first date that has the
    maximum number of active trips.
    """
    f = feed.compute_trip_activity(dates)
    s = [(f[c].sum(), c) for c in f.columns if c != "trip_id"]
    return max(s)[1]


def name_stop_patterns(feed: "Feed") -> pd.DataFrame:
    """
    For each (route ID, direction ID) pair, find the distinct stop patterns of its
    trips, and assign them each an integer *pattern rank* based on the stop pattern's
    frequency rank, where 1 is the most frequent stop pattern, 2 is the second most
    frequent, etc.
    Return the DataFrame ``feed.trips`` with the additional column
    ``stop_pattern_name``, which equals the trip's 'direction_id' concatenated with a
    dash and its stop pattern rank.

    If ``feed.trips`` has no 'direction_id' column, then temporarily create one equal
    to all zeros, proceed as above, then delete the column.
    """
    # Collect stop patterns
    trips = feed.trips.copy()
    if "direction_id" in trips.columns:
        has_direction = True
    else:
        has_direction = False
        trips["direction_id"] = 0  # placeholder to ease calculations below
    f = (
        feed.stop_times.sort_values(["trip_id", "stop_sequence"])
        .groupby("trip_id")
        .apply(
            lambda x: pd.Series({"stop_pattern": "-".join(x["stop_id"])}),
            include_groups=False,
        )
        .reset_index()
        .merge(trips[["route_id", "trip_id", "direction_id"]])
    )
    # Compute pattern ranks
    f["rank"] = f.groupby(["route_id", "direction_id"])["stop_pattern"].transform(
        lambda x: pd.Series(x).map(
            {pattern: idx + 1 for idx, pattern in enumerate(x.value_counts().index)}
        )
    )
    # Assign pattern names
    f["stop_pattern_name"] = (
        f["direction_id"].astype(str).str.cat(f["rank"].astype(str), sep="-")
    )
    ff = trips.merge(f[["route_id", "trip_id", "direction_id", "stop_pattern_name"]])
    # Clean up
    if not has_direction:
        del ff["direction_id"]

    return ff


def compute_trip_stats(
    feed: "Feed",
    route_ids: list[str | None] = None,
    *,
    compute_dist_from_shapes: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame with the following columns:

    - ``'trip_id'``
    - ``'route_id'``
    - ``'route_short_name'``
    - ``'route_type'``
    - ``'direction_id'``: NaN if missing from feed
    - ``'shape_id'``: NaN if missing from feed
    - ``'stop_pattern_name'``: output from :func:`name_stop_patterns`
    - ``'num_stops'``: number of stops on trip
    - ``'start_time'``: first departure time of the trip
    - ``'end_time'``: last departure time of the trip
    - ``'start_stop_id'``: stop ID of the first stop of the trip
    - ``'end_stop_id'``: stop ID of the last stop of the trip
    - ``'is_loop'``: 1 if the start and end stop are less than 400m apart and
      0 otherwise
    - ``'distance'``: distance of the trip;
      measured in kilometers if ``feed.dist_units`` is metric;
      otherwise measured in miles;
      contains all ``np.nan`` entries if ``feed.shapes is None``
    - ``'duration'``: duration of the trip in hours
    - ``'speed'``: distance/duration

    If ``feed.stop_times`` has a ``shape_dist_traveled`` column with at
    least one non-NaN value and ``compute_dist_from_shapes == False``,
    then use that column to compute the distance column.
    Else if ``feed.shapes is not None``, then compute the distance
    column using the shapes and Shapely.
    Otherwise, set the distances to NaN.

    If route IDs are given, then restrict to trips on those routes.

    Notes
    -----
    - Assume the following feed attributes are not ``None``:

        * ``feed.trips``
        * ``feed.routes``
        * ``feed.stop_times``
        * ``feed.shapes`` (optionally)

    - Calculating trip distances with ``compute_dist_from_shapes=True``
      seems pretty accurate.  For example, calculating trip distances on
      `this Portland feed
      <https://transitfeeds.com/p/trimet/43/1400947517>`_
      using ``compute_dist_from_shapes=False`` and
      ``compute_dist_from_shapes=True``,
      yields a difference of at most 0.83km from the original values.

    """
    f = name_stop_patterns(feed)

    # Restrict to given route IDs
    if route_ids is not None:
        f = f[f["route_id"].isin(route_ids)].copy()

    # Merge with stop times and extra trip info.
    # Convert departure times to seconds past midnight to
    # compute trip durations later.
    if "direction_id" not in f.columns:
        f["direction_id"] = np.nan
    if "shape_id" not in f.columns:
        f["shape_id"] = np.nan

    f = (
        f[["route_id", "trip_id", "direction_id", "shape_id", "stop_pattern_name"]]
        .merge(feed.routes[["route_id", "route_short_name", "route_type"]])
        .merge(feed.stop_times)
        .sort_values(["trip_id", "stop_sequence"])
        .assign(departure_time=lambda x: x["departure_time"].map(hp.timestr_to_seconds))
    )

    # Compute all trips stats except distance,
    # which is possibly more involved
    geometry_by_stop = feed.build_geometry_by_stop(use_utm=True)
    g = f.groupby("trip_id")

    def my_agg(group):
        d = dict()
        d["route_id"] = group["route_id"].iat[0]
        d["route_short_name"] = group["route_short_name"].iat[0]
        d["route_type"] = group["route_type"].iat[0]
        d["direction_id"] = group["direction_id"].iat[0]
        d["shape_id"] = group["shape_id"].iat[0]
        d["stop_pattern_name"] = group["stop_pattern_name"].iat[0]
        d["num_stops"] = group.shape[0]
        d["start_time"] = group["departure_time"].iat[0]
        d["end_time"] = group["departure_time"].iat[-1]
        d["start_stop_id"] = group["stop_id"].iat[0]
        d["end_stop_id"] = group["stop_id"].iat[-1]
        dist = geometry_by_stop[d["start_stop_id"]].distance(
            geometry_by_stop[d["end_stop_id"]]
        )
        d["is_loop"] = int(dist < 400)
        d["duration"] = (d["end_time"] - d["start_time"]) / 3600
        return pd.Series(d)

    # Apply my_agg, but don't reset index yet.
    # Need trip ID as index to line up the results of the
    # forthcoming distance calculation
    h = g.apply(my_agg, include_groups=False)

    # Compute distance
    if hp.is_not_null(f, "shape_dist_traveled") and not compute_dist_from_shapes:
        # Compute distances using shape_dist_traveled column, converting to km or mi
        if hp.is_metric(feed.dist_units):
            convert_dist = hp.get_convert_dist(feed.dist_units, "km")
        else:
            convert_dist = hp.get_convert_dist(feed.dist_units, "mi")
        h["distance"] = g.apply(
            lambda group: convert_dist(group["shape_dist_traveled"].max()),
            include_groups=False,
        )
    elif feed.shapes is not None:
        # Compute distances using the shapes and Shapely
        geometry_by_shape = feed.build_geometry_by_shape(use_utm=True)
        # Convert to km or mi
        if hp.is_metric(feed.dist_units):
            m_to_dist = hp.get_convert_dist("m", "km")
        else:
            m_to_dist = hp.get_convert_dist("m", "mi")

        def compute_dist(group):
            """
            Return the distance traveled along the trip between the
            first and last stops.
            If that distance is negative or if the trip's linestring
            intersects itfeed, then return the length of the trip's
            linestring instead.
            """
            shape = group["shape_id"].iat[0]
            try:
                # Get the linestring for this trip
                linestring = geometry_by_shape[shape]
            except KeyError:
                # Shape ID is NaN or doesn't exist in shapes.
                # No can do.
                return np.nan

            # If the linestring intersects itfeed, then that can cause
            # errors in the computation below, so just
            # return the length of the linestring as a good approximation
            D = linestring.length
            if not linestring.is_simple:
                return D

            # Otherwise, return the difference of the distances along
            # the linestring of the first and last stop
            start_stop = group["stop_id"].iat[0]
            end_stop = group["stop_id"].iat[-1]
            try:
                start_point = geometry_by_stop[start_stop]
                end_point = geometry_by_stop[end_stop]
            except KeyError:
                # One of the two stop IDs is NaN, so just
                # return the length of the linestring
                return D
            d1 = linestring.project(start_point)
            d2 = linestring.project(end_point)
            d = d2 - d1
            if 0 < d < D + 100:
                return d
            else:
                # Something is probably wrong, so just
                # return the length of the linestring
                return D

        h["distance"] = g.apply(compute_dist, include_groups=False)
        # Convert from meters
        h["distance"] = h["distance"].map(m_to_dist)
    else:
        h["distance"] = np.nan

    # Reset index and compute final stats
    h = h.reset_index()
    h["speed"] = h["distance"] / h["duration"]
    h[["start_time", "end_time"]] = h[["start_time", "end_time"]].map(
        lambda x: hp.seconds_to_timestr(x)
    )

    return h.sort_values(["route_id", "direction_id", "start_time"], ignore_index=True)


def _locate_trips_calibrated(feed: "Feed", date: str, times: list[str]) -> pd.DataFrame:
    """
    Implementation of :func:`locate_trips` with ``calibrated=True``.

    Positions are computed from the calibrated vertex distance ruler of each
    shape (see :func:`.shapes.append_dist_to_shapes`): stop times are
    interpolated linearly in time between stops that have departure times,
    and the resulting distance is mapped to a point using the vertex ruler.
    """
    from . import calibration as cb

    st = feed.stop_times
    if st is None or feed.trips is None:
        raise ValueError("locate_trips requires feed.stop_times and feed.trips")

    if "shape_id" not in feed.trips.columns:
        raise ValueError("feed.trips.shape_id must exist.")

    if "shape_dist_traveled" not in st.columns:
        # Point at a trip that would have been located, to satisfy the
        # "name the offending trip_id" rule.
        shape_by_trip = dict(zip(feed.trips["trip_id"], feed.trips["shape_id"]))
        for trip_id in st["trip_id"].unique():
            if not pd.isna(shape_by_trip.get(trip_id, np.nan)):
                cb.check_trip_id(
                    trip_id,
                    "stop_times has no shape_dist_traveled column; run "
                    "append_dist_to_stop_times(calibrated=True, ...) first",
                )

    active = feed.get_stop_times(date)
    if active is None or active.empty:
        empty = pd.DataFrame(
            columns=[
                "route_id",
                "trip_id",
                "direction_id",
                "shape_id",
                "time",
                "rel_dist",
                "lon",
                "lat",
            ]
        )
        return empty

    if feed.shapes is not None and not feed.shapes.empty:
        shapes_by_id = {
            shape_id: group
            for shape_id, group in feed.shapes.groupby("shape_id", sort=False)
        }
    else:
        shapes_by_id = {}
    records = {}

    trip_info = feed.trips.copy()
    if "direction_id" not in trip_info.columns:
        trip_info["direction_id"] = np.nan
    info_by_trip = {
        row["trip_id"]: row
        for _, row in trip_info[
            ["trip_id", "route_id", "direction_id", "shape_id"]
        ].iterrows()
    }

    sample_times = np.array([hp.timestr_to_seconds(s) for s in times], dtype=float)

    rows = []
    geoms = {}

    def record_for(shape_id, trip_id):
        """
        Prepare (once) and return the calibrated vertex ruler for a shape,
        reporting problems against the trip that uses it.
        """
        if shape_id not in records:
            if shape_id not in shapes_by_id:
                cb.check_trip_id(trip_id, f"references unknown shape_id {shape_id!r}")
            try:
                prepared = cb.prepare_shape_distances(
                    shapes_by_id[shape_id], feed.dist_units, generate_missing=False
                )
            except ValueError as exc:
                cb.check_trip_id(trip_id, f"unusable shape mileage ({exc})")
            records[shape_id] = prepared[shape_id]
        return records[shape_id]

    def tol_for(shape_id):
        rec = records[shape_id]
        return cb.dist_tolerance(rec["dist"][0], rec["dist"][-1])

    active = active.sort_values(["trip_id", "stop_sequence"], kind="stable")
    for trip_id, group in active.groupby("trip_id", sort=False):
        info = info_by_trip.get(trip_id)
        shape_id = info["shape_id"] if info is not None else np.nan
        if pd.isna(shape_id):
            # Trips without a shape cannot be located.
            continue

        record = record_for(shape_id, trip_id)
        try:
            dists_all = group["shape_dist_traveled"].to_numpy(
                dtype=float, na_value=np.nan
            )
        except (TypeError, ValueError):
            cb.check_trip_id(
                trip_id,
                "stop_times.shape_dist_traveled is not numeric",
            )
        seq = group["stop_sequence"].to_numpy()

        # Missing mileage is an error for a trip that has a shape.
        missing = pd.isna(dists_all)
        if missing.any():
            bad = seq[missing]
            cb.check_trip_id(
                trip_id,
                "missing shape_dist_traveled at stop_sequence "
                + ", ".join(str(s) for s in bad[:5])
                + (" ..." if len(bad) > 5 else ""),
            )

        shape_dist = record["dist"]
        d_min, d_max = float(shape_dist[0]), float(shape_dist[-1])
        tol = tol_for(shape_id)

        # Out-of-bounds and decreasing distances.
        if np.any(dists_all < d_min - tol) or np.any(dists_all > d_max + tol):
            bad = seq[(dists_all < d_min - tol) | (dists_all > d_max + tol)]
            cb.check_trip_id(
                trip_id,
                "shape_dist_traveled outside shape "
                f"{shape_id!r} at stop_sequence " + ", ".join(str(s) for s in bad[:5]),
            )
        if len(dists_all) >= 2 and np.any(np.diff(dists_all) < -tol):
            bad = seq[1:][np.diff(dists_all) < -tol]
            cb.check_trip_id(
                trip_id,
                "shape_dist_traveled decreases with stop_sequence at "
                + ", ".join(str(s) for s in bad[:5]),
            )

        # Departure times; only timed stops participate in interpolation.
        dep = group["departure_time"].map(hp.timestr_to_seconds)
        dep_values = dep.to_numpy(dtype=float, na_value=np.nan)
        timed = ~pd.isna(dep_values)
        if not timed.any():
            continue
        timed_times = dep_values[timed]
        timed_dists = dists_all[timed]

        # Departure times must not go backwards along the trip.
        if len(timed_times) >= 2 and np.any(np.diff(timed_times) < 0):
            cb.check_trip_id(
                trip_id,
                "departure_time decreases with stop_sequence",
            )

        # Collapse repeated times, but reject one time pointing at two
        # different distances.
        points = {}
        for t, d in zip(timed_times, timed_dists):
            if t in points:
                if abs(points[t] - d) > tol:
                    cb.check_trip_id(
                        trip_id,
                        f"departure_time {hp.seconds_to_timestr(t)} points at "
                        "two different shape_dist_traveled values "
                        f"({points[t]!r} and {float(d)!r})",
                    )
                continue
            points[t] = float(d)

        if len(points) < 2:
            # Not enough distinct times to interpolate.
            continue

        point_times = np.array(sorted(points), dtype=float)
        point_dists = np.array([points[t] for t in point_times], dtype=float)

        keep = (sample_times >= point_times[0]) & (sample_times <= point_times[-1])
        ts = sample_times[keep]
        if ts.size == 0:
            continue
        ds = np.interp(ts, point_times, point_dists)

        d_first = point_dists[0]
        d_last = point_dists[-1]
        if d_last > d_first:
            rel = (ds - d_first) / (d_last - d_first)
            rel = np.clip(rel, 0.0, 1.0)
        else:
            # First and last distances coincide: report zero relative
            # distance, as specified.
            rel = np.zeros_like(ds)

        if shape_id not in geoms:
            geoms[shape_id] = cb.build_shape_geometries({shape_id: record}, [shape_id])[
                shape_id
            ]
        geom = geoms[shape_id]

        route_id = info["route_id"] if info is not None else np.nan
        direction_id = info["direction_id"] if info is not None else np.nan
        for t, r, d in zip(ts, rel, ds):
            lon, lat, _, _, _, _ = geom.locate_distance(float(d))
            rows.append(
                {
                    "route_id": route_id,
                    "trip_id": trip_id,
                    "direction_id": direction_id,
                    "shape_id": shape_id,
                    "time": hp.seconds_to_timestr(int(round(float(t)))),
                    "rel_dist": float(r),
                    "lon": lon,
                    "lat": lat,
                }
            )

    columns = [
        "shape_id",
        "trip_id",
        "time",
        "rel_dist",
        "route_id",
        "direction_id",
        "lon",
        "lat",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["trip_id", "time", "rel_dist"], ignore_index=True
    )


def locate_trips(
    feed: "Feed", date: str, times: list[str], *, calibrated: bool = False
) -> pd.DataFrame:
    """
    Return the positions of all trips active on the
    given date (YYYYMMDD date string) and times (HH:MM:SS time strings,
    possibly with HH > 23).

    Return a DataFrame with the columns

    - ``'trip_id'``
    - ``'route_id'``
    - ``'direction_id'``: all NaNs if ``feed.trips.direction_id`` is
      missing
    - ``'time'``
    - ``'rel_dist'``: number between 0 (start) and 1 (end)
      indicating the relative distance of the trip along its path
    - ``'lon'``: longitude of trip at given time
    - ``'lat'``: latitude of trip at given time

    Assume ``feed.stop_times`` has an accurate
    ``shape_dist_traveled`` column.

    If ``calibrated`` is true, then use the calibrated vertex distance ruler
    of each shape, as prepared by
    :func:`~gtfs_kit.shapes.append_dist_to_shapes` with ``calibrated=True``
    and filled in by
    :func:`~gtfs_kit.stop_times.append_dist_to_stop_times` with
    ``calibrated=True``:

    - Only trips active on ``date`` are considered; times past 24:00 still
      belong to that service day.
    - Distances are interpolated linearly in time between stops that have
      departure times, and mapped to longitude/latitude with the vertex
      ruler.
    - ``rel_dist`` runs from 0 at the first timed stop to 1 at the last timed
      stop.  If those two distances coincide, ``rel_dist`` is 0.
    - Query times outside a trip's first/last timed stop produce no row, and
      trips without a ``shape_id`` or with fewer than two distinct timed
      stops are skipped.
    - Missing, out-of-bounds, or decreasing (along ``stop_sequence``)
      distances, decreasing departure times, and a single departure time
      mapped to two different distances raise a ``ValueError`` naming the
      ``trip_id``.

    In both modes the input feed is not modified.
    """
    if calibrated:
        return _locate_trips_calibrated(feed, date, times)

    if not hp.is_not_null(feed.stop_times, "shape_dist_traveled"):
        raise ValueError(
            "feed.stop_times needs to have a non-null shape_dist_traveled "
            "column. You can create it, possibly with some inaccuracies, "
            "via feed2 = feed.append_dist_to_stop_times()."
        )

    if "shape_id" not in feed.trips.columns:
        raise ValueError("feed.trips.shape_id must exist.")

    # Start with stop times active on date
    f = feed.get_stop_times(date)
    f["departure_time"] = f["departure_time"].map(hp.timestr_to_seconds)

    # Compute relative distance of each trip along its path
    # at the given time times.
    # Use linear interpolation based on stop departure times and
    # shape distance traveled.
    geometry_by_shape = feed.build_geometry_by_shape(use_utm=False)
    sample_times = np.array([hp.timestr_to_seconds(s) for s in times])

    def compute_rel_dist(group):
        dists = sorted(group["shape_dist_traveled"].values)
        times = sorted(group["departure_time"].values)
        ts = sample_times[(sample_times >= times[0]) & (sample_times <= times[-1])]
        ds = np.interp(ts, times, dists)
        return pd.DataFrame({"time": ts, "rel_dist": ds / dists[-1]})

    # return f.groupby('trip_id', group_keys=False).\
    #   apply(compute_rel_dist).reset_index()
    g = f.groupby("trip_id").apply(compute_rel_dist, include_groups=False).reset_index()

    # Delete extraneous multi-index column
    del g["level_1"]

    # Convert times back to time strings
    g["time"] = g["time"].map(lambda x: hp.seconds_to_timestr(x))

    # Merge in more trip info and
    # compute longitude and latitude of trip from relative distance
    t = feed.trips.copy()
    if "direction_id" not in t.columns:
        t["direction_id"] = np.nan

    h = pd.merge(g, t[["trip_id", "route_id", "direction_id", "shape_id"]])
    if not h.shape[0]:
        # Return a DataFrame with the promised headers but no data.
        # Without this check, result below could be an empty DataFrame.
        h["lon"] = pd.Series()
        h["lat"] = pd.Series()
        return h

    def get_lonlat(group):
        shape = group.name
        linestring = geometry_by_shape[shape]
        lonlats = [
            linestring.interpolate(d, normalized=True).coords[0]
            for d in group["rel_dist"].values
        ]
        group["lon"], group["lat"] = zip(*lonlats)
        return group

    return (
        h.groupby("shape_id")
        .apply(get_lonlat, include_groups=False)
        .reset_index()
        .drop("level_1", axis=1)  # Where did this column come from?
    )


def trips_to_geojson(
    feed: "Feed",
    trip_ids: Iterable[str] | None = None,
    *,
    include_stops: bool = False,
) -> dict:
    """
    Return a GeoJSON FeatureCollection (in WGS84 coordinates) of LineString features
    representing this Feed's trips.

    If an iterable of trip IDs is given, then subset to those trips, which could yield
    an empty FeatureCollection in case all trip IDs are invalid.
    If ``include_stops``, then include the trip stops as Point features.
    If the Feed has no shapes, then raise a ValueError.
    """
    g = get_trips(feed, as_gdf=True)
    if trip_ids is not None:
        g = g.loc[lambda x: x["trip_id"].isin(trip_ids)]

    if g is None or g.empty:
        gj = {
            "type": "FeatureCollection",
            "features": [],
        }
    else:
        gj = json.loads(g.to_json(drop_id=True))
        if include_stops:
            st_gj = feed.stop_times_to_geojson(trip_ids)
            gj["features"].extend(st_gj["features"])

    return gj


def map_trips(
    feed: "Feed",
    trip_ids: Iterable[str],
    color_palette: list[str] = cs.COLORS_SET2,
    *,
    show_stops: bool = False,
    show_direction: bool = False,
):
    """
    Return a Folium map showing the given trips and (optionally)
    their stops.
    If any of the given trip IDs are not found in the feed, then raise a ValueError.
    If ``include_direction``, then use the Folium plugin PolyLineTextPath to draw arrows
    on each trip polyline indicating its direction of travel; this fails to work in some
    browsers, such as Brave 0.68.132.
    """
    # Initialize map
    my_map = fl.Map(tiles="cartodbpositron")

    # Create colors
    n = len(trip_ids)
    colors = [color_palette[i % len(color_palette)] for i in range(n)]

    # Collect bounding boxes to set map zoom later
    bboxes = []

    # Create a feature group for each route and add it to the map
    for i, trip_id in enumerate(trip_ids):
        collection = trips_to_geojson(feed, [trip_id], include_stops=show_stops)

        group = fl.FeatureGroup(name=f"Trip {trip_id}")
        color = colors[i]

        for f in collection["features"]:
            prop = f["properties"]

            # Add stop if present
            if f["geometry"]["type"] == "Point":
                lon, lat = f["geometry"]["coordinates"]
                fl.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    fill=True,
                    color=color,
                    weight=1,
                    popup=fl.Popup(hp.make_html(prop)),
                ).add_to(group)

            # Add trip
            else:
                path = fl.PolyLine(
                    [[x[1], x[0]] for x in f["geometry"]["coordinates"]],
                    color=color,
                    popup=hp.make_html(prop),
                )

                path.add_to(group)
                bboxes.append(sg.box(*sg.shape(f["geometry"]).bounds))

                if show_direction:
                    # Direction arrows, assuming, as GTFS does, that
                    # trip direction equals LineString direction
                    fp.PolyLineTextPath(
                        path,
                        "        \u27a4        ",
                        repeat=True,
                        offset=5.5,
                        attributes={"fill": color, "font-size": "18"},
                    ).add_to(group)

        group.add_to(my_map)

    fl.LayerControl().add_to(my_map)

    # Fit map to bounds
    bounds = so.unary_union(bboxes).bounds
    # Folium wants a different ordering
    bounds = [(bounds[1], bounds[0]), (bounds[3], bounds[2])]
    my_map.fit_bounds(bounds)

    return my_map

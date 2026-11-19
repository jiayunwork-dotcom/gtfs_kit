"""
Functions about stop times.
"""

from __future__ import annotations
from typing import Iterable, TYPE_CHECKING
import json

import numpy as np
import pandas as pd

from . import calibration as cb
from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def get_stop_times(feed: "Feed", date: str | None = None) -> pd.DataFrame:
    """
    Return ``feed.stop_times``.
    If a date (YYYYMMDD date string) is given, then subset the result to only those
    stop times with trips active on the date.
    """
    if date is None:
        f = feed.stop_times.copy()
    else:
        trip_ids = feed.get_trips(date).trip_id
        f = feed.stop_times.loc[lambda x: x.trip_id.isin(trip_ids)].copy()

    return f


def _validate_max_offset_m(max_offset_m) -> float:
    """
    Return ``max_offset_m`` after checking that it is a finite positive number.
    """
    if max_offset_m is None:
        raise ValueError(
            "append_dist_to_stop_times(calibrated=True) requires a finite "
            "positive max_offset_m (in meters)"
        )
    try:
        value = float(max_offset_m)
    except (TypeError, ValueError):
        raise ValueError(
            "max_offset_m must be a finite positive number of meters; "
            f"got {max_offset_m!r}"
        )
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            "max_offset_m must be a finite positive number of meters; "
            f"got {max_offset_m!r}"
        )
    return value


def _calibrated_stop_times(feed: "Feed", max_offset_m: float) -> "Feed":
    """
    Implementation of the calibrated branch of
    :func:`append_dist_to_stop_times`.
    """
    if feed.stop_times is None:
        return feed

    has_shapes = feed.shapes is not None and not feed.shapes.empty
    trips = feed.trips
    if trips is None:
        return feed

    if "shape_id" not in trips.columns:
        # No shape references at all: everything stays blank.
        new_feed = feed.copy()
        new_feed.stop_times = feed.stop_times.assign(shape_dist_traveled=np.nan)
        return new_feed

    shape_id_by_trip = dict(zip(trips["trip_id"], trips["shape_id"]))

    def referenced_shape(trip_id):
        return shape_id_by_trip.get(trip_id, np.nan)

    # A trip that references a shape requires a shapes table.
    if not has_shapes:
        for trip_id in feed.stop_times["trip_id"].unique():
            shape_id = referenced_shape(trip_id)
            if not pd.isna(shape_id):
                cb.check_trip_id(
                    trip_id,
                    f"references shape_id {shape_id!r}, but the feed has no shapes",
                )
        new_feed = feed.copy()
        new_feed.stop_times = feed.stop_times.assign(shape_dist_traveled=np.nan)
        return new_feed

    # Prepare the vertex distance ruler for every shape from the mileage that
    # append_dist_to_shapes(calibrated=True) prepared.  A shape whose mileage
    # is missing or invalid raises an error naming that shape, pointing the
    # caller back at the preparation step.
    records = cb.prepare_shape_distances(
        feed.shapes, feed.dist_units, generate_missing=False
    )

    # Stops in WGS84, later reprojected to the shapes' common UTM CRS.
    import geopandas as gpd
    import shapely

    if feed.stops is None:
        raise ValueError("append_dist_to_stop_times(calibrated=True) requires feed.stops")
    stop_points = gpd.GeoSeries(
        gpd.points_from_xy(feed.stops["stop_lon"], feed.stops["stop_lat"]),
        index=feed.stops["stop_id"].to_numpy(),
        crs="EPSG:4326",
    )
    stop_points = stop_points[~stop_points.index.duplicated(keep="first")]
    known_stops = set(stop_points.index)

    geoms = cb.build_shape_geometries(records)
    # Reproject stops into the same UTM CRS used for the shapes.
    utm_crs = next(iter(geoms.values())).utm_crs if geoms else None
    if stop_points.empty or utm_crs is None:
        stop_xy = {}
    else:
        projected = stop_points.to_crs(utm_crs)
        values = projected.values
        stop_xy = {
            stop_id: (float(x), float(y))
            for stop_id, x, y in zip(
                stop_points.index,
                shapely.get_x(values),
                shapely.get_y(values),
            )
        }

    tol = {s: cb.dist_tolerance(g.d_min, g.d_max) for s, g in geoms.items()}

    had_column = "shape_dist_traveled" in feed.stop_times.columns
    st = feed.stop_times.copy()
    if not had_column:
        st["shape_dist_traveled"] = np.nan
    st["shape_dist_traveled"] = st["shape_dist_traveled"].astype("float64")

    # Work on a canonical ordering, then restore it, so that permuting the
    # input rows cannot change the result keyed by (trip_id, stop_sequence).
    order = np.arange(len(st))
    st = st.assign(__order=order).sort_values(["trip_id", "stop_sequence"], kind="stable")
    out_values = {}

    for trip_id, group in st.groupby("trip_id", sort=False):
        shape_id = referenced_shape(trip_id)
        if pd.isna(shape_id):
            # Leave trips without a shape blank.
            continue
        if shape_id not in geoms:
            cb.check_trip_id(trip_id, f"references unknown shape_id {shape_id!r}")
        geom = geoms[shape_id]

        stop_ids = group["stop_id"].tolist()
        for stop_id in stop_ids:
            if pd.isna(stop_id) or stop_id not in known_stops:
                cb.check_trip_id(
                    trip_id,
                    f"references unknown stop_id {stop_id!r}",
                )
        stops_xy = [stop_xy[stop_id] for stop_id in stop_ids]
        anchors = [
            None if pd.isna(v) else float(v)
            for v in group["shape_dist_traveled"].tolist()
        ]

        matched = cb.match_trip_distances(
            geom,
            stops_xy,
            stop_ids,
            max_offset_m,
            anchors,
            trip_id=trip_id,
        )
        if matched is None:
            cb.check_trip_id(
                trip_id,
                f"no stop-to-shape matching stays within {max_offset_m} m of "
                f"shape_id {shape_id!r} in stop_sequence order",
            )

        # Validate distances against the shape's ruler once more (cheap) and
        # store by original row position.
        g_tol = tol[shape_id]
        for (_, row), d in zip(group.iterrows(), matched):
            if d < geom.d_min - g_tol or d > geom.d_max + g_tol:
                cb.check_trip_id(
                    trip_id,
                    f"shape_dist_traveled {d!r} at stop_sequence "
                    f"{row['stop_sequence']} lies outside shape "
                    f"{shape_id!r}",
                )
            out_values[row["__order"]] = d

    new_st = st.copy()
    new_st["shape_dist_traveled"] = [
        out_values.get(position, np.nan) for position in new_st["__order"]
    ]
    # Restore original row order.
    new_st = new_st.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    # Keep column order: original columns, then shape_dist_traveled.
    cols = [c for c in feed.stop_times.columns if c != "shape_dist_traveled"]
    cols = cols + ["shape_dist_traveled"]
    new_st = new_st[cols]

    new_feed = feed.copy()
    new_feed.stop_times = new_st
    return new_feed


def append_dist_to_stop_times(
    feed: "Feed",
    *,
    calibrated: bool = False,
    max_offset_m: float | None = None,
) -> "Feed":
    """
    Calculate and append the optional ``shape_dist_traveled`` column in
    ``feed.stop_times`` in terms of the distance units ``feed.dist_units``.
    Trips without shapes will have NaN distances.
    Return the resulting Feed.
    Uses ``feed.shapes``, so if that is missing, then return the original feed.

    This does not always give accurate results.
    The algorithm works as follows.
    Compute the ``shape_dist_traveled`` field by using Shapely to
    measure the distance of a stop along its trip LineString.
    If for a given trip this process produces a non-monotonically
    increasing, hence incorrect, list of (cumulative) distances, then
    fall back to estimating the distances as follows.

    Set the first distance to 0, the last to the length of the trip shape,
    and leave the remaining ones computed above.
    Choose the longest increasing subsequence of that new set of
    distances and use them and their corresponding departure times to linearly
    interpolate the rest of the distances.

    If ``calibrated`` is true, then match each stop of each trip against the
    *calibrated* vertex distance ruler prepared by
    :func:`.shapes.append_dist_to_shapes` with ``calibrated=True``.
    In that mode ``max_offset_m`` (a finite positive number of meters) is
    required, and:

    - Each trip is matched independently, walking its stops in
      ``stop_sequence`` order.  Distances may stay equal but must not
      decrease.
    - A stop may be matched to any pass of the shape within ``max_offset_m``
      meters, so loops and fold-backs that revisit a stop are handled: the
      stop's permitted distances are all the parts of the polyline inside the
      matching circle.
    - Existing ``shape_dist_traveled`` values are treated as anchors: they
      are kept verbatim, but must still lie on the shape and within the
      radius of their stop.
    - All other stops are filled in, including stops without departure times.
    - If a trip cannot be matched, or references an unknown shape or stop,
      a ``ValueError`` naming the ``trip_id`` is raised.
    - Trips without a ``shape_id`` keep blank (NaN) distances.

    The input feed is never modified; a copy is returned.
    """
    if calibrated:
        radius = _validate_max_offset_m(max_offset_m)
        return _calibrated_stop_times(feed, radius)

    if feed.shapes is None or feed.shapes.empty:
        return feed

    # Get stop and shape geometries as dictionaries
    geom_by_stop = feed.build_geometry_by_stop(use_utm=True)
    geom_by_shape = feed.build_geometry_by_shape(use_utm=True)

    # Memoize distance by stop by shape to avoid repeating calculations
    dist_by_stop_by_shape = {shape: {} for shape in geom_by_shape}

    def compute_dist(group):
        g = group.copy()

        # Compute the distances of the stops along this trip and memoize.
        shape = g.shape_id.iat[0]
        if pd.isna(shape):
            g["shape_dist_traveled"] = np.nan
        else:
            linestring = geom_by_shape[shape]
            dists = []
            for stop in g.stop_id.values:
                if stop in dist_by_stop_by_shape[shape]:
                    d = dist_by_stop_by_shape[shape][stop]
                else:
                    d = linestring.project(geom_by_stop[stop])
                    dist_by_stop_by_shape[shape][stop] = d
                dists.append(d)

            s = sorted(dists)
            D = linestring.length
            dists_are_reasonable = all([d < D + 100 for d in dists])

            if dists_are_reasonable and s == dists:
                # Good
                g["shape_dist_traveled"] = dists
            elif dists_are_reasonable and s == dists[::-1]:
                # Good after reversal.
                # This happens when the direction of the linestring
                # opposes the direction of the vehicle trip.
                dists = dists[::-1]
                g["shape_dist_traveled"] = dists
            else:
                # Bad. Redo using interpolation on a good subset of dists.
                dists = np.array([0] + dists[1:-1] + [D])
                ix = hp.longest_subsequence(dists, index=True)
                good_dists = np.take(dists, ix)
                g["shape_dist_traveled"] = np.interp(
                    g["departure_time_s"], g.iloc[ix]["departure_time_s"], good_dists
                )

                # Update dist dictionary with new and improved dists
                for row in g[["stop_id", "shape_dist_traveled"]].itertuples(index=False):
                    dist_by_stop_by_shape[shape][row.stop_id] = row.shape_dist_traveled

        return g

    cols = [c for c in feed.stop_times.columns if c != "shape_dist_traveled"]
    new_cols = cols + ["shape_dist_traveled"]

    m_to_dist = hp.get_convert_dist("m", feed.dist_units)
    st = (
        feed.stop_times.filter(cols)
        .merge(feed.trips.filter(["trip_id", "shape_id"]))
        # Convert departure times to seconds to ease calculatios
        .assign(departure_time_s=lambda x: x.departure_time.map(hp.timestr_to_seconds))
        .sort_values(["trip_id", "stop_sequence"])
        .groupby("trip_id")
        .apply(compute_dist, include_groups=False)
        .reset_index()
        # Convert distances from meters to feed's distance units
        .assign(
            shape_dist_traveled=lambda x: x.shape_dist_traveled.map(
                m_to_dist, na_action="ignore"
            )
        )
        .filter(new_cols)
    )

    # Create new feed
    new_feed = feed.copy()
    new_feed.stop_times = st

    return new_feed


def get_start_and_end_times(feed: "Feed", date: str | None = None) -> list[str]:
    """
    Return the first departure time and last arrival time
    (HH:MM:SS time strings) listed in ``feed.stop_times``, respectively.
    Restrict to the given date (YYYYMMDD string) if specified.
    """
    st = feed.get_stop_times(date)
    return (st["departure_time"].dropna().min(), st["arrival_time"].dropna().max())


def stop_times_to_geojson(
    feed: "Feed",
    trip_ids: Iterable[str] | None = None,
) -> dict:
    """
    Return a GeoJSON FeatureCollection (in WGS84 coordinates) of Point features
    representing all the trip-stop pairs in ``feed.stop_times``.

    For every trip, drop duplicate stop IDs within that trip.
    In particular, a looping trip will lack its final stop.

    If an iterable of trip IDs is given, then subset to those trips, which could yield
    an empty FeatureCollection in case of all invalid trips.
    """
    from .stops import get_stops

    st = feed.stop_times
    if trip_ids is not None:
        st = st.loc[lambda x: x.trip_id.isin(trip_ids)]

    g = (
        get_stops(feed, as_gdf=True)
        .drop_duplicates("stop_id")
        .merge(st)
        .sort_values(["trip_id", "stop_sequence"])
        .drop_duplicates(subset=["trip_id", "stop_id"])
    )

    return json.loads(g.to_json(drop_id=True))

"""
Functions about routes.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Iterable

import folium as fl
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as sg
import shapely.ops as so

from . import constants as cs
from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def build_route_timetable(feed: "Feed", route_id: str, dates: list[str]) -> pd.DataFrame:
    """
    Return a timetable for the given route and dates (YYYYMMDD date strings).

    Return a table whose columns are all those in ``feed.trips`` plus
    those in ``feed.stop_times`` plus ``'date'``.
    The trip IDs are restricted to the given route ID.

    The result is sorted first by date then by grouping by trip ID and
    sorting the groups by their first departure time.
    Skip dates outside of the Feed's dates.
    If there is no route activity on the given dates, then return an empty table.
    """
    dates = feed.subset_dates(dates)
    final_cols = ["date"] + feed.trips.columns.tolist() + feed.stop_times.columns.tolist()

    if not dates:
        return pd.DataFrame(columns=final_cols)

    t = pd.merge(feed.trips, feed.stop_times)
    t = t[t["route_id"] == route_id].copy()
    a = feed.compute_trip_activity(dates)

    frames = []
    for date in dates:
        # Slice to trips active on date
        ids = a.loc[a[date] == 1, "trip_id"]
        f = t[t["trip_id"].isin(ids)].copy()
        f["date"] = date

        # Group by trip ID and sort groups by their minimum departure time.
        # Fill missing departure times only within each trip so values do not
        # bleed across trip boundaries.
        f["dt"] = (
            f.groupby("trip_id", sort=False)["departure_time"]
            .ffill()
            .map(hp.timestr_to_seconds)
        )
        f["min_dt"] = f.groupby("trip_id", sort=False)["dt"].transform("min")
        frames.append(f)

    return (
        pd.concat(frames)
        .sort_values(["date", "min_dt", "stop_sequence"], ignore_index=True)
        .filter(final_cols)
    )


def get_routes(
    feed: "Feed",
    date: str | None = None,
    time: str | None = None,
    *,
    as_gdf: bool = False,
    use_utm: bool = False,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Return ``feed.routes`` or a subset thereof.
    If a YYYYMMDD date string is given, then restrict routes to only those active on
    the date.
    If a HH:MM:SS time string is given, possibly with HH > 23, then restrict routes
    to only those active during the time.

    Given a Feed, return a GeoDataFrame with all the columns of ``feed.routes``
    plus a geometry column of (Multi)LineStrings, each of which represents the
    corresponding routes's shape.

    If ``as_gdf`` and ``feed.shapes`` is not ``None``,
    then return a GeoDataFrame with all the columns of ``feed.routes``
    plus a geometry column of (Multi)LineStrings, each of which represents the
    corresponding routes's union of trip shapes.
    The GeoDataFrame will have a local UTM CRS if ``use_utm``; otherwise it will have
    CRS WGS84.
    If ``split_directions`` and ``as_gdf``, then add the column ``direction_id`` and
    split each route into the union of its direction 0 shapes
    and the union of its direction 1 shapes.
    If ``as_gdf`` and ``feed.shapes`` is ``None``, then raise a ValueError.
    """
    from .trips import get_trips

    trips = get_trips(feed, date=date, time=time, as_gdf=as_gdf, use_utm=use_utm)
    f = feed.routes[lambda x: x["route_id"].isin(trips["route_id"])]

    if as_gdf:
        if feed.shapes is None:
            raise ValueError("This Feed has no shapes.")

        if split_directions:
            groupby_cols = ["route_id", "direction_id"]
            final_cols = f.columns.tolist() + ["direction_id", "geometry"]
        else:
            groupby_cols = ["route_id"]
            final_cols = f.columns.tolist() + ["geometry"]

        def merge_lines(group):
            lines = [
                g
                for g in group["geometry"]
                if g and g.geom_type in ["LineString", "MultiLineString"]
            ]
            if not lines:
                return pd.Series({"geometry": None})
            return pd.Series({"geometry": so.linemerge(lines)})

        geom_cols = groupby_cols + ["shape_id", "geometry"]
        trip_geoms = trips.loc[:, geom_cols].copy()

        # Deduplicate shape geometries at the same aggregation grain used below.
        trip_geoms = trip_geoms.drop_duplicates(subset=["shape_id"] + groupby_cols)

        f = (
            trip_geoms.filter(groupby_cols + ["geometry"])
            .groupby(groupby_cols)
            .apply(merge_lines, include_groups=False)
            .reset_index()
            .merge(f, how="right")
            .pipe(gpd.GeoDataFrame)
            .set_crs(trips.crs)
            .filter(final_cols)
        )

    return f


def routes_to_geojson(
    feed: "Feed",
    route_ids: Iterable[str | None] = None,
    route_short_names: Iterable[str] | None = None,
    *,
    split_directions: bool = False,
    include_stops: bool = False,
) -> dict:
    """
    Return a GeoJSON FeatureCollection (in WGS84 coordinates) of MultiLineString
    features representing this Feed's routes.

    If an iterable of route IDs or route short names is given, then subset to the
    union of those routes, which could yield an empty FeatureCollection in case of
    all invalid route IDs / route short names.
    If ``include_stops``, then include the route stops as Point features.
    If the Feed has no shapes, then raise a ValueError.
    """
    g = get_routes(feed, as_gdf=True, split_directions=split_directions)

    # Restrict routes if given
    R = set()
    if route_ids is not None:
        R |= set(route_ids)
    if route_short_names is not None:
        R |= set(
            feed.routes.loc[
                lambda x: x["route_short_name"].isin(route_short_names), "route_id"
            ]
        )
    if R:
        g = g.loc[lambda x: x["route_id"].isin(R)]

    if g is None or g.empty:
        gj = {
            "type": "FeatureCollection",
            "features": [],
        }
    else:
        gj = json.loads(g.to_json(drop_id=True))
        if include_stops:
            stop_ids = (
                feed.stop_times.merge(feed.trips.filter(["trip_id", "route_id"]))
                .loc[lambda x: x.route_id.isin(route_ids), "stop_id"]
                .unique()
            )
            stops_gj = feed.stops_to_geojson(stop_ids=stop_ids)
            gj["features"].extend(stops_gj["features"])

    return gj


def map_routes(
    feed: "Feed",
    route_ids: Iterable[str] | None = None,
    route_short_names: Iterable[str] | None = None,
    color_palette: Iterable[str] = cs.COLORS_SET2,
    *,
    show_stops: bool = False,
):
    """
    Return a Folium map showing the given routes and (optionally) their stops.
    At least one of ``route_ids`` and ``route_short_names`` must be given.
    If both are given, then combine the two into a single set of routes.
    If any of the given route IDs are not found in the feed, then raise a ValueError.
    """
    # Compile route IDs
    R = set()
    if route_short_names is not None:
        R |= set(
            feed.routes.loc[
                lambda x: x["route_short_name"].isin(route_short_names), "route_id"
            ]
        )
    if route_ids is not None:
        R |= set(route_ids)
    route_ids = sorted(R)
    if not R:
        raise ValueError("Route IDs or route short names must be given")

    # Initialize map
    my_map = fl.Map(tiles="cartodbpositron", prefer_canvas=True)

    # Create route colors
    n = len(route_ids)
    colors = [color_palette[i % len(color_palette)] for i in range(n)]

    # Collect route bounding boxes to set map zoom later
    bboxes = []

    # Create a feature group for each route and add it to the map
    for i, route_id in enumerate(route_ids):
        collection = feed.routes_to_geojson(
            route_ids=[route_id], include_stops=show_stops
        )

        # Use route short name for group name if possible; otherwise use route ID
        route_name = route_id
        for f in collection["features"]:
            if "route_short_name" in f["properties"]:
                route_name = f["properties"]["route_short_name"]
                break

        group = fl.FeatureGroup(name=f"Route {route_name}")
        color = colors[i]

        for f in collection["features"]:
            prop = f["properties"]

            # Add stop
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

            # Add path
            else:
                prop["color"] = color
                path = fl.GeoJson(
                    f,
                    name=prop["route_short_name"],
                    style_function=lambda x: {"color": x["properties"]["color"]},
                )
                path.add_child(fl.Popup(hp.make_html(prop)))
                path.add_to(group)
                bboxes.append(sg.box(*sg.shape(f["geometry"]).bounds))

        group.add_to(my_map)

    fl.LayerControl().add_to(my_map)

    # Fit map to bounds
    bounds = so.unary_union(bboxes).bounds
    bounds2 = [bounds[1::-1], bounds[3:1:-1]]  # Folium expects this ordering
    my_map.fit_bounds(bounds2)

    return my_map


def compute_route_stats_0(
    trip_stats: pd.DataFrame,
    headway_start_time: str = "07:00:00",
    headway_end_time: str = "19:00:00",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Compute stats for the given subset of trips stats of the form output by the
    function :func:`.trips.compute_trip_stats`.
    Ignore trips with zero duration.

    If ``split_directions``, then separate the stats by trip direction (0 or 1).
    Use the headway start and end times to specify the time period for computing
    headway stats.

    Return a DataFrame with the columns

    - ``'route_id'``
    - ``'route_short_name'``
    - ``'route_type'``
    - ``'direction_id'``: present if only if ``split_directions``
    - ``'num_trips'``: number of trips on the route in the subset
    - ``'num_trip_starts'``: number of trips on the route with
      nonnull start times
    - ``'num_trip_ends'``: number of trips on the route with nonnull
      end times that end before 23:59:59
    - ``'num_stop_patterns'``: number of stop pattern across trips
    - ``'is_loop'``: 1 if at least one of the trips on the route has
      its ``is_loop`` field equal to 1; 0 otherwise
    - ``'is_bidirectional'``: 1 if the route has trips in both
      directions; 0 otherwise; present if only if not ``split_directions``
    - ``'start_time'``: start time of the earliest trip on the route
    - ``'end_time'``: end time of latest trip on the route
    - ``'max_headway'``: maximum of the durations (in minutes)
      between trip starts on the route between
      ``headway_start_time`` and ``headway_end_time`` on the given
      dates
    - ``'min_headway'``: minimum of the durations (in minutes)
      mentioned above
    - ``'mean_headway'``: mean of the durations (in minutes)
      mentioned above
    - ``'peak_num_trips'``: maximum number of simultaneous trips in
      service (for the given direction, or for both directions when
      ``split_directions==False``)
    - ``'peak_start_time'``: start time of first longest period
      during which the peak number of trips occurs
    - ``'peak_end_time'``: end time of first longest period during
      which the peak number of trips occurs
    - ``'service_duration'``: total of the duration of each trip on
      the route in the given subset of trips; measured in hours
    - ``'service_distance'``: total of the distance traveled by each
      trip on the route in the given subset of trips;
      measured in kilometers if ``feed.dist_units`` is metric;
      otherwise measured in miles;
      contains all ``np.nan`` entries if ``feed.shapes is None``
    - ``'service_speed'``: service_distance/service_duration when defined; 0 otherwise
    - ``'mean_trip_distance'``: service_distance/num_trips
    - ``'mean_trip_duration'``: service_duration/num_trips

    If ``trip_stats`` is empty, return an empty DataFrame.

    If not ``split_directions``, then compute each route's stats,
    except for headways, using its trips running in both directions.
    For headways, (1) compute max headway by taking the max of the
    max headways in both directions; (2) compute mean headway by
    taking the weighted mean of the mean headways in both
    directions.

    Raise a ValueError if ``split_directions`` and no non-null
    direction ID values present.
    """
    final_cols = [
        "route_id",
        "route_short_name",
        "route_type",
        "num_trips",
        "num_trip_starts",
        "num_trip_ends",
        "num_stop_patterns",
        "is_loop",
        "start_time",  # HH:MM:SS
        "end_time",  # HH:MM:SS
        "max_headway",  # minutes
        "min_headway",  # minutes
        "mean_headway",  # minutes
        "peak_num_trips",
        "peak_start_time",  # HH:MM:SS
        "peak_end_time",  # HH:MM:SS
        "service_distance",
        "service_duration",  # hours
        "service_speed",
        "mean_trip_distance",
        "mean_trip_duration",
    ]
    if split_directions:
        final_cols.append("direction_id")
    else:
        final_cols.append("is_bidirectional")

    null_stats = pd.DataFrame(data=[], columns=final_cols)

    # Handle defunct case
    if trip_stats.empty:
        return null_stats

    # Remove defunct trips
    f = trip_stats.loc[lambda x: x["duration"] > 0].copy()

    # Convert trip start and end times to seconds to ease calculations below
    f[["start_time", "end_time"]] = f[["start_time", "end_time"]].map(
        hp.timestr_to_seconds
    )

    headway_start = hp.timestr_to_seconds(headway_start_time)
    headway_end = hp.timestr_to_seconds(headway_end_time)

    def agg_sd(group):
        # Take this group of all trips stats for a single route
        # and compute route-level stats.
        d = dict()
        d["route_short_name"] = group["route_short_name"].iat[0]
        d["route_type"] = group["route_type"].iat[0]
        d["num_trips"] = group.shape[0]
        d["num_trip_starts"] = group["start_time"].count()
        d["num_trip_ends"] = group.loc[group["end_time"] < 24 * 3600, "end_time"].count()
        d["num_stop_patterns"] = group["stop_pattern_name"].nunique()
        d["is_loop"] = int(group["is_loop"].any())
        d["start_time"] = group["start_time"].min()
        d["end_time"] = group["end_time"].max()

        # Compute max and mean headway
        stimes = group["start_time"].values
        stimes = sorted(
            [stime for stime in stimes if headway_start <= stime <= headway_end]
        )
        headways = np.diff(stimes)
        if headways.size:
            d["max_headway"] = np.max(headways) / 60  # minutes
            d["min_headway"] = np.min(headways) / 60  # minutes
            d["mean_headway"] = np.mean(headways) / 60  # minutes
        else:
            d["max_headway"] = np.nan
            d["min_headway"] = np.nan
            d["mean_headway"] = np.nan

        # Compute peak num trips
        active_trips = hp.get_active_trips_df(group[["start_time", "end_time"]])
        times, counts = active_trips.index.values, active_trips.values
        start, end = hp.get_peak_indices(times, counts)
        d["peak_num_trips"] = counts[start]
        d["peak_start_time"] = times[start]
        d["peak_end_time"] = times[end]

        d["service_distance"] = group["distance"].sum()
        d["service_duration"] = group["duration"].sum()

        return pd.Series(d)

    def agg(group):
        d = dict()
        d["route_short_name"] = group["route_short_name"].iat[0]
        d["route_type"] = group["route_type"].iat[0]
        d["num_trips"] = group.shape[0]
        d["num_trip_starts"] = group["start_time"].count()
        d["num_trip_ends"] = group.loc[group["end_time"] < 24 * 3600, "end_time"].count()
        d["num_stop_patterns"] = group["stop_pattern_name"].nunique()
        d["is_loop"] = int(group["is_loop"].any())
        d["is_bidirectional"] = int(group["direction_id"].unique().size > 1)
        d["start_time"] = group["start_time"].min()
        d["end_time"] = group["end_time"].max()

        # Compute headway stats
        headways = np.array([])
        for direction in [0, 1]:
            stimes = group[group["direction_id"] == direction]["start_time"].values
            stimes = sorted(
                [stime for stime in stimes if headway_start <= stime <= headway_end]
            )
            headways = np.concatenate([headways, np.diff(stimes)])
        if headways.size:
            d["max_headway"] = np.max(headways) / 60  # minutes
            d["min_headway"] = np.min(headways) / 60  # minutes
            d["mean_headway"] = np.mean(headways) / 60  # minutes
        else:
            d["max_headway"] = np.nan
            d["min_headway"] = np.nan
            d["mean_headway"] = np.nan

        # Compute peak num trips
        active_trips = hp.get_active_trips_df(group[["start_time", "end_time"]])
        times, counts = active_trips.index.values, active_trips.values
        start, end = hp.get_peak_indices(times, counts)
        d["peak_num_trips"] = counts[start]
        d["peak_start_time"] = times[start]
        d["peak_end_time"] = times[end]

        d["service_distance"] = group["distance"].sum()
        d["service_duration"] = group["duration"].sum()

        return pd.Series(d)

    if split_directions:
        f = f.loc[lambda x: x.direction_id.notnull()].assign(
            direction_id=lambda x: x.direction_id.astype(int)
        )
        if f.empty:
            raise ValueError(
                "At least one trip stats direction ID value must be non-NaN."
            )

        g = (
            f.groupby(["route_id", "direction_id"])
            .apply(agg_sd, include_groups=False)
            .reset_index()
        )
    else:
        g = f.groupby("route_id").apply(agg, include_groups=False).reset_index()

    # Compute a few more stats
    g["service_speed"] = (
        g["service_distance"]
        .div(g["service_duration"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    g["mean_trip_distance"] = g["service_distance"] / g["num_trips"]
    g["mean_trip_duration"] = g["service_duration"] / g["num_trips"]

    # Convert route times to time strings
    g[["start_time", "end_time", "peak_start_time", "peak_end_time"]] = g[
        ["start_time", "end_time", "peak_start_time", "peak_end_time"]
    ].map(lambda x: hp.seconds_to_timestr(x))

    return g.filter(final_cols)


def compute_route_stats(
    feed: "Feed",
    dates: list[str],
    trip_stats: pd.DataFrame | None = None,
    headway_start_time: str = "07:00:00",
    headway_end_time: str = "19:00:00",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Compute route stats for all the trips that lie in the given subset
    of trip stats, which defaults to ``feed.compute_trip_stats()``,
    and that start on the given dates (YYYYMMDD date strings).

    If ``split_directions``, then separate the stats by trip direction (0 or 1).
    Use the headway start and end times to specify the time period for computing
    headway stats.

    Return a DataFrame with the columns

    - ``'date'``
    - ``'route_id'``
    - ``'route_short_name'``
    - ``'route_type'``
    - ``'direction_id'``: present if only if ``split_directions``
    - ``'num_trips'``: number of trips on the route in the subset
    - ``'num_trip_starts'``: number of trips on the route with
      nonnull start times
    - ``'num_trip_ends'``: number of trips on the route with nonnull
      end times that end before 23:59:59
    - ``'num_stop_patterns'``: number of stop pattern across trips
    - ``'is_loop'``: 1 if at least one of the trips on the route has
      its ``is_loop`` field equal to 1; 0 otherwise
    - ``'is_bidirectional'``: 1 if the route has trips in both
      directions; 0 otherwise; present if only if not ``split_directions``
    - ``'start_time'``: start time of the earliest trip on the route
    - ``'end_time'``: end time of latest trip on the route
    - ``'max_headway'``: maximum of the durations (in minutes)
      between trip starts on the route between
      ``headway_start_time`` and ``headway_end_time`` on the given
      dates
    - ``'min_headway'``: minimum of the durations (in minutes)
      mentioned above
    - ``'mean_headway'``: mean of the durations (in minutes)
      mentioned above
    - ``'peak_num_trips'``: maximum number of simultaneous trips in
      service (for the given direction, or for both directions when
      ``split_directions==False``)
    - ``'peak_start_time'``: start time of first longest period
      during which the peak number of trips occurs
    - ``'peak_end_time'``: end time of first longest period during
      which the peak number of trips occurs
    - ``'service_duration'``: total of the duration of each trip on
      the route in the given subset of trips; measured in hours
    - ``'service_distance'``: total of the distance traveled by each
      trip on the route in the given subset of trips;
      measured in kilometers if ``feed.dist_units`` is metric;
      otherwise measured in miles;
      contains all ``np.nan`` entries if ``feed.shapes is None``
    - ``'service_speed'``: service_distance/service_duration when defined; 0 otherwise
    - ``'mean_trip_distance'``: service_distance/num_trips
    - ``'mean_trip_duration'``: service_duration/num_trips


    Exclude dates with no active trips, which could yield the empty DataFrame.

    If not ``split_directions``, then compute each route's stats,
    except for headways, using its trips running in both directions.
    For headways, (1) compute max headway by taking the max of the
    max headways in both directions; (2) compute mean headway by
    taking the weighted mean of the mean headways in both
    directions.

    Notes
    -----
    - If you've already computed trip stats in your workflow, then you should pass
      that table into this function to speed things up significantly.
    - The route stats for date d contain stats for trips that start on
      date d only and ignore trips that start on date d-1 and end on
      date d.
    - Raise a ValueError if ``split_directions`` and no non-null
      direction ID values present.

    """
    null_stats = compute_route_stats_0(
        feed.trips.head(0), split_directions=split_directions
    )
    final_cols = ["date"] + list(null_stats.columns)
    null_stats = null_stats.assign(date=None).filter(final_cols)
    dates = feed.subset_dates(dates)

    # Handle defunct case
    if not dates:
        return null_stats

    if trip_stats is None:
        trip_stats = feed.compute_trip_stats()

    # Collect stats for each date,
    # memoizing stats the sequence of trip IDs active on the date
    # to avoid unnecessary recomputations.
    # Store in a dictionary of the form
    # trip ID sequence -> stats DataFrame.
    stats_by_ids = {}
    activity = feed.compute_trip_activity(dates)
    frames = []
    for date in dates:
        ids = tuple(sorted(activity.loc[activity[date] > 0, "trip_id"].values))
        if ids in stats_by_ids:
            # Reuse stats with updated date
            stats = stats_by_ids[ids].assign(date=date)
        elif ids:
            # Compute stats afresh
            t = trip_stats.loc[lambda x: x.trip_id.isin(ids)].copy()
            stats = compute_route_stats_0(
                t,
                split_directions=split_directions,
                headway_start_time=headway_start_time,
                headway_end_time=headway_end_time,
            ).assign(date=date)
            # Remember stats
            stats_by_ids[ids] = stats
        else:
            stats = null_stats

        frames.append(stats)

    # Collate stats
    sort_by = (
        ["date", "route_id", "direction_id"] if split_directions else ["date", "route_id"]
    )
    return pd.concat(frames).filter(final_cols).sort_values(sort_by)


def compute_route_time_series_0(
    trip_stats: pd.DataFrame,
    date_label: str = "20010101",
    freq: str = "h",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Compute stats in a 24-hour time series form at the given Pandas frequency
    for the given subset of trip stats of the
    form output by the function :func:`.trips.compute_trip_stats`.

    If ``split_directions``, then separate each routes's stats by trip direction.
    Use the given YYYYMMDD date label as the date in the time series index.

    Return a long-format DataFrame with the columns

    - ``datetime``: datetime object
    - ``route_id``
    - ``direction_id``: direction of route; presest if and only if ``split_directions``
    - ``num_trips``: number of trips in service on the route
      at any time within the time bin
    - ``num_trip_starts``: number of trips that start within
      the time bin
    - ``num_trip_ends``: number of trips that end within the
      time bin, ignoring trips that end past midnight
    - ``service_distance``: sum of the service distance accrued
      during the time bin across all trips on the route;
      measured in kilometers if ``feed.dist_units`` is metric;
      otherwise measured in miles;
    - ``service_duration``: sum of the service duration accrued
      during the time bin across all trips on the route;
      measured in hours
    - ``service_speed``: ``service_distance/service_duration``
      for the route


    Notes
    -----
    - Bin ``k`` covers the half-open interval
      ``[k * freq, (k + 1) * freq)`` of the service day.  Bin boundaries
      are measured from midnight and are computed with second (actually
      nanosecond) precision, so trip start/end times that fall on
      seconds are handled correctly.
    - Trips that lack start or end times, or that have zero duration,
      are ignored, so the aggregate ``num_trips`` across the day could be
      less than the ``num_trips`` column of :func:`compute_route_stats_0`
    - ``num_trips`` counts the distinct trips that run for a positive
      amount of time inside the bin.  A trip that ends exactly at the
      left edge of a bin is not counted as running in that bin.
    - ``num_trip_starts`` and ``num_trip_ends`` are event counts binned
      by the time at which the event occurs.
    - ``service_duration`` only includes the part of each trip that falls
      inside the bin, and ``service_distance`` is allocated in proportion
      to time (constant-speed assumption within each trip).
    - All trip departure times are taken modulo 24 hours.
      So trips that end past 23:59:59 will have their stats wrap around
      to the early morning of the time series, except for their
      ``num_trip_ends`` indicator: trip endings past 23:59:59 are not
      binned, preserving the original cross-midnight convention.
    - Because a trip's start event never precedes the first bin in which
      it runs, the distinct-trip count of a coarser bin that is a union
      of fine bins is
      ``num_trips(first fine bin) + sum(num_trip_starts of the other
      fine bins)``.  This is what :func:`.helpers.downsample` uses to
      coarsen the table exactly without recording trip IDs.
    - Raise a ValueError if ``split_directions`` and no non-null
      direction ID values present

    """
    final_cols = [
        "datetime",
        "route_id",
        "num_trips",
        "num_trip_starts",
        "num_trip_ends",
        "service_distance",
        "service_duration",
        "service_speed",
    ]
    if split_directions:
        final_cols.insert(2, "direction_id")

    null_stats = pd.DataFrame([], columns=final_cols)

    # Handle defunct case
    if trip_stats.empty:
        return null_stats

    freq_ns = hp.freq_to_ns(freq)
    day_ns = hp._SECONDS_PER_DAY * hp._NS_PER_SECOND
    num_bins = -(-day_ns // freq_ns)  # Ceiling division
    ns_per_sec = hp._NS_PER_SECOND

    tss = trip_stats.copy()
    if split_directions:
        if "direction_id" not in tss.columns:
            tss["direction_id"] = pd.NA
        tss = tss.loc[lambda x: x.direction_id.notnull()].assign(
            direction_id=lambda x: x.direction_id.astype(int)
        )
        if tss.empty:
            raise ValueError(
                "At least one trip stats direction ID value must be non-NaN."
            )

        # Alter route IDs to encode direction:
        # <route ID>-0 and <route ID>-1 or <route ID>-NA
        tss["route_id"] = (
            tss["route_id"] + "-" + tss["direction_id"].map(lambda x: str(int(x)))
        )

    tss = tss.loc[lambda x: x["route_id"].notnull()]
    routes = sorted(tss["route_id"].unique().tolist())

    indicators = [
        "num_trip_starts",
        "num_trip_ends",
        "num_trips",
        "service_duration",
        "service_distance",
    ]

    # One zero-filled array per route per indicator, indexed by bin number.
    # This is O(number of bins) per route, not per trip per second.
    values = {
        indicator: {route: np.zeros(num_bins, dtype=float) for route in routes}
        for indicator in indicators
    }

    wrapped_any = False
    for row in tss.itertuples(index=False):
        route = row.route_id
        start = hp.timestr_to_seconds(row.start_time, mod24=True)
        end = hp.timestr_to_seconds(row.end_time, mod24=True)

        # Ignore defunct trips
        if pd.isna(start) or pd.isna(end):
            continue
        start = int(start)
        end = int(end)
        if start == end:
            continue

        start_ns = start * ns_per_sec
        end_ns = end * ns_per_sec
        wrapped = start > end
        wrapped_any = wrapped_any or wrapped
        total_ns = end_ns - start_ns if not wrapped else day_ns - start_ns + end_ns
        if total_ns <= 0:
            continue

        distance = row.distance
        if pd.isna(distance):
            distance = 0.0

        # Active intervals of the trip within the service day
        segments = (
            [(start_ns, day_ns), (0, end_ns)]
            if wrapped
            else [(start_ns, end_ns)]
        )

        for segment_number, (a, b) in enumerate(segments):
            if b <= a:
                continue
            # Bins touched by [a, b), with b exclusive
            first_bin = a // freq_ns
            last_bin = (b - 1) // freq_ns
            weight = distance / total_ns

            # The two segments of a wrapped trip are disjoint in time, so
            # they share a bin only when that single bin covers the whole
            # day.  In that case count the trip once.
            count_trip = not (num_bins == 1 and segment_number > 0)

            def add_overlap(k, overlap_ns):
                if count_trip:
                    values["num_trips"][route][k] += 1
                values["service_duration"][route][k] += (
                    overlap_ns / ns_per_sec / 3600
                )
                values["service_distance"][route][k] += weight * overlap_ns

            if first_bin == last_bin:
                add_overlap(first_bin, b - a)
                continue

            # Partial first and last bins
            add_overlap(first_bin, (first_bin + 1) * freq_ns - a)
            add_overlap(last_bin, b - last_bin * freq_ns)

            # Interior bins are fully covered; fill them with slice arithmetic
            # rather than an O(bins) Python loop.
            interior = slice(first_bin + 1, last_bin)
            if interior.stop > interior.start:
                if count_trip:
                    values["num_trips"][route][interior] += 1
                values["service_duration"][route][interior] += (
                    freq_ns / ns_per_sec / 3600
                )
                values["service_distance"][route][interior] += weight * freq_ns

        # Trip start event
        values["num_trip_starts"][route][start_ns // freq_ns] += 1

        # Trip end event (cross-midnight ends keep being excluded)
        if not wrapped:
            values["num_trip_ends"][route][end_ns // freq_ns] += 1

    # Build per-indicator DataFrames indexed by bin start across the given date
    base = pd.Timestamp(pd.to_datetime(date_label, format="%Y%m%d"))
    rng = pd.date_range(base, periods=num_bins, freq=freq)
    series_by_indicator = {}
    for indicator, by_route in values.items():
        frame = pd.DataFrame(by_route, index=rng)
        if indicator in ("num_trip_starts", "num_trip_ends", "num_trips"):
            # These are event/entity counts; keep them integral, as before.
            frame = frame.round().astype("int64")
        series_by_indicator[indicator] = frame

    # Combine into a single long-form time series per route (and direction if requested);
    # hp.combine_time_series is expected to compute derived fields like service_speed
    g = hp.combine_time_series(
        series_by_indicator, kind="route", split_directions=split_directions
    )
    # Present the documented column order
    g = g.filter(final_cols)
    return hp.set_time_series_meta(
        g, kind="route", freq_ns=freq_ns, wrapped=wrapped_any
    )


def compute_route_time_series(
    feed: "Feed",
    dates: list[str],
    trip_stats: pd.DataFrame | None = None,
    freq: str = "h",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Compute route stats in time series form for the trips that lie in
    the trip stats subset, which defaults to the output of
    :func:`.trips.compute_trip_stats`, and that start on the given dates
    (YYYYMMDD date strings).

    If ``split_directions``, then separate each routes's stats by trip direction.
    Specify the time series frequency with a Pandas frequency string, e.g. ``'5Min'``.

    Return a time series DataFrame with the following columns.

    - ``datetime``: datetime object
    - ``route_id``
    - ``direction_id``: direction of route; presest if and only if ``split_directions``
    - ``num_trips``: number of trips in service on the route
      at any time within the time bin
    - ``num_trip_starts``: number of trips that start within
      the time bin
    - ``num_trip_ends``: number of trips that end within the
      time bin, ignoring trips that end past midnight
    - ``service_distance``: sum of the service distance accrued
      during the time bin across all trips on the route;
      measured in kilometers if ``feed.dist_units`` is metric;
      otherwise measured in miles;
    - ``service_duration``: sum of the service duration accrued
      during the time bin across all trips on the route;
      measured in hours
    - ``service_speed``: ``service_distance/service_duration``
      for the route

    Exclude dates that lie outside of the Feed's date range.
    If all dates lie outside the Feed's date range, then return an
    empty DataFrame.

    The result also carries provenance metadata in ``.attrs`` (see
    :func:`.helpers.downsample`), so it can later be coarsened exactly,
    e.g. from '1min' to '5min' to '15min', with counts matching a table
    computed directly at the coarse frequency.

    Notes
    -----
    - If you've already computed trip stats in your workflow, then you should pass
      that table into this function to speed things up significantly.
    - See the notes for :func:`compute_route_time_series_0`
    - Raise a ValueError if ``split_directions`` and no non-null
      direction ID values present

    """
    dates = feed.subset_dates(dates)
    null_stats = compute_route_time_series_0(
        pd.DataFrame(), split_directions=split_directions
    )

    # Handle defunct case
    if not dates:
        return null_stats

    freq_ns = hp.freq_to_ns(freq)
    activity = feed.compute_trip_activity(dates)
    if trip_stats is None:
        trip_stats = feed.compute_trip_stats()
    else:
        trip_stats = trip_stats.copy()

    # Collect stats for each date, memoizing stats by trip ID sequence
    # to avoid unnecessary re-computations.
    # Store in dictionary of the form
    # trip ID sequence -> stats table
    null_stats = pd.DataFrame()
    stats_by_ids = {}
    activity = feed.compute_trip_activity(dates)
    frames = []
    wrapped = False
    for date in dates:
        ids = tuple(sorted(activity.loc[activity[date] > 0, "trip_id"].values))
        if ids in stats_by_ids:
            # Reuse stats with updated date
            stats = stats_by_ids[ids].pipe(hp.replace_date, date=date)
        elif ids:
            # Compute stats afresh
            t = trip_stats.loc[lambda x: x.trip_id.isin(ids)].copy()
            stats = compute_route_time_series_0(
                t, split_directions=split_directions, freq=freq, date_label=date
            )
            wrapped = wrapped or bool(hp.get_time_series_meta(stats).get("wrapped"))
            stats = stats.pipe(hp.replace_date, date=date)
            # Remember stats
            stats_by_ids[ids] = stats
        else:
            stats = null_stats

        frames.append(stats)

    # Collate stats
    result = pd.concat(frames, ignore_index=True)
    return hp.set_time_series_meta(
        result, kind="route", freq_ns=freq_ns, wrapped=wrapped
    )

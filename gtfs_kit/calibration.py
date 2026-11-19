"""
Shared machinery for *calibrated* shape-distance calculations.

The ordinary :func:`.shapes.append_dist_to_shapes` computes
``shape_dist_traveled`` as the cumulative planar length of the shape, and
:func:`.stop_times.append_dist_to_stop_times` snaps stops to a shape with
``shapely.project``.  Both assume that the shape is not self-intersecting and
that the whole shape is traversed by every trip.

Calibrated mode instead treats the ``shape_dist_traveled`` values on the shape
vertices as an arbitrary but strictly increasing *ruler* along the polyline.
This makes it possible to

- let a feed supply its own (possibly non-uniform) mileage ticks,
- match a stop to a particular *pass* of the polyline rather than always to
  the same projection (important for loop and fold-back shapes), and
- locate a vehicle between two vertices by using the ruler values rather than
  the raw geometric length.

The helpers in this module are deliberately free of any ``Feed`` imports so
that they can be used from :mod:`gtfs_kit.shapes`,
:mod:`gtfs_kit.stop_times` and :mod:`gtfs_kit.trips` without creating import
cycles.

Choosing among ambiguous matches
--------------------------------
Near a self-intersection or fold-back, several points of the polyline can be
within the matching radius of one stop.  Rather than pick one blindly, the
matcher (:func:`match_trip_distances`) treats each stop's in-radius pieces as
a set of distance *intervals* and looks for a non-decreasing chain through
them in ``stop_sequence`` order:

- the forward pass records the least distance reachable at each stop;
- the backward pass records the greatest distance from which the trip can
  still finish.

Their intersection is a corridor of feasible distances.  Inside it the
perpendicular projection of the stop is used, clamped so distances never
decrease.  A trip is reported unmatchable exactly when the corridor is empty,
never because of a tie-break.  The choice is deterministic and depends only
on this trip's stops, so two trips sharing a shape cannot influence each
other's result.

Cost and reuse
--------------
For a shape with ``n`` vertices and a stop list of length ``k``:

- Each stop's in-radius intervals are found with a Shapely ``STRtree`` over
  the shape's segments, so a stop tests ``O(log n + c)`` candidate segments
  instead of all ``n`` (very short shapes, at most 32 segments, skip the index
  and scan linearly, which is cheaper to set up).
- The per-stop intervals depend only on the shape geometry and the stop
  coordinates, never on the trip, so they are cached on the
  :class:`ShapeGeometry` object.  A shape used by many trips computes each
  stop's geometry once and reuses it for every trip.
- The chain search itself is ``O(k)`` once the intervals are known, because
  both passes are single sweeps over the interval lists.

The :class:`ShapeGeometry` for a shape is likewise worth building once per
feed and sharing across trips; :func:`build_shape_geometries` projects all of
a feed's vertices in one batch for exactly that reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as cs
from . import helpers as hp

#: Relative slack used when comparing distances that may have been round
#: tripped through a CSV file.
DIST_REL_TOL = 1e-9


def dist_tolerance(*values: float) -> float:
    """
    Return a small absolute tolerance suitable for comparing distances whose
    magnitude is at most ``max(values)``.
    """
    scale = max([abs(float(v)) for v in values] + [1.0])
    return DIST_REL_TOL * scale


def check_shape_id(shape_id, message: str):
    """
    Raise a ``ValueError`` about the given shape ID.
    """
    raise ValueError(f"shape_id {shape_id!r}: {message}")


def check_trip_id(trip_id, message: str):
    """
    Raise a ``ValueError`` about the given trip ID.
    """
    raise ValueError(f"trip_id {trip_id!r}: {message}")


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------


def project_to_utm(
    lon: np.ndarray, lat: np.ndarray, crs=None
) -> tuple[np.ndarray, np.ndarray, object]:
    """
    Project WGS84 longitude/latitude arrays to a local UTM CRS.

    Return ``(x, y, utm_crs)``.  If ``crs`` is not given, estimate a UTM CRS
    from the given points.
    """
    import geopandas as gpd
    import shapely

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.size == 0:
        return np.array([]), np.array([]), crs

    gs = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=cs.WGS84)
    if crs is None:
        crs = gs.estimate_utm_crs()
    projected = gs.to_crs(crs)
    values = projected.values
    return shapely.get_x(values), shapely.get_y(values), crs


# ---------------------------------------------------------------------------
# Shape vertex validation and distance preparation
# ---------------------------------------------------------------------------


def prepare_shape_distances(
    shapes: pd.DataFrame | None,
    dist_units: str,
    *,
    generate_missing: bool = False,
) -> dict:
    """
    Validate the shape vertices and return a dictionary

      ``shape_id -> {"shape_id", "sequence", "lon", "lat", "dist"}``

    where ``dist`` is a strictly increasing NumPy array in ``dist_units``.

    The rules are:

    - Vertices are ordered by ``shape_pt_sequence`` and that sequence must be
      unique within a shape.
    - Adjacent vertices must not have identical coordinates.
    - If a shape has a complete ``shape_dist_traveled`` column (all values
      present, finite, non-negative, and strictly increasing along the
      sequence), those values are kept verbatim; the first value need not be
      zero.
    - If the column is entirely missing for a shape and ``generate_missing``
      is true, distances are generated from the shape geometry in
      ``dist_units``.
    - A partially filled column, invalid values, duplicate vertex sequences or
      coincident adjacent vertices raise a ``ValueError`` naming the shape ID.
    """
    result = {}
    if shapes is None or shapes.empty:
        return result

    required = {"shape_id", "shape_pt_sequence", "shape_pt_lon", "shape_pt_lat"}
    missing = required - set(shapes.columns)
    if missing:
        raise ValueError(
            "shapes must have columns "
            + ", ".join(sorted(required))
            + f"; missing: {', '.join(sorted(missing))}"
        )

    has_dist_col = "shape_dist_traveled" in shapes.columns

    ordered = shapes.sort_values(["shape_id", "shape_pt_sequence"], kind="stable")
    groups = list(ordered.groupby("shape_id", sort=False))

    # Project every vertex once, to a single UTM CRS, for generating any
    # missing distances.  Doing the projection in one shot keeps the result
    # independent of the per-shape CRS that ``estimate_utm_crs`` might pick.
    generated = {}
    if generate_missing:
        all_lon = np.concatenate(
            [group["shape_pt_lon"].to_numpy(dtype=float) for _, group in groups]
        )
        all_lat = np.concatenate(
            [group["shape_pt_lat"].to_numpy(dtype=float) for _, group in groups]
        )
        all_x, all_y, _ = project_to_utm(all_lon, all_lat)
        position = 0
        for shape_id, group in groups:
            n = len(group)
            x = all_x[position : position + n]
            y = all_y[position : position + n]
            position += n
            seg = np.hypot(np.diff(x), np.diff(y))
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            generated[shape_id] = hp.get_convert_dist("m", dist_units)(cum)

    for shape_id, group in groups:
        n = len(group)
        sequence = group["shape_pt_sequence"].to_numpy()
        if len(np.unique(sequence)) != n:
            check_shape_id(shape_id, "duplicate shape_pt_sequence values")

        lon = group["shape_pt_lon"].to_numpy(dtype=float)
        lat = group["shape_pt_lat"].to_numpy(dtype=float)
        if np.isnan(lon).any() or np.isnan(lat).any():
            check_shape_id(shape_id, "missing shape vertex coordinates")

        if n >= 2:
            same = (lon[1:] == lon[:-1]) & (lat[1:] == lat[:-1])
            if same.any():
                i = int(np.flatnonzero(same)[0])
                check_shape_id(
                    shape_id,
                    "adjacent vertices at shape_pt_sequence "
                    f"{sequence[i]} and {sequence[i + 1]} have identical "
                    "coordinates",
                )

        if has_dist_col:
            raw = group["shape_dist_traveled"]
            values = raw.to_numpy(dtype=float, na_value=np.nan)
            n_missing = int(pd.isna(values).sum())
        else:
            values = np.full(n, np.nan)
            n_missing = n

        if n_missing == n:
            if not generate_missing:
                check_shape_id(
                    shape_id,
                    "has no shape_dist_traveled values; run "
                    "append_dist_to_shapes(calibrated=True) first",
                )
            dist = generated[shape_id]
        elif n_missing:
            bad_seq = sequence[pd.isna(values)]
            check_shape_id(
                shape_id,
                "shape_dist_traveled is only partially filled; missing at "
                "shape_pt_sequence "
                + ", ".join(str(s) for s in bad_seq[:5])
                + (" ..." if len(bad_seq) > 5 else ""),
            )
        else:
            if not np.all(np.isfinite(values)):
                check_shape_id(shape_id, "shape_dist_traveled contains non-finite values")
            if np.any(values < 0):
                check_shape_id(shape_id, "shape_dist_traveled contains negative values")
            if n >= 2 and np.any(np.diff(values) <= 0):
                bad = np.flatnonzero(np.diff(values) <= 0)[0]
                check_shape_id(
                    shape_id,
                    "shape_dist_traveled is not strictly increasing at "
                    f"shape_pt_sequence {sequence[bad]} -> {sequence[bad + 1]}",
                )
            dist = values.astype(float)

        result[shape_id] = {
            "shape_id": shape_id,
            "sequence": sequence,
            "lon": lon,
            "lat": lat,
            "dist": dist,
        }

    return result


# ---------------------------------------------------------------------------
# Shape geometry used for stop matching and locating
# ---------------------------------------------------------------------------


class ShapeGeometry:
    """
    A polyline with a calibrated distance ruler.

    The vertex arrays ``lon``/``lat`` and ``dist`` are in feed distance units,
    while ``x``/``y`` are the same vertices in a local UTM CRS, used to
    measure offsets in meters.

    The object caches, per stop ID, the set of distance intervals along the
    polyline that lie within the matching radius.  That cache depends only on
    the shape geometry and the stop location, never on a particular trip, so
    it is safe (and valuable) to share it among all trips that use the shape.
    """

    def __init__(
        self,
        shape_id,
        lon: np.ndarray,
        lat: np.ndarray,
        dist: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        crs=None,
    ):
        self.shape_id = shape_id
        self.lon = np.asarray(lon, dtype=float)
        self.lat = np.asarray(lat, dtype=float)
        self.dist = np.asarray(dist, dtype=float)
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.utm_crs = crs

        self._segment_tree = None
        self._segment_geoms = None
        self._interval_cache = {}

    @property
    def n_vertices(self) -> int:
        return len(self.dist)

    @property
    def d_min(self) -> float:
        return float(self.dist[0])

    @property
    def d_max(self) -> float:
        return float(self.dist[-1])

    @property
    def n_segments(self) -> int:
        return max(0, len(self.dist) - 1)

    # -- spatial index ------------------------------------------------------

    def _build_segment_index(self):
        """
        Build a Shapely STRtree over the shape's UTM segments.

        Very short shapes are scanned linearly, which is faster than building
        an index.  For long shapes the index keeps the number of candidate
        segments per stop small: ``O(log n + k)`` instead of ``O(n)``.
        """
        if self._segment_tree is not None or self.n_segments <= 32:
            return
        import shapely

        geoms = [
            shapely.geometry.LineString(
                [(self.x[i], self.y[i]), (self.x[i + 1], self.y[i + 1])]
            )
            for i in range(self.n_segments)
        ]
        self._segment_geoms = geoms
        self._segment_tree = shapely.STRtree(geoms)

    def _candidate_segments(self, sx: float, sy: float, radius_m: float):
        """
        Return an iterable of segment indices that may be within
        ``radius_m`` of the UTM point ``(sx, sy)``.
        """
        self._build_segment_index()
        if self._segment_tree is None:
            return range(self.n_segments)
        import shapely.geometry as sg

        return self._segment_tree.query(sg.Point(sx, sy).buffer(radius_m))

    # -- distance <-> location ---------------------------------------------

    def _segment_index(self, d: float) -> int:
        """
        Return the index ``j`` of the segment containing distance ``d``.
        Clamp to the first/last segment for values at the ends.
        """
        if self.n_segments == 0:
            return 0
        j = int(np.searchsorted(self.dist, d, side="right") - 1)
        return min(max(j, 0), self.n_segments - 1)

    def locate_distance(self, d: float):
        """
        Map distance ``d`` to a point on the polyline.

        Return ``(lon, lat, x, y, j, t)`` where ``j`` is the segment index and
        ``t`` the fractional position within that segment.  Coordinates are
        linearly interpolated in longitude/latitude (and in UTM for offsets)
        using the calibrated fraction of the segment's distance span.
        """
        j = self._segment_index(d)
        if self.n_segments == 0:
            return (
                float(self.lon[0]),
                float(self.lat[0]),
                float(self.x[0]),
                float(self.y[0]),
                0,
                0.0,
            )
        d0, d1 = self.dist[j], self.dist[j + 1]
        t = 0.0 if d1 == d0 else (d - d0) / (d1 - d0)
        t = float(min(max(t, 0.0), 1.0))
        lon = self.lon[j] + t * (self.lon[j + 1] - self.lon[j])
        lat = self.lat[j] + t * (self.lat[j + 1] - self.lat[j])
        x = self.x[j] + t * (self.x[j + 1] - self.x[j])
        y = self.y[j] + t * (self.y[j + 1] - self.y[j])
        return float(lon), float(lat), float(x), float(y), j, t

    # -- stop matching ------------------------------------------------------

    def _segment_interval(self, j: int, sx: float, sy: float, radius_m: float):
        """
        Return ``(lo, hi, preferred, offset)`` for segment ``j`` if some part
        of it is within ``radius_m`` of ``(sx, sy)``; otherwise return None.

        ``lo`` and ``hi`` are distances along the shape, ``preferred`` is the
        distance of the perpendicular (or endpoint) projection of the stop,
        and ``offset`` is the distance in meters between the stop and that
        preferred point.
        """
        ax, ay = self.x[j], self.y[j]
        bx, by = self.x[j + 1], self.y[j + 1]
        vx, vy = bx - ax, by - ay
        a = vx * vx + vy * vy
        if a <= 0:
            return None

        wx, wy = sx - ax, sy - ay
        b = -2.0 * (wx * vx + wy * vy)
        c = wx * wx + wy * wy - radius_m * radius_m
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return None

        root = np.sqrt(disc)
        t_lo = max((-b - root) / (2.0 * a), 0.0)
        t_hi = min((-b + root) / (2.0 * a), 1.0)
        if t_lo > t_hi:
            return None

        t_pref = min(max((wx * vx + wy * vy) / a, 0.0), 1.0)

        d0, d1 = self.dist[j], self.dist[j + 1]
        span = d1 - d0
        lo = d0 + t_lo * span
        hi = d0 + t_hi * span
        preferred = d0 + t_pref * span
        px = ax + t_pref * vx
        py = ay + t_pref * vy
        offset = float(np.hypot(sx - px, sy - py))
        return float(lo), float(hi), float(preferred), offset

    def intervals_for_stop(
        self, stop_id, sx: float, sy: float, radius_m: float
    ) -> list[tuple[float, float, float, float]]:
        """
        Return the list of ``(lo, hi, preferred, offset)`` distance intervals
        along the shape that lie within ``radius_m`` of the given stop.

        The list is sorted by distance and merged where intervals overlap or
        touch.  It is cached per ``(stop_id, radius_m)``.  It contains only
        geometry-dependent information, so sharing it between trips does not
        leak one trip's chosen position into another.
        """
        key = (stop_id, radius_m)
        if key in self._interval_cache:
            return self._interval_cache[key]

        intervals = []
        if self.n_segments == 0:
            # Degenerate one-vertex shape: a point.
            offset = float(np.hypot(sx - self.x[0], sy - self.y[0]))
            if offset <= radius_m:
                d = float(self.dist[0])
                intervals.append((d, d, d, offset))
        else:
            for j in self._candidate_segments(sx, sy, radius_m):
                interval = self._segment_interval(int(j), sx, sy, radius_m)
                if interval is not None:
                    intervals.append(interval)
            intervals.sort(key=lambda z: (z[0], z[1]))
        intervals = _merge_intervals(intervals)
        self._interval_cache[key] = intervals
        return intervals

    def nearest_offset(self, stop_id, sx: float, sy: float, radius_m: float) -> float:
        """Return the smallest offset in meters from the stop to the shape."""
        intervals = self.intervals_for_stop(stop_id, sx, sy, radius_m)
        if not intervals:
            return np.inf
        return min(z[3] for z in intervals)

    def anchor_is_valid(
        self,
        stop_id,
        sx: float,
        sy: float,
        radius_m: float,
        d: float,
    ) -> bool:
        """
        Return whether the anchored distance ``d`` lies on the shape and its
        point is within ``radius_m`` of the stop.
        """
        tol = dist_tolerance(self.d_min, self.d_max)
        if d < self.d_min - tol or d > self.d_max + tol:
            return False
        _, _, x, y, _, _ = self.locate_distance(d)
        return bool(np.hypot(sx - x, sy - y) <= radius_m + 1e-9)


def build_shape_geometries(records: dict, shape_ids=None, crs=None) -> dict:
    """
    Given the output of :func:`prepare_shape_distances`, return a dictionary
    ``shape_id -> ShapeGeometry``.

    If ``shape_ids`` is given, only build those shapes.  All vertices are
    projected to a single UTM CRS to keep offsets in meters comparable.
    """
    if not records:
        return {}

    if shape_ids is None:
        shape_ids = list(records)
    else:
        shape_ids = [s for s in shape_ids if s in records]

    if not shape_ids:
        return {}

    lon_parts = [records[s]["lon"] for s in shape_ids]
    lat_parts = [records[s]["lat"] for s in shape_ids]
    lengths = [len(x) for x in lon_parts]
    all_lon = np.concatenate(lon_parts)
    all_lat = np.concatenate(lat_parts)
    all_x, all_y, crs = project_to_utm(all_lon, all_lat, crs=crs)

    result = {}
    start = 0
    for shape_id, length in zip(shape_ids, lengths):
        stop = start + length
        rec = records[shape_id]
        result[shape_id] = ShapeGeometry(
            shape_id,
            rec["lon"],
            rec["lat"],
            rec["dist"],
            all_x[start:stop],
            all_y[start:stop],
            crs=crs,
        )
        start = stop

    return result


def _merge_intervals(intervals):
    """
    Sort and merge ``(lo, hi, preferred, offset)`` intervals that overlap or
    touch, keeping the smallest offset and the preferred value of the tightest
    piece.
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda z: (z[0], z[1]))
    merged = [list(intervals[0])]
    for lo, hi, preferred, offset in intervals[1:]:
        last = merged[-1]
        if lo <= last[1] + 1e-12:
            if hi > last[1]:
                last[1] = hi
            if offset < last[3]:
                last[2] = preferred
                last[3] = offset
        else:
            merged.append([lo, hi, preferred, offset])
    return [tuple(z) for z in merged]


# ---------------------------------------------------------------------------
# Trip matching
# ---------------------------------------------------------------------------


def match_trip_distances(
    geom: ShapeGeometry,
    stops_xy: list[tuple[float, float]],
    stop_ids: list,
    radius_m: float,
    anchors: list[float | None],
    trip_id=None,
) -> list[float] | None:
    """
    Match the stops of a single trip to ``geom`` and return one distance per
    stop, or ``None`` if no consistent matching exists.

    Parameters
    ----------
    geom : ShapeGeometry
    stops_xy : list of (x, y)
        Stop coordinates in the same UTM CRS as ``geom``.
    stop_ids : list
        Stop IDs, used only to key the geometry cache.
    radius_m : float
        Maximum allowed offset in meters from the polyline.
    anchors : list of float or None
        Distances already present for the trip's stops in feed units.  They
        are kept verbatim, but must lie on the shape and within ``radius_m``
        of their stop.

    Notes
    -----
    Stops are visited in the given order (i.e. ``stop_sequence`` order) and
    the chosen distances must be non-decreasing.  Each stop's allowed
    distances are the union of the intervals within ``radius_m`` of it, so a
    later pass of a loop or fold-back shape is available.

    Select a deterministic matching that never depends on other trips.  The
    forward pass finds, at each stop, the least distance reachable given all
    earlier stops; the backward pass finds the greatest distance from which
    the trip can still finish.  Together they form a corridor through the
    per-stop intervals.  Within that corridor the perpendicular projection of
    the stop is used, clamped into the corridor so distances stay
    non-decreasing.  This is linear in the number of stops once the per-stop
    intervals are known, and a trip is reported as unmatchable exactly when
    the corridor is empty.
    """
    n = len(stop_ids)
    if n == 0:
        return []

    tol = dist_tolerance(geom.d_min, geom.d_max)

    # Compute the allowed interval lists, restricted by anchors.
    choices = []
    last_anchor = None
    last_anchor_id = None
    for i in range(n):
        stop_id = stop_ids[i]
        sx, sy = stops_xy[i]
        anchor = anchors[i]
        if anchor is not None and not pd.isna(anchor):
            anchor = float(anchor)
            if not geom.anchor_is_valid(stop_id, sx, sy, radius_m, anchor):
                check_trip_id(
                    trip_id,
                    f"existing shape_dist_traveled {anchor!r} for stop "
                    f"{stop_id!r} is off the shape or outside the "
                    f"{radius_m} m matching radius",
                )
            if last_anchor is not None and anchor < last_anchor - tol:
                check_trip_id(
                    trip_id,
                    "existing shape_dist_traveled decreases with "
                    "stop_sequence: "
                    f"{last_anchor!r} at stop {last_anchor_id!r} then "
                    f"{anchor!r} at stop {stop_id!r}",
                )
            last_anchor = anchor
            last_anchor_id = stop_id
            choices.append([(anchor, anchor, anchor, 0.0)])
            continue

        intervals = geom.intervals_for_stop(stop_id, sx, sy, radius_m)
        if not intervals:
            return None
        choices.append(intervals)

    # Forward pass: the least distance reachable at each stop.
    prefix_min = [0.0] * n
    current_min = -np.inf
    for i in range(n):
        found = None
        for lo, hi, _, _ in choices[i]:
            if hi + tol >= current_min:
                found = max(lo, current_min)
                break
        if found is None:
            return None
        prefix_min[i] = found
        current_min = found

    # Backward pass: the greatest distance from which the trip can finish.
    suffix_max = [0.0] * n
    current_max = np.inf
    for i in range(n - 1, -1, -1):
        found = None
        for lo, hi, _, _ in reversed(choices[i]):
            if lo - tol <= current_max:
                found = min(hi, current_max)
                break
        if found is None:
            return None
        suffix_max[i] = found
        current_max = found

    # Greedy selection of the pointwise-least chain.
    result = []
    prev = -np.inf
    for i in range(n):
        low = max(prefix_min[i], prev)
        high = suffix_max[i]
        found = None
        for lo, hi, preferred, _ in choices[i]:
            if hi + tol < low:
                continue
            if lo - tol > high:
                break
            lo2 = max(lo, low)
            hi2 = min(hi, high)
            if lo2 > hi2 + tol:
                continue
            value = min(max(preferred, lo2), hi2)
            found = max(value, lo2)
            break
        if found is None:
            return None
        found = float(found)
        result.append(found)
        prev = found

    return result

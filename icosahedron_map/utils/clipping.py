"""
Spherical polygon clipping for icosahedron faces.

Clips country polygons to the Voronoi region of each face.
"""

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, box
from shapely.ops import unary_union
from typing import Dict, List, Optional
import geopandas as gpd

from ..projection.face_assignment import FaceAssignment
from ..projection.gnomonic import FaceProjection


class SphericalClipper:
    """
    Clips geographic polygons to icosahedron face boundaries.
    """

    def __init__(self, face_assignment: FaceAssignment,
                 face_projections: List[FaceProjection]):
        """
        Initialize clipper.

        Args:
            face_assignment: FaceAssignment instance
            face_projections: List of FaceProjection for all 20 faces
        """
        self.face_assignment = face_assignment
        self.face_projections = face_projections
        self.icosahedron = face_assignment.icosahedron

        # Build clip polygons for each face
        self.clip_polygons = self._build_clip_polygons()

    def _build_clip_polygons(self) -> Dict[int, Polygon]:
        """
        Build Shapely polygons for clipping each face.

        Returns:
            Dict mapping face_idx to clip Polygon
        """
        clip_polygons = {}

        for face_idx in range(20):
            # Get densified boundary
            boundary = self.face_assignment.get_face_boundary_polygon(face_idx, n_points=30)

            if not boundary:
                # Fallback: create approximate boundary
                boundary = self._create_approximate_boundary(face_idx)

            if len(boundary) < 3:
                clip_polygons[face_idx] = Polygon()
                continue

            # Convert to (lon, lat) for Shapely, handling antimeridian and poles
            center_lon = self.face_projections[face_idx].center_lon

            coords = []
            for lat, lon in boundary:
                # At poles, longitude is undefined - use face center's longitude
                if abs(lat) > 89.9:
                    lon = center_lon
                else:
                    # Normalize longitude relative to face center
                    # This keeps coordinates continuous even across the antimeridian
                    while lon - center_lon > 180:
                        lon -= 360
                    while lon - center_lon < -180:
                        lon += 360
                coords.append((lon, lat))

            try:
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                clip_polygons[face_idx] = poly
            except Exception:
                clip_polygons[face_idx] = Polygon()

        return clip_polygons

    def _create_approximate_boundary(self, face_idx: int) -> List[tuple]:
        """
        Create an approximate face boundary when Voronoi fails.

        Uses a circle around the face center.
        """
        center_lat, center_lon = self.face_projections[face_idx].center_lat, \
                                  self.face_projections[face_idx].center_lon

        # Approximate angular radius of face (icosahedron face spans ~37 degrees)
        radius = 37  # degrees

        points = []
        for angle in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            # Simple approximation
            lat = center_lat + radius * np.cos(angle) * 0.5
            lon = center_lon + radius * np.sin(angle) / np.cos(np.radians(center_lat))
            lat = np.clip(lat, -89, 89)
            points.append((lat, lon))

        return points

    def _shift_geometry_lon(self, geometry, center_lon: float):
        """
        Shift geometry longitudes to be centered around center_lon.

        Uses standard ±180° normalization relative to center.
        """
        from shapely.ops import transform

        def shift_coords(x, y):
            x = np.array(x, dtype=float)
            # Standard normalization relative to center
            x = np.where(x - center_lon > 180, x - 360, x)
            x = np.where(x - center_lon < -180, x + 360, x)
            return x, y

        return transform(shift_coords, geometry)

    def _split_polar_geometry_at_meridians(self, geometry):
        """
        Split polar-spanning geometries into narrow slices before rotation.

        This is ONLY needed for longitude (Z-axis) rotation, not for pole-on-face
        rotation. Pole-on-face rotation tilts the entire coordinate system in 3D,
        so the "critical longitude" concept doesn't apply.

        For longitude rotation: geometries that span near the poles and also span
        wide longitude ranges can become self-intersecting after rotation. We split
        at meridians that account for the rotation angle to ensure no slice crosses
        the post-rotation antimeridian.
        """
        # Only apply this splitting for longitude rotation, not pole-on-face
        rotation = self.icosahedron.longitude_rotation
        if rotation == 0.0:
            # No longitude rotation - skip splitting
            # (pole_on_face alone doesn't need this splitting)
            return geometry

        minx, miny, maxx, maxy = geometry.bounds

        # Only process if geometry reaches near a pole AND spans wide longitudes
        is_polar = miny < -80 or maxy > 80
        is_wide = maxx - minx > 90  # Spans more than 1/4 of the globe

        if not is_polar or not is_wide:
            return geometry

        # Calculate the critical longitude that becomes ±180 after rotation
        # For rotation by angle θ, original lon = -180 - θ becomes rotated lon = -180
        critical_lon = -180 - rotation  # This longitude maps to -180 after rotation
        while critical_lon > 180:
            critical_lon -= 360
        while critical_lon < -180:
            critical_lon += 360

        # Use narrow 30° slices for better control
        slice_width = 30
        num_slices = 12

        # Create slices that AVOID having the critical longitude on a boundary
        # Points exactly at critical_lon map to ±180 which causes coordinate discontinuity
        # By offsetting the slice boundaries slightly, we keep critical_lon inside a slice
        epsilon = 0.5  # Half-degree offset
        slice_starts = []

        for i in range(num_slices):
            lon = critical_lon + epsilon + i * slice_width
            while lon > 180:
                lon -= 360
            while lon < -180:
                lon += 360
            slice_starts.append(lon)
        slice_starts.sort()

        # Create slices
        slices = []
        for i in range(len(slice_starts)):
            lon1 = slice_starts[i]
            lon2 = slice_starts[(i + 1) % len(slice_starts)]

            if lon2 < lon1:
                # Slice crosses -180/180 boundary
                # Split into two pieces
                slices.append(box(lon1, -90, 180, 90))
                slices.append(box(-180, -90, lon2, 90))
            else:
                slices.append(box(lon1, -90, lon2, 90))

        parts = []
        try:
            for slc in slices:
                part = geometry.intersection(slc)
                if part.is_empty:
                    continue
                if isinstance(part, (Polygon, MultiPolygon)):
                    if isinstance(part, MultiPolygon):
                        parts.extend(part.geoms)
                    else:
                        parts.append(part)
                elif hasattr(part, 'geoms'):
                    for g in part.geoms:
                        if isinstance(g, Polygon):
                            parts.append(g)

            if len(parts) == 0:
                return geometry
            elif len(parts) == 1:
                return parts[0]
            else:
                return MultiPolygon(parts)
        except Exception:
            return geometry

    def _split_invalid_at_antimeridian(self, geometry):
        """
        Split invalid geometries at the antimeridian.

        If a geometry is invalid due to coordinates wrapping around the
        antimeridian, split it into eastern and western parts.
        """
        minx, miny, maxx, maxy = geometry.bounds
        width = maxx - minx

        # For valid geometries, only split if very wide
        if geometry.is_valid and width < 300:
            return geometry

        # For invalid geometries, try buffer(0) fix first
        if not geometry.is_valid:
            try:
                fixed = geometry.buffer(0)
                if fixed.is_valid and not fixed.is_empty:
                    # Check if buffer(0) destroyed too much area
                    if isinstance(fixed, (Polygon, MultiPolygon)):
                        new_minx, new_miny, new_maxx, new_maxy = fixed.bounds
                        new_width = new_maxx - new_minx
                        # If it's now narrower and valid, use it
                        if new_width < width:
                            return fixed
            except Exception:
                pass

        # Only try splitting if geometry spans nearly the whole globe
        if width < 300:
            if not geometry.is_valid:
                # Last resort: try buffer(0)
                try:
                    return geometry.buffer(0)
                except Exception:
                    pass
            return geometry

        # First try: shift all coordinates to one side of antimeridian
        fixed = self._try_shift_to_hemisphere(geometry)
        if fixed is not None and fixed.is_valid:
            return fixed

        # Second try: fix with buffer(0) then split
        work_geom = geometry
        if not geometry.is_valid:
            try:
                work_geom = geometry.buffer(0)
                if work_geom.is_empty:
                    return geometry
            except Exception:
                return geometry

        # Split at the antimeridian
        west_box = box(-180, -90, 0, 90)
        east_box = box(0, -90, 180, 90)

        try:
            west_part = work_geom.intersection(west_box)
            east_part = work_geom.intersection(east_box)

            parts = []
            for part in [west_part, east_part]:
                if part.is_empty:
                    continue
                if isinstance(part, Polygon):
                    parts.append(part)
                elif isinstance(part, MultiPolygon):
                    parts.extend(part.geoms)
                elif hasattr(part, 'geoms'):
                    for g in part.geoms:
                        if isinstance(g, Polygon):
                            parts.append(g)

            if len(parts) == 0:
                return work_geom if work_geom.is_valid else geometry
            elif len(parts) == 1:
                return parts[0]
            else:
                return MultiPolygon(parts)
        except Exception:
            return work_geom if work_geom.is_valid else geometry

    def _try_shift_to_hemisphere(self, geometry):
        """
        Try to fix a geometry that falsely spans the globe by shifting coords.

        When coordinates cluster near ±180, some might be at +179.9 and others
        at -179.9 due to rotation precision. This makes the bounding box span
        360° even though the actual geometry is small. We fix by shifting all
        coordinates to one side.
        """
        from shapely.ops import transform

        coords = list(geometry.exterior.coords)
        lons = [c[0] for c in coords]

        # Count how many coords are near each side of antimeridian
        near_pos_180 = sum(1 for lon in lons if lon > 170)
        near_neg_180 = sum(1 for lon in lons if lon < -170)

        if near_pos_180 == 0 and near_neg_180 == 0:
            return None  # Not an antimeridian issue

        # Check if most coords are in one hemisphere
        in_east = sum(1 for lon in lons if lon > 0)
        in_west = sum(1 for lon in lons if lon < 0)

        if in_east > in_west:
            # Shift western outliers to eastern hemisphere
            def shift_east(x, y):
                x = np.array(x, dtype=float)
                x = np.where(x < -90, x + 360, x)
                return x, y
            shifted = transform(shift_east, geometry)
        else:
            # Shift eastern outliers to western hemisphere
            def shift_west(x, y):
                x = np.array(x, dtype=float)
                x = np.where(x > 90, x - 360, x)
                return x, y
            shifted = transform(shift_west, geometry)

        return shifted

    def _split_at_antimeridian_pre_rotation(self, geometry):
        """
        Split geometries that cross the antimeridian before any rotation.

        Wide-spanning geometries that cross ±180 can become invalid after 3D
        rotation because coordinates wrap around unexpectedly. Split them at
        multiple meridians to create narrower pieces.
        """
        minx, miny, maxx, maxy = geometry.bounds
        width = maxx - minx

        # Only process wide geometries
        if width < 270:
            return geometry

        # Split at the antimeridian only (0° and ±180° boundaries)
        # More aggressive splitting can cause more issues with 3D rotation
        slices = [
            box(-180, -90, 0, 90),
            box(0, -90, 180, 90),
        ]

        try:
            parts = []
            for slc in slices:
                part = geometry.intersection(slc)
                if part.is_empty:
                    continue
                if isinstance(part, Polygon):
                    parts.append(part)
                elif isinstance(part, MultiPolygon):
                    parts.extend(part.geoms)
                elif hasattr(part, 'geoms'):
                    for g in part.geoms:
                        if isinstance(g, Polygon):
                            parts.append(g)

            if len(parts) == 0:
                return geometry
            elif len(parts) == 1:
                return parts[0]
            else:
                return MultiPolygon(parts)
        except Exception:
            return geometry

    def _fix_post_rotation_spans(self, geometry):
        """
        Fix geometries that span too much longitude after rotation.

        After 3D rotation, coordinates near the rotated pole can end up on
        opposite sides of the antimeridian, creating bounding boxes that span
        most of the globe. Fix by shifting all coordinates to the same hemisphere.
        """
        from shapely.ops import transform

        if isinstance(geometry, MultiPolygon):
            fixed_parts = []
            for part in geometry.geoms:
                fixed = self._fix_single_polygon_span(part)
                if fixed is not None:
                    if isinstance(fixed, MultiPolygon):
                        fixed_parts.extend(fixed.geoms)
                    elif isinstance(fixed, Polygon):
                        fixed_parts.append(fixed)
            if fixed_parts:
                return MultiPolygon(fixed_parts)
            return geometry
        elif isinstance(geometry, Polygon):
            fixed = self._fix_single_polygon_span(geometry)
            return fixed if fixed is not None else geometry
        return geometry

    def _fix_single_polygon_span(self, polygon):
        """Fix a single polygon that may span too much longitude."""
        from shapely.ops import transform

        if polygon.is_empty:
            return polygon

        # First, normalize all coordinates to -180 to 180 range
        def normalize_lon(x, y):
            x = np.array(x, dtype=float)
            x = ((x + 180) % 360) - 180
            return x, y

        try:
            polygon = transform(normalize_lon, polygon)
        except Exception:
            pass

        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx

        if width < 180:
            return polygon

        # Check if this is a "false span" - coordinates clustered near antimeridian
        coords = list(polygon.exterior.coords)
        lons = [c[0] for c in coords]

        # Count coordinates in each region
        near_pos_180 = sum(1 for lon in lons if lon > 120)
        near_neg_180 = sum(1 for lon in lons if lon < -120)

        if near_pos_180 > 0 and near_neg_180 > 0:
            # Coordinates clustered near both +180 and -180
            # Shift all to one side
            if near_pos_180 > near_neg_180:
                # Shift negative coords to positive (add 360 to negative coords)
                def shift_pos(x, y):
                    x = np.array(x, dtype=float)
                    x = np.where(x < 0, x + 360, x)
                    return x, y
                shift_func = shift_pos
            else:
                # Shift positive coords to negative (subtract 360 from positive coords)
                def shift_neg(x, y):
                    x = np.array(x, dtype=float)
                    x = np.where(x > 0, x - 360, x)
                    return x, y
                shift_func = shift_neg

            try:
                shifted = transform(shift_func, polygon)
                # Check if shifting reduced the width
                new_width = shifted.bounds[2] - shifted.bounds[0]
                if new_width < width:
                    # Check if valid or can be made valid
                    if shifted.is_valid:
                        return shifted
                    fixed = shifted.buffer(0)
                    if fixed.is_valid and not fixed.is_empty:
                        return fixed
            except Exception:
                pass

        # If shifting didn't help and polygon is invalid, try splitting at antimeridian
        if not polygon.is_valid:
            fixed = self._split_invalid_at_antimeridian(polygon)
            return fixed

        # If valid but too wide, also try splitting
        if width > 270:
            return self._split_invalid_at_antimeridian(polygon)

        return polygon

    def _rotate_geometry(self, geometry):
        """
        Apply coordinate rotation for pole_on_face or longitude rotation mode.

        Rotates all coordinates in the geometry using the icosahedron's
        coordinate rotation matrix.
        """
        if self.icosahedron._coord_rotation is None:
            return geometry

        # Split wide-spanning geometries at antimeridian before any rotation
        # This prevents invalid geometries after 3D rotation
        geometry = self._split_at_antimeridian_pre_rotation(geometry)

        # For longitude rotation only: split polar geometries into slices
        # to prevent self-intersection at the post-rotation antimeridian
        geometry = self._split_polar_geometry_at_meridians(geometry)

        from shapely.ops import transform

        def rotate_coords(x, y):
            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)

            # x is longitude, y is latitude
            rotated_lats, rotated_lons = self.icosahedron.rotate_latlon_arrays(y, x)

            # Fix pole points: at lat = ±90, longitude is undefined
            # After rotation, arctan2 assigns arbitrary lons to pole points
            # We fix this by inheriting longitude from nearest non-pole neighbor
            pole_mask = np.abs(rotated_lats) > 89.9

            if np.any(pole_mask):
                n = len(rotated_lons)
                for i in np.where(pole_mask)[0]:
                    # Find nearest non-pole neighbor
                    for offset in range(1, n):
                        prev_idx = (i - offset) % n
                        next_idx = (i + offset) % n
                        if not pole_mask[prev_idx]:
                            rotated_lons[i] = rotated_lons[prev_idx]
                            break
                        if not pole_mask[next_idx]:
                            rotated_lons[i] = rotated_lons[next_idx]
                            break

            return rotated_lons, rotated_lats

        rotated = transform(rotate_coords, geometry)

        # Post-process: fix geometries that span too much longitude or are invalid
        rotated = self._fix_post_rotation_spans(rotated)

        # Handle any remaining invalid geometries
        if isinstance(rotated, MultiPolygon):
            fixed_parts = []
            for part in rotated.geoms:
                if part.is_valid:
                    fixed_parts.append(part)
                else:
                    fixed = self._split_invalid_at_antimeridian(part)
                    if isinstance(fixed, MultiPolygon):
                        fixed_parts.extend(fixed.geoms)
                    elif isinstance(fixed, Polygon):
                        fixed_parts.append(fixed)
            if fixed_parts:
                return MultiPolygon(fixed_parts)
            return rotated
        elif isinstance(rotated, Polygon) and not rotated.is_valid:
            return self._split_invalid_at_antimeridian(rotated)

        return rotated

    def clip_geometry_to_face(self, geometry, face_idx: int):
        """
        Clip a geometry to a specific face.

        Args:
            geometry: Shapely geometry (Polygon or MultiPolygon)
            face_idx: Face index (0-19)

        Returns:
            Clipped geometry or None if no intersection
        """
        clip_poly = self.clip_polygons[face_idx]

        if clip_poly.is_empty:
            return None

        # Apply coordinate rotation for pole_on_face mode
        rotated_geometry = self._rotate_geometry(geometry)

        # Shift geometry to match clip polygon's coordinate system
        center_lon = self.face_projections[face_idx].center_lon
        shifted_geometry = self._shift_geometry_lon(rotated_geometry, center_lon)

        # Check if we're in rotated mode (pole_on_face)
        is_rotated = self.icosahedron._coord_rotation is not None

        # Check if shifted geometry was incorrectly stretched by straddling shift boundary
        # This happens when a small polygon straddles center_lon ± 180 and gets
        # some points shifted while others don't, creating a huge invalid polygon
        orig_minx, _, orig_maxx, _ = rotated_geometry.bounds
        orig_width = orig_maxx - orig_minx
        shift_minx, _, shift_maxx, _ = shifted_geometry.bounds
        shift_width = shift_maxx - shift_minx

        # If width increased dramatically (more than 180° increase), it's invalid
        if shift_width - orig_width > 180:
            return None

        try:
            clipped = shifted_geometry.intersection(clip_poly)

            if clipped.is_empty:
                return None

            # Filter out non-polygonal results
            if isinstance(clipped, (Polygon, MultiPolygon)):
                return self._fill_polar_gaps(clipped, clip_poly)
            elif isinstance(clipped, GeometryCollection):
                polys = [g for g in clipped.geoms
                        if isinstance(g, (Polygon, MultiPolygon))]
                if polys:
                    return self._fill_polar_gaps(unary_union(polys), clip_poly)
                return None
            return None

        except Exception:
            # The shifted geometry may be invalid (self-intersecting) for
            # polygons that span wide longitude ranges.
            if is_rotated:
                # For rotated mode, use simple buffer fix on rotated geometry
                result = self._clip_simple_fallback(rotated_geometry, clip_poly, center_lon)
            else:
                # For normal mode, use split-at-boundary fallback
                result = self._clip_geometry_fallback(rotated_geometry, clip_poly, center_lon)
            return self._fill_polar_gaps(result, clip_poly) if result else None

    def _clip_simple_fallback(self, geometry, clip_poly, center_lon: float):
        """
        Simple fallback using buffer(0) to fix invalid geometries.

        Used for rotated coordinate mode where split-at-boundary doesn't apply.
        For wide-spanning polygons (>180°), uses split-at-boundary approach
        WITHOUT buffer(0) since that destroys polar regions.
        """
        try:
            # Check width on ORIGINAL geometry before any fix attempt
            # buffer(0) can destroy polar regions on wide invalid geometries
            minx, _, maxx, _ = geometry.bounds
            width = maxx - minx

            if width > 180:
                # For wide polygons, use split-at-boundary approach directly
                # Don't use buffer(0) as it destroys the polar region
                return self._clip_geometry_fallback(geometry, clip_poly, center_lon)

            # For narrow polygons, use buffer(0) fix + simple shift
            fixed_geom = geometry.buffer(0)
            if not fixed_geom.is_valid:
                return None

            shifted_geom = self._shift_geometry_lon(fixed_geom, center_lon)
            fixed_clip = clip_poly.buffer(0)
            result = shifted_geom.intersection(fixed_clip)
            if result.is_empty:
                return None
            if isinstance(result, (Polygon, MultiPolygon)):
                return result
            elif isinstance(result, GeometryCollection):
                polys = [g for g in result.geoms
                        if isinstance(g, (Polygon, MultiPolygon))]
                if polys:
                    return unary_union(polys)
            return None
        except Exception:
            return None

    def _fill_polar_gaps(self, geometry, clip_poly, pole_lat: float = -90.0,
                          threshold: float = 5.0):
        """
        Fill triangular gaps at the poles by extending geometry to clip polygon boundary.

        Currently disabled - just returns geometry unchanged.
        """
        return geometry

    def _split_at_shift_boundary(self, geometry, center_lon: float):
        """
        Split geometry at the longitude where shifting creates a discontinuity.

        The discontinuity occurs at center_lon + 180° (or center_lon - 180°).
        Splitting here ensures each piece shifts continuously.
        """
        # The shift boundary is where lon - center_lon = 180 (or -180)
        # i.e., at lon = center_lon + 180
        boundary_lon = center_lon + 180
        if boundary_lon > 180:
            boundary_lon -= 360
        elif boundary_lon < -180:
            boundary_lon += 360

        # Check if geometry crosses this boundary
        minx, miny, maxx, maxy = geometry.bounds

        # Normalize bounds relative to boundary_lon
        crosses_boundary = False
        if boundary_lon > 0:
            # Boundary is in eastern hemisphere
            # Geometry crosses if it spans across boundary_lon
            if minx < boundary_lon < maxx:
                crosses_boundary = True
        else:
            # Boundary is in western hemisphere
            if minx < boundary_lon < maxx:
                crosses_boundary = True

        # Also check if geometry wraps around (spans most of globe)
        if maxx - minx > 180:
            crosses_boundary = True

        if not crosses_boundary:
            return geometry

        # Split into two parts: east and west of boundary
        # Use boxes that cover each hemisphere relative to boundary
        # Add tiny overlap (epsilon) to prevent gaps at the boundary
        epsilon = 1e-6
        west_box = box(boundary_lon - 360, -90, boundary_lon + epsilon, 90)
        east_box = box(boundary_lon - epsilon, -90, boundary_lon + 360, 90)

        try:
            west_part = geometry.intersection(west_box)
            east_part = geometry.intersection(east_box)

            parts = []
            if not west_part.is_empty:
                parts.append(west_part)
            if not east_part.is_empty:
                parts.append(east_part)

            if len(parts) == 0:
                return geometry
            elif len(parts) == 1:
                return parts[0]
            else:
                # Return as MultiPolygon or GeometryCollection
                return GeometryCollection(parts)
        except Exception:
            return geometry

    def _shift_geometry_unconditional(self, geometry, shift_amount: float):
        """
        Shift all geometry coordinates by a fixed amount.
        Used for parts that are entirely on one side of the shift boundary.
        """
        from shapely.ops import transform

        def shift_coords(x, y):
            x = np.array(x, dtype=float)
            return x + shift_amount, y

        return transform(shift_coords, geometry)

    def _clip_geometry_fallback(self, geometry, clip_poly, center_lon: float):
        """
        Fallback clipping that splits geometry at shift boundary first.
        """
        # Split the geometry at the shift boundary
        split_geom = self._split_at_shift_boundary(geometry, center_lon)

        # Process all parts
        parts_to_process = []
        if isinstance(split_geom, (MultiPolygon, GeometryCollection)):
            for part in split_geom.geoms:
                if isinstance(part, (Polygon, MultiPolygon)):
                    parts_to_process.append(part)
                elif isinstance(part, GeometryCollection):
                    for subpart in part.geoms:
                        if isinstance(subpart, (Polygon, MultiPolygon)):
                            parts_to_process.append(subpart)
        elif isinstance(split_geom, Polygon):
            parts_to_process.append(split_geom)

        results = []
        for part in parts_to_process:
            try:
                # Determine the correct shift for this part based on its position
                # relative to center_lon. The target range is [center_lon-180, center_lon+180].
                minx, _, maxx, _ = part.bounds

                # Check if part needs shifting and in which direction
                # Use the centroid to determine which side of center this part is on
                cx = (minx + maxx) / 2

                if cx - center_lon > 180:
                    # Part is too far east, shift down by 360
                    shifted_part = self._shift_geometry_unconditional(part, -360)
                elif cx - center_lon < -180:
                    # Part is too far west, shift up by 360
                    shifted_part = self._shift_geometry_unconditional(part, +360)
                else:
                    # Part is already in the correct range
                    shifted_part = part

                if shifted_part.is_valid:
                    clipped = shifted_part.intersection(clip_poly)
                    if not clipped.is_empty:
                        if isinstance(clipped, (Polygon, MultiPolygon)):
                            results.append(clipped)
                        elif isinstance(clipped, GeometryCollection):
                            for g in clipped.geoms:
                                if isinstance(g, (Polygon, MultiPolygon)):
                                    results.append(g)
            except Exception:
                continue

        if results:
            return unary_union(results)
        return None

    def clip_all_countries(self, gdf: gpd.GeoDataFrame) -> Dict[int, gpd.GeoDataFrame]:
        """
        Clip all countries to each face.

        Args:
            gdf: GeoDataFrame with country geometries

        Returns:
            Dict mapping face_idx to GeoDataFrame of clipped countries
        """
        results = {i: [] for i in range(20)}

        for idx, row in gdf.iterrows():
            geometry = row.geometry
            name = row.get('NAME', f'Country_{idx}')

            if geometry is None or geometry.is_empty:
                continue

            for face_idx in range(20):
                clipped = self.clip_geometry_to_face(geometry, face_idx)

                if clipped is not None and not clipped.is_empty:
                    # Check if the clipped area is significant
                    if clipped.area > 1e-6:  # Filter tiny slivers
                        results[face_idx].append({
                            'geometry': clipped,
                            'name': name,
                            'original_idx': idx
                        })

        # Convert to GeoDataFrames
        gdfs = {}
        for face_idx, data in results.items():
            if data:
                gdfs[face_idx] = gpd.GeoDataFrame(data, crs="EPSG:4326")
            else:
                gdfs[face_idx] = gpd.GeoDataFrame(
                    columns=['geometry', 'name', 'original_idx'],
                    crs="EPSG:4326"
                )

        return gdfs

    def get_countries_for_face(self, gdf: gpd.GeoDataFrame,
                                face_idx: int) -> gpd.GeoDataFrame:
        """
        Get countries clipped to a specific face.

        Args:
            gdf: GeoDataFrame with country geometries
            face_idx: Face index (0-19)

        Returns:
            GeoDataFrame of clipped countries for this face
        """
        results = []

        for idx, row in gdf.iterrows():
            geometry = row.geometry
            name = row.get('NAME', f'Country_{idx}')

            if geometry is None or geometry.is_empty:
                continue

            clipped = self.clip_geometry_to_face(geometry, face_idx)

            if clipped is not None and not clipped.is_empty:
                if clipped.area > 1e-6:
                    results.append({
                        'geometry': clipped,
                        'name': name,
                        'original_idx': idx
                    })

        if results:
            return gpd.GeoDataFrame(results, crs="EPSG:4326")
        return gpd.GeoDataFrame(columns=['geometry', 'name', 'original_idx'], crs="EPSG:4326")

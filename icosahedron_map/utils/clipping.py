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
            # Use > for positive boundary (exact boundary handled by fallback)
            x = np.where(x - center_lon > 180, x - 360, x)
            x = np.where(x - center_lon < -180, x + 360, x)
            return x, y

        return transform(shift_coords, geometry)

    def _rotate_geometry(self, geometry):
        """
        Apply coordinate rotation for pole_on_face mode.

        Rotates all coordinates in the geometry using the icosahedron's
        coordinate rotation matrix.
        """
        if self.icosahedron._coord_rotation is None:
            return geometry

        from shapely.ops import transform

        def rotate_coords(x, y):
            x = np.array(x)
            y = np.array(y)
            # x is longitude, y is latitude
            rotated_lats, rotated_lons = self.icosahedron.rotate_latlon_arrays(y, x)
            return rotated_lons, rotated_lats

        return transform(rotate_coords, geometry)

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
        """
        try:
            # Fix geometry with buffer(0), then shift and clip
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

"""
Spherical polygon clipping for icosahedron faces.

Clips country polygons to the Voronoi region of each face.
"""

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
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
                    # Normalize longitude relative to face center to avoid antimeridian issues
                    # Shift lon to be within ±180° of center_lon
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

        This handles antimeridian crossing by ensuring all coordinates
        are within ±180° of the center longitude.
        """
        from shapely.ops import transform

        def shift_coords(x, y):
            x = np.array(x)
            # Shift to be within ±180° of center
            x = np.where(x - center_lon > 180, x - 360, x)
            x = np.where(x - center_lon < -180, x + 360, x)
            return x, y

        return transform(shift_coords, geometry)

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

        # Shift geometry to match clip polygon's coordinate system
        center_lon = self.face_projections[face_idx].center_lon
        shifted_geometry = self._shift_geometry_lon(geometry, center_lon)

        try:
            clipped = shifted_geometry.intersection(clip_poly)

            if clipped.is_empty:
                return None

            # Filter out non-polygonal results
            if isinstance(clipped, (Polygon, MultiPolygon)):
                return clipped
            elif isinstance(clipped, GeometryCollection):
                polys = [g for g in clipped.geoms
                        if isinstance(g, (Polygon, MultiPolygon))]
                if polys:
                    return unary_union(polys)
                return None
            return None

        except Exception:
            # Try with buffer(0) fix
            try:
                fixed_geom = geometry.buffer(0)
                fixed_clip = clip_poly.buffer(0)
                return fixed_geom.intersection(fixed_clip)
            except Exception:
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

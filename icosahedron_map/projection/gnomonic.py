"""
Gnomonic projection for each icosahedron face using pyproj.

The gnomonic projection projects points from a sphere onto a tangent plane,
with the projection center at the center of each face.
"""

import numpy as np
from pyproj import CRS, Transformer
from typing import Tuple, Optional, List
from ..geometry.icosahedron import Icosahedron


class FaceProjection:
    """
    Gnomonic projection for a single icosahedron face.

    Uses pyproj with +proj=gnom centered on the face center.
    """

    def __init__(self, icosahedron: Icosahedron, face_idx: int):
        """
        Initialize projection for a specific face.

        Args:
            icosahedron: Icosahedron instance
            face_idx: Face index (0-19)
        """
        self.face_idx = face_idx
        self.icosahedron = icosahedron

        # Get face center in lat/lon
        self.center_lat, self.center_lon = icosahedron.get_face_latlon_center(face_idx)

        # Create gnomonic projection centered on this face
        proj_string = f"+proj=gnom +lat_0={self.center_lat} +lon_0={self.center_lon} +x_0=0 +y_0=0 +ellps=WGS84"
        self.crs_gnom = CRS.from_proj4(proj_string)
        self.crs_wgs84 = CRS.from_epsg(4326)

        # Transformers
        self.to_gnom = Transformer.from_crs(self.crs_wgs84, self.crs_gnom, always_xy=True)
        self.from_gnom = Transformer.from_crs(self.crs_gnom, self.crs_wgs84, always_xy=True)

        # Compute normalization scale based on face vertices
        self._compute_scale()

    def _compute_scale(self):
        """
        Compute scale factor to normalize projected coordinates.

        We scale so that the projected triangle vertices have a consistent size.
        """
        # Project the face vertices
        vertices_latlon = self.icosahedron.get_face_vertices_latlon(self.face_idx)
        projected = []
        for lat, lon in vertices_latlon:
            x, y = self.to_gnom.transform(lon, lat)
            projected.append((x, y))

        projected = np.array(projected)

        # Compute centroid of projected triangle
        self.proj_centroid = np.mean(projected, axis=0)

        # Compute average distance from centroid (for scaling)
        distances = np.linalg.norm(projected - self.proj_centroid, axis=1)
        self.scale = 1.0 / np.mean(distances)

        # Determine which flips are needed based on face position and pattern orientation
        # North polar faces (0-4): point UP in pattern, need y-flip
        # Equatorial faces 5,7,9,11,13: point DOWN in pattern, need x-flip
        # Equatorial faces 6,8,10,12,14: point UP in pattern, need y-flip
        # South polar faces (15-19): point DOWN in pattern, need x-flip
        if self.face_idx < 5:
            # North polar - flip y
            self.flip_x = False
            self.flip_y = True
        elif self.face_idx < 15:
            # Equatorial
            if self.face_idx in (5, 7, 9, 11, 13):
                # Points DOWN, flip x
                self.flip_x = True
                self.flip_y = False
            else:
                # Points UP (6,8,10,12,14), flip y
                self.flip_x = False
                self.flip_y = True
        else:
            # South polar - flip x (same as equatorial down-pointing)
            self.flip_x = True
            self.flip_y = False

        # Store projected vertices for reference (with flips applied)
        self.proj_vertices = (projected - self.proj_centroid) * self.scale
        if self.flip_x:
            self.proj_vertices[:, 0] *= -1
        if self.flip_y:
            self.proj_vertices[:, 1] *= -1

    def project(self, lat: float, lon: float, already_rotated: bool = False) -> Tuple[float, float]:
        """
        Project a point from lat/lon to normalized face coordinates.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            already_rotated: If True, skip the pole_on_face coordinate rotation
                            (for coordinates that were already rotated, e.g. clipped geometries)

        Returns:
            (x, y) in normalized face coordinates (centered, scaled)
        """
        # Apply coordinate rotation for pole_on_face mode
        if not already_rotated:
            lat, lon = self.icosahedron.rotate_latlon(lat, lon)

        # Project using pyproj
        x, y = self.to_gnom.transform(lon, lat)

        # Normalize (center and scale)
        x = (x - self.proj_centroid[0]) * self.scale
        y = (y - self.proj_centroid[1]) * self.scale

        # Apply flips for correct orientation in the 2D pattern
        if self.flip_x:
            x = -x
        if self.flip_y:
            y = -y

        return (x, y)

    def project_array(self, lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project arrays of coordinates.

        Args:
            lats: Array of latitudes in degrees
            lons: Array of longitudes in degrees

        Returns:
            (x_array, y_array) in normalized face coordinates
        """
        # Apply coordinate rotation for pole_on_face mode
        lats, lons = self.icosahedron.rotate_latlon_arrays(lats, lons)

        x, y = self.to_gnom.transform(lons, lats)

        x = (x - self.proj_centroid[0]) * self.scale
        y = (y - self.proj_centroid[1]) * self.scale

        # Apply flips for correct orientation in the 2D pattern
        if self.flip_x:
            x = -x
        if self.flip_y:
            y = -y

        return x, y

    def inverse(self, x: float, y: float) -> Tuple[float, float]:
        """
        Inverse projection: from face coordinates to lat/lon.

        Args:
            x, y: Normalized face coordinates

        Returns:
            (lat, lon) in degrees
        """
        # Denormalize
        x_proj = x / self.scale + self.proj_centroid[0]
        y_proj = y / self.scale + self.proj_centroid[1]

        # Inverse transform
        lon, lat = self.from_gnom.transform(x_proj, y_proj)

        return (lat, lon)

    def get_projected_triangle(self) -> np.ndarray:
        """
        Get the projected face triangle vertices in normalized coordinates.

        Returns:
            3x2 array of (x, y) coordinates
        """
        return self.proj_vertices.copy()


def create_all_projections(icosahedron: Icosahedron) -> List[FaceProjection]:
    """
    Create gnomonic projections for all 20 faces.

    Args:
        icosahedron: Icosahedron instance

    Returns:
        List of FaceProjection instances, one per face
    """
    return [FaceProjection(icosahedron, i) for i in range(20)]

"""
Face assignment for geographic points.

Determines which icosahedron face each point belongs to using
spherical Voronoi partitioning based on face centers.
"""

import numpy as np
from scipy.spatial import SphericalVoronoi
from typing import List, Tuple
from ..geometry.icosahedron import Icosahedron


class FaceAssignment:
    """
    Assigns geographic points to icosahedron faces.

    Uses the nearest face center approach (equivalent to spherical Voronoi).
    """

    def __init__(self, icosahedron: Icosahedron):
        """
        Initialize face assignment.

        Args:
            icosahedron: Icosahedron instance
        """
        self.icosahedron = icosahedron
        self.centers = icosahedron.face_centers

        # Create spherical Voronoi diagram for face boundaries
        try:
            self.voronoi = SphericalVoronoi(
                self.centers,
                radius=1.0,
                center=np.array([0, 0, 0])
            )
            self.voronoi.sort_vertices_of_regions()
            self._has_voronoi = True
        except Exception:
            self._has_voronoi = False

    def assign_point(self, lat: float, lon: float) -> int:
        """
        Determine which face a point belongs to.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Face index (0-19)
        """
        # Apply coordinate rotation for pole_on_face mode
        lat, lon = self.icosahedron.rotate_latlon(lat, lon)

        point_3d = self.icosahedron.latlon_to_cartesian(lat, lon)

        # Find nearest face center (dot product comparison)
        dots = np.dot(self.centers, point_3d)
        return int(np.argmax(dots))

    def assign_points_array(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """
        Assign multiple points to faces (vectorized).

        Args:
            lats: Array of latitudes in degrees
            lons: Array of longitudes in degrees

        Returns:
            Array of face indices
        """
        # Apply coordinate rotation for pole_on_face mode
        lats, lons = self.icosahedron.rotate_latlon_arrays(lats, lons)

        # Convert to cartesian
        lat_rad = np.radians(lats)
        lon_rad = np.radians(lons)

        points_3d = np.column_stack([
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad)
        ])

        # Dot product with all centers
        dots = points_3d @ self.centers.T

        return np.argmax(dots, axis=1)

    def get_face_boundary(self, face_idx: int) -> List[Tuple[float, float]]:
        """
        Get the boundary of a face's Voronoi region in lat/lon.

        Args:
            face_idx: Face index (0-19)

        Returns:
            List of (lat, lon) tuples defining the boundary
        """
        if not self._has_voronoi:
            return []

        region = self.voronoi.regions[face_idx]
        vertices = self.voronoi.vertices[region]

        boundary = []
        for v in vertices:
            lat, lon = self.icosahedron.vertex_to_latlon(v)
            boundary.append((lat, lon))

        return boundary

    def get_face_boundary_polygon(self, face_idx: int, n_points: int = 50) -> List[Tuple[float, float]]:
        """
        Get a densified face boundary (with interpolated points along great circles).

        Args:
            face_idx: Face index (0-19)
            n_points: Points per edge for densification

        Returns:
            List of (lat, lon) tuples
        """
        boundary = self.get_face_boundary(face_idx)
        if not boundary:
            return []

        densified = []
        for i in range(len(boundary)):
            p1 = boundary[i]
            p2 = boundary[(i + 1) % len(boundary)]
            edge_points = self._interpolate_great_circle(p1, p2, n_points)
            densified.extend(edge_points[:-1])  # Avoid duplicating endpoints

        return densified

    def _interpolate_great_circle(self, p1: Tuple[float, float],
                                   p2: Tuple[float, float],
                                   n_points: int) -> List[Tuple[float, float]]:
        """
        Interpolate points along a great circle arc.

        Args:
            p1, p2: (lat, lon) endpoints in degrees
            n_points: Number of points to generate

        Returns:
            List of (lat, lon) tuples
        """
        # Convert to cartesian
        v1 = self.icosahedron.latlon_to_cartesian(p1[0], p1[1])
        v2 = self.icosahedron.latlon_to_cartesian(p2[0], p2[1])

        # SLERP (spherical linear interpolation)
        dot = np.clip(np.dot(v1, v2), -1, 1)
        omega = np.arccos(dot)

        if omega < 1e-10:
            return [p1]

        points = []
        for t in np.linspace(0, 1, n_points):
            if omega > 1e-10:
                v = (np.sin((1 - t) * omega) * v1 + np.sin(t * omega) * v2) / np.sin(omega)
            else:
                v = v1

            v = v / np.linalg.norm(v)
            lat, lon = self.icosahedron.vertex_to_latlon(v)
            points.append((lat, lon))

        return points

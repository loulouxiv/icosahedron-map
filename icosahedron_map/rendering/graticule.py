"""
Graticule (lat/lon grid) generation for icosahedron faces.

Generates parallels (constant latitude) and meridians (constant longitude)
projected onto each face.
"""

import numpy as np
from typing import List, Tuple
from shapely.geometry import LineString

from ..projection.gnomonic import FaceProjection
from ..projection.face_assignment import FaceAssignment


class GraticuleGenerator:
    """
    Generates latitude/longitude grid lines for icosahedron faces.
    """

    def __init__(self, lat_step: float = 15.0, lon_step: float = 15.0):
        """
        Initialize graticule generator.

        Args:
            lat_step: Spacing between parallels in degrees
            lon_step: Spacing between meridians in degrees
        """
        self.lat_step = lat_step
        self.lon_step = lon_step

    def generate_parallels(self, face_proj: FaceProjection,
                           face_assignment: FaceAssignment,
                           face_idx: int) -> List[List[Tuple[float, float]]]:
        """
        Generate parallel lines (constant latitude) for a face.

        Args:
            face_proj: FaceProjection for this face
            face_assignment: FaceAssignment instance
            face_idx: Face index

        Returns:
            List of line segments, each as [(x1, y1), (x2, y2), ...]
        """
        parallels = []

        for lat in np.arange(-90 + self.lat_step, 90, self.lat_step):
            if abs(lat) > 89:
                continue

            # Sample points along the parallel
            lons = np.linspace(-180, 180, 361)
            segment = []

            for lon in lons:
                # Check if point belongs to this face
                assigned_face = face_assignment.assign_point(lat, lon)
                if assigned_face != face_idx:
                    if segment:
                        if len(segment) >= 2:
                            parallels.append(segment)
                        segment = []
                    continue

                # Project point
                try:
                    x, y = face_proj.project(lat, lon)
                    # Check for reasonable values
                    if abs(x) < 10 and abs(y) < 10:
                        segment.append((x, y))
                except Exception:
                    if segment and len(segment) >= 2:
                        parallels.append(segment)
                    segment = []

            if len(segment) >= 2:
                parallels.append(segment)

        return parallels

    def generate_meridians(self, face_proj: FaceProjection,
                           face_assignment: FaceAssignment,
                           face_idx: int) -> List[List[Tuple[float, float]]]:
        """
        Generate meridian lines (constant longitude) for a face.

        Args:
            face_proj: FaceProjection for this face
            face_assignment: FaceAssignment instance
            face_idx: Face index

        Returns:
            List of line segments
        """
        meridians = []

        for lon in np.arange(-180, 180, self.lon_step):
            # Sample points along the meridian
            lats = np.linspace(-89, 89, 179)
            segment = []

            for lat in lats:
                # Check if point belongs to this face
                assigned_face = face_assignment.assign_point(lat, lon)
                if assigned_face != face_idx:
                    if segment:
                        if len(segment) >= 2:
                            meridians.append(segment)
                        segment = []
                    continue

                # Project point
                try:
                    x, y = face_proj.project(lat, lon)
                    if abs(x) < 10 and abs(y) < 10:
                        segment.append((x, y))
                except Exception:
                    if segment and len(segment) >= 2:
                        meridians.append(segment)
                    segment = []

            if len(segment) >= 2:
                meridians.append(segment)

        return meridians

    def generate_for_face(self, face_proj: FaceProjection,
                          face_assignment: FaceAssignment,
                          face_idx: int) -> Tuple[List, List]:
        """
        Generate complete graticule for a face.

        Args:
            face_proj: FaceProjection for this face
            face_assignment: FaceAssignment instance
            face_idx: Face index

        Returns:
            (parallels, meridians) as lists of line segments
        """
        parallels = self.generate_parallels(face_proj, face_assignment, face_idx)
        meridians = self.generate_meridians(face_proj, face_assignment, face_idx)
        return parallels, meridians


class EquatorHighlighter:
    """
    Generates special highlighting for the equator.
    """

    def generate_equator(self, face_proj: FaceProjection,
                         face_assignment: FaceAssignment,
                         face_idx: int) -> List[List[Tuple[float, float]]]:
        """
        Generate equator line for a face.

        Args:
            face_proj: FaceProjection for this face
            face_assignment: FaceAssignment instance
            face_idx: Face index

        Returns:
            List of line segments for the equator
        """
        segments = []
        lons = np.linspace(-180, 180, 721)
        segment = []

        for lon in lons:
            assigned_face = face_assignment.assign_point(0, lon)
            if assigned_face != face_idx:
                if segment and len(segment) >= 2:
                    segments.append(segment)
                segment = []
                continue

            try:
                x, y = face_proj.project(0, lon)
                if abs(x) < 10 and abs(y) < 10:
                    segment.append((x, y))
            except Exception:
                if segment and len(segment) >= 2:
                    segments.append(segment)
                segment = []

        if len(segment) >= 2:
            segments.append(segment)

        return segments


class SpecialParallelsGenerator:
    """
    Generates special latitude lines: polar circles and tropics.

    Special latitudes:
    - Arctic Circle: ~66.56°N (90° - obliquity)
    - Tropic of Cancer: ~23.44°N (obliquity)
    - Tropic of Capricorn: ~23.44°S (-obliquity)
    - Antarctic Circle: ~66.56°S (-90° + obliquity)
    """

    # Earth's axial tilt (obliquity) in degrees
    OBLIQUITY = 23.436

    # Named latitudes
    ARCTIC_CIRCLE = 90.0 - OBLIQUITY      # ~66.564°N
    TROPIC_OF_CANCER = OBLIQUITY           # ~23.436°N
    TROPIC_OF_CAPRICORN = -OBLIQUITY       # ~23.436°S
    ANTARCTIC_CIRCLE = -90.0 + OBLIQUITY  # ~66.564°S

    def __init__(self):
        """Initialize the special parallels generator."""
        self.special_latitudes = {
            'arctic_circle': self.ARCTIC_CIRCLE,
            'tropic_of_cancer': self.TROPIC_OF_CANCER,
            'tropic_of_capricorn': self.TROPIC_OF_CAPRICORN,
            'antarctic_circle': self.ANTARCTIC_CIRCLE
        }

    def generate_parallel_at_latitude(
            self,
            latitude: float,
            face_proj: FaceProjection,
            face_assignment: FaceAssignment,
            face_idx: int
    ) -> List[List[Tuple[float, float]]]:
        """
        Generate a parallel line at a specific latitude for a face.

        Args:
            latitude: Latitude in degrees
            face_proj: FaceProjection for this face
            face_assignment: FaceAssignment instance
            face_idx: Face index

        Returns:
            List of line segments
        """
        segments = []
        lons = np.linspace(-180, 180, 721)
        segment = []

        for lon in lons:
            assigned_face = face_assignment.assign_point(latitude, lon)
            if assigned_face != face_idx:
                if segment and len(segment) >= 2:
                    segments.append(segment)
                segment = []
                continue

            try:
                x, y = face_proj.project(latitude, lon)
                if abs(x) < 10 and abs(y) < 10:
                    segment.append((x, y))
            except Exception:
                if segment and len(segment) >= 2:
                    segments.append(segment)
                segment = []

        if len(segment) >= 2:
            segments.append(segment)

        return segments

    def generate_polar_circles(
            self,
            face_proj: FaceProjection,
            face_assignment: FaceAssignment,
            face_idx: int
    ) -> dict:
        """
        Generate Arctic and Antarctic circles for a face.

        Args:
            face_proj: FaceProjection for this face
            face_assignment: FaceAssignment instance
            face_idx: Face index

        Returns:
            Dict with 'arctic_circle' and 'antarctic_circle' keys,
            each containing a list of line segments
        """
        return {
            'arctic_circle': self.generate_parallel_at_latitude(
                self.ARCTIC_CIRCLE, face_proj, face_assignment, face_idx
            ),
            'antarctic_circle': self.generate_parallel_at_latitude(
                self.ANTARCTIC_CIRCLE, face_proj, face_assignment, face_idx
            )
        }

    def generate_tropics(
            self,
            face_proj: FaceProjection,
            face_assignment: FaceAssignment,
            face_idx: int
    ) -> dict:
        """
        Generate Tropic of Cancer and Tropic of Capricorn for a face.

        Args:
            face_proj: FaceProjection for this face
            face_assignment: FaceAssignment instance
            face_idx: Face index

        Returns:
            Dict with 'tropic_of_cancer' and 'tropic_of_capricorn' keys,
            each containing a list of line segments
        """
        return {
            'tropic_of_cancer': self.generate_parallel_at_latitude(
                self.TROPIC_OF_CANCER, face_proj, face_assignment, face_idx
            ),
            'tropic_of_capricorn': self.generate_parallel_at_latitude(
                self.TROPIC_OF_CAPRICORN, face_proj, face_assignment, face_idx
            )
        }

    def generate_all(
            self,
            face_proj: FaceProjection,
            face_assignment: FaceAssignment,
            face_idx: int
    ) -> dict:
        """
        Generate all special parallels (polar circles and tropics) for a face.

        Args:
            face_proj: FaceProjection for this face
            face_assignment: FaceAssignment instance
            face_idx: Face index

        Returns:
            Dict with keys for each special parallel, containing line segments
        """
        result = {}
        for name, latitude in self.special_latitudes.items():
            result[name] = self.generate_parallel_at_latitude(
                latitude, face_proj, face_assignment, face_idx
            )
        return result

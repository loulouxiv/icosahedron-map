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

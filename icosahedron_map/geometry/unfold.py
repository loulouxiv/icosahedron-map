"""
Icosahedron unfolding to a 2D pattern (net).

Pattern layout: 5-10-5
- Top row: 5 triangles pointing up (north polar faces)
- Middle row: 10 triangles alternating up/down (equatorial faces)
- Bottom row: 5 triangles pointing down (south polar faces)
"""

import numpy as np
from typing import Dict, Tuple, List


class IcosahedronUnfolder:
    """
    Computes the 2D layout of the unfolded icosahedron pattern.
    """

    def __init__(self, edge_length: float = 100.0, margin: float = 0.1):
        """
        Initialize the unfolder.

        Args:
            edge_length: Length of triangle edge in output units (pixels).
            margin: Margin as fraction of edge_length. Use 0.0 for no margin.
        """
        self.edge_length = edge_length
        self.margin = margin
        self.triangle_height = edge_length * np.sqrt(3) / 2

        # Compute face positions in the 2D pattern
        self.face_positions = self._compute_face_positions()

    def _compute_face_positions(self) -> Dict[int, Tuple[np.ndarray, float]]:
        """
        Compute position and rotation of each face in the 2D pattern.

        Pattern 5-10-5:
        - Row 1: 5 triangles pointing UP (north polar faces 0-4)
        - Row 2: 10 triangles alternating DOWN/UP (equatorial faces 5-14)
        - Row 3: 5 triangles pointing DOWN (south polar faces 15-19)

        Returns:
            Dict mapping face_idx to (center_position, rotation_angle_degrees)
        """
        positions = {}
        e = self.edge_length
        h = self.triangle_height

        # Row 1: 5 triangles pointing UP (faces 0-4, north polar)
        # Base at y=h, apex at y=0
        for i in range(5):
            x = (i + 0.5) * e
            y = 2 * h / 3  # Center is 2/3 from apex towards base
            positions[i] = (np.array([x, y]), 0)

        # Row 2: 10 triangles alternating (faces 5-14, equatorial)
        # Triangles pointing DOWN share base with row 1
        # Triangles pointing UP are interleaved
        for i in range(10):
            if i % 2 == 0:
                # Triangles pointing DOWN (indices 5,7,9,11,13 -> i=0,2,4,6,8)
                # These share base with row 1 triangles
                x = (i // 2 + 0.5) * e
                y = h + h / 3  # Center is 1/3 from base
                rotation = 180
            else:
                # Triangles pointing UP (indices 6,8,10,12,14 -> i=1,3,5,7,9)
                # These are offset by e/2
                x = (i // 2 + 1) * e
                y = h + 2 * h / 3  # Center is 2/3 from apex
                rotation = 0
            positions[5 + i] = (np.array([x, y]), rotation)

        # Row 3: 5 triangles pointing DOWN (faces 15-19, south polar)
        # These connect to the UP triangles of row 2
        for i in range(5):
            x = (i + 1) * e  # Offset by e/2 from row 1
            y = 2 * h + h / 3
            positions[15 + i] = (np.array([x, y]), 180)

        return positions

    def get_triangle_vertices(self, face_idx: int) -> np.ndarray:
        """
        Get the 3 vertices of a triangle in the 2D pattern.

        Args:
            face_idx: Face index (0-19)

        Returns:
            3x2 array of vertex coordinates
        """
        center, rotation = self.face_positions[face_idx]
        e = self.edge_length
        h = self.triangle_height

        if rotation == 0:
            # Pointing up: apex at top
            vertices = np.array([
                [0, -2 * h / 3],     # Top vertex
                [-e / 2, h / 3],     # Bottom left
                [e / 2, h / 3]       # Bottom right
            ])
        else:
            # Pointing down: apex at bottom
            vertices = np.array([
                [0, 2 * h / 3],      # Bottom vertex
                [-e / 2, -h / 3],    # Top left
                [e / 2, -h / 3]      # Top right
            ])

        return vertices + center

    def transform_point(self, face_idx: int,
                        local_x: float, local_y: float) -> Tuple[float, float]:
        """
        Transform a point from face-local coordinates to pattern coordinates.

        The face-local coordinate system has:
        - Origin at face center
        - Scale normalized so triangle circumradius ≈ 1

        Args:
            face_idx: Face index (0-19)
            local_x, local_y: Coordinates in face-local system

        Returns:
            (x, y) in pattern coordinates
        """
        center, rotation = self.face_positions[face_idx]

        # Scale from normalized to pixels
        # The circumradius of equilateral triangle with edge e is e/sqrt(3)
        scale = self.edge_length / np.sqrt(3)

        x = local_x * scale
        y = local_y * scale

        # Apply rotation if triangle is flipped
        if rotation == 180:
            x = -x
            y = -y

        return (center[0] + x, center[1] + y)

    def get_pattern_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get the bounding box of the entire pattern.

        Returns:
            (min_x, min_y, width, height)
        """
        # Calculate actual bounds from all triangle vertices
        all_x = []
        all_y = []
        for face_idx in range(20):
            verts = self.get_triangle_vertices(face_idx)
            all_x.extend(v[0] for v in verts)
            all_y.extend(v[1] for v in verts)

        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)

        # Add margins
        margin_size = self.edge_length * self.margin
        return (min_x - margin_size, min_y - margin_size,
                max_x - min_x + 2 * margin_size, max_y - min_y + 2 * margin_size)

    def get_face_label_position(self, face_idx: int) -> Tuple[float, float]:
        """Get position for face label (at center)."""
        center, _ = self.face_positions[face_idx]
        return (center[0], center[1])

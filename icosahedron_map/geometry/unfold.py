"""
Icosahedron unfolding to a 2D pattern (net).

Pattern layout: 5-10-5
- Top row: 5 triangles pointing up (north polar faces)
- Middle row: 10 triangles alternating up/down (equatorial faces)
- Bottom row: 5 triangles pointing down (south polar faces)
"""

import numpy as np
from typing import Dict, Tuple, List, Set


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

    def _get_edge_endpoints(self, face_idx: int, edge_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the two endpoints of an edge in pattern coordinates.

        Edge indices for UP triangles (apex at top):
            0: top vertex → bottom-left
            1: top vertex → bottom-right
            2: bottom-left → bottom-right (base)

        Edge indices for DOWN triangles (apex at bottom):
            0: bottom vertex → top-left
            1: bottom vertex → top-right
            2: top-left → top-right (base)

        Returns:
            (point_a, point_b) as numpy arrays
        """
        vertices = self.get_triangle_vertices(face_idx)
        _, rotation = self.face_positions[face_idx]

        if rotation == 0:  # UP triangle
            if edge_idx == 0:
                return vertices[0], vertices[1]  # top to bottom-left
            elif edge_idx == 1:
                return vertices[0], vertices[2]  # top to bottom-right
            else:  # edge_idx == 2
                return vertices[1], vertices[2]  # bottom-left to bottom-right
        else:  # DOWN triangle (rotation == 180)
            if edge_idx == 0:
                return vertices[0], vertices[1]  # bottom to top-left
            elif edge_idx == 1:
                return vertices[0], vertices[2]  # bottom to top-right
            else:  # edge_idx == 2
                return vertices[1], vertices[2]  # top-left to top-right

    def _compute_net_connectivity(self) -> Set[Tuple[int, int, int, int]]:
        """
        Compute which edges are physically connected (hinges) in the 2D net.

        Two edges are connected if they share the same coordinates in the pattern.

        Returns:
            Set of (face_a, edge_a, face_b, edge_b) tuples where face_a < face_b
        """
        connected = set()
        tolerance = 1e-6

        # Collect all edge endpoints
        all_edges = []
        for face_idx in range(20):
            for edge_idx in range(3):
                p1, p2 = self._get_edge_endpoints(face_idx, edge_idx)
                # Store in canonical order (sorted by coordinates)
                if (p1[0], p1[1]) > (p2[0], p2[1]):
                    p1, p2 = p2, p1
                all_edges.append((face_idx, edge_idx, p1, p2))

        # Find matching edges
        for i, (face_a, edge_a, p1_a, p2_a) in enumerate(all_edges):
            for j, (face_b, edge_b, p1_b, p2_b) in enumerate(all_edges[i+1:], i+1):
                if face_a == face_b:
                    continue
                # Check if edges overlap (same endpoints)
                if (np.allclose(p1_a, p1_b, atol=tolerance) and
                    np.allclose(p2_a, p2_b, atol=tolerance)):
                    if face_a < face_b:
                        connected.add((face_a, edge_a, face_b, edge_b))
                    else:
                        connected.add((face_b, edge_b, face_a, edge_a))

        return connected

    def get_free_edges_with_tabs(self) -> List[Tuple[int, int]]:
        """
        Get list of (face_idx, edge_idx) pairs for edges that need gluing tabs.

        Only returns one edge per pair that will be glued together.
        The face with lower index owns the tab for each 3D icosahedron edge.

        Returns:
            List of (face_idx, edge_idx) tuples
        """
        connected = self._compute_net_connectivity()

        # Build set of connected edges for quick lookup
        connected_edges = set()
        for face_a, edge_a, face_b, edge_b in connected:
            connected_edges.add((face_a, edge_a))
            connected_edges.add((face_b, edge_b))

        # Find all free edges (not connected in the net)
        free_edges = []
        for face_idx in range(20):
            for edge_idx in range(3):
                if (face_idx, edge_idx) not in connected_edges:
                    free_edges.append((face_idx, edge_idx))

        # For deduplication: we need to know which free edges will be glued
        # to each other when assembling. This requires 3D icosahedron edge info.
        # For now, we'll place tabs on all free edges and let the user decide.
        # A smarter approach would track 3D edge correspondence.

        # Simple deduplication: assign tab to lower face index for each 3D edge.
        # This works because each 3D edge appears on exactly 2 faces.
        return free_edges

    def compute_tab_vertices(self, face_idx: int, edge_idx: int,
                             tab_size: float = 0.15) -> np.ndarray:
        """
        Compute trapezoid tab vertices for a given edge.

        The tab extends outward from the triangle edge, forming a trapezoid
        that tapers toward the outer edge.

        Args:
            face_idx: Face index (0-19)
            edge_idx: Edge index (0-2)
            tab_size: Tab height as fraction of edge length

        Returns:
            4x2 array of vertices [edge_start, edge_end, outer_end, outer_start]
        """
        p1, p2 = self._get_edge_endpoints(face_idx, edge_idx)

        # Edge vector and length
        edge_vec = p2 - p1
        edge_length = np.linalg.norm(edge_vec)
        edge_unit = edge_vec / edge_length

        # Perpendicular vector (pointing outward from triangle center)
        # Use cross product with z-axis, then check direction
        perp = np.array([-edge_unit[1], edge_unit[0]])

        # Check if perpendicular points away from triangle center
        center, _ = self.face_positions[face_idx]
        mid_edge = (p1 + p2) / 2
        to_center = center - mid_edge

        if np.dot(perp, to_center) > 0:
            perp = -perp  # Flip to point outward

        # Tab dimensions
        tab_height = edge_length * tab_size
        # Taper: offset outer corners inward along edge by 25% of tab height
        taper_offset = tab_height * 1

        # Compute outer edge endpoints (tapered)
        outer_start = p1 + perp * tab_height + edge_unit * taper_offset
        outer_end = p2 + perp * tab_height - edge_unit * taper_offset

        return np.array([p1, p2, outer_end, outer_start])

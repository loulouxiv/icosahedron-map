"""
Icosahedron geometry with north pole on a vertex.

An icosahedron has:
- 12 vertices
- 20 triangular faces
- 30 edges

With north pole on vertex: 5 faces around north pole, 5 around south pole, 10 equatorial.
"""

import numpy as np
from typing import List, Tuple, Optional


class Icosahedron:
    """
    Regular icosahedron inscribed in a unit sphere, oriented with
    the north pole (90°N) on a vertex.

    Supports an optional coordinate rotation to effectively place the
    north pole at the center of a face instead.
    """

    # Golden ratio
    PHI = (1 + np.sqrt(5)) / 2

    def __init__(self, pole_on_face: bool = False, longitude_rotation: float = 0.0):
        """
        Initialize icosahedron with vertices and faces.

        Args:
            pole_on_face: If True, apply a coordinate rotation so that
                          geographic north pole maps to a face center.
            longitude_rotation: Rotation angle around north-south axis in degrees.
                               Positive values rotate eastward.
        """
        self.pole_on_face = pole_on_face
        self.longitude_rotation = longitude_rotation
        self.vertices = self._compute_vertices()
        self.face_indices = self._define_face_indices()
        self.faces = self._compute_faces()
        self.face_centers = self._compute_face_centers()

        # Compute combined rotation matrix
        self._coord_rotation = self._compute_coord_rotation()

    def _compute_vertices(self) -> np.ndarray:
        """
        Compute the 12 vertices of the icosahedron on a unit sphere.

        Standard icosahedron vertices use coordinates (0, ±1, ±φ) and permutations.
        We then rotate so that one vertex is at the north pole (0, 0, 1).
        """
        phi = self.PHI

        # Standard icosahedron vertices (before rotation)
        # These are permutations of (0, ±1, ±φ)
        raw_vertices = np.array([
            [0, 1, phi],
            [0, -1, phi],
            [0, 1, -phi],
            [0, -1, -phi],
            [1, phi, 0],
            [-1, phi, 0],
            [1, -phi, 0],
            [-1, -phi, 0],
            [phi, 0, 1],
            [-phi, 0, 1],
            [phi, 0, -1],
            [-phi, 0, -1]
        ])

        # Normalize to unit sphere
        norms = np.linalg.norm(raw_vertices, axis=1, keepdims=True)
        vertices = raw_vertices / norms

        # Standard orientation: vertex at north pole
        north_idx = np.argmax(vertices[:, 2])
        north_vertex = vertices[north_idx]

        # Rotate so this vertex is exactly at (0, 0, 1)
        rotation = self._rotation_to_north_pole(north_vertex)
        vertices = (rotation @ vertices.T).T

        return vertices

    def _compute_pole_on_face_rotation(self) -> np.ndarray:
        """
        Compute rotation matrix that maps geographic coordinates so that
        the north pole (0,0,1) maps to a face center of the icosahedron.

        This is the INVERSE of rotating a face center to the north pole.
        """
        # Find the face center with highest z (closest to north pole)
        north_face_idx = np.argmax(self.face_centers[:, 2])
        face_center = self.face_centers[north_face_idx]

        # We want: rotation @ (0,0,1) = face_center
        # This is the inverse of rotating face_center to (0,0,1)
        rotation_to_pole = self._rotation_to_north_pole(face_center)
        # Inverse of rotation matrix is its transpose
        return rotation_to_pole.T

    def _compute_z_rotation(self, angle_deg: float) -> np.ndarray:
        """
        Compute rotation matrix for rotation around Z-axis (north-south).

        Args:
            angle_deg: Rotation angle in degrees (positive = eastward)

        Returns:
            3x3 rotation matrix
        """
        theta = np.radians(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        return np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0, 0, 1]
        ])

    def _compute_coord_rotation(self) -> Optional[np.ndarray]:
        """
        Compute combined coordinate rotation matrix.

        Composes longitude rotation with pole_on_face rotation.
        Returns None if no rotation is needed.
        """
        matrices = []

        # Longitude rotation applied first (to input coordinates)
        if self.longitude_rotation != 0.0:
            matrices.append(self._compute_z_rotation(self.longitude_rotation))

        # Pole-on-face rotation applied second
        if self.pole_on_face:
            matrices.append(self._compute_pole_on_face_rotation())

        if not matrices:
            return None
        elif len(matrices) == 1:
            return matrices[0]
        else:
            # Compose: R_combined = R_pole @ R_lon
            # Applied right-to-left: first lon rotation, then pole rotation
            return matrices[1] @ matrices[0]

    def _rotation_to_north_pole(self, vertex: np.ndarray) -> np.ndarray:
        """
        Compute rotation matrix that moves vertex to north pole (0, 0, 1).
        """
        north_pole = np.array([0, 0, 1])
        v = vertex / np.linalg.norm(vertex)

        # If already at north pole
        if np.allclose(v, north_pole):
            return np.eye(3)

        # If at south pole
        if np.allclose(v, -north_pole):
            return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

        # Rotation axis (cross product)
        axis = np.cross(v, north_pole)
        axis = axis / np.linalg.norm(axis)

        # Rotation angle
        cos_angle = np.dot(v, north_pole)
        sin_angle = np.sqrt(1 - cos_angle ** 2)

        # Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * (K @ K)
        return R

    def _define_face_indices(self) -> List[Tuple[int, int, int]]:
        """
        Define the 20 faces as triplets of vertex indices.

        After rotation, vertex 0 is at north pole.
        Faces are ordered: 5 north polar, 10 equatorial, 5 south polar.
        """
        # Find vertices by latitude bands
        z_coords = self.vertices[:, 2]

        # North pole vertex (highest z)
        north_idx = int(np.argmax(z_coords))

        # South pole vertex (lowest z)
        south_idx = int(np.argmin(z_coords))

        # Upper ring (5 vertices connected to north pole)
        # Lower ring (5 vertices connected to south pole)
        other_indices = [i for i in range(12) if i not in [north_idx, south_idx]]
        other_z = [(i, z_coords[i]) for i in other_indices]
        other_z.sort(key=lambda x: -x[1])  # Sort by z descending

        upper_ring = [x[0] for x in other_z[:5]]
        lower_ring = [x[0] for x in other_z[5:]]

        # Sort rings by angle around z-axis for consistent ordering
        def angle_around_z(idx):
            v = self.vertices[idx]
            return np.arctan2(v[1], v[0])

        upper_ring.sort(key=angle_around_z)
        lower_ring.sort(key=angle_around_z)

        faces = []

        # 5 faces around north pole
        for i in range(5):
            v1 = upper_ring[i]
            v2 = upper_ring[(i + 1) % 5]
            faces.append((north_idx, v1, v2))

        # 10 equatorial faces (alternating up/down triangles)
        for i in range(5):
            # Triangle pointing down (connecting upper to lower)
            u1 = upper_ring[i]
            u2 = upper_ring[(i + 1) % 5]

            # Find closest lower vertex to midpoint of u1-u2
            mid = (self.vertices[u1] + self.vertices[u2]) / 2
            mid_angle = np.arctan2(mid[1], mid[0])

            # Find lower vertex with closest angle
            lower_angles = [(j, angle_around_z(lower_ring[j])) for j in range(5)]
            closest_lower = min(lower_angles,
                              key=lambda x: abs(((x[1] - mid_angle + np.pi) % (2*np.pi)) - np.pi))
            l_idx = lower_ring[closest_lower[0]]

            faces.append((u1, l_idx, u2))

            # Triangle pointing up
            l1 = lower_ring[i]
            l2 = lower_ring[(i + 1) % 5]

            # Find closest upper vertex
            mid = (self.vertices[l1] + self.vertices[l2]) / 2
            mid_angle = np.arctan2(mid[1], mid[0])

            upper_angles = [(j, angle_around_z(upper_ring[j])) for j in range(5)]
            closest_upper = min(upper_angles,
                              key=lambda x: abs(((x[1] - mid_angle + np.pi) % (2*np.pi)) - np.pi))
            u_idx = upper_ring[closest_upper[0]]

            faces.append((l1, u_idx, l2))

        # 5 faces around south pole
        for i in range(5):
            v1 = lower_ring[i]
            v2 = lower_ring[(i + 1) % 5]
            faces.append((south_idx, v2, v1))

        return faces

    def _compute_faces(self) -> List[np.ndarray]:
        """Return face vertices as 3x3 arrays."""
        return [self.vertices[list(face)] for face in self.face_indices]

    def _compute_face_centers(self) -> np.ndarray:
        """
        Compute the center of each face, projected onto the unit sphere.
        """
        centers = []
        for face in self.faces:
            center = np.mean(face, axis=0)
            center = center / np.linalg.norm(center)
            centers.append(center)
        return np.array(centers)

    def vertex_to_latlon(self, vertex: np.ndarray) -> Tuple[float, float]:
        """
        Convert cartesian vertex (x, y, z) to (latitude, longitude) in degrees.
        """
        x, y, z = vertex
        lat = np.degrees(np.arcsin(np.clip(z, -1, 1)))
        lon = np.degrees(np.arctan2(y, x))
        return lat, lon

    def latlon_to_cartesian(self, lat: float, lon: float) -> np.ndarray:
        """
        Convert (latitude, longitude) in degrees to cartesian (x, y, z).
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        return np.array([x, y, z])

    def rotate_latlon(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Apply coordinate rotation for pole_on_face mode.

        In pole_on_face mode, rotates coordinates so that the geographic
        north pole maps to a face center of the icosahedron.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            (rotated_lat, rotated_lon) in degrees
        """
        if self._coord_rotation is None:
            return lat, lon

        # Convert to cartesian
        cart = self.latlon_to_cartesian(lat, lon)

        # Apply rotation
        rotated = self._coord_rotation @ cart

        # Convert back to lat/lon
        return self.vertex_to_latlon(rotated)

    def rotate_latlon_arrays(self, lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply coordinate rotation to arrays of coordinates.

        Args:
            lats: Array of latitudes in degrees
            lons: Array of longitudes in degrees

        Returns:
            (rotated_lats, rotated_lons) as arrays in degrees
        """
        if self._coord_rotation is None:
            return lats, lons

        # Convert to cartesian
        lat_rad = np.radians(lats)
        lon_rad = np.radians(lons)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)

        # Stack and rotate
        cart = np.stack([x, y, z], axis=0)  # 3 x N
        rotated = self._coord_rotation @ cart  # 3 x N

        # Convert back to lat/lon
        rotated_lats = np.degrees(np.arcsin(np.clip(rotated[2], -1, 1)))
        rotated_lons = np.degrees(np.arctan2(rotated[1], rotated[0]))

        return rotated_lats, rotated_lons

    def get_face_latlon_center(self, face_idx: int) -> Tuple[float, float]:
        """Get the center of a face in lat/lon coordinates."""
        center = self.face_centers[face_idx]
        return self.vertex_to_latlon(center)

    def get_face_vertices_latlon(self, face_idx: int) -> List[Tuple[float, float]]:
        """Get the vertices of a face in lat/lon coordinates."""
        face = self.faces[face_idx]
        return [self.vertex_to_latlon(v) for v in face]

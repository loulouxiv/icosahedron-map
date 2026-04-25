"""
SVG generator for the icosahedron map pattern.

Creates an SVG file with the unfolded icosahedron net, projected countries,
and graticule.
"""

import svgwrite
import numpy as np
from typing import List, Dict, Optional, Tuple
from shapely.geometry import Polygon, MultiPolygon

from ..geometry.unfold import IcosahedronUnfolder
from ..projection.gnomonic import FaceProjection


class IcosahedronSVGGenerator:
    """
    Generates SVG output for the icosahedron map pattern.
    """

    def __init__(self, unfolder: IcosahedronUnfolder,
                 face_projections: List[FaceProjection],
                 output_path: str = "icosahedron_map.svg",
                 country_colors: Dict[str, str] = None):
        """
        Initialize SVG generator.

        Args:
            unfolder: IcosahedronUnfolder instance
            face_projections: List of FaceProjection for all 20 faces
            output_path: Output SVG file path
            country_colors: Optional dict mapping country name to fill color
        """
        self.unfolder = unfolder
        self.face_projections = face_projections
        self.output_path = output_path
        self.country_colors = country_colors or {}
        self._face_triangles: Dict[int, Polygon] = {}  # Cache for 2D clip triangles

        # Get pattern bounds (tabs are intentionally excluded - they spill outside)
        min_x, min_y, width, height = unfolder.get_pattern_bounds()

        # Create SVG document
        self.dwg = svgwrite.Drawing(
            output_path,
            size=(f"{width}px", f"{height}px"),
            viewBox=f"{min_x} {min_y} {width} {height}"
        )

        # Define styles
        self._define_styles()

        # Create layer groups
        self.background_group = self.dwg.g(id="background")
        self.tabs_group = self.dwg.g(id="glue-tabs")
        self.countries_group = self.dwg.g(id="countries")
        self.graticule_group = self.dwg.g(id="graticule")
        self.face_outlines_group = self.dwg.g(id="face-outlines")
        self.labels_group = self.dwg.g(id="labels")

        # Add groups in order (background first, tabs behind countries)
        self.dwg.add(self.background_group)
        self.dwg.add(self.tabs_group)
        self.dwg.add(self.countries_group)
        self.dwg.add(self.graticule_group)
        self.dwg.add(self.face_outlines_group)
        self.dwg.add(self.labels_group)

    def _define_styles(self):
        """Define CSS styles for SVG elements."""
        self.dwg.defs.add(self.dwg.style("""
            .face-outline {
                fill: none;
                stroke: #333;
                stroke-width: 0.5;
            }
            .face-background {
                fill: #d4e8f0;
                stroke: none;
            }
            .country {
                fill: #c8d8c0;
                stroke: #666;
                stroke-width: 0.2;
            }
            .graticule {
                fill: none;
                stroke: #999;
                stroke-width: 0.15;
                stroke-dasharray: 2,1;
            }
            .equator {
                fill: none;
                stroke: #c44;
                stroke-width: 0.3;
            }
            .polar-circle {
                fill: none;
                stroke: #4488cc;
                stroke-width: 0.3;
                stroke-dasharray: 4,2;
            }
            .tropic {
                fill: none;
                stroke: #cc8844;
                stroke-width: 0.3;
                stroke-dasharray: 4,2;
            }
            .face-label {
                font-family: Arial, sans-serif;
                font-size: 6px;
                fill: #666;
            }
            .glue-tab {
                fill: #e8e8e8;
                stroke: #999;
                stroke-width: 0.3;
                stroke-dasharray: 2,2;
            }
        """))

    def draw_face_backgrounds(self):
        """Draw ocean background for all faces."""
        for face_idx in range(20):
            vertices = self.unfolder.get_triangle_vertices(face_idx)
            points = [(v[0], v[1]) for v in vertices]

            triangle = self.dwg.polygon(
                points=points,
                class_="face-background"
            )
            self.background_group.add(triangle)

    def draw_gluing_tabs(self, tab_size: float = 0.15):
        """
        Draw gluing tabs on free edges of the net.

        Tabs are trapezoidal flaps that extend outward from edges that
        are not connected to adjacent triangles in the unfolded pattern.
        These tabs are used for gluing the paper model together.

        Note: Tabs intentionally extend beyond the viewbox bounds.

        Args:
            tab_size: Tab height as fraction of edge length (default: 0.15)
        """
        free_edges = self.unfolder.get_free_edges_with_tabs()

        for face_idx, edge_idx in free_edges:
            vertices = self.unfolder.compute_tab_vertices(face_idx, edge_idx, tab_size)
            points = [(v[0], v[1]) for v in vertices]

            tab = self.dwg.polygon(
                points=points,
                class_="glue-tab"
            )
            self.tabs_group.add(tab)

    def draw_face_outlines(self):
        """Draw outlines of all triangular faces."""
        for face_idx in range(20):
            vertices = self.unfolder.get_triangle_vertices(face_idx)
            points = [(v[0], v[1]) for v in vertices]
            points.append(points[0])  # Close polygon

            triangle = self.dwg.polygon(
                points=points,
                class_="face-outline"
            )
            self.face_outlines_group.add(triangle)

    def _get_face_triangle(self, face_idx: int) -> Polygon:
        """Get the 2D triangle polygon for a face (for clipping)."""
        if face_idx not in self._face_triangles:
            vertices = self.unfolder.get_triangle_vertices(face_idx)
            self._face_triangles[face_idx] = Polygon(vertices)
        return self._face_triangles[face_idx]

    def _extend_to_polar_apex(self, polygon: Polygon, face_idx: int,
                               face_triangle: Polygon) -> Polygon:
        """
        Extend a polygon to the triangle apex if it's near the pole.

        This fixes triangular gaps at the south pole for faces 15-19.
        """
        from shapely.ops import unary_union

        # Get triangle vertices - apex is vertex 0 for all faces
        vertices = self.unfolder.get_triangle_vertices(face_idx)
        apex = vertices[0]

        # Check if polygon reaches near the apex (within 10% of triangle height)
        triangle_height = abs(vertices[0][1] - vertices[1][1])
        threshold = triangle_height * 0.1

        # Get polygon bounds
        minx, miny, maxx, maxy = polygon.bounds

        # For southern faces, apex is at max y
        dist_to_apex = abs(maxy - apex[1])

        if dist_to_apex > threshold:
            # Polygon not near apex, return unchanged
            return polygon

        # Create a small triangle from the polygon's top edge to the apex
        # Find points on the polygon closest to the apex
        top_points = []
        for x, y in polygon.exterior.coords:
            if abs(y - maxy) < threshold:
                top_points.append((x, y))

        if len(top_points) < 2:
            return polygon

        # Get leftmost and rightmost top points
        top_points.sort(key=lambda p: p[0])
        left_top = top_points[0]
        right_top = top_points[-1]

        # Create fill triangle from left_top to apex to right_top
        fill_triangle = Polygon([left_top, apex, right_top])

        if not fill_triangle.is_valid or fill_triangle.area < 0.01:
            return polygon

        # Union the polygon with the fill triangle
        try:
            extended = unary_union([polygon, fill_triangle])
            if extended.is_valid:
                return extended
        except Exception:
            pass

        return polygon

    def draw_country(self, face_idx: int, geometry, name: str = "",
                     already_rotated: bool = False):
        """
        Draw a country polygon on a face.

        Args:
            face_idx: Face index
            geometry: Shapely geometry (Polygon or MultiPolygon)
            name: Country name for data attribute
            already_rotated: If True, geometry coordinates are already in
                            icosahedron space (rotated for pole_on_face mode)
        """
        color = self.country_colors.get(name)
        if isinstance(geometry, MultiPolygon):
            for poly in geometry.geoms:
                self._draw_single_polygon(face_idx, poly, name, color, already_rotated)
        elif isinstance(geometry, Polygon):
            self._draw_single_polygon(face_idx, geometry, name, color, already_rotated)

    def _draw_single_polygon(self, face_idx: int, polygon: Polygon, name: str,
                              color: str = None, already_rotated: bool = False):
        """Draw a single polygon."""
        face_proj = self.face_projections[face_idx]

        # Transform exterior ring
        exterior_points = []
        for lon, lat in polygon.exterior.coords:
            try:
                local_x, local_y = face_proj.project(lat, lon, already_rotated=already_rotated)
                x, y = self.unfolder.transform_point(face_idx, local_x, local_y)
                exterior_points.append((x, y))
            except Exception:
                continue

        if len(exterior_points) < 3:
            return

        # Handle interior rings (holes)
        holes = []
        for interior in polygon.interiors:
            hole_points = []
            for lon, lat in interior.coords:
                try:
                    local_x, local_y = face_proj.project(lat, lon, already_rotated=already_rotated)
                    x, y = self.unfolder.transform_point(face_idx, local_x, local_y)
                    hole_points.append((x, y))
                except Exception:
                    continue
            if len(hole_points) >= 3:
                holes.append(hole_points)

        # Clip projected polygon against face triangle to prevent spilling
        try:
            face_triangle = self._get_face_triangle(face_idx)
            if holes:
                projected_poly = Polygon(exterior_points, holes)
            else:
                projected_poly = Polygon(exterior_points)

            if not projected_poly.is_valid:
                projected_poly = projected_poly.buffer(0)

            # For southern faces (15-19), fill polar gaps by extending to apex
            if face_idx >= 15:
                projected_poly = self._extend_to_polar_apex(
                    projected_poly, face_idx, face_triangle
                )

            clipped = projected_poly.intersection(face_triangle)

            if clipped.is_empty:
                return

            # Draw clipped result (may be Polygon or MultiPolygon)
            self._draw_clipped_geometry(clipped, color)
        except Exception:
            # Fallback: draw without clipping
            self._draw_polygon_element(exterior_points, holes, color)

    def _draw_clipped_geometry(self, geometry, color: str = None):
        """Draw a clipped geometry (Polygon or MultiPolygon)."""
        from shapely.geometry import GeometryCollection

        if isinstance(geometry, Polygon):
            if geometry.is_empty:
                return
            exterior = list(geometry.exterior.coords)
            holes = [list(interior.coords) for interior in geometry.interiors]
            self._draw_polygon_element(exterior, holes, color)
        elif isinstance(geometry, MultiPolygon):
            for poly in geometry.geoms:
                self._draw_clipped_geometry(poly, color)
        elif isinstance(geometry, GeometryCollection):
            for geom in geometry.geoms:
                if isinstance(geom, (Polygon, MultiPolygon)):
                    self._draw_clipped_geometry(geom, color)

    def _draw_polygon_element(self, exterior_points: List, holes: List, color: str = None):
        """Draw a polygon SVG element."""
        if len(exterior_points) < 3:
            return

        if holes:
            d = self._points_to_path_d(exterior_points, holes)
            if color:
                path = self.dwg.path(d=d, class_="country", style=f"fill:{color}")
            else:
                path = self.dwg.path(d=d, class_="country")
            self.countries_group.add(path)
        else:
            if color:
                poly = self.dwg.polygon(points=exterior_points, class_="country",
                                        style=f"fill:{color}")
            else:
                poly = self.dwg.polygon(points=exterior_points, class_="country")
            self.countries_group.add(poly)

    def _points_to_path_d(self, exterior: List, holes: List) -> str:
        """Convert polygon with holes to SVG path d attribute."""
        d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in exterior) + " Z"

        for hole in holes:
            d += " M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in hole) + " Z"

        return d

    def draw_graticule_line(self, face_idx: int,
                             points: List[Tuple[float, float]],
                             is_equator: bool = False):
        """
        Draw a graticule line segment.

        Args:
            face_idx: Face index
            points: List of (x, y) points in face-local coordinates
            is_equator: Whether this is the equator (special styling)
        """
        if len(points) < 2:
            return

        # Transform points to pattern coordinates
        pattern_points = []
        for local_x, local_y in points:
            x, y = self.unfolder.transform_point(face_idx, local_x, local_y)
            pattern_points.append((x, y))

        css_class = "equator" if is_equator else "graticule"
        polyline = self.dwg.polyline(
            points=pattern_points,
            class_=css_class
        )
        self.graticule_group.add(polyline)

    def draw_graticule(self, face_idx: int,
                       parallels: List[List[Tuple[float, float]]],
                       meridians: List[List[Tuple[float, float]]]):
        """
        Draw complete graticule for a face.

        Args:
            face_idx: Face index
            parallels: List of parallel line segments
            meridians: List of meridian line segments
        """
        for segment in parallels:
            self.draw_graticule_line(face_idx, segment)

        for segment in meridians:
            self.draw_graticule_line(face_idx, segment)

    def draw_special_parallels(self, face_idx: int, special_parallels: dict):
        """
        Draw special parallels (equator, polar circles, and tropics) for a face.

        Args:
            face_idx: Face index
            special_parallels: Dict with keys like 'equator', 'arctic_circle',
                              'tropic_of_cancer', etc., each containing a list of line segments
        """
        for name, segments in special_parallels.items():
            if name == 'equator':
                css_class = 'equator'
            elif 'circle' in name:
                css_class = 'polar-circle'
            else:
                css_class = 'tropic'

            for segment in segments:
                if len(segment) < 2:
                    continue

                # Transform points to pattern coordinates
                pattern_points = []
                for local_x, local_y in segment:
                    x, y = self.unfolder.transform_point(face_idx, local_x, local_y)
                    pattern_points.append((x, y))

                polyline = self.dwg.polyline(
                    points=pattern_points,
                    class_=css_class
                )
                self.graticule_group.add(polyline)

    def add_face_labels(self):
        """Add face number labels."""
        for face_idx in range(20):
            x, y = self.unfolder.get_face_label_position(face_idx)

            text = self.dwg.text(
                str(face_idx),
                insert=(x, y),
                class_="face-label",
                text_anchor="middle",
                dominant_baseline="middle"
            )
            self.labels_group.add(text)

    def save(self):
        """Save the SVG file."""
        self.dwg.save()
        print(f"SVG saved to: {self.output_path}")

    def get_svg_string(self) -> str:
        """Get SVG as string."""
        return self.dwg.tostring()

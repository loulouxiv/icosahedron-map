"""
PDF generator for the icosahedron map pattern.

Converts SVG output to A4 PDF format using cairosvg.
"""

import re
import math
import cairosvg
import numpy as np
from ..geometry.unfold import IcosahedronUnfolder

# A4 dimensions in points (72 points per inch)
A4_WIDTH_PT = 595.28  # 210mm
A4_HEIGHT_PT = 841.89  # 297mm

# A4 dimensions in mm
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297


def _parse_svg_dimensions(svg_string: str) -> tuple:
    """Extract viewBox dimensions from SVG."""
    match = re.search(r'viewBox="([^"]+)"', svg_string)
    if match:
        parts = match.group(1).split()
        min_x, min_y, width, height = map(float, parts)
        return min_x, min_y, width, height
    return 0, 0, 100, 100


def _get_rotated_bounds(vertices: np.ndarray, angle_deg: float,
                         center: tuple = None) -> tuple:
    """Get bounding box after rotating vertices around a center point.

    Args:
        vertices: Nx2 array of vertex coordinates
        angle_deg: Rotation angle in degrees
        center: (cx, cy) rotation center. If None, rotates around origin.

    Returns:
        (min_x, min_y, width, height) of rotated bounding box
    """
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    if center is not None:
        cx, cy = center
        # Rotate around center point
        dx = vertices[:, 0] - cx
        dy = vertices[:, 1] - cy
        rotated_x = cx + dx * cos_t - dy * sin_t
        rotated_y = cy + dx * sin_t + dy * cos_t
    else:
        # Rotate around origin
        rotated_x = vertices[:, 0] * cos_t - vertices[:, 1] * sin_t
        rotated_y = vertices[:, 0] * sin_t + vertices[:, 1] * cos_t

    min_x = rotated_x.min()
    max_x = rotated_x.max()
    min_y = rotated_y.min()
    max_y = rotated_y.max()

    return min_x, min_y, max_x - min_x, max_y - min_y


def _find_optimal_angle(page_w: float, page_h: float) -> tuple:
    """
    Find rotation angle that maximizes scale on page using actual net vertices.

    Args:
        page_w, page_h: Page dimensions

    Returns:
        (angle_degrees, scale_factor relative to horizontal)
    """
    # Collect all triangle vertices
    unfolder = IcosahedronUnfolder()
    all_verts = []
    for face_idx in range(20):
        all_verts.extend(unfolder.get_triangle_vertices(face_idx))
    vertices = np.array(all_verts)

    # Horizontal placement
    _, _, w0, h0 = _get_rotated_bounds(vertices, 0)
    horiz_scale = min(page_w / w0, page_h / h0)

    best_angle = 0
    best_scale = horiz_scale

    # Search angles from 0 to 45 degrees
    for angle_deg in range(0, 451):
        angle = angle_deg / 10.0
        _, _, w, h = _get_rotated_bounds(vertices, angle)
        scale = min(page_w / w, page_h / h)

        if scale > best_scale:
            best_scale = scale
            best_angle = angle

    return best_angle, best_scale / horiz_scale


def _apply_rotation(svg_string: str, angle: float, width: float, height: float,
                     scale: float = 1.0, unfolder: IcosahedronUnfolder = None) -> str:
    """
    Apply rotation and scale transform to SVG content and adjust viewBox.

    Args:
        svg_string: Original SVG
        angle: Rotation angle in degrees
        width, height: Original content dimensions
        scale: Scale factor to apply
        unfolder: Optional IcosahedronUnfolder for accurate vertex-based rotation bounds

    Returns:
        Modified SVG string with rotation and scale applied
    """
    if angle == 0 and scale == 1.0:
        return svg_string

    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # Parse original viewBox to get center for rotation
    min_x, min_y, orig_w, orig_h = _parse_svg_dimensions(svg_string)
    cx = min_x + orig_w / 2
    cy = min_y + orig_h / 2

    # Calculate new bounding box dimensions after rotation and scale
    # Use actual net vertices for accurate bounds, not viewBox rectangle
    if unfolder is not None:
        all_verts = []
        for face_idx in range(20):
            all_verts.extend(unfolder.get_triangle_vertices(face_idx))
        vertices = np.array(all_verts)
        # Rotate around the same center as the SVG transform
        rot_min_x, rot_min_y, rot_width, rot_height = _get_rotated_bounds(
            vertices, angle, center=(cx, cy))
        # Scale the viewBox around the center of rotated content
        rot_cx = rot_min_x + rot_width / 2
        rot_cy = rot_min_y + rot_height / 2
        new_width = rot_width / scale
        new_height = rot_height / scale
        new_min_x = rot_cx - new_width / 2
        new_min_y = rot_cy - new_height / 2
    else:
        # Fallback to rectangle-based calculation if unfolder not available
        rot_width = abs(width * cos_t) + abs(height * sin_t)
        rot_height = abs(width * sin_t) + abs(height * cos_t)
        new_width = rot_width / scale
        new_height = rot_height / scale
        new_min_x = cx - new_width / 2
        new_min_y = cy - new_height / 2

    # Create transform: rotate around center of original content
    transform = f"rotate({angle} {cx} {cy})"

    # Update viewBox (smaller viewBox = larger content on page)
    new_viewbox = f"{new_min_x} {new_min_y} {new_width} {new_height}"
    svg_string = re.sub(r'viewBox="[^"]+"', f'viewBox="{new_viewbox}"', svg_string)

    # Update size attributes
    svg_string = re.sub(r'width="[^"]+"', f'width="{new_width}px"', svg_string)
    svg_string = re.sub(r'height="[^"]+"', f'height="{new_height}px"', svg_string)

    # Wrap content in a group with rotation transform
    svg_string = re.sub(
        r'(<svg[^>]*>)',
        r'\1<g transform="' + transform + '">',
        svg_string
    )
    svg_string = svg_string.replace('</svg>', '</g></svg>')

    return svg_string


def svg_to_pdf(svg_string: str, output_path: str, landscape: bool = True,
               oblique: bool = False, no_margin: bool = False,
               unfolder: IcosahedronUnfolder = None) -> None:
    """
    Convert SVG string to A4 PDF.

    Args:
        svg_string: SVG content as string
        output_path: Output PDF file path
        landscape: If True, use landscape orientation (default for wide patterns)
        oblique: If True, rotate content to maximize size on page
        no_margin: If True, remove margins for tight bounding box
        unfolder: Optional IcosahedronUnfolder instance for accurate rotation bounds
    """
    if landscape:
        page_width = A4_HEIGHT_PT  # 297mm in points
        page_height = A4_WIDTH_PT  # 210mm in points
        page_w_mm = A4_HEIGHT_MM
        page_h_mm = A4_WIDTH_MM
    else:
        page_width = A4_WIDTH_PT
        page_height = A4_HEIGHT_PT
        page_w_mm = A4_WIDTH_MM
        page_h_mm = A4_HEIGHT_MM

    if oblique:
        # Get content dimensions from SVG (now tight if no_margin was applied)
        min_x, min_y, width, height = _parse_svg_dimensions(svg_string)

        # Find optimal angle
        angle, rel_scale = _find_optimal_angle(page_w_mm, page_h_mm)

        if angle > 0.5:  # Only rotate if meaningful improvement
            improvement = (rel_scale - 1) * 100
            print(f"   Rotating {angle:.1f}° for {improvement:.1f}% larger output")

            # Apply rotation and scale
            svg_string = _apply_rotation(svg_string, angle, width, height, 1, unfolder)

    # Convert SVG to PDF, scaling to fit A4
    cairosvg.svg2pdf(
        bytestring=svg_string.encode('utf-8'),
        write_to=output_path,
        output_width=page_width,
        output_height=page_height,
    )
    print(f"PDF saved to: {output_path}")


def svg_file_to_pdf(svg_path: str, output_path: str, landscape: bool = True,
                    oblique: bool = False) -> None:
    """
    Convert SVG file to A4 PDF.

    Args:
        svg_path: Input SVG file path
        output_path: Output PDF file path
        landscape: If True, use landscape orientation
        oblique: If True, rotate content to maximize size on page
    """
    with open(svg_path, 'r', encoding='utf-8') as f:
        svg_string = f.read()
    svg_to_pdf(svg_string, output_path, landscape, oblique)

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


def _get_rotated_bounds(vertices: np.ndarray, angle_deg: float) -> tuple:
    """Get bounding box dimensions after rotating vertices."""
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    rotated_x = vertices[:, 0] * cos_t - vertices[:, 1] * sin_t
    rotated_y = vertices[:, 0] * sin_t + vertices[:, 1] * cos_t

    width = rotated_x.max() - rotated_x.min()
    height = rotated_y.max() - rotated_y.min()
    return width, height


def _get_tight_bounds(svg_string: str) -> tuple:
    """Get tight bounding box from all points in the SVG."""
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')
    found_point = False

    # Extract points from polygon elements: points="x1,y1 x2,y2 ..."
    for match in re.finditer(r'points="([^"]+)"', svg_string):
        points_str = match.group(1)
        for point in points_str.split():
            if ',' in point:
                x, y = point.split(',')
                x, y = float(x), float(y)
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                found_point = True

    # Extract points from path elements: d="M x,y L x,y ..."
    for match in re.finditer(r'\bd="([^"]+)"', svg_string):
        path_str = match.group(1)
        # Find all coordinate pairs (number,number or number number)
        for coord in re.finditer(r'(-?\d+\.?\d*)[,\s](-?\d+\.?\d*)', path_str):
            x, y = float(coord.group(1)), float(coord.group(2))
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            found_point = True

    if not found_point:
        return _parse_svg_dimensions(svg_string)

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
    w0, h0 = _get_rotated_bounds(vertices, 0)
    horiz_scale = min(page_w / w0, page_h / h0)

    best_angle = 0
    best_scale = horiz_scale

    # Search angles from 0 to 45 degrees
    for angle_deg in range(0, 451):
        angle = angle_deg / 10.0
        w, h = _get_rotated_bounds(vertices, angle)
        scale = min(page_w / w, page_h / h)

        if scale > best_scale:
            best_scale = scale
            best_angle = angle

    return best_angle, best_scale / horiz_scale


def _apply_rotation(svg_string: str, angle: float, width: float, height: float,
                     scale: float = 1.0) -> str:
    """
    Apply rotation and scale transform to SVG content and adjust viewBox.

    Args:
        svg_string: Original SVG
        angle: Rotation angle in degrees
        width, height: Original content dimensions
        scale: Scale factor to apply

    Returns:
        Modified SVG string with rotation and scale applied
    """
    if angle == 0 and scale == 1.0:
        return svg_string

    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # Calculate new bounding box dimensions after rotation and scale
    rot_width = abs(width * cos_t) + abs(height * sin_t)
    rot_height = abs(width * sin_t) + abs(height * cos_t)
    new_width = rot_width / scale
    new_height = rot_height / scale

    # Parse original viewBox
    min_x, min_y, orig_w, orig_h = _parse_svg_dimensions(svg_string)

    # Center of original content
    cx = min_x + orig_w / 2
    cy = min_y + orig_h / 2

    # New viewBox centered on rotated/scaled content
    new_min_x = cx - new_width / 2
    new_min_y = cy - new_height / 2

    # Create transform: rotate around center
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
               oblique: bool = False, no_margin: bool = False) -> None:
    """
    Convert SVG string to A4 PDF.

    Args:
        svg_string: SVG content as string
        output_path: Output PDF file path
        landscape: If True, use landscape orientation (default for wide patterns)
        oblique: If True, rotate content to maximize size on page
        no_margin: If True, remove margins for tight bounding box
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

    if no_margin:
        # Recompute viewBox from tight bounding box
        min_x, min_y, width, height = _get_tight_bounds(svg_string)
        new_viewbox = f"{min_x} {min_y} {width} {height}"
        svg_string = re.sub(r'viewBox="[^"]+"', f'viewBox="{new_viewbox}"', svg_string)
        svg_string = re.sub(r'width="[^"]+"', f'width="{width}px"', svg_string)
        svg_string = re.sub(r'height="[^"]+"', f'height="{height}px"', svg_string)

    if oblique:
        # Get content dimensions from SVG
        min_x, min_y, width, height = _parse_svg_dimensions(svg_string)

        # Find optimal angle
        angle, rel_scale = _find_optimal_angle(page_w_mm, page_h_mm)

        if angle > 0.5:  # Only rotate if meaningful improvement
            improvement = (rel_scale - 1) * 100
            print(f"   Rotating {angle:.1f}° for {improvement:.1f}% larger output")

            # Apply rotation and scale
            svg_string = _apply_rotation(svg_string, angle, width, height, rel_scale)

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

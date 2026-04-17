"""
PDF generator for the icosahedron map pattern.

Converts SVG output to A4 PDF format using cairosvg.
"""

import cairosvg

# A4 dimensions in points (72 points per inch)
A4_WIDTH_PT = 595.28  # 210mm
A4_HEIGHT_PT = 841.89  # 297mm

# A4 dimensions in mm
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297


def svg_to_pdf(svg_string: str, output_path: str, landscape: bool = True) -> None:
    """
    Convert SVG string to A4 PDF.

    Args:
        svg_string: SVG content as string
        output_path: Output PDF file path
        landscape: If True, use landscape orientation (default for wide patterns)
    """
    if landscape:
        page_width = A4_HEIGHT_PT  # 297mm in points
        page_height = A4_WIDTH_PT  # 210mm in points
    else:
        page_width = A4_WIDTH_PT
        page_height = A4_HEIGHT_PT

    # Convert SVG to PDF, scaling to fit A4
    cairosvg.svg2pdf(
        bytestring=svg_string.encode('utf-8'),
        write_to=output_path,
        output_width=page_width,
        output_height=page_height,
    )
    print(f"PDF saved to: {output_path}")


def svg_file_to_pdf(svg_path: str, output_path: str, landscape: bool = True) -> None:
    """
    Convert SVG file to A4 PDF.

    Args:
        svg_path: Input SVG file path
        output_path: Output PDF file path
        landscape: If True, use landscape orientation
    """
    with open(svg_path, 'r', encoding='utf-8') as f:
        svg_string = f.read()
    svg_to_pdf(svg_string, output_path, landscape)

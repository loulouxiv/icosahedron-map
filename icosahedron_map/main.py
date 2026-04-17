"""
Main CLI for the icosahedron map generator.

Usage:
    python -m icosahedron_map -r 50m -o map.svg --lat-step 15 --lon-step 15
"""

import argparse
import sys

from .geometry.icosahedron import Icosahedron
from .geometry.unfold import IcosahedronUnfolder
from .projection.gnomonic import create_all_projections
from .projection.face_assignment import FaceAssignment
from .data.downloader import NaturalEarthDownloader
from .utils.clipping import SphericalClipper
from .utils.coloring import assign_country_colors
from .rendering.graticule import GraticuleGenerator
from .rendering.svg_generator import IcosahedronSVGGenerator
from .rendering.pdf_generator import svg_to_pdf


def main():
    parser = argparse.ArgumentParser(
        description="Generate an icosahedron map pattern in SVG or A4 PDF format"
    )
    parser.add_argument(
        '-r', '--resolution',
        choices=['110m', '50m', '10m'],
        default='50m',
        help="Natural Earth data resolution (default: 50m)"
    )
    parser.add_argument(
        '-o', '--output',
        default='icosahedron_map.svg',
        help="Output SVG file (default: icosahedron_map.svg)"
    )
    parser.add_argument(
        '-s', '--scale',
        type=float,
        default=100.0,
        help="Triangle edge length in pixels (default: 100)"
    )
    parser.add_argument(
        '--lat-step',
        type=float,
        default=15.0,
        help="Latitude grid spacing in degrees (default: 15)"
    )
    parser.add_argument(
        '--lon-step',
        type=float,
        default=15.0,
        help="Longitude grid spacing in degrees (default: 15)"
    )
    parser.add_argument(
        '--no-graticule',
        action='store_true',
        help="Disable latitude/longitude grid"
    )
    parser.add_argument(
        '--no-countries',
        action='store_true',
        help="Disable country rendering"
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help="Disable face number labels"
    )
    parser.add_argument(
        '--color-countries',
        action='store_true',
        help="Colorize countries with contrasting colors"
    )
    parser.add_argument(
        '--pdf',
        action='store_true',
        help="Output A4 PDF instead of SVG (landscape orientation)"
    )
    parser.add_argument(
        '--oblique',
        action='store_true',
        help="Rotate PDF output to maximize size on page (~7%% larger)"
    )
    parser.add_argument(
        '--no-margin',
        action='store_true',
        help="Remove margins in PDF output for tight bounding box"
    )

    args = parser.parse_args()

    print("=== Icosahedron Map Generator ===\n")

    # 1. Build icosahedron
    print("1. Building icosahedron geometry...")
    icosahedron = Icosahedron()

    # Print some info about the orientation
    north_vertex = icosahedron.vertices[0]  # Should be at north pole
    lat, lon = icosahedron.vertex_to_latlon(north_vertex)
    print(f"   North pole vertex at: ({lat:.1f}N, {lon:.1f}E)")

    # 2. Create projections for each face
    print("2. Setting up gnomonic projections for 20 faces...")
    face_projections = create_all_projections(icosahedron)

    # 3. Create face assignment
    print("3. Creating face assignment (spherical Voronoi)...")
    face_assignment = FaceAssignment(icosahedron)

    # 4. Create unfolder
    print("4. Computing 2D pattern layout (5-10-5)...")
    unfolder = IcosahedronUnfolder(edge_length=args.scale)

    # 5. Download and process country data
    countries_by_face = None
    country_colors = {}
    if not args.no_countries:
        print(f"5. Loading Natural Earth data ({args.resolution})...")
        try:
            downloader = NaturalEarthDownloader()
            countries_gdf = downloader.load_countries(args.resolution)
            print(f"   Loaded {len(countries_gdf)} countries")

            # Compute country colors if requested
            if args.color_countries:
                print("   Assigning country colors...")
                country_colors = assign_country_colors(countries_gdf)

            # 6. Clip countries to faces
            print("6. Clipping countries to faces...")
            clipper = SphericalClipper(face_assignment, face_projections)
            countries_by_face = clipper.clip_all_countries(countries_gdf)

            total_fragments = sum(len(gdf) for gdf in countries_by_face.values())
            print(f"   Created {total_fragments} country fragments")
        except Exception as e:
            print(f"   Warning: Could not load countries: {e}")
            print("   Continuing without country data...")

    # 7. Generate SVG (or PDF)
    output_format = "PDF" if args.pdf else "SVG"
    print(f"7. Generating {output_format}...")

    # Determine output path
    if args.pdf:
        if args.output == 'icosahedron_map.svg':
            output_path = 'icosahedron_map.pdf'
        elif args.output.lower().endswith('.svg'):
            output_path = args.output[:-4] + '.pdf'
        elif args.output.lower().endswith('.pdf'):
            output_path = args.output
        else:
            output_path = args.output + '.pdf'
        svg_output = None  # Will use in-memory SVG
    else:
        output_path = args.output
        svg_output = args.output

    svg_gen = IcosahedronSVGGenerator(unfolder, face_projections, svg_output or "temp.svg",
                                       country_colors=country_colors)

    # Draw background (ocean)
    svg_gen.draw_face_backgrounds()

    # Draw countries
    if countries_by_face:
        print("   Drawing countries...")
        for face_idx, gdf in countries_by_face.items():
            for _, row in gdf.iterrows():
                svg_gen.draw_country(face_idx, row.geometry, row.get('name', ''))

    # Draw graticule
    if not args.no_graticule:
        print("   Drawing graticule...")
        grat_gen = GraticuleGenerator(lat_step=args.lat_step, lon_step=args.lon_step)
        for face_idx, face_proj in enumerate(face_projections):
            parallels, meridians = grat_gen.generate_for_face(
                face_proj, face_assignment, face_idx
            )
            svg_gen.draw_graticule(face_idx, parallels, meridians)

    # Draw face outlines
    svg_gen.draw_face_outlines()

    # Add labels
    if not args.no_labels:
        svg_gen.add_face_labels()

    # Save
    if args.pdf:
        svg_string = svg_gen.get_svg_string()
        svg_to_pdf(svg_string, output_path, landscape=True, oblique=args.oblique,
                   no_margin=args.no_margin, unfolder=unfolder)
    else:
        svg_gen.save()

    print(f"\n=== Done! Output: {output_path} ===")


if __name__ == "__main__":
    main()

# Icosahedron Map Generator

Generate printable SVG patterns for paper globe construction by projecting world map data onto an icosahedron net.

![Example output](example.png)

## Features

- Projects Natural Earth country data onto 20 icosahedron faces
- Gnomonic projection centered on each face for minimal distortion
- Generates a foldable 5-10-5 triangle layout
- Includes latitude/longitude graticule
- Handles antimeridian crossing and polar regions correctly

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python -m icosahedron_map -o map.svg
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output SVG file | `icosahedron_map.svg` |
| `-r, --resolution` | Natural Earth resolution (110m, 50m, 10m) | `50m` |
| `-s, --scale` | Triangle edge length in pixels | `100` |
| `--lat-step` | Latitude grid spacing in degrees | `15` |
| `--lon-step` | Longitude grid spacing in degrees | `15` |
| `--no-graticule` | Disable latitude/longitude grid | |
| `--no-countries` | Disable country rendering | |
| `--no-labels` | Disable face number labels | |
| `--color-countries` | Colorize countries with contrasting colors | |

### Examples

High resolution with fine graticule:
```bash
python -m icosahedron_map -r 10m -s 200 --lat-step 10 --lon-step 10 -o detailed.svg
```

Colorful map:
```bash
python -m icosahedron_map --color-countries -o colorful.svg
```

Simple outline only:
```bash
python -m icosahedron_map --no-countries --no-labels -o graticule_only.svg
```

## Architecture

```
icosahedron_map/
├── geometry/
│   ├── icosahedron.py   # Icosahedron vertices, faces, and coordinate conversion
│   └── unfold.py        # 2D pattern layout (5-10-5 net)
├── projection/
│   ├── gnomonic.py      # Gnomonic projection per face
│   └── face_assignment.py # Spherical Voronoi face assignment
├── rendering/
│   ├── svg_generator.py # SVG output with layers
│   └── graticule.py     # Lat/lon grid generation
├── utils/
│   ├── clipping.py      # Polygon clipping to face boundaries
│   └── coloring.py      # Graph coloring for country colors
└── data/
    └── downloader.py    # Natural Earth data fetching
```

## How It Works

1. **Icosahedron Construction**: Creates a regular icosahedron with one vertex at the north pole, defining 20 triangular faces.

2. **Face Assignment**: Uses spherical Voronoi partitioning to assign each point on Earth to its nearest face center.

3. **Gnomonic Projection**: Projects each face's portion of the globe onto a tangent plane, preserving straight lines (great circles become straight).

4. **Polygon Clipping**: Clips country polygons to face boundaries, handling antimeridian crossing and polar singularities.

5. **Pattern Unfolding**: Arranges the 20 triangular faces into a flat 5-10-5 pattern suitable for cutting and folding.

## License

MIT

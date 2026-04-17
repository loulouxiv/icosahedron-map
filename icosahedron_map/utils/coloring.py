"""
Graph coloring for country maps.

Assigns colors to countries such that adjacent countries have different colors.
Uses a greedy algorithm with typically 4-6 colors.
"""

import geopandas as gpd
from typing import Dict, List

# Contrasting color palette (colorblind-friendly)
DEFAULT_COLORS = [
    "#e6194B",  # Red
    "#3cb44b",  # Green
    "#ffe119",  # Yellow
    "#4363d8",  # Blue
    "#f58231",  # Orange
    "#911eb4",  # Purple
    "#42d4f4",  # Cyan
    "#f032e6",  # Magenta
]


def build_adjacency(gdf: gpd.GeoDataFrame) -> Dict[int, List[int]]:
    """
    Build adjacency graph from country geometries.

    Two countries are adjacent if their geometries touch or intersect.

    Args:
        gdf: GeoDataFrame with country geometries

    Returns:
        Dict mapping country index to list of adjacent country indices
    """
    adjacency = {i: [] for i in range(len(gdf))}

    for i in range(len(gdf)):
        geom_i = gdf.iloc[i].geometry
        if geom_i is None or geom_i.is_empty:
            continue

        for j in range(i + 1, len(gdf)):
            geom_j = gdf.iloc[j].geometry
            if geom_j is None or geom_j.is_empty:
                continue

            try:
                if geom_i.touches(geom_j) or geom_i.intersects(geom_j):
                    adjacency[i].append(j)
                    adjacency[j].append(i)
            except Exception:
                continue

    return adjacency


def greedy_color(adjacency: Dict[int, List[int]], num_colors: int = 8) -> Dict[int, int]:
    """
    Assign colors using greedy graph coloring.

    Args:
        adjacency: Adjacency graph (node -> list of neighbors)
        num_colors: Maximum number of colors to use

    Returns:
        Dict mapping node index to color index (0 to num_colors-1)
    """
    colors = {}

    # Sort nodes by degree (most constrained first)
    nodes = sorted(adjacency.keys(), key=lambda n: len(adjacency[n]), reverse=True)

    for node in nodes:
        # Find colors used by neighbors
        neighbor_colors = {colors[n] for n in adjacency[node] if n in colors}

        # Assign first available color
        for color in range(num_colors):
            if color not in neighbor_colors:
                colors[node] = color
                break
        else:
            # Fallback if all colors used (shouldn't happen with 8 colors)
            colors[node] = 0

    return colors


def assign_country_colors(countries_gdf: gpd.GeoDataFrame,
                          palette: List[str] = None) -> Dict[str, str]:
    """
    Assign contrasting colors to countries.

    Args:
        countries_gdf: GeoDataFrame with country geometries and 'NAME' column
        palette: Optional list of color hex codes

    Returns:
        Dict mapping country name to color hex code
    """
    if palette is None:
        palette = DEFAULT_COLORS

    # Build adjacency graph
    adjacency = build_adjacency(countries_gdf)

    # Assign color indices
    color_indices = greedy_color(adjacency, len(palette))

    # Map to actual colors
    country_colors = {}
    for i, row in countries_gdf.iterrows():
        name = row.get('NAME', f'Country_{i}')
        idx = countries_gdf.index.get_loc(i) if hasattr(countries_gdf.index, 'get_loc') else i
        color_idx = color_indices.get(idx, 0)
        country_colors[name] = palette[color_idx % len(palette)]

    return country_colors

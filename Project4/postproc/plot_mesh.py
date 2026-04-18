#!/usr/bin/env python3
"""
Plot a .gri mesh, including curved q=2/q=3 edges.

Usage:
    python postproc/plot_mesh.py <mesh.gri> [output.png]

Examples:
    python postproc/plot_mesh.py grids/base.gri report_figures/base_mesh.png
    python postproc/plot_mesh.py grids/base.gri --view leading --output report_figures/base_leading.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dg_utils import map_to_physical, read_gri_mesh


def unique_corner_edges(mesh):
    seen = set()
    unique = []
    for elem_idx, element in enumerate(mesh["elements"]):
        c = element["corners"]
        for edge in ((c[0], c[1]), (c[1], c[2]), (c[2], c[0])):
            key = tuple(sorted(edge))
            if key in seen:
                continue
            seen.add(key)
            unique.append((elem_idx, edge))
    return unique


def directed_ref_edge(element, edge):
    v0, v1, v2 = element["corners"]
    n1, n2 = edge
    if (n1, n2) == (v0, v1):
        return np.array([0.0, 0.0]), np.array([1.0, 0.0])
    if (n1, n2) == (v1, v0):
        return np.array([1.0, 0.0]), np.array([0.0, 0.0])
    if (n1, n2) == (v1, v2):
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    if (n1, n2) == (v2, v1):
        return np.array([0.0, 1.0]), np.array([1.0, 0.0])
    if (n1, n2) == (v2, v0):
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    if (n1, n2) == (v0, v2):
        return np.array([0.0, 0.0]), np.array([0.0, 1.0])
    raise ValueError(f"Edge {edge} not found in element corners {element['corners']}")


def build_edge_segments(mesh, samples_per_edge=31):
    nodes = mesh["nodes"]
    tvals = np.linspace(0.0, 1.0, samples_per_edge)
    segments = []
    for elem_idx, edge in unique_corner_edges(mesh):
        element = mesh["elements"][elem_idx]
        ref0, ref1 = directed_ref_edge(element, edge)
        xy = []
        for t in tvals:
            xi_eta = (1.0 - t) * ref0 + t * ref1
            xy.append(map_to_physical(element, nodes, float(xi_eta[0]), float(xi_eta[1])))
        segments.append(np.asarray(xy))
    return segments


def wall_nodes(mesh):
    wall_edges = next(
        (edges for name, edges in mesh["boundary_groups"].items() if name.lower() == "wall"),
        None,
    )
    if not wall_edges:
        return None
    wall_node_ids = sorted({n for edge in wall_edges for n in edge})
    return mesh["nodes"][wall_node_ids]


def infer_view_box(mesh, view):
    if view == "full":
        return None

    wnodes = wall_nodes(mesh)
    if wnodes is None:
        return None

    x_min = float(np.min(wnodes[:, 0]))
    x_max = float(np.max(wnodes[:, 0]))
    y_mid = 0.5 * float(np.min(wnodes[:, 1]) + np.max(wnodes[:, 1]))
    chord = max(x_max - x_min, 1e-8)

    if view == "leading":
        half_w = 0.08 * chord
        half_h = 0.06 * chord
        return (x_min - half_w, x_min + half_w, y_mid - half_h, y_mid + half_h)
    if view == "trailing":
        half_w = 0.06 * chord
        half_h = 0.05 * chord
        return (x_max - half_w, x_max + half_w, y_mid - half_h, y_mid + half_h)
    return None


def plot_mesh(meshfile, output, view="full", samples_per_edge=31, figscale=1.0, dpi=300):
    mesh = read_gri_mesh(str(meshfile))
    segments = build_edge_segments(mesh, samples_per_edge=samples_per_edge)
    box = infer_view_box(mesh, view)

    fig, ax = plt.subplots(1, 1, figsize=(6.0 * figscale, 5.5 * figscale))
    for seg in segments:
        ax.plot(seg[:, 0], seg[:, 1], color="black", linewidth=0.55)

    ax.set_aspect("equal", adjustable="box")
    if box is not None:
        xmin, xmax, ymin, ymax = box
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    mesh_name = Path(meshfile).name
    ne = len(mesh["elements"])
    qmax = max(elem["q"] for elem in mesh["elements"]) if mesh["elements"] else 1
    ax.set_title(f"{mesh_name} | {ne} elements | q_max={qmax} | view={view}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot a .gri mesh with curved-edge rendering.")
    parser.add_argument("meshfile")
    parser.add_argument("output", nargs="?", default="mesh.png")
    parser.add_argument("--view", choices=["full", "leading", "trailing"], default="full")
    parser.add_argument("--samples-per-edge", type=int, default=31)
    parser.add_argument("--figscale", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_mesh(
        args.meshfile,
        output,
        view=args.view,
        samples_per_edge=args.samples_per_edge,
        figscale=args.figscale,
        dpi=args.dpi,
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

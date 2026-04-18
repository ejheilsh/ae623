#!/usr/bin/env python3
"""
Create a simple two-frame GIF comparing an original .gri mesh against one saved
adapted companion mesh from the AMR run.

Usage:
    python postproc/animate_mesh_refine_compare.py <original_mesh.gri> <adapted_mesh_cycle.bin> [output.gif]

Example:
    python postproc/animate_mesh_refine_compare.py \
        grids/2k.gri \
        data_steady/steady_2k_p0_adjoint_mesh_cycle0.bin \
        postproc_out/2k_refine_compare.gif
"""

from __future__ import annotations

import argparse
import io
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dg_utils import map_to_physical, read_gri_mesh


def read_companion_mesh(filename):
    with open(filename, "rb") as f:
        nn = struct.unpack("i", f.read(4))[0]
        nodes = np.frombuffer(f.read(8 * 2 * nn), dtype=np.float64).reshape(nn, 2)
        ne = struct.unpack("i", f.read(4))[0]
        elements = []
        q_orders = []
        for _ in range(ne):
            q, v0, v1, v2 = struct.unpack("iiii", f.read(16))
            q_orders.append(q)
            elements.append({"q": q, "row": [v0, v1, v2], "corners": [v0, v1, v2]})
    return {
        "nodes": nodes,
        "elements": elements,
        "boundary_groups": {},
        "q_orders": np.asarray(q_orders, dtype=int),
    }


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


def mesh_limits(meshes):
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    for mesh in meshes:
        nodes = mesh["nodes"]
        xmin = min(xmin, float(nodes[:, 0].min()))
        xmax = max(xmax, float(nodes[:, 0].max()))
        ymin = min(ymin, float(nodes[:, 1].min()))
        ymax = max(ymax, float(nodes[:, 1].max()))
    xpad = 0.03 * max(xmax - xmin, 1e-12)
    ypad = 0.03 * max(ymax - ymin, 1e-12)
    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)


def render_frame(mesh, title, xlim, ylim, dpi, figscale):
    segments = build_edge_segments(mesh)
    fig, ax = plt.subplots(figsize=(7.5 * figscale, 5.5 * figscale))
    for seg in segments:
        ax.plot(seg[:, 0], seg[:, 1], color="black", linewidth=0.45)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Make a two-frame mesh comparison GIF.")
    parser.add_argument("original_mesh")
    parser.add_argument("adapted_mesh")
    parser.add_argument("output", nargs="?", default="mesh_refine_compare.gif")
    parser.add_argument("--duration", type=int, default=900, help="Frame duration in ms [default: 900]")
    parser.add_argument("--figscale", type=float, default=1.2, help="Figure size scale factor [default: 1.2]")
    parser.add_argument("--dpi", type=int, default=170, help="Rendered frame DPI [default: 170]")
    args = parser.parse_args()

    orig = read_gri_mesh(args.original_mesh)
    adapted = read_companion_mesh(args.adapted_mesh)
    xlim, ylim = mesh_limits([orig, adapted])

    frames = [
        render_frame(orig, f"Original mesh | {len(orig['elements'])} elements", xlim, ylim, args.dpi, args.figscale),
        render_frame(adapted, f"Adapted mesh | {len(adapted['elements'])} elements", xlim, ylim, args.dpi, args.figscale),
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=[args.duration, args.duration],
        loop=0,
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

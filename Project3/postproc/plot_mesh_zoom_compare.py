#!/usr/bin/env python3
"""
Create report-style mesh comparison plots for q=1/q=2/q=3 .gri meshes.

The script renders actual curved element edges by sampling the geometry map,
so q=2/q=3 edges appear curved rather than just connecting corner vertices.

Example:
  python3 postproc/plot_mesh_zoom_compare.py \
    --meshes grids/2k.gri grids/2k_q2.gri grids/2k_q3.gri \
    --labels q=1 q=2 q=3 \
    --outdir postproc_out/task1/2k
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dg_utils import map_to_physical, read_gri_mesh


def _unique_corner_edges(mesh):
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


def _directed_ref_edge(element, edge):
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


def build_edge_segments(mesh, samples_per_edge=25):
    nodes = mesh["nodes"]
    tvals = np.linspace(0.0, 1.0, samples_per_edge)
    segments = []
    for elem_idx, edge in _unique_corner_edges(mesh):
        element = mesh["elements"][elem_idx]
        ref0, ref1 = _directed_ref_edge(element, edge)
        xy = []
        for t in tvals:
            xi_eta = (1.0 - t) * ref0 + t * ref1
            xy.append(map_to_physical(element, nodes, float(xi_eta[0]), float(xi_eta[1])))
        segments.append(np.array(xy))
    return segments


def wall_nodes(mesh):
    wall_edges = next(
        (edges for name, edges in mesh["boundary_groups"].items() if name.lower() == "wall"),
        None,
    )
    if not wall_edges:
        raise RuntimeError("No wall boundary found in mesh.")
    wall_node_ids = sorted({n for edge in wall_edges for n in edge})
    return mesh["nodes"][wall_node_ids]


def infer_zoom_boxes(mesh):
    wnodes = wall_nodes(mesh)
    x_min = float(np.min(wnodes[:, 0]))
    x_max = float(np.max(wnodes[:, 0]))
    y_mid = 0.5 * float(np.min(wnodes[:, 1]) + np.max(wnodes[:, 1]))
    chord = max(x_max - x_min, 1e-8)

    lead_half_w = 0.08 * chord
    lead_half_h = 0.06 * chord
    trail_half_w = 0.06 * chord
    trail_half_h = 0.05 * chord

    # return {
    #     "full": None,
    #     "leading": (x_min - lead_half_w, x_min + lead_half_w, y_mid - lead_half_h, y_mid + lead_half_h),
    #     "trailing": (x_max - trail_half_w, x_max + trail_half_w, y_mid - trail_half_h, y_mid + trail_half_h),
    # }
    return {
        "full": None,
        "leading": (-1, 2.5, 16, 18.5),
        "trailing": (x_max - trail_half_w, x_max + trail_half_w, y_mid - trail_half_h, y_mid + trail_half_h),
    }


def style_axes(ax, label, box=None):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if box is not None:
        xmin, xmax, ymin, ymax = box
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        x_text = 0.5 * (xmin + xmax)
        y_text = 17.5 if ymin <= 17.5 <= ymax else ymin + 0.78 * (ymax - ymin)
    else:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        x_text = 0.5 * (xmin + xmax)
        y_text = ymin + 0.78 * (ymax - ymin)
    ax.text(
        x_text,
        y_text,
        label,
        ha="center",
        va="center",
        fontsize=18,
        family="serif",
    )


def plot_single_mesh(ax, segments, title, box=None, lw=0.8):
    for seg in segments:
        ax.plot(seg[:, 0], seg[:, 1], color="black", linewidth=lw)
    style_axes(ax, title, box=box)


def save_single_mesh_figure(mesh_path, label, box, outpath, samples_per_edge=25, keep_open=False):
    mesh = read_gri_mesh(str(mesh_path))
    segments = build_edge_segments(mesh, samples_per_edge=samples_per_edge)

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    plot_single_mesh(ax, segments, label, box=box)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    if not keep_open:
        plt.close(fig)


def comparison_figure(mesh_paths, labels, box, outpath, samples_per_edge=25, keep_open=False):
    meshes = [read_gri_mesh(str(p)) for p in mesh_paths]
    segments = [build_edge_segments(mesh, samples_per_edge=samples_per_edge) for mesh in meshes]

    fig, axes = plt.subplots(1, len(meshes), figsize=(4.2 * len(meshes), 4.5))
    if len(meshes) == 1:
        axes = [axes]

    for ax, segs, label in zip(axes, segments, labels):
        plot_single_mesh(ax, segs, label, box=box)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    if not keep_open:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create q=1/q=2/q=3 mesh comparison plots")
    parser.add_argument("--meshes", nargs="+", required=True, help="List of .gri meshes to compare")
    parser.add_argument("--labels", nargs="+", help="Labels for each mesh; defaults to filenames")
    parser.add_argument("--outdir", default="postproc_out/task1", help="Output directory")
    parser.add_argument(
        "--views",
        nargs="+",
        choices=["full", "leading", "trailing"],
        default=["full", "leading", "trailing"],
        help="Which views to generate",
    )
    parser.add_argument("--samples-per-edge", type=int, default=31, help="Geometry samples per edge")
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening interactive windows")
    args = parser.parse_args()

    mesh_paths = [Path(m) for m in args.meshes]
    labels = args.labels if args.labels is not None else [p.stem for p in mesh_paths]
    if len(labels) != len(mesh_paths):
      raise SystemExit("--labels must match --meshes in length")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    reference_mesh = read_gri_mesh(str(mesh_paths[0]))
    zoom_boxes = infer_zoom_boxes(reference_mesh)

    for view in args.views:
        for mesh_path, label in zip(mesh_paths, labels):
            safe_label = (
                label.replace("$", "")
                .replace("\\", "")
                .replace(" ", "_")
                .replace("=", "")
            )
            stem_parts = mesh_path.stem.split("_")
            mesh_base = stem_parts[0]
            outpath = outdir / f"mesh_{mesh_base}_{safe_label}_{view}.png"
            save_single_mesh_figure(
                mesh_path,
                label,
                zoom_boxes[view],
                outpath,
                samples_per_edge=args.samples_per_edge,
                keep_open=not args.no_show,
            )
            print(f"Saved {outpath}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

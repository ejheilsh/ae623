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


def _polyline_point_distances(points, polyline):
    dmin = np.full(points.shape[0], np.inf)
    for i in range(len(polyline) - 1):
        a = polyline[i]
        b = polyline[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-16:
            cand = np.linalg.norm(points - a, axis=1)
        else:
            t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
            proj = a + t[:, None] * ab
            cand = np.linalg.norm(points - proj, axis=1)
        dmin = np.minimum(dmin, cand)
    return dmin


def load_blade_reference(mesh=None):
    candidates = [
        (Path("data/bladeupper.txt"), Path("data/bladelower.txt")),
        (Path("../Project1/data/bladeupper.txt"), Path("../Project1/data/bladelower.txt")),
    ]
    for up_path, lo_path in candidates:
        if not up_path.exists() or not lo_path.exists():
            continue
        up = np.loadtxt(up_path)
        lo = np.loadtxt(lo_path)
        le_idx = np.argmin(up[:, 0])
        sx, sy = up[le_idx, 0], up[le_idx, 1]
        upper = np.column_stack((up[:, 0] - sx, up[:, 1] - sy))
        lower = np.column_stack((lo[:, 0] - sx, lo[:, 1] - sy + 18.0))
        branches = {
            "upper": upper,
            "upper_shifted": upper + np.array([0.0, 18.0]),
            "lower": lower,
            "lower_shifted": lower - np.array([0.0, 18.0]),
        }
        selected_names = ["upper", "upper_shifted"]
        if mesh is not None:
            wall_pts = wall_nodes(mesh)
            names = list(branches.keys())
            dmat = np.vstack([_polyline_point_distances(wall_pts, branches[name]) for name in names])
            closest = np.argmin(dmat, axis=0)
            counts = np.bincount(closest, minlength=len(names))
            ranked = sorted(
                range(len(names)),
                key=lambda idx: (-counts[idx], float(np.mean(dmat[idx]))),
            )
            selected_names = [names[idx] for idx in ranked[:2]]
        return {
            "curves": [branches[name] for name in selected_names],
            "names": selected_names,
        }
    return None


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


def high_order_nodes(mesh, include_corners=False):
    node_ids = set()
    corner_ids = set()
    for element in mesh["elements"]:
        row = element["row"]
        corners = set(element["corners"])
        corner_ids.update(corners)
        if element["q"] <= 1:
            continue
        for nid in row:
            node_ids.add(nid)
    if not include_corners:
        node_ids.difference_update(corner_ids)
    if not node_ids:
        return np.empty((0, 2))
    return mesh["nodes"][sorted(node_ids)]


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


def plot_single_mesh(ax, mesh, segments, title, box=None, lw=0.8, blade_ref=None,
                     show_ho_nodes=False, show_ho_corners=False,
                     show_wall_nodes=False):
    for seg in segments:
        ax.plot(seg[:, 0], seg[:, 1], color="black", linewidth=lw)
    if blade_ref is not None:
        for curve in blade_ref["curves"]:
            ax.plot(curve[:, 0], curve[:, 1], color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    if show_ho_nodes:
        ho = high_order_nodes(mesh, include_corners=show_ho_corners)
        if ho.size:
            ax.scatter(
                ho[:, 0],
                ho[:, 1],
                s=14,
                color="blue",
                edgecolors="none",
                zorder=4,
            )
    if show_wall_nodes:
        mesh_wall_nodes = wall_nodes(mesh)
        ax.scatter(
            mesh_wall_nodes[:, 0],
            mesh_wall_nodes[:, 1],
            s=18,
            color="magenta",
            edgecolors="none",
            zorder=5,
        )
    style_axes(ax, title, box=box)


def save_single_mesh_figure(mesh_path, label, box, outpath, samples_per_edge=25,
                            keep_open=False, blade_ref=None,
                            show_ho_nodes=False, show_ho_corners=False,
                            show_wall_nodes=False):
    mesh = read_gri_mesh(str(mesh_path))
    segments = build_edge_segments(mesh, samples_per_edge=samples_per_edge)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
    plot_single_mesh(
        ax,
        mesh,
        segments,
        label,
        box=box,
        blade_ref=blade_ref,
        show_ho_nodes=show_ho_nodes,
        show_ho_corners=show_ho_corners,
        show_wall_nodes=show_wall_nodes,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=600, bbox_inches="tight")
    if not keep_open:
        plt.close(fig)


def comparison_figure(mesh_paths, labels, box, outpath, samples_per_edge=25,
                      keep_open=False, blade_ref=None,
                      show_ho_nodes=False, show_ho_corners=False,
                      show_wall_nodes=False):
    meshes = [read_gri_mesh(str(p)) for p in mesh_paths]
    segments = [build_edge_segments(mesh, samples_per_edge=samples_per_edge) for mesh in meshes]

    fig, axes = plt.subplots(1, len(meshes), figsize=(6.0 * len(meshes), 6.5))
    if len(meshes) == 1:
        axes = [axes]

    for ax, mesh, segs, label in zip(axes, meshes, segments, labels):
        plot_single_mesh(
            ax,
            mesh,
            segs,
            label,
            box=box,
            blade_ref=blade_ref,
            show_ho_nodes=show_ho_nodes,
            show_ho_corners=show_ho_corners,
            show_wall_nodes=show_wall_nodes,
        )

    fig.tight_layout()
    fig.savefig(outpath, dpi=600, bbox_inches="tight")
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
    parser.add_argument("--show-blade", action="store_true", help="Overlay the blade spline reference if available")
    parser.add_argument(
        "--show-ho-nodes",
        action="store_true",
        help="Overlay high-order geometry nodes (e.g. q=2 midside nodes)",
    )
    parser.add_argument(
        "--show-ho-corners",
        action="store_true",
        help="When used with --show-ho-nodes, include corner nodes too",
    )
    parser.add_argument(
        "--show-wall-nodes",
        action="store_true",
        help="Overlay wall boundary nodes",
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
    blade_ref = load_blade_reference(reference_mesh) if args.show_blade else None

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
                blade_ref=blade_ref,
                show_ho_nodes=args.show_ho_nodes,
                show_ho_corners=args.show_ho_corners,
                show_wall_nodes=args.show_wall_nodes,
            )
            print(f"Saved {outpath}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

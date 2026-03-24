#!/usr/bin/env python3
"""
Create a report-style entropy contour plot for one steady or unsteady solution file.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable

from dg_utils import (
    build_dg_triangulation,
    element_polygon,
    infer_dg_filename,
    maybe_read_dg_results,
    primitive_from_state,
    read_cell_results,
    read_gri_mesh,
)


def plot_entropy_contour(
    meshfile,
    resultsfile,
    output_file="entropy_contour.png",
    show_plot=True,
    cmap="viridis",
    vmin=None,
    vmax=None,
    title=None,
    xlim=None,
    ylim=None,
    dpi=300,
    colorbar_output_file=None,
    label_text=None,
):
    mesh = read_gri_mesh(meshfile)
    dg_filename, U_dg, p_order, _ = maybe_read_dg_results(resultsfile)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    if U_dg is not None:
        x, y, tris, entropy_log = build_dg_triangulation(mesh, U_dg, p_order, "entropy")
        entropy = np.exp(entropy_log)
        triang = mtri.Triangulation(x, y, tris)
        artist = ax.tripcolor(
            triang, entropy, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax
        )
        print(f"Using DG coefficients from {dg_filename}")
    else:
        u = read_cell_results(resultsfile)
        entropy = np.exp(primitive_from_state(u)["entropy"])
        verts = [element_polygon(elem, mesh["nodes"]) for elem in mesh["elements"]]
        artist = PolyCollection(verts, cmap=cmap, edgecolors="none")
        artist.set_array(entropy)
        artist.set_clim(vmin, vmax)
        ax.add_collection(artist)
        ax.autoscale()
        print(
            f"No DG coefficient file found; using cell averages. Expected {infer_dg_filename(resultsfile)}"
        )

    ax.set_aspect("equal", adjustable="box")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if title is not None:
        ax.set_title(title)

    if label_text is not None:
        wall_edges = next(
            (edges for name, edges in mesh["boundary_groups"].items() if name.lower() == "wall"),
            None,
        )
        if wall_edges:
            wall_nodes = sorted({node for edge in wall_edges for node in edge})
            x_coords = mesh["nodes"][wall_nodes, 0]
            label_x = 0.5 * (float(x_coords.min()) + float(x_coords.max())) - 1.5
        else:
            label_x = 0.0
        ax.text(
            label_x,
            -5.0,
            label_text,
            ha="center",
            va="bottom",
            fontsize=24,
            family="serif",
            zorder=5,
        )

    ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight", pad_inches=0)
    print(f"Saved {output_file}")

    if colorbar_output_file is not None:
        cfig, cax = plt.subplots(figsize=(8.5, 0.5))
        sm = ScalarMappable(norm=artist.norm, cmap=artist.cmap)
        sm.set_array([])
        cbar = cfig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label("Entropy", fontsize=13)
        cbar.ax.tick_params(labelsize=11)
        cfig.tight_layout()
        cfig.savefig(colorbar_output_file, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved {colorbar_output_file}")
        plt.close(cfig)

    if show_plot:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create a report-style entropy contour plot for one case.")
    parser.add_argument("meshfile")
    parser.add_argument("resultsfile")
    parser.add_argument("--output", default=None)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--xlim", nargs=2, type=float, default=None)
    parser.add_argument("--ylim", nargs=2, type=float, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--label-text", default=None)
    parser.add_argument("--colorbar-output", default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        results_stem = Path(args.resultsfile).stem
        output_path = Path("postproc_out") / "task5" / f"entropy_{results_stem}.png"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    colorbar_output = (
        args.colorbar_output
        if args.colorbar_output is not None
        else str(output_path.with_name(f"{output_path.stem}_colorbar{output_path.suffix}"))
    )

    plot_entropy_contour(
        args.meshfile,
        args.resultsfile,
        output_file=str(output_path),
        show_plot=not args.no_show,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        title=args.title,
        xlim=args.xlim,
        ylim=args.ylim,
        dpi=args.dpi,
        colorbar_output_file=colorbar_output,
        label_text=args.label_text,
    )


if __name__ == "__main__":
    main()

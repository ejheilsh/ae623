#!/usr/bin/env python3
"""Plot a single-overlay Cp figure for multiple p-orders."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
except ImportError:
    sns = None

from dg_utils import integrate_wall_forces, maybe_read_dg_results, read_gri_mesh
from plot_cp import (
    compute_force_coefficients,
    get_boundary_edges_from_gri,
    get_elements_from_gri,
    get_upper_lower_wall_paths,
    read_mesh,
    read_results,
    sample_cellavg_cp_along_path,
    sample_dg_cp_along_path,
)


def _case_label_from_results(resultsfile):
    stem = Path(resultsfile).stem
    if "_p" in stem:
        return f"p={stem.rsplit('_p', 1)[1].split('_', 1)[0]}"
    return stem


def _extract_p_order(label, resultsfile):
    if "p=" in label:
        try:
            return int(label.split("p=", 1)[1].split()[0].split(",")[0])
        except ValueError:
            pass
    stem = Path(resultsfile).stem
    if "_p" in stem:
        try:
            return int(stem.rsplit("_p", 1)[1].split("_", 1)[0])
        except ValueError:
            pass
    return 10**9


def _wall_chord_data(mesh):
    wall_edges = next(
        (edges for name, edges in mesh["boundary_groups"].items() if name.lower() == "wall"),
        None,
    )
    if wall_edges is None:
        raise RuntimeError("No wall boundary found for chord normalization")
    wall_nodes = sorted({node for edge in wall_edges for node in edge})
    x_coords = mesh["nodes"][wall_nodes, 0]
    x0 = float(np.min(x_coords))
    chord = float(np.max(x_coords) - x0)
    if chord <= 0.0:
        raise RuntimeError("Non-positive chord length")
    return x0, chord


def _collect_cp_data(meshfile, resultsfile):
    mesh = read_gri_mesh(meshfile)
    wall_edges = next(
        (edges for name, edges in mesh["boundary_groups"].items() if name.lower() == "wall"),
        None,
    )
    if wall_edges is None:
        raise RuntimeError(f"No wall boundary found in {meshfile}")

    dg_filename, U_dg, p_order, _ = maybe_read_dg_results(resultsfile)
    if U_dg is not None:
        upper_path, lower_path = get_upper_lower_wall_paths(mesh["nodes"], wall_edges)
        x_upper, cp_upper = sample_dg_cp_along_path(mesh, U_dg, p_order, upper_path)
        x_lower, cp_lower = sample_dg_cp_along_path(mesh, U_dg, p_order, lower_path)
        c_x, c_y = integrate_wall_forces(mesh, U_dg, p_order)
        return mesh, x_upper, cp_upper, x_lower, cp_lower, c_x, c_y, dg_filename

    v = read_mesh(meshfile)
    e = get_elements_from_gri(meshfile)
    boundary_groups = get_boundary_edges_from_gri(meshfile)
    u = read_results(resultsfile)
    upper_path, lower_path = get_upper_lower_wall_paths(v, wall_edges)
    x_upper, cp_upper = sample_cellavg_cp_along_path(v, e, u, upper_path)
    x_lower, cp_lower = sample_cellavg_cp_along_path(v, e, u, lower_path)
    c_x, c_y = compute_force_coefficients(v, e, boundary_groups, u)
    return mesh, x_upper, cp_upper, x_lower, cp_lower, c_x, c_y, None


def plot_cp_series(
    meshfile,
    resultsfiles,
    labels=None,
    output_file=None,
    show_plot=True,
    title=None,
    dpi=300,
):
    if labels is None:
        labels = [_case_label_from_results(rf) for rf in resultsfiles]
    if len(labels) != len(resultsfiles):
        raise ValueError("labels and resultsfiles must have the same length")

    ordering = np.argsort([_extract_p_order(label, resultsfile) for label, resultsfile in zip(labels, resultsfiles)])
    resultsfiles = [resultsfiles[i] for i in ordering]
    labels = [labels[i] for i in ordering]

    if output_file is None:
        grid_tag = Path(meshfile).stem
        output_path = Path("postproc_out") / "task3" / f"cp_series_{grid_tag}.png"
    else:
        output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 15,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 16,
            "legend.title_fontsize": 18,
        }
    )

    if sns is not None:
        bright = sns.color_palette("bright", n_colors=6)
        colors = [bright[0], bright[3], bright[2], "black"]
    else:
        tab10 = list(plt.get_cmap("tab10").colors)
        colors = [tab10[0], tab10[3], tab10[2], "black"]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))

    cp_min = np.inf
    cp_max = -np.inf
    x0 = chord = None
    upper_handles = []
    lower_handles = []

    for i, (resultsfile, label) in enumerate(zip(resultsfiles, labels)):
        color = colors[i % len(colors)]
        p_value = _extract_p_order(label, resultsfile)
        mesh, x_upper, cp_upper, x_lower, cp_lower, c_x, c_y, dg_filename = _collect_cp_data(meshfile, resultsfile)
        if x0 is None:
            x0, chord = _wall_chord_data(mesh)

        x_upper = (x_upper - x0) / chord if len(x_upper) else x_upper
        x_lower = (x_lower - x0) / chord if len(x_lower) else x_lower

        if dg_filename is not None:
            print(f"Using DG coefficients from {dg_filename} for p={p_value}")
        print(f"p={p_value}: c_x={c_x:.6f}, c_y={c_y:.6f}")

        if len(x_upper):
            line_u, = ax.plot(x_upper, cp_upper, color=color, linewidth=1.8, linestyle="-", zorder=3, clip_on=False)
            upper_handles.append((line_u, rf"$p={p_value}$"))
            cp_min = min(cp_min, float(np.min(cp_upper)))
            cp_max = max(cp_max, float(np.max(cp_upper)))
        if len(x_lower):
            line_l, = ax.plot(x_lower, cp_lower, color=color, linewidth=1.8, linestyle="--", zorder=3, clip_on=False)
            lower_handles.append((line_l, rf"$p={p_value}$"))
            cp_min = min(cp_min, float(np.min(cp_lower)))
            cp_max = max(cp_max, float(np.max(cp_lower)))

    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$C_p$")
    ax.invert_yaxis()
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_zorder(1)
    ax.spines["bottom"].set_zorder(1)
    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        width=0,
        bottom=False,
        top=False,
        left=False,
        right=False,
    )
    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", pad=10)

    ax.set_xlim(0.0, 1.0)
    if np.isfinite(cp_min) and np.isfinite(cp_max):
        pad = 0.04 * max(cp_max - cp_min, 1e-6)
        ax.set_ylim(cp_max + pad, cp_min - pad)

    if title is not None:
        ax.set_title(title)

    upper_legend = ax.legend(
        [h for h, _ in upper_handles],
        [l for _, l in upper_handles],
        frameon=False,
        title="Upper",
        handlelength=0.9,
        handletextpad=0.7,
        labelspacing=0.18,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        borderaxespad=0.0,
        markerfirst=False,
    )
    ax.add_artist(upper_legend)

    lower_legend = ax.legend(
        [h for h, _ in lower_handles],
        [l for _, l in lower_handles],
        frameon=False,
        title="Lower",
        handlelength=0.9,
        handletextpad=0.7,
        labelspacing=0.18,
        loc="lower right",
        bbox_to_anchor=(1.02, 0.005),
        borderaxespad=0.0,
        markerfirst=False,
    )
    ax.add_artist(lower_legend)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {output_path}")

    if show_plot:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot a combined Cp overlay for multiple cases.")
    parser.add_argument("meshfile")
    parser.add_argument("resultsfiles", nargs="+")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    plot_cp_series(
        args.meshfile,
        args.resultsfiles,
        labels=args.labels,
        output_file=args.output,
        show_plot=not args.no_show,
        title=args.title,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

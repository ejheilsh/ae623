#!/usr/bin/env python3
"""Generate flat, report-style Cp distribution figures for final grid cases."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

mpl_config = Path(tempfile.gettempdir()) / "project4_matplotlib"
mpl_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

from plot_cp_series import _collect_cp_data, _wall_chord_data


DEFAULT_REFERENCE_MESH = Path(
    "/Users/eheilshorn/Courses/AE623/Code/ae623/Project3/grids/128k_q3.gri"
)
DEFAULT_REFERENCE_RESULTS = Path(
    "/Users/eheilshorn/Courses/AE623/Code/ae623/Project3/data_steady/steady_128k_q3_p0_results.bin"
)
DEFAULT_REFERENCE_LABEL = "reference"
STAGE_COLORS = {
    "initial": "#D62728",
    "final": "#2CA02C",
}


@dataclass(frozen=True)
class Snapshot:
    case: str
    label: str
    iteration: int
    meshfile: Path
    resultsfile: Path


def natural_key(value: str) -> list[object]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts]


def infer_iteration(path: Path) -> int:
    match = re.fullmatch(r"iter(\d+)_soln\.bin", path.name)
    if not match:
        raise ValueError(f"Cannot infer iteration from {path}")
    return int(match.group(1))


def flat_case_name(case: str) -> str:
    return case.replace("/", "_")


def normalize_curvature_filter(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip("/")
    if re.fullmatch(r"\d+", value):
        value = f"q{value}"
    if not re.fullmatch(r"q\d+", value):
        raise ValueError("curvature must look like q1, q2, q3, or a number such as 1")
    return value


def discover_snapshots(
    final_grids: Path,
    final_solutions: Path,
    case_filter: str | None,
    curvature_filter: str | None,
    include_initial: bool,
    include_final: bool,
) -> list[Snapshot]:
    by_case: dict[str, list[tuple[int, Path]]] = {}
    for resultsfile in final_solutions.glob("*/*/*/iter*_soln.bin"):
        rel = resultsfile.relative_to(final_solutions)
        case = "/".join(rel.parts[:3])
        if case_filter is not None and case != case_filter.strip("/"):
            continue
        if curvature_filter is not None and rel.parts[1] != curvature_filter:
            continue
        by_case.setdefault(case, []).append((infer_iteration(resultsfile), resultsfile))

    snapshots: list[Snapshot] = []
    for case, entries in sorted(by_case.items(), key=lambda item: natural_key(item[0])):
        entries.sort(key=lambda item: item[0])
        ncells, curvature, solver_order = case.split("/")
        entries = [
            (iteration, resultsfile)
            for iteration, resultsfile in entries
            if (final_grids / ncells / curvature / solver_order / f"iter{iteration}.gri").exists()
        ]
        if not entries:
            print(f"Skipping {case}: no solved iterations have matching final_grids")
            continue
        selected: list[tuple[str, int, Path]] = []
        if include_initial:
            iteration, resultsfile = entries[0]
            selected.append(("initial", iteration, resultsfile))
        if include_final:
            iteration, resultsfile = entries[-1]
            if not selected or selected[-1][1] != iteration:
                selected.append(("final", iteration, resultsfile))

        for label, iteration, resultsfile in selected:
            meshfile = final_grids / ncells / curvature / solver_order / f"iter{iteration}.gri"
            if not meshfile.exists():
                print(f"Skipping {case} {label}: missing {meshfile}")
                continue
            snapshots.append(Snapshot(case, label, iteration, meshfile, resultsfile))

    return snapshots


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 15,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "legend.title_fontsize": 15,
        }
    )


def style_cp_axes(ax) -> None:
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


def normalized_cp_from_files(meshfile: Path, resultsfile: Path) -> dict:
    mesh, x_upper, cp_upper, x_lower, cp_lower, c_x, c_y, dg_filename = _collect_cp_data(
        str(meshfile), str(resultsfile)
    )
    x0, chord = _wall_chord_data(mesh)
    return {
        "x_upper": (x_upper - x0) / chord if len(x_upper) else x_upper,
        "cp_upper": cp_upper,
        "x_lower": (x_lower - x0) / chord if len(x_lower) else x_lower,
        "cp_lower": cp_lower,
        "c_x": c_x,
        "c_y": c_y,
        "dg_filename": dg_filename,
    }


def normalized_cp(snapshot: Snapshot) -> dict:
    return normalized_cp_from_files(snapshot.meshfile, snapshot.resultsfile)


def cp_limits(data_items: list[dict]) -> tuple[float, float]:
    values = []
    for data in data_items:
        if len(data["cp_upper"]):
            values.append(data["cp_upper"])
        if len(data["cp_lower"]):
            values.append(data["cp_lower"])
    if not values:
        return -1.0, 1.0
    values = np.concatenate(values)
    cp_min = float(np.nanmin(values))
    cp_max = float(np.nanmax(values))
    pad = 0.04 * max(cp_max - cp_min, 1.0e-6)
    return cp_max + pad, cp_min - pad


def save_single_cp(
    snapshot: Snapshot,
    data: dict,
    outdir: Path,
    dpi: int,
    title: bool,
    reference_data: dict | None,
    reference_label: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    data_items = [data]
    handles = []
    labels = []
    color = STAGE_COLORS.get(snapshot.label, "#0072B2")
    snapshot_label = snapshot.label
    snapshot_handle = None
    if len(data["x_upper"]):
        snapshot_handle, = ax.plot(
            data["x_upper"],
            data["cp_upper"],
            color=color,
            linewidth=1.8,
            linestyle="-",
            clip_on=False,
        )
    if len(data["x_lower"]):
        lower_line, = ax.plot(
            data["x_lower"],
            data["cp_lower"],
            color=color,
            linewidth=1.8,
            linestyle="--",
            clip_on=False,
        )
        if snapshot_handle is None:
            snapshot_handle = lower_line
    if snapshot_handle is not None:
        handles.append(snapshot_handle)
        labels.append(snapshot_label)

    if reference_data is not None:
        data_items.append(reference_data)
        reference_handle = None
        if len(reference_data["x_upper"]):
            reference_handle, = ax.plot(
                reference_data["x_upper"],
                reference_data["cp_upper"],
                color="black",
                linewidth=2.0,
                linestyle="-",
                zorder=4,
                clip_on=False,
            )
        if len(reference_data["x_lower"]):
            ref_lower_line, = ax.plot(
                reference_data["x_lower"],
                reference_data["cp_lower"],
                color="black",
                linewidth=2.0,
                linestyle="--",
                zorder=4,
                clip_on=False,
            )
            if reference_handle is None:
                reference_handle = ref_lower_line
        if reference_handle is not None:
            handles.append(reference_handle)
            labels.append(reference_label)

    style_cp_axes(ax)
    ax.set_ylim(*cp_limits(data_items))
    if title:
        ax.set_title(f"{snapshot.case} {snapshot.label} iter{snapshot.iteration}")

    if handles:
        stage_legend = ax.legend(
            handles,
            labels,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.015, 0.985),
            borderaxespad=0.0,
            handlelength=1.2,
        )
        ax.add_artist(stage_legend)

    upper_proxy, = ax.plot([], [], color="0.25", linestyle="-", linewidth=1.8, label="Upper")
    lower_proxy, = ax.plot([], [], color="0.25", linestyle="--", linewidth=1.8, label="Lower")
    ax.legend(
        [upper_proxy, lower_proxy],
        ["Upper", "Lower"],
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.02, 0.005),
        borderaxespad=0.0,
        handlelength=1.2,
    )
    fig.tight_layout()

    output = outdir / f"cp_{flat_case_name(snapshot.case)}_{snapshot.label}_iter{snapshot.iteration}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    return output


def save_overlay_cp(
    case: str,
    snapshots: list[Snapshot],
    data_by_snapshot: dict[Snapshot, dict],
    reference_data: dict | None,
    reference_label: str,
    outdir: Path,
    dpi: int,
    title: bool,
) -> Path:
    if sns is not None:
        colors = sns.color_palette("bright", n_colors=max(2, len(snapshots)))
    else:
        colors = list(plt.get_cmap("tab10").colors)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    data_items = []
    handles = []
    labels = []
    for i, snapshot in enumerate(sorted(snapshots, key=lambda snap: snap.iteration)):
        data = data_by_snapshot[snapshot]
        data_items.append(data)
        color = STAGE_COLORS.get(snapshot.label, colors[i % len(colors)])
        label = snapshot.label
        if len(data["x_upper"]):
            line_u, = ax.plot(
                data["x_upper"],
                data["cp_upper"],
                color=color,
                linewidth=1.8,
                linestyle="-",
                clip_on=False,
            )
            handles.append(line_u)
            labels.append(label)
        if len(data["x_lower"]):
            ax.plot(
                data["x_lower"],
                data["cp_lower"],
                color=color,
                linewidth=1.8,
                linestyle="--",
                clip_on=False,
            )

    if reference_data is not None:
        data_items.append(reference_data)
        if len(reference_data["x_upper"]):
            ref_line, = ax.plot(
                reference_data["x_upper"],
                reference_data["cp_upper"],
                color="black",
                linewidth=2.0,
                linestyle="-",
                zorder=4,
                clip_on=False,
            )
            handles.append(ref_line)
            labels.append(reference_label)
        if len(reference_data["x_lower"]):
            ax.plot(
                reference_data["x_lower"],
                reference_data["cp_lower"],
                color="black",
                linewidth=2.0,
                linestyle="--",
                zorder=4,
                clip_on=False,
            )

    style_cp_axes(ax)
    ax.set_ylim(*cp_limits(data_items))
    if title:
        ax.set_title(case)

    if handles:
        stage_legend = ax.legend(
            handles,
            labels,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.015, 0.985),
            borderaxespad=0.0,
            handlelength=1.2,
        )
        ax.add_artist(stage_legend)

    upper_proxy, = ax.plot([], [], color="0.25", linestyle="-", linewidth=1.8, label="Upper")
    lower_proxy, = ax.plot([], [], color="0.25", linestyle="--", linewidth=1.8, label="Lower")
    ax.legend(
        [upper_proxy, lower_proxy],
        ["Upper", "Lower"],
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.02, 0.005),
        borderaxespad=0.0,
        handlelength=1.2,
    )

    fig.tight_layout()
    output = outdir / f"cp_{flat_case_name(case)}_initial_final.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Cp distributions for final grid cases.")
    parser.add_argument("--final-grids", default="final_grids")
    parser.add_argument("--final-solutions", default="final_solutions")
    parser.add_argument("--outdir", default="final_figures")
    parser.add_argument("--case", default=None, help="Optional case filter like 8k/q3/p0")
    parser.add_argument("--curvature", default=None, help="Optional curvature filter like q1")
    parser.add_argument("--initial-only", action="store_true")
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--single-only", action="store_true", help="Skip overlay figures")
    parser.add_argument("--overlay-only", action="store_true", help="Skip separate per-snapshot figures")
    parser.add_argument("--no-reference", action="store_true", help="Do not add the reference solution to overlay plots")
    parser.add_argument("--reference-mesh", default=str(DEFAULT_REFERENCE_MESH))
    parser.add_argument("--reference-results", default=str(DEFAULT_REFERENCE_RESULTS))
    parser.add_argument("--reference-label", default=DEFAULT_REFERENCE_LABEL)
    parser.add_argument("--title", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    include_initial = not args.final_only
    include_final = not args.initial_only
    snapshots = discover_snapshots(
        Path(args.final_grids),
        Path(args.final_solutions),
        args.case,
        normalize_curvature_filter(args.curvature),
        include_initial,
        include_final,
    )
    if not snapshots:
        raise SystemExit("No matching snapshots found")

    configure_style()
    outdir = Path(args.outdir)

    reference_data = None
    if not args.no_reference:
        reference_mesh = Path(args.reference_mesh)
        reference_results = Path(args.reference_results)
        if not reference_mesh.exists():
            raise SystemExit(f"Reference mesh not found: {reference_mesh}")
        if not reference_results.exists():
            raise SystemExit(f"Reference results not found: {reference_results}")
        reference_data = normalized_cp_from_files(reference_mesh, reference_results)
        print(
            f"reference {args.reference_label}: "
            f"c_x={reference_data['c_x']:.6f}, c_y={reference_data['c_y']:.6f}"
        )

    data_by_snapshot = {}
    for snapshot in snapshots:
        data = normalized_cp(snapshot)
        data_by_snapshot[snapshot] = data
        print(
            f"{snapshot.case} {snapshot.label} iter{snapshot.iteration}: "
            f"c_x={data['c_x']:.6f}, c_y={data['c_y']:.6f}"
        )
        if not args.overlay_only:
            save_single_cp(
                snapshot,
                data,
                outdir,
                args.dpi,
                args.title,
                reference_data,
                args.reference_label,
            )

    if not args.no_overlay and not args.single_only:
        by_case: dict[str, list[Snapshot]] = {}
        for snapshot in snapshots:
            by_case.setdefault(snapshot.case, []).append(snapshot)
        for case, case_snapshots in sorted(by_case.items(), key=lambda item: natural_key(item[0])):
            if len(case_snapshots) >= 2:
                save_overlay_cp(
                    case,
                    case_snapshots,
                    data_by_snapshot,
                    reference_data,
                    args.reference_label,
                    outdir,
                    args.dpi,
                    args.title,
                )

    print(f"Processed {len(snapshots)} Cp snapshot(s)")


if __name__ == "__main__":
    main()

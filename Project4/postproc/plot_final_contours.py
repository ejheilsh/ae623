#!/usr/bin/env python3
"""
Generate report-style contours for initial and final adapted-grid cases.

The default pass writes Mach contours for iter0 and the highest available
iteration in each final_solutions/<ncells>/<curvature>/<solver_order>/ case.
"""

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
import matplotlib.tri as mtri
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection

from dg_utils import (
    build_dg_triangulation,
    element_polygon,
    infer_dg_filename,
    maybe_read_dg_results,
    primitive_from_state,
    read_cell_results,
    read_mesh,
)


FIELD_LABELS = {
    "mach": "Mach Number",
    "pressure": "Pressure",
    "aspect_ratio": "Aspect Ratio",
    "adjoint_indicator": r"$\log_{10}(|\epsilon_e|)$",
}

FIELD_CMAPS = {
    "mach": "turbo",
    "pressure": "viridis",
    "aspect_ratio": "magma",
    "adjoint_indicator": "viridis",
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


def case_file_prefix(case: str) -> str:
    ncells, curvature, solver_order = case.split("/")
    if curvature == "q1":
        return f"steady_{ncells}_{solver_order}_"
    return f"steady_{ncells}_{curvature}_{solver_order}_"


def indicator_filename_for_case(case: str, iteration: int) -> str:
    return f"{case_file_prefix(case)}adjoint_indicators_cycle{iteration}.bin"


def indicator_file_for_case(indicator_roots: list[Path], case: str, iteration: int) -> Path:
    filename = indicator_filename_for_case(case, iteration)
    for root in indicator_roots:
        candidate = root / filename
        if candidate.exists():
            return candidate
    return indicator_roots[0] / filename


def case_from_companion_mesh_name(path: Path) -> tuple[str, int] | None:
    match = re.fullmatch(
        r"steady_(?P<ncells>[^_]+)(?:_(?P<curvature>q\d+))?_(?P<order>p\d+)_adjoint_mesh_cycle(?P<cycle>\d+)\.bin",
        path.name,
    )
    if not match:
        return None
    curvature = match.group("curvature") or "q1"
    case = f"{match.group('ncells')}/{curvature}/{match.group('order')}"
    return case, int(match.group("cycle"))


def available_indicator_cycles(indicator_roots: list[Path], case: str) -> list[int]:
    prefix = case_file_prefix(case)
    cycles = set()
    for root in indicator_roots:
        for path in root.glob(f"{prefix}adjoint_indicators_cycle*.bin"):
            match = re.search(r"cycle(\d+)\.bin$", path.name)
            if match:
                cycles.add(int(match.group(1)))
    return sorted(cycles)


def default_indicator_roots() -> list[Path]:
    return [Path("final_cases/indicators"), Path("data_steady")]


def discover_snapshots(
    final_grids: Path,
    final_solutions: Path,
    case_filter: str | None,
    curvature_filter: str | None,
    include_initial: bool,
    include_final: bool,
    field: str,
    indicator_roots: list[Path],
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
        if field == "adjoint_indicator":
            indicator_cycles = [
                iteration
                for iteration in available_indicator_cycles(indicator_roots, case)
                if (final_grids / ncells / curvature / solver_order / f"iter{iteration}.gri").exists()
            ]
            if not indicator_cycles:
                roots = ", ".join(str(root) for root in indicator_roots)
                print(f"Skipping {case}: no indicator files with matching final_grids in {roots}")
                continue
            by_iter = {iteration: resultsfile for iteration, resultsfile in entries}
            selected: list[tuple[str, int, Path]] = []
            if include_initial:
                iteration = indicator_cycles[0]
                selected.append(("initial", iteration, by_iter.get(iteration, entries[0][1])))
            if include_final:
                iteration = indicator_cycles[-1]
                if not selected or selected[-1][1] != iteration:
                    selected.append(("final", iteration, by_iter.get(iteration, entries[-1][1])))
        else:
            selected = []
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
            snapshots.append(
                Snapshot(
                    case=case,
                    label=label,
                    iteration=iteration,
                    meshfile=meshfile,
                    resultsfile=resultsfile,
                )
            )

    return snapshots


def discover_companion_snapshots(
    mesh_snapshot_root: Path,
    case_filter: str | None,
    curvature_filter: str | None,
    include_initial: bool,
    include_final: bool,
    field: str,
    indicator_roots: list[Path],
) -> list[Snapshot]:
    if field in {"mach", "pressure"}:
        raise SystemExit(
            "--mesh-snapshot-root only provides meshes. Use final_grids/final_solutions "
            "for Mach or pressure contours."
        )

    by_case: dict[str, list[tuple[int, Path]]] = {}
    for meshfile in mesh_snapshot_root.glob("*_adjoint_mesh_cycle*.bin"):
        parsed = case_from_companion_mesh_name(meshfile)
        if parsed is None:
            continue
        case, iteration = parsed
        if case_filter is not None and case != case_filter.strip("/"):
            continue
        if curvature_filter is not None and case.split("/")[1] != curvature_filter:
            continue
        by_case.setdefault(case, []).append((iteration, meshfile))

    snapshots: list[Snapshot] = []
    for case, entries in sorted(by_case.items(), key=lambda item: natural_key(item[0])):
        entries.sort(key=lambda item: item[0])
        available_mesh_cycles = {iteration for iteration, _ in entries}
        if field == "adjoint_indicator":
            indicator_cycles = [
                iteration
                for iteration in available_indicator_cycles(indicator_roots, case)
                if iteration in available_mesh_cycles
            ]
            if not indicator_cycles:
                roots = ", ".join(str(root) for root in indicator_roots)
                print(
                    f"Skipping {case}: no indicator files with matching companion "
                    f"mesh snapshots in {roots}"
                )
                continue
            selected_cycles = []
            if include_initial:
                selected_cycles.append(("initial", indicator_cycles[0]))
            if include_final:
                iteration = indicator_cycles[-1]
                if not selected_cycles or selected_cycles[-1][1] != iteration:
                    selected_cycles.append(("final", iteration))
        else:
            selected_cycles = []
            if include_initial:
                selected_cycles.append(("initial", entries[0][0]))
            if include_final:
                iteration = entries[-1][0]
                if not selected_cycles or selected_cycles[-1][1] != iteration:
                    selected_cycles.append(("final", iteration))

        mesh_by_iter = dict(entries)
        for label, iteration in selected_cycles:
            snapshots.append(
                Snapshot(
                    case=case,
                    label=label,
                    iteration=iteration,
                    meshfile=mesh_by_iter[iteration],
                    resultsfile=Path(),
                )
            )

    return snapshots


def field_values(meshfile: Path, resultsfile: Path, field: str) -> np.ndarray:
    mesh = read_mesh(str(meshfile))

    if field == "aspect_ratio":
        return element_aspect_ratios(mesh)

    if field in {"mach", "pressure"}:
        primitive_field = "mach" if field == "mach" else "p"
        _, u_dg, p_order, _ = maybe_read_dg_results(str(resultsfile))
        if u_dg is not None:
            _, _, _, values = build_dg_triangulation(mesh, u_dg, p_order, primitive_field)
            return values

        u_avg = read_cell_results(str(resultsfile))
        return primitive_from_state(u_avg)[primitive_field]

    raise ValueError(f"Unsupported field: {field}")


def read_indicators(filename: Path) -> np.ndarray:
    with filename.open("rb") as f:
        count = np.fromfile(f, dtype=np.int32, count=1)
        if count.size == 0:
            return np.array([], dtype=np.float64)
        return np.fromfile(f, dtype=np.float64, count=int(count[0]))


def indicator_values(mesh: dict, indicator_file: Path) -> np.ndarray:
    indicators = np.abs(read_indicators(indicator_file))
    if len(indicators) != len(mesh["elements"]):
        raise ValueError(
            f"Indicator count mismatch for {indicator_file}: "
            f"{len(indicators)} indicators vs {len(mesh['elements'])} elements"
        )
    return np.log10(np.maximum(indicators, 1.0e-30))


def element_aspect_ratios(mesh: dict) -> np.ndarray:
    values = []
    nodes = mesh["nodes"]
    for element in mesh["elements"]:
        a, b, c = nodes[element["corners"]]
        l0 = np.linalg.norm(b - a)
        l1 = np.linalg.norm(c - b)
        l2 = np.linalg.norm(a - c)
        lmax = max(l0, l1, l2)
        twice_area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if twice_area <= 1.0e-14:
            values.append(np.inf)
            continue
        shortest_altitude = twice_area / max(lmax, 1.0e-14)
        values.append(lmax / max(shortest_altitude, 1.0e-14))
    return np.array(values)


def auto_limits(
    snapshots: list[Snapshot],
    field: str,
    indicator_roots: list[Path],
) -> tuple[float, float]:
    all_values = []
    for snap in snapshots:
        if field == "adjoint_indicator":
            mesh = read_mesh(str(snap.meshfile))
            values = indicator_values(
                mesh,
                indicator_file_for_case(indicator_roots, snap.case, snap.iteration),
            )
        else:
            values = field_values(snap.meshfile, snap.resultsfile, field)
        finite = values[np.isfinite(values)]
        if finite.size:
            all_values.append(finite)

    if not all_values:
        raise RuntimeError("No finite field values found")

    values = np.concatenate(all_values)
    return float(values.min()), float(values.max())


def case_label(snapshot: Snapshot) -> str:
    _, _, solver_order = snapshot.case.split("/")
    return f"{snapshot.case} {snapshot.label} iter{snapshot.iteration} ({solver_order})"


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


def figure_paths(
    outdir: Path,
    snapshot: Snapshot,
    field: str,
    separate_colorbars: bool,
    wrote_shared_colorbar: bool,
    nested_output: bool,
) -> tuple[Path, Path | None]:
    case_name = flat_case_name(snapshot.case)
    stem = f"{field}_{case_name}_{snapshot.label}_iter{snapshot.iteration}"
    if nested_output:
        case_dir = outdir / field / snapshot.case
        output_file = case_dir / f"{snapshot.label}_iter{snapshot.iteration}_{field}.png"
        shared_colorbar = outdir / field / f"{field}_colorbar.png"
    else:
        output_file = outdir / f"{stem}.png"
        shared_colorbar = outdir / f"{field}_colorbar.png"

    colorbar_file = None
    if separate_colorbars:
        colorbar_file = output_file.with_name(f"{output_file.stem}_colorbar.png")
    elif not wrote_shared_colorbar:
        colorbar_file = shared_colorbar
    return output_file, colorbar_file


def plot_field(
    snapshot: Snapshot,
    field: str,
    output_file: Path,
    colorbar_file: Path | None,
    vmin: float | None,
    vmax: float | None,
    cmap: str,
    dpi: int,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    title: str | None,
    label_text: str | None,
    indicator_roots: list[Path],
) -> None:
    mesh = read_mesh(str(snapshot.meshfile))

    fig, ax = plt.subplots(figsize=(10, 5.5))

    if field == "aspect_ratio":
        values = element_aspect_ratios(mesh)
        verts = [element_polygon(elem, mesh["nodes"]) for elem in mesh["elements"]]
        artist = PolyCollection(
            verts,
            cmap=cmap,
            edgecolors=(0.0, 0.0, 0.0, 0.16),
            linewidths=0.05,
        )
        artist.set_array(values)
        artist.set_clim(vmin, vmax)
        ax.add_collection(artist)
        ax.autoscale()
    elif field == "adjoint_indicator":
        values = indicator_values(
            mesh,
            indicator_file_for_case(indicator_roots, snapshot.case, snapshot.iteration),
        )
        verts = [element_polygon(elem, mesh["nodes"]) for elem in mesh["elements"]]
        artist = PolyCollection(verts, cmap=cmap, edgecolors="none")
        artist.set_array(values)
        artist.set_clim(vmin, vmax)
        ax.add_collection(artist)
        ax.autoscale()
    elif field in {"mach", "pressure"}:
        primitive_field = "mach" if field == "mach" else "p"
        dg_filename, u_dg, p_order, _ = maybe_read_dg_results(str(snapshot.resultsfile))
        if u_dg is not None:
            x, y, tris, values = build_dg_triangulation(mesh, u_dg, p_order, primitive_field)
            triang = mtri.Triangulation(x, y, tris)
            artist = ax.tripcolor(triang, values, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
            print(f"Using DG coefficients from {dg_filename}")
        else:
            u_avg = read_cell_results(str(snapshot.resultsfile))
            values = primitive_from_state(u_avg)[primitive_field]
            verts = [element_polygon(elem, mesh["nodes"]) for elem in mesh["elements"]]
            artist = PolyCollection(verts, cmap=cmap, edgecolors="none")
            artist.set_array(values)
            artist.set_clim(vmin, vmax)
            ax.add_collection(artist)
            ax.autoscale()
            print(
                "No DG coefficient file found; using cell averages. "
                f"Expected {infer_dg_filename(str(snapshot.resultsfile))}"
            )
    else:
        raise ValueError(f"Unsupported field: {field}")

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
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved {output_file}")

    if colorbar_file is not None:
        cfig, cax = plt.subplots(figsize=(4.8, 1.0))
        sm = ScalarMappable(norm=artist.norm, cmap=artist.cmap)
        sm.set_array([])
        cbar = cfig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label(FIELD_LABELS[field])
        cfig.tight_layout()
        cfig.savefig(colorbar_file, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(cfig)
        print(f"Saved {colorbar_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create initial/final report-style contours for final adapted-grid cases."
    )
    parser.add_argument("--final-grids", default="final_grids")
    parser.add_argument("--final-solutions", default="final_solutions")
    parser.add_argument(
        "--mesh-snapshot-root",
        default=None,
        help=(
            "Directory containing solver companion mesh snapshots named "
            "steady_<case>_adjoint_mesh_cycle<N>.bin. Useful for plotting "
            "aspect_ratio or adjoint_indicator without exporting .gri files."
        ),
    )
    parser.add_argument(
        "--indicator-root",
        action="append",
        default=None,
        help=(
            "Directory containing *_adjoint_indicators_cycle<N>.bin files. "
            "Repeat to add search roots. Defaults to final_cases/indicators, then data_steady."
        ),
    )
    parser.add_argument(
        "--outdir",
        default="final_figures",
        help="Output figure directory. Defaults to flat LaTeX-friendly files in final_figures.",
    )
    parser.add_argument("--field", default="mach", choices=sorted(FIELD_LABELS))
    parser.add_argument("--case", default=None, help="Optional case filter like 8k/q3/p0")
    parser.add_argument("--curvature", default=None, help="Optional curvature filter like q1")
    parser.add_argument("--initial-only", action="store_true")
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument(
        "--cmap",
        default=None,
        help="Matplotlib colormap name. Defaults: turbo for Mach, viridis for pressure, magma for aspect_ratio.",
    )
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--xlim", nargs=2, type=float, default=None)
    parser.add_argument("--ylim", nargs=2, type=float, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--separate-colorbars",
        action="store_true",
        help="Write a colorbar next to every contour instead of one shared colorbar",
    )
    parser.add_argument(
        "--nested-output",
        action="store_true",
        help="Use the older nested layout: <outdir>/<field>/<case>/figure.png",
    )
    parser.add_argument("--title", action="store_true", help="Add matplotlib titles")
    parser.add_argument("--label", action="store_true", help="Add large case labels below the airfoil")
    args = parser.parse_args()
    cmap = args.cmap or FIELD_CMAPS[args.field]

    include_initial = not args.final_only
    include_final = not args.initial_only
    if args.indicator_root is not None:
        indicator_roots = [Path(root) for root in args.indicator_root]
    else:
        indicator_roots = default_indicator_roots()
        if args.mesh_snapshot_root is not None:
            indicator_roots = [Path(args.mesh_snapshot_root), *indicator_roots]

    if args.mesh_snapshot_root is not None:
        snapshots = discover_companion_snapshots(
            Path(args.mesh_snapshot_root),
            args.case,
            normalize_curvature_filter(args.curvature),
            include_initial,
            include_final,
            args.field,
            indicator_roots,
        )
    else:
        snapshots = discover_snapshots(
            Path(args.final_grids),
            Path(args.final_solutions),
            args.case,
            normalize_curvature_filter(args.curvature),
            include_initial,
            include_final,
            args.field,
            indicator_roots,
        )
    if not snapshots:
        raise SystemExit("No matching snapshots found")

    vmin = args.vmin
    vmax = args.vmax
    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = auto_limits(snapshots, args.field, indicator_roots)
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax
    print(f"{args.field} color scale: vmin={vmin:.6g}, vmax={vmax:.6g}")

    outdir = Path(args.outdir)
    wrote_shared_colorbar = False
    for snap in snapshots:
        output_file, colorbar_file = figure_paths(
            outdir,
            snap,
            args.field,
            args.separate_colorbars,
            wrote_shared_colorbar,
            args.nested_output,
        )
        if colorbar_file is not None and not args.separate_colorbars:
            wrote_shared_colorbar = True

        plot_field(
            snap,
            args.field,
            output_file,
            colorbar_file,
            vmin,
            vmax,
            cmap,
            args.dpi,
            tuple(args.xlim) if args.xlim is not None else None,
            tuple(args.ylim) if args.ylim is not None else None,
            case_label(snap) if args.title else None,
            case_label(snap) if args.label else None,
            indicator_roots,
        )

    print(f"Processed {len(snapshots)} contour snapshot(s)")


if __name__ == "__main__":
    main()

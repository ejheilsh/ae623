#!/usr/bin/env python3
"""
Create a mesh-adaptation animation from saved adjoint adaptation cycles.

Usage:
    python postproc/animate_mesh_adaptation.py <data_dir> [output.gif]

Examples:
    python postproc/animate_mesh_adaptation.py data_steady mesh_adaptation.gif
    python postproc/animate_mesh_adaptation.py data_steady mesh_adaptation.mp4 --show-indicators
    python postproc/animate_mesh_adaptation.py data_steady mesh_frames.gif --dpi 150 --figscale 1.8
"""

import argparse
import io
import os
import re
import struct
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from PIL import Image


def load_blade_reference(upper_path=None, lower_path=None, lower_shift=18.0):
    candidates = []
    if upper_path and lower_path:
        candidates.append((Path(upper_path), Path(lower_path)))
    env_upper = os.environ.get("BLADE_UPPER_PATH")
    env_lower = os.environ.get("BLADE_LOWER_PATH")
    if env_upper and env_lower:
        candidates.append((Path(env_upper), Path(env_lower)))
    candidates.extend(
        [
            (Path("grids/bladeupper.txt"), Path("grids/bladelower.txt")),
            (Path("../Project1/data/bladeupper.txt"), Path("../Project1/data/bladelower.txt")),
            (Path("../Project1_submit/data/bladeupper.txt"), Path("../Project1_submit/data/bladelower.txt")),
            (Path("../Project1_submit 3/data/bladeupper.txt"), Path("../Project1_submit 3/data/bladelower.txt")),
        ]
    )

    for upath, lpath in candidates:
        if upath.is_file() and lpath.is_file():
            upper = np.loadtxt(upath)
            lower = np.loadtxt(lpath)
            idx_shift = np.argmin(upper[:, 0])
            shift_x = upper[idx_shift, 0]
            shift_y = upper[idx_shift, 1]
            upper = upper.copy()
            lower = lower.copy()
            upper[:, 0] -= shift_x
            upper[:, 1] -= shift_y
            lower[:, 0] -= shift_x
            lower[:, 1] -= shift_y
            lower[:, 1] += lower_shift
            return {
                "upper": upper,
                "lower": lower,
                "upper_path": str(upath),
                "lower_path": str(lpath),
                "lower_shift": lower_shift,
                "shift_x": float(shift_x),
                "shift_y": float(shift_y),
            }
    return None


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
            elements.append([v0, v1, v2])
    return nodes, elements, np.asarray(q_orders, dtype=int)


def read_indicators(filename):
    with open(filename, "rb") as f:
        ne = struct.unpack("i", f.read(4))[0]
        return np.frombuffer(f.read(8 * ne), dtype=np.float64)


def discover_cycles(data_dir):
    data_dir = Path(data_dir)
    meshes = {}
    indicators = {}

    for path in data_dir.glob("*adjoint_mesh_cycle*.bin"):
        m = re.search(r"cycle(\d+)\.bin$", path.name)
        if m:
            meshes[int(m.group(1))] = path

    for path in data_dir.glob("*adjoint_indicators_cycle*.bin"):
        m = re.search(r"cycle(\d+)\.bin$", path.name)
        if m:
            indicators[int(m.group(1))] = path

    cycles = sorted(meshes.keys())
    if not cycles:
        raise SystemExit(f"No adjoint mesh cycle files found in {data_dir}")

    return cycles, meshes, indicators


def frame_durations_ms(nframes, duration_ms):
    return [duration_ms] * nframes


def save_mp4(frames, durations_ms, output):
    output = str(Path(output).resolve())
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        concat_path = tmp / "frames.ffconcat"
        with open(concat_path, "w") as f:
            f.write("ffconcat version 1.0\n")
            for i, (frame, duration_ms) in enumerate(zip(frames, durations_ms)):
                frame_path = tmp / f"frame_{i:05d}.png"
                frame.save(frame_path, format="PNG")
                f.write(f"file '{frame_path.name}'\n")
                f.write(f"duration {max(duration_ms / 1000.0, 1e-3):.6f}\n")
            last_frame_path = tmp / f"frame_{len(frames)-1:05d}.png"
            f.write(f"file '{last_frame_path.name}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-safe",
            "0",
            "-f",
            "concat",
            "-i",
            str(concat_path),
            "-vsync",
            "vfr",
            "-pix_fmt",
            "yuv420p",
            output,
        ]
        subprocess.run(cmd, cwd=tmpdir, check=True)


def compute_boundary_edges(elements):
    """Return list of (v0, v1) pairs that appear in exactly one triangle."""
    from collections import Counter
    counts = Counter()
    for tri in elements:
        v = list(tri)
        for i in range(3):
            counts[tuple(sorted([v[i], v[(i + 1) % 3]]))] += 1
    return [e for e, c in counts.items() if c == 1]


def render_frame(nodes, elements, q_orders, cycle, figscale, dpi,
                 xlim, ylim, indicators=None, ind_limits=None, blade=None):
    verts = [nodes[e] for e in elements]
    ne = len(elements)
    n_curved = int(np.count_nonzero(q_orders > 1))
    q_global = int(q_orders.max()) if len(q_orders) else 1

    # Boundary edges — drawn in orange over the mesh to show wall node positions.
    bdry_edges = compute_boundary_edges(elements)
    # Collect unique boundary nodes for scatter plot.
    bdry_node_idx = sorted({v for e in bdry_edges for v in e})
    bdry_nodes = nodes[bdry_node_idx] if bdry_node_idx else np.empty((0, 2))

    if indicators is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14 * figscale, 5 * figscale))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(7 * figscale, 5 * figscale))
        axes = [axes]

    fig.suptitle(f"Adaptation cycle {cycle} | {ne} elements | q_max={q_global} | curved={n_curved}")

    for ax_idx, ax in enumerate(axes):
        is_indicator_panel = (ax_idx == 1 and indicators is not None)

        if is_indicator_panel:
            ind_log = np.log10(np.maximum(indicators, 1e-30))
            vmin, vmax = ind_limits
            pc = PolyCollection(verts, cmap="hot", edgecolors="black", linewidths=0.15)
            pc.set_array(ind_log)
            pc.set_clim(vmin, vmax)
            ax.add_collection(pc)
            fig.colorbar(pc, ax=ax, label="log10(indicator)")
            ax.set_title("Indicators")
        else:
            pc = PolyCollection(verts, facecolors="white", edgecolors="#aaaaaa",
                                linewidths=0.35)
            ax.add_collection(pc)
            ax.set_title("Mesh")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Blade reference splines — true geometry the wall nodes should lie on.
        if blade is not None:
            blade_col = "cyan" if is_indicator_panel else "crimson"
            ax.plot(blade["upper"][:, 0], blade["upper"][:, 1],
                    color=blade_col, lw=1.5, label="blade surface", zorder=4)
            ax.plot(blade["lower"][:, 0], blade["lower"][:, 1],
                    color=blade_col, lw=1.5, zorder=4)

        # Boundary edges — orange so wall snapping quality is easy to see.
        from matplotlib.collections import LineCollection
        segs = [[nodes[e[0]], nodes[e[1]]] for e in bdry_edges]
        if segs:
            lc = LineCollection(segs, colors="darkorange", linewidths=1.4, zorder=3)
            ax.add_collection(lc)

        # Boundary nodes — dots to show exact vertex positions on/near the blade.
        if len(bdry_nodes):
            ax.scatter(bdry_nodes[:, 0], bdry_nodes[:, 1],
                       s=14, color="darkorange", zorder=5)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Create a mesh-adaptation animation from saved cycles.")
    parser.add_argument("data_dir", help="Directory containing *adjoint_mesh_cycle*.bin files")
    parser.add_argument("output", nargs="?", default="mesh_adaptation.gif")
    parser.add_argument("--duration", type=int, default=500,
                        help="Frame duration in ms [default: 500]")
    parser.add_argument("--figscale", type=float, default=1.6,
                        help="Figure size scale factor [default: 1.6]")
    parser.add_argument("--dpi", type=int, default=140,
                        help="Rendered frame DPI [default: 140]")
    parser.add_argument("--show-indicators", action="store_true",
                        help="Plot the saved error indicators beside the mesh")
    parser.add_argument("--show-blade", action="store_true",
                        help="Overlay blade reference curves if bladeupper/bladelower files are available")
    parser.add_argument("--blade-upper", default=None,
                        help="Optional explicit path to bladeupper.txt")
    parser.add_argument("--blade-lower", default=None,
                        help="Optional explicit path to bladelower.txt")
    parser.add_argument("--blade-lower-shift", type=float, default=18.0,
                        help="Vertical shift applied to lower blade surface [default: 18.0]")
    args = parser.parse_args()

    cycles, mesh_files, ind_files = discover_cycles(args.data_dir)
    print(f"Found adaptation cycles: {cycles}")

    mesh_data = {}
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    ind_vmin = np.inf
    ind_vmax = -np.inf

    for cycle in cycles:
        nodes, elements, q_orders = read_companion_mesh(mesh_files[cycle])
        mesh_data[cycle] = (nodes, elements, q_orders)
        xmin = min(xmin, float(nodes[:, 0].min()))
        xmax = max(xmax, float(nodes[:, 0].max()))
        ymin = min(ymin, float(nodes[:, 1].min()))
        ymax = max(ymax, float(nodes[:, 1].max()))

        if args.show_indicators and cycle in ind_files:
            indicators = read_indicators(ind_files[cycle])
            ind_log = np.log10(np.maximum(indicators, 1e-30))
            ind_vmin = min(ind_vmin, float(ind_log.min()))
            ind_vmax = max(ind_vmax, float(ind_log.max()))

    xpad = 0.03 * max(xmax - xmin, 1e-12)
    ypad = 0.03 * max(ymax - ymin, 1e-12)
    xlim = (xmin - xpad, xmax + xpad)
    ylim = (ymin - ypad, ymax + ypad)

    if not np.isfinite(ind_vmin):
        ind_limits = (0.0, 1.0)
    else:
        ind_limits = (ind_vmin, ind_vmax)

    # Always try to load blade reference — show_blade flag kept for back-compat.
    blade = load_blade_reference(
        upper_path=args.blade_upper,
        lower_path=args.blade_lower,
        lower_shift=args.blade_lower_shift,
    )
    if blade is not None:
        print(
            "Using blade reference:",
            blade["upper_path"],
            blade["lower_path"],
            f"(lower shift={blade['lower_shift']})",
        )
    else:
        print("No blade reference files found — blade curves will not be shown.")

    frames = []
    for idx, cycle in enumerate(cycles, start=1):
        print(f"  frame {idx}/{len(cycles)}: cycle {cycle}")
        nodes, elements, q_orders = mesh_data[cycle]
        indicators = None
        if args.show_indicators and cycle in ind_files:
            indicators = read_indicators(ind_files[cycle])
        frames.append(
            render_frame(nodes, elements, q_orders, cycle,
                         args.figscale, args.dpi, xlim, ylim,
                         indicators=indicators, ind_limits=ind_limits, blade=blade)
        )

    durations = frame_durations_ms(len(frames), args.duration)
    suffix = Path(args.output).suffix.lower()
    if suffix == ".mp4":
        save_mp4(frames, durations, args.output)
    else:
        frames[0].save(
            args.output,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

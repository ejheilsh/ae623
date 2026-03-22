#!/usr/bin/env python3
"""
Create an entropy animation from unsteady snapshots.

Usage:
    python postproc/animate_unsteady_entropy.py <meshfile> <results_dir> [output.gif]

Examples:
    python postproc/animate_unsteady_entropy.py grids/2k.gri unsteady_data/p1_2k entropy.gif
    python postproc/animate_unsteady_entropy.py grids/2k.gri unsteady_data/p1_2k entropy.gif --stride 5
    python postproc/animate_unsteady_entropy.py grids/2k.gri unsteady_data/p1_2k entropy.gif --tmin 20 --tmax 40
    python postproc/animate_unsteady_entropy.py grids/2k.gri unsteady_data/p1_2k entropy.gif --playback-scale 0.5
"""

import argparse
import io
import os
import struct
import glob
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import PolyCollection
from PIL import Image

from dg_utils import build_dg_triangulation, maybe_read_dg_results, primitive_from_state, read_gri_mesh, element_polygon


def read_results(filename):
    with open(filename, "rb") as f:
        ne = struct.unpack("i", f.read(4))[0]
        data = struct.unpack("d" * 4 * ne, f.read(8 * 4 * ne))
        return np.array(data).reshape(ne, 4)


def extract_time_from_filename(filename):
    basename = os.path.basename(filename)
    try:
        parts = basename.replace("results_", "").replace(".bin", "").split("_")
        return float(parts[0])
    except Exception:
        return 0.0


def frame_durations_ms(times, playback_scale, fallback_duration):
    if len(times) <= 1:
        return [fallback_duration]

    durations = []
    for i in range(len(times) - 1):
        dt = max(times[i + 1] - times[i], 0.0)
        durations.append(max(int(round(1000.0 * dt * playback_scale)), 20))

    durations.append(durations[-1] if durations else fallback_duration)
    return durations


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
            # Repeat last frame once; concat demuxer uses the last duration on the prior line.
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


def load_entropy_field(mesh, snapshot_file):
    dg_filename, U_dg, p_order, _ = maybe_read_dg_results(snapshot_file)
    if U_dg is not None:
      x, y, tris, entropy_log = build_dg_triangulation(mesh, U_dg, p_order, "entropy")
      return {
          "type": "dg",
          "x": x,
          "y": y,
          "tris": tris,
          "entropy": np.exp(entropy_log),
      }

    u = read_results(snapshot_file)
    entropy = np.exp(primitive_from_state(u)["entropy"])
    verts = [element_polygon(elem, mesh["nodes"]) for elem in mesh["elements"]]
    return {
        "type": "cell",
        "verts": verts,
        "entropy": entropy,
    }


def render_frame(mesh, snapshot_file, entropy_min, entropy_max, figscale, dpi):
    field = load_entropy_field(mesh, snapshot_file)
    t = extract_time_from_filename(snapshot_file)

    fig, ax = plt.subplots(figsize=(9 * figscale, 5 * figscale))
    if field["type"] == "dg":
        triang = mtri.Triangulation(field["x"], field["y"], field["tris"])
        pc = ax.tripcolor(triang, field["entropy"], shading="gouraud", cmap="viridis")
        pc.set_clim(entropy_min, entropy_max)
        fig.colorbar(pc, ax=ax, label=r"$e^s = p/\rho^\gamma$")
    else:
        pc = PolyCollection(field["verts"], cmap="viridis", edgecolors="none")
        pc.set_array(field["entropy"])
        pc.set_clim(entropy_min, entropy_max)
        ax.add_collection(pc)
        ax.autoscale()
        fig.colorbar(pc, ax=ax, label=r"$e^s = p/\rho^\gamma$")

    ax.set_aspect("equal")
    ax.set_title(f"Entropy at t = {t:.6f}")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    # Keep frames in RGB and let Pillow quantize once during GIF save.
    # Pre-quantizing each frame independently exaggerates banding.
    return Image.open(buf).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Create an entropy animation (GIF or MP4) from unsteady snapshots.")
    parser.add_argument("meshfile")
    parser.add_argument("results_dir")
    parser.add_argument("output", nargs="?", default="entropy_animation.gif")
    parser.add_argument("--stride", type=int, default=1,
                        help="Use every Nth snapshot [default: 1]")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit the number of frames after striding")
    parser.add_argument("--duration", type=int, default=250,
                        help="Fallback frame duration in ms for GIF output [default: 250]")
    parser.add_argument("--tmin", type=float, default=None,
                        help="Minimum snapshot time to include")
    parser.add_argument("--tmax", type=float, default=None,
                        help="Maximum snapshot time to include")
    parser.add_argument("--playback-scale", type=float, default=1.0,
                        help="Real seconds per one unit of simulation time [default: 1.0]")
    parser.add_argument("--figscale", type=float, default=2.0,
                        help="Figure size scale factor [default: 2.0]")
    parser.add_argument("--dpi", type=int, default=120,
                        help="Rendered frame DPI [default: 120]")
    args = parser.parse_args()

    mesh = read_gri_mesh(args.meshfile)
    result_files = sorted(
        [f for f in glob.glob(os.path.join(args.results_dir, "results_*.bin")) if not f.endswith("_dg.bin")],
        key=extract_time_from_filename,
    )
    if not result_files:
        raise SystemExit(f"No snapshots found in {args.results_dir}")

    if args.tmin is not None or args.tmax is not None:
        tmin = -np.inf if args.tmin is None else args.tmin
        tmax = np.inf if args.tmax is None else args.tmax
        result_files = [
            f for f in result_files
            if tmin <= extract_time_from_filename(f) <= tmax
        ]
        if not result_files:
            raise SystemExit(
                f"No snapshots found in {args.results_dir} within [{tmin}, {tmax}]"
            )

    result_files = result_files[:: max(args.stride, 1)]
    if args.max_frames is not None:
        result_files = result_files[: args.max_frames]

    print(f"Animating {len(result_files)} snapshots from {args.results_dir}")
    times = [extract_time_from_filename(f) for f in result_files]

    entropy_min = np.inf
    entropy_max = -np.inf
    for snap in result_files:
        field = load_entropy_field(mesh, snap)
        entropy_min = min(entropy_min, float(np.min(field["entropy"])))
        entropy_max = max(entropy_max, float(np.max(field["entropy"])))

    print(f"Entropy range across frames: [{entropy_min:.4f}, {entropy_max:.4f}]")

    frames = []
    for i, snap in enumerate(result_files):
        print(f"  frame {i+1}/{len(result_files)}: {os.path.basename(snap)}")
        frames.append(render_frame(mesh, snap, entropy_min, entropy_max,
                                   args.figscale, args.dpi))

    if not frames:
        raise SystemExit("No frames rendered.")

    durations = frame_durations_ms(times, args.playback_scale, args.duration)
    if len(durations) != len(frames):
        raise SystemExit("Internal error: frame/duration count mismatch.")

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

#!/usr/bin/env python3
# /// script
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pillow",
# ]
# ///
"""
Create a mesh-adaptation animation from saved adjoint adaptation cycles.

Usage:
    python postproc/animate_mesh_adaptation.py <data_dir> [output.gif] [--show-blade] [--show-ar-cleanup]
"""

import argparse
import glob
import os
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for reliability
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import PolyCollection, LineCollection
from PIL import Image

def read_companion_mesh(filename):
    with open(filename, "rb") as f:
        data = f.read()
    offset = 0
    nn = struct.unpack_from("i", data, offset)[0]; offset += 4
    nodes = np.frombuffer(data[offset : offset + 8 * 2 * nn], dtype=np.float64).reshape(nn, 2)
    offset += 8 * 2 * nn
    ne = struct.unpack_from("i", data, offset)[0]; offset += 4
    elements = []
    for _ in range(ne):
        q, v0, v1, v2 = struct.unpack_from("iiii", data, offset); offset += 16
        elements.append([v0, v1, v2])
    
    bdry_edges = []
    bdry_names = {} # index -> name
    if offset < len(data):
        nb = struct.unpack_from("i", data, offset)[0]; offset += 4
        for _ in range(nb):
            v0, v1, bidx = struct.unpack_from("iii", data, offset); offset += 12
            bdry_edges.append((v0, v1, bidx))
        
        if offset < len(data):
            nnames = struct.unpack_from("i", data, offset)[0]; offset += 4
            for i in range(nnames):
                slen = struct.unpack_from("i", data, offset)[0]; offset += 4
                name = data[offset:offset+slen].decode("utf-8")
                bdry_names[i] = name.lower()
                offset += slen
                
    return nodes, elements, bdry_edges, bdry_names

def find_cycles(data_dir):
    pattern = os.path.join(data_dir, "*adjoint_mesh_cycle*.bin")
    files = glob.glob(pattern)
    cycles = []
    for f in files:
        base = os.path.basename(f)
        try:
            c = int(base.split("cycle")[-1].split(".bin")[0])
            cycles.append(c)
        except: pass
    return sorted(list(set(cycles)))

def read_marked_mask(filename):
    with open(filename, "rb") as f:
        ne = struct.unpack("i", f.read(4))[0]
        mask = np.frombuffer(f.read(ne), dtype=np.uint8).astype(bool)
    return mask

def read_indicators(filename):
    with open(filename, "rb") as f:
        data = f.read()
    ne = struct.unpack_from("i", data, 0)[0]
    vals = np.frombuffer(data, dtype=np.float64, count=ne, offset=4)
    return vals

def get_color_info(bidx, name):
    if "wall" in name: return "crimson", "Wall"
    if "inflow" in name: return "dodgerblue", "Inflow"
    if "outflow" in name: return "forestgreen", "Outflow"
    if not name:
        if bidx == 0: return "dodgerblue", "Inflow"
        if bidx == 1: return "forestgreen", "Outflow"
        if bidx == 2: return "crimson", "Wall"
    return "orange", f"Group {bidx} ({name})"

def build_boundary_groups(nodes, bdry_edges, bdry_names):
    groups = {}
    for v0, v1, bidx in bdry_edges:
        if bidx not in groups:
            groups[bidx] = {"segs": [], "name": bdry_names.get(bidx, "")}
        groups[bidx]["segs"].append([nodes[v0], nodes[v1]])
    return groups

def draw_mesh_axes(ax, verts, groups, blade_ref=None, show_blade=False, add_legend=False,
                   scalar_values=None, scalar_norm=None, scalar_cmap='viridis'):
    if scalar_values is not None and len(scalar_values) == len(verts):
        pc = PolyCollection(
            verts,
            array=np.asarray(scalar_values),
            cmap=scalar_cmap,
            norm=scalar_norm,
            edgecolors='gray',
            linewidths=0.2,
        )
    else:
        pc = PolyCollection(verts, facecolors='white', edgecolors='gray', linewidths=0.2)
    ax.add_collection(pc)
    handles = []
    labels = []
    for bidx, info in groups.items():
        color, label = get_color_info(bidx, info["name"])
        lc = LineCollection(info["segs"], colors=color, linewidths=1.5, label=label)
        ax.add_collection(lc)
        if add_legend and label not in labels:
            handles.append(lc)
            labels.append(label)

    if show_blade and blade_ref:
        h1, = ax.plot(blade_ref['ux'], blade_ref['uy'], 'k--', alpha=0.5, label='Blade Ref')
        ax.plot(blade_ref['lx'], blade_ref['ly'], 'k--', alpha=0.5)
        if add_legend and 'Blade Ref' not in labels:
            handles.append(h1)
            labels.append('Blade Ref')

    ax.set_aspect('equal')
    ax.autoscale_view()
    if add_legend and handles:
        ax.legend(handles, labels, loc='upper right')
    return pc

def blade_inset_specs(blade_ref):
    if blade_ref is None:
        return []

    specs = []
    surfaces = [
        ("Top LE", np.column_stack([blade_ref['lx'], blade_ref['ly']])),
        ("Top TE", np.column_stack([blade_ref['lx'], blade_ref['ly']])),
        ("Bottom LE", np.column_stack([blade_ref['ux'], blade_ref['uy']])),
        ("Bottom TE", np.column_stack([blade_ref['ux'], blade_ref['uy']])),
    ]
    for title, pts in surfaces:
        target = pts[np.argmin(pts[:, 0])] if "LE" in title else pts[np.argmax(pts[:, 0])]
        if "LE" in title:
            x_half = 1.45
            y_half = 1.0
        else:
            x_half = 2.0
            y_half = 1.35
        specs.append({
            "title": title,
            "xlim": (target[0] - x_half, target[0] + x_half),
            "ylim": (target[1] - y_half, target[1] + y_half),
        })
    return specs

def render_frame(data_dir, cycle, show_blade=False, blade_ref=None, show_ar_cleanup=False,
                 show_indicators=False, indicator_norm=None):
    mesh_file = glob.glob(os.path.join(data_dir, f"*adjoint_mesh_cycle{cycle}.bin"))[0]
    nodes, elements, bdry_edges, bdry_names = read_companion_mesh(mesh_file)
    
    verts = [nodes[e] for e in elements]
    groups = build_boundary_groups(nodes, bdry_edges, bdry_names)
    indicators = None
    if show_indicators:
        ind_files = glob.glob(os.path.join(data_dir, f"*adjoint_indicators_cycle{cycle}.bin"))
        if ind_files:
            candidate = read_indicators(ind_files[0])
            if len(candidate) == len(elements):
                indicators = np.maximum(np.abs(candidate), 1e-30)
    
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[2.1, 2.0, 2.1],
        height_ratios=[1, 1],
        wspace=0.12,
        hspace=0.18,
    )
    ax = fig.add_subplot(gs[:, 1])
    pc = draw_mesh_axes(
        ax, verts, groups,
        blade_ref=blade_ref,
        show_blade=show_blade,
        add_legend=False,
        scalar_values=indicators,
        scalar_norm=indicator_norm,
    )

    has_ar_overlay = False
    if show_ar_cleanup:
        ar_files = glob.glob(os.path.join(data_dir, f"*adjoint_ar_cleanup_marked_cycle{cycle}.bin"))
        if ar_files:
            marked = read_marked_mask(ar_files[0])
            if len(marked) == len(elements) and np.any(marked):
                marked_verts = [verts[i] for i, is_marked in enumerate(marked) if is_marked]
                ar_pc = PolyCollection(
                    marked_verts,
                    facecolors='gold',
                    edgecolors='darkorange',
                    linewidths=0.6,
                    alpha=0.45,
                    label='AR Cleanup Target',
                )
                ax.add_collection(ar_pc)
                has_ar_overlay = True

    title = f"Cycle {cycle} - Elements: {len(elements)}"
    if indicators is not None:
        title += " | Colored by |eps_e|"
    if has_ar_overlay:
        title += " | AR Cleanup Targets"
    ax.set_title(title)

    if show_blade and blade_ref:
        inset_specs = blade_inset_specs(blade_ref)
        inset_axes_list = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 2]),
        ]
        for iax, spec in zip(inset_axes_list, inset_specs):
            draw_mesh_axes(
                iax, verts, groups,
                blade_ref=blade_ref,
                show_blade=show_blade,
                add_legend=False,
                scalar_values=indicators,
                scalar_norm=indicator_norm,
            )
            iax.set_xlim(*spec["xlim"])
            iax.set_ylim(*spec["ylim"])
            iax.set_xticks([])
            iax.set_yticks([])
            iax.set_facecolor('white')
            for spine in iax.spines.values():
                spine.set_linewidth(1.0)
            iax.set_title(spec["title"], fontsize=11, pad=4.0)

    legend_handles = []
    legend_labels = []
    for bidx, info in groups.items():
        color, label = get_color_info(bidx, info["name"])
        if label in legend_labels:
            continue
        legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=1.5))
        legend_labels.append(label)
    if show_blade and blade_ref:
        legend_handles.append(plt.Line2D([0], [0], color='k', linestyle='--', alpha=0.5, linewidth=1.5))
        legend_labels.append('Blade Ref')
    ax.legend(legend_handles, legend_labels, loc='upper right')
    if indicators is not None:
        cbar = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(r'$|\epsilon_e|$')
    
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    image = np.ascontiguousarray(rgba[:, :, :3])
    plt.close(fig)
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("output", nargs="?", default="mesh_adaptation.gif")
    parser.add_argument("--show-blade", action="store_true")
    parser.add_argument("--show-ar-cleanup", action="store_true")
    parser.add_argument("--show-indicators", action="store_true")
    args = parser.parse_args()

    cycles = find_cycles(args.data_dir)
    print(f"Found adaptation cycles: {cycles}")
    
    blade_ref = None
    indicator_norm = None
    if args.show_indicators:
        indicator_values = []
        for c in cycles:
            ind_files = glob.glob(os.path.join(args.data_dir, f"*adjoint_indicators_cycle{c}.bin"))
            if not ind_files:
                continue
            vals = np.abs(read_indicators(ind_files[0]))
            vals = vals[np.isfinite(vals) & (vals > 0.0)]
            if vals.size:
                indicator_values.append(vals)
        if indicator_values:
            all_vals = np.concatenate(indicator_values)
            vmin = max(np.percentile(all_vals, 5.0), 1e-30)
            vmax = max(np.percentile(all_vals, 99.5), vmin * 10.0)
            indicator_norm = LogNorm(vmin=vmin, vmax=vmax)

    if args.show_blade:
        candidates = [
            ("../Project1/data/bladeupper.txt", "../Project1/data/bladelower.txt"),
            ("data/bladeupper.txt", "data/bladelower.txt"),
        ]
        for up_path, lo_path in candidates:
            if not os.path.exists(up_path) or not os.path.exists(lo_path):
                continue
            print(f"Using blade reference: {up_path} {lo_path}")
            up = np.loadtxt(up_path)
            lo = np.loadtxt(lo_path)
            # Center and shift exactly as MeshRefinement.cpp does
            le_idx = np.argmin(up[:,0])
            sx, sy = up[le_idx, 0], up[le_idx, 1]
            blade_ref = {
                'ux': up[:,0]-sx, 'uy': up[:,1]-sy,
                'lx': lo[:,0]-sx, 'ly': lo[:,1]-sy + 18.0
            }
            break

    frames = []
    for i, c in enumerate(cycles):
        print(f"  frame {i+1}/{len(cycles)}: cycle {c}")
        frames.append(Image.fromarray(render_frame(
            args.data_dir,
            c,
            args.show_blade,
            blade_ref,
            args.show_ar_cleanup,
            args.show_indicators,
            indicator_norm,
        )))
    
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=200, loop=0)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()

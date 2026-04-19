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

def render_frame(data_dir, cycle, show_blade=False, blade_ref=None, show_ar_cleanup=False):
    mesh_file = glob.glob(os.path.join(data_dir, f"*adjoint_mesh_cycle{cycle}.bin"))[0]
    nodes, elements, bdry_edges, bdry_names = read_companion_mesh(mesh_file)
    
    verts = [nodes[e] for e in elements]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    pc = PolyCollection(verts, facecolors='white', edgecolors='gray', linewidths=0.2)
    ax.add_collection(pc)

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
    
    # Boundary coloring logic — dynamic mapping based on names
    # Fallback to standard indices if names are missing
    def get_color_info(bidx, name):
        if "wall" in name: return "crimson", "Wall"
        if "inflow" in name: return "dodgerblue", "Inflow"
        if "outflow" in name: return "forestgreen", "Outflow"
        # Fallback to standard 2k.gri indices if names weren't found
        if not name:
            if bidx == 0: return "dodgerblue", "Inflow"
            if bidx == 1: return "forestgreen", "Outflow"
            if bidx == 2: return "crimson", "Wall"
        return "orange", f"Group {bidx} ({name})"

    groups = {}
    for v0, v1, bidx in bdry_edges:
        if bidx not in groups:
            groups[bidx] = {"segs": [], "name": bdry_names.get(bidx, "")}
        groups[bidx]["segs"].append([nodes[v0], nodes[v1]])

    for bidx, info in groups.items():
        color, label = get_color_info(bidx, info["name"])
        lc = LineCollection(info["segs"], colors=color, linewidths=1.5, label=label)
        ax.add_collection(lc)

    if show_blade and blade_ref:
        ax.plot(blade_ref['ux'], blade_ref['uy'], 'k--', alpha=0.5, label='Blade Ref')
        ax.plot(blade_ref['lx'], blade_ref['ly'], 'k--', alpha=0.5)

    ax.set_aspect('equal')
    ax.autoscale_view()
    title = f"Cycle {cycle} - Elements: {len(elements)}"
    if has_ar_overlay:
        title += " | AR Cleanup Targets"
    ax.set_title(title)
    ax.legend(loc='upper right')
    
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
    args = parser.parse_args()

    cycles = find_cycles(args.data_dir)
    print(f"Found adaptation cycles: {cycles}")
    
    blade_ref = None
    if args.show_blade:
        # Hardcoded relative paths for project structure
        up_path = "../Project1/data/bladeupper.txt"
        lo_path = "../Project1/data/bladelower.txt"
        if os.path.exists(up_path):
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

    frames = []
    for i, c in enumerate(cycles):
        print(f"  frame {i+1}/{len(cycles)}: cycle {c}")
        frames.append(Image.fromarray(render_frame(
            args.data_dir,
            c,
            args.show_blade,
            blade_ref,
            args.show_ar_cleanup,
        )))
    
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=200, loop=0)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()

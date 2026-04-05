"""
plot_q1_q2.py
=======================
q=1 vs q=2 mesh comparison for the turbine-passage uniform refinement sequence.

Strategy for q=2:
  Walk the wall boundary edges as connected chains (graph traversal, no y-split).
  Fit arc-length CubicSplines x(s), y(s) along each chain.
  For every wall edge A->B insert a midpoint by evaluating the spline at the
  midpoint arc-length -- this is the q=2 curved midpoint.
  Render q=1 edges as straight lines; q=2 wall edges as A->M_curved->B arcs.
  Overlay the true wall spline in black for reference.

Uses only utils/plotgri.py read_gri_file for mesh reading (existing code).

Usage:
    python3 plot_q1_q2.py [--meshes coarse 2k 8k] [--out .]
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import CubicSpline

sys.path.insert(0, "utils")
from plotgri import read_gri_file

parser = argparse.ArgumentParser()
parser.add_argument("--meshes", nargs="+", default=["coarse", "2k", "8k"])
parser.add_argument("--out", default=".")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)


def trace_wall_chains(wall_edges_1based):
    if len(wall_edges_1based) == 0:
        return []
    edges = wall_edges_1based - 1
    adj = {}
    for a, b in edges:
        a, b = int(a), int(b)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    visited = set()
    chains = []
    for seed in list(adj.keys()):
        if seed in visited:
            continue
        component = set()
        stack = [seed]
        while stack:
            nn = stack.pop()
            if nn in component:
                continue
            component.add(nn)
            for nb in adj[nn]:
                if nb not in component:
                    stack.append(nb)
        visited |= component
        ep = next((nn for nn in component if len(adj[nn] & component) == 1),
                  next(iter(component)))
        ordered, prev, cur = [ep], None, ep
        while True:
            nxt_set = (adj[cur] & component) - ({prev} if prev is not None else set())
            if not nxt_set:
                break
            nxt = next(iter(nxt_set))
            ordered.append(nxt)
            prev, cur = cur, nxt
        chains.append(np.array(ordered, dtype=int))
    return chains


def chain_splines(nodes_xy, chain):
    pts = nodes_xy[chain]
    ds = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    s /= s[-1]
    return CubicSpline(s, pts[:, 0]), CubicSpline(s, pts[:, 1]), s


def build_q2_lookup(nodes_xy, chains):
    lookup = {}
    for chain in chains:
        sx, sy, s = chain_splines(nodes_xy, chain)
        node_s = {int(chain[i]): s[i] for i in range(len(chain))}
        for i in range(len(chain) - 1):
            a, b = int(chain[i]), int(chain[i + 1])
            sm = 0.5 * (node_s[a] + node_s[b])
            key = (min(a, b), max(a, b))
            lookup[key] = (float(sx(sm)), float(sy(sm)))
    return lookup


def q1_segs(nodes, elements):
    seen, segs = set(), []
    for tri in elements:
        v = tri - 1
        for i in range(3):
            a, b = v[i], v[(i + 1) % 3]
            k = (min(a, b), max(a, b))
            if k not in seen:
                seen.add(k)
                segs.append([nodes[a], nodes[b]])
    return np.array(segs)


def q2_segs(nodes, elements, q2_lookup):
    seen, segs = set(), []
    for tri in elements:
        v = tri - 1
        for i in range(3):
            a, b = v[i], v[(i + 1) % 3]
            k = (min(a, b), max(a, b))
            if k in seen:
                continue
            seen.add(k)
            A, B = nodes[a], nodes[b]
            if k in q2_lookup:
                M = np.array(q2_lookup[k])
                segs.append([A, M])
                segs.append([M, B])
            else:
                segs.append([A, B])
    return np.array(segs)


def chain_curve(nodes_xy, chain, n=400):
    sx, sy, _ = chain_splines(nodes_xy, chain)
    s = np.linspace(0, 1, n)
    return sx(s), sy(s)


def load_mesh(name):
    path = "grids/{}.gri".format(name)
    if not os.path.exists(path):
        return None
    nodes, elements, bgroups, _ = read_gri_file(path)
    wall = bgroups.get("wall", np.empty((0, 2), dtype=int))
    chains = trace_wall_chains(wall)
    lookup = build_q2_lookup(nodes, chains)
    return dict(name=name, nodes=nodes, elements=elements,
                chains=chains, lookup=lookup)


def overlay_splines(ax, m, **kw):
    for ch in m["chains"]:
        cx, cy = chain_curve(m["nodes"], ch)
        ax.plot(cx, cy, **kw)


mesh_names = args.meshes
n = len(mesh_names)

# Figure 1: full domain
# fig1, axes1 = plt.subplots(n, 2, figsize=(14, 5 * n))
# if n == 1:
#     axes1 = axes1[None, :]
# fig1.suptitle("q=1 straight edges vs q=2 curved wall edges",
#               fontsize=13, fontweight="bold")
# 
# for row, name in enumerate(mesh_names):
#     m = load_mesh(name)
#     for col in range(2):
#         ax = axes1[row, col]
#         if m is None:
#             ax.text(0.5, 0.5, "{}.gri not found".format(name),
#                     ha="center", va="center", transform=ax.transAxes)
#             continue
#         if col == 0:
#             segs = q1_segs(m["nodes"], m["elements"])
#             color, label = "#2271b2", "q=1  {}.gri".format(name)
#         else:
#             segs = q2_segs(m["nodes"], m["elements"], m["lookup"])
#             color, label = "#d62728", "q=2  {}.gri".format(name)
#         if len(segs):
#             ax.add_collection(LineCollection(segs, linewidths=0.35, color=color, alpha=0.8))
#         # Only overlay the true spline on the q=2 panel so q=1 edges stay unobscured
#         if col == 1:
#             overlay_splines(ax, m, color="k", lw=0.9, zorder=5)
#         ax.autoscale(); ax.set_aspect("equal")
#         ax.set_title(label, fontsize=11); ax.set_xlabel("x"); ax.set_ylabel("y")
# 
# fig1.tight_layout()
# p1 = os.path.join(args.out, "mesh_q1_vs_q2_full.png")
# fig1.savefig(p1, dpi=150, bbox_inches="tight")
# print("Saved", p1)

# Figure 2: zoomed
ref_name = "2k" if os.path.exists("grids/2k.gri") else mesh_names[0]
ref = load_mesh(ref_name)
ZOOMS = []
if ref is not None:
    chains_sorted = sorted(ref["chains"], key=lambda c: ref["nodes"][c, 1].mean())
    for ch in chains_sorted:
        pts = ref["nodes"][ch]
        pad = 1.5
        ZOOMS.append((pts[:, 0].min() - pad, pts[:, 0].max() + pad,
                      pts[:, 1].min() - pad, pts[:, 1].max() + pad))
if not ZOOMS:
    ZOOMS = [(-1, 20, -14, 5), (-1, 20, 4, 20)]

nz = len(ZOOMS)
fig2, axes2 = plt.subplots(nz, 2 * n, figsize=(7 * n, 5 * nz))
if nz == 1:
    axes2 = axes2[None, :]
if 2 * n == 1:
    axes2 = axes2[:, None]
fig2.suptitle("Near-wall zoom: q=1 (blue) vs q=2 curved (red) | black=true spline", fontsize=12)

for row, (xmin, xmax, ymin, ymax) in enumerate(ZOOMS):
    for col_m, name in enumerate(mesh_names):
        m = load_mesh(name)
        for qi, (is_q2, color) in enumerate([(False, "#2271b2"), (True, "#d62728")]):
            ax = axes2[row, col_m * 2 + qi]
            if m is None:
                ax.text(0.5, 0.5, "not found", ha="center", va="center", transform=ax.transAxes)
                continue
            nd = m["nodes"]
            v0 = m["elements"][:, 0] - 1
            v1 = m["elements"][:, 1] - 1
            v2 = m["elements"][:, 2] - 1
            def in_win(ni, x0=xmin, x1=xmax, y0=ymin, y1=ymax):
                return ((nd[ni, 0] >= x0) & (nd[ni, 0] <= x1) &
                        (nd[ni, 1] >= y0) & (nd[ni, 1] <= y1))
            mask = in_win(v0) | in_win(v1) | in_win(v2)
            E_z = m["elements"][mask]
            segs = q2_segs(nd, E_z, m["lookup"]) if is_q2 else q1_segs(nd, E_z)
            if len(segs):
                ax.add_collection(LineCollection(segs, linewidths=0.7, color=color, alpha=0.9))
            # Only overlay true spline on q=2 panels
            if is_q2:
                for ch in m["chains"]:
                    cx, cy = chain_curve(nd, ch)
                    in_v = (cx >= xmin) & (cx <= xmax) & (cy >= ymin) & (cy <= ymax)
                    if in_v.any():
                        ax.plot(cx, cy, "k-", lw=1.5, zorder=6)
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal")
            ax.set_title("{} {}".format("q=2" if is_q2 else "q=1", name), fontsize=10)
            ax.set_xlabel("x"); ax.set_ylabel("y")

fig2.tight_layout()
p2 = os.path.join(args.out, "mesh_q1_vs_q2_zoom.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
print("Saved", p2)

# Figure 3: refinement sequence
fig3, axes3 = plt.subplots(1, n, figsize=(6 * n, 5))
if n == 1:
    axes3 = [axes3]
fig3.suptitle("Uniform refinement sequence -- q=1 meshes", fontsize=13)
for ax, name in zip(axes3, mesh_names):
    m = load_mesh(name)
    if m is None:
        ax.text(0.5, 0.5, "not found", ha="center", va="center", transform=ax.transAxes)
        continue
    segs = q1_segs(m["nodes"], m["elements"])
    if len(segs):
        ax.add_collection(LineCollection(segs, linewidths=0.2, color="#2271b2", alpha=0.7))
    ax.autoscale(); ax.set_aspect("equal")
    ax.set_title("{}.gri  ({} elems)".format(name, len(m["elements"])), fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
fig3.tight_layout()
p3 = os.path.join(args.out, "mesh_refinement_sequence.png")
fig3.savefig(p3, dpi=150, bbox_inches="tight")
print("Saved", p3)

plt.close("all")
print("Done.")

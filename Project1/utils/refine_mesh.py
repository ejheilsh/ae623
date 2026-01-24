# Pass down the Mesh class

# One of the files has how to get element to element distance

# Take the mesh objects (V, B, E) and then use the function in section 1.5 of the project specs to implement the refinement
# Note: examine edgehash() in E2N.py to fully understand it better

import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import CubicSpline
import numpy as np


class SizingFunction:
    def __init__(self, blade_coords, pitch, h_min=0.05, h_max=2.0, alpha=0.3):
        self.tree = KDTree(blade_coords)
        self.h_min = h_min
        self.h_mid = h_min * 4.0
        self.h_max = h_max
        self.alpha = alpha
        
        # Blade bounds
        self.x_min = np.min(blade_coords[:, 0])
        self.x_max = np.max(blade_coords[:, 0])
        self.chord = self.x_max - self.x_min

        # --- PERIODIC ADDITION ---
        self.pitch = pitch
        # -------------------------

    def get_h(self, point):
        x, y = point
        # Calculate distance to central blade, ghost blade above, and ghost blade below
        d0, _ = self.tree.query([x, y])
        d_up, _ = self.tree.query([x, y - self.pitch])
        d_down, _ = self.tree.query([x, y + self.pitch])
        
        # Use the smallest of the three
        d = min(d0, d_up, d_down)
        
        # Normalized x for LE/TE detection
        xn = (x - self.x_min) / self.chord
        xn = max(0.0, min(1.0, xn))
        
        # h_wall calculation
        h_wall = self.h_min + (self.h_mid - self.h_min) * (4.0 * xn * (1.0 - xn))
        
        # Final Blend
        h = h_wall + (self.h_max - h_wall) * (1.0 - np.exp(-self.alpha * d))
        return h


def mark_edges_for_refinement(V, E, sizing_func, periodic_pairs_nodes):
    nelem = E.shape[0]
    tri_edge_marks = np.zeros((nelem, 3), dtype=bool)
    edge_status = {}
    
    # 1. Create a 0-based partner map for nodes
    p_map = {p[0]-1: p[1]-1 for p in periodic_pairs_nodes}
    p_map.update({p[1]-1: p[0]-1 for p in periodic_pairs_nodes})

    # 2. Pass 1: Initial marking based on sizing function
    for elem_idx in range(nelem):
        nodes = E[elem_idx] 
        local_edges = [(nodes[1], nodes[2]), (nodes[2], nodes[0]), (nodes[0], nodes[1])]
        for local_idx, e_nodes in enumerate(local_edges):
            e_key = tuple(sorted(e_nodes))
            if e_key not in edge_status:
                mid = (V[e_key[0]] + V[e_key[1]]) / 2.0
                edge_status[e_key] = np.linalg.norm(V[e_key[0]] - V[e_key[1]]) > sizing_func.get_h(mid)

    # 3. Pass 2: Symmetry enforcement for periodic edges
    # If a bottom edge is marked, force the top edge to be marked too
    for e_key, is_marked in list(edge_status.items()):
        if is_marked:
            n1, n2 = e_key
            if n1 in p_map and n2 in p_map:
                partner_key = tuple(sorted((p_map[n1], p_map[n2])))
                edge_status[partner_key] = True

    # 4. Final Pass: Apply forced marks back to the triangles
    for elem_idx in range(nelem):
        nodes = E[elem_idx]
        local_edges = [(nodes[1], nodes[2]), (nodes[2], nodes[0]), (nodes[0], nodes[1])]
        for local_idx, e_nodes in enumerate(local_edges):
            if edge_status[tuple(sorted(e_nodes))]:
                tri_edge_marks[elem_idx, local_idx] = True

    marked_edges_list = [edge for edge, marked in edge_status.items() if marked]
    return marked_edges_list, tri_edge_marks

def refine_mesh_locally(Mesh_obj, sf, periodic_pairs_nodes):
    """
    Conforming local refinement with periodic symmetry preserved.

    Inputs
    ------
    Mesh_obj.V : (N,2) float, 0-based nodes
    Mesh_obj.E : (M,3) int,   0-based triangles
    sf         : sizing function with get_h(point)
    periodic_pairs_nodes : list of [nbot, ntop] in 1-based indexing (from .gri)

    Returns
    -------
    V_new, E_new, periodic_pairs_new (pairs are 1-based for writing)
    """
    V = Mesh_obj.V
    E = Mesh_obj.E
    nelem = E.shape[0]

    # Build 0-based periodic node map
    p_map = {p[0]-1: p[1]-1 for p in periodic_pairs_nodes}
    p_map.update({p[1]-1: p[0]-1 for p in periodic_pairs_nodes})

    def tri_edges(tri):
        a, b, c = tri
        return [tuple(sorted((b, c))), tuple(sorted((c, a))), tuple(sorted((a, b)))]

    # 1) Initial edge marking
    split_edges = set()
    for tri in E:
        for e in tri_edges(tri):
            p1, p2 = V[e[0]], V[e[1]]
            mid = 0.5 * (p1 + p2)
            if np.linalg.norm(p1 - p2) > sf.get_h(mid):
                split_edges.add(e)

    # Enforce periodic symmetry on the marked set
    def enforce_periodic_on_edges(edge_set):
        changed = False
        for e in list(edge_set):
            n1, n2 = e
            if n1 in p_map and n2 in p_map:
                pe = tuple(sorted((p_map[n1], p_map[n2])))
                if pe not in edge_set:
                    edge_set.add(pe)
                    changed = True
        return changed

    enforce_periodic_on_edges(split_edges)

    # 2) If any edge in a triangle is split, split ALL 3
    changed = True
    while changed:
        changed = False
        for tri in E:
            edges = tri_edges(tri)
            if any(e in split_edges for e in edges):
                for e in edges:
                    if e not in split_edges:
                        split_edges.add(e)
                        changed = True
        # mirror across periodic too
        if enforce_periodic_on_edges(split_edges):
            changed = True

    # 3) Create midpoints globally, with periodic pairing
    V_list = V.tolist()
    edge_to_mid = {}

    def get_or_make_midpoint(e):
        e = tuple(sorted(e))
        if e in edge_to_mid:
            return edge_to_mid[e]

        n1, n2 = e
        # If this is a periodic boundary edge (both nodes have partners),
        # create BOTH midpoints and pair them in p_map.
        if n1 in p_map and n2 in p_map:
            pe = tuple(sorted((p_map[n1], p_map[n2])))

            if pe in edge_to_mid:
                # partner already created => just create this one and map
                mid = 0.5 * (V[n1] + V[n2])
                idx = len(V_list)
                V_list.append(mid.tolist())
                edge_to_mid[e] = idx

                pidx = edge_to_mid[pe]
                p_map[idx] = pidx
                p_map[pidx] = idx
                return idx

            # create both
            mid  = 0.5 * (V[n1] + V[n2])
            pmid = 0.5 * (V[pe[0]] + V[pe[1]])

            idx  = len(V_list); V_list.append(mid.tolist())
            pidx = len(V_list); V_list.append(pmid.tolist())

            edge_to_mid[e]  = idx
            edge_to_mid[pe] = pidx

            p_map[idx] = pidx
            p_map[pidx] = idx
            return idx

        # non-periodic edge
        mid = 0.5 * (V[n1] + V[n2])
        idx = len(V_list)
        V_list.append(mid.tolist())
        edge_to_mid[e] = idx
        return idx

    # build all midpoints first (order-independent)
    for e in split_edges:
        get_or_make_midpoint(e)

    # 4) Rebuild triangles 
    E_new = []
    for tri in E:
        edges = tri_edges(tri)
        if any(e in split_edges for e in edges):
            m0 = edge_to_mid[edges[0]]  # (b,c)
            m1 = edge_to_mid[edges[1]]  # (c,a)
            m2 = edge_to_mid[edges[2]]  # (a,b)
            a, b, c = tri
            E_new.extend([
                [a, m2, m1],
                [b, m0, m2],
                [c, m1, m0],
                [m0, m1, m2],
            ])
        else:
            E_new.append(tri.tolist())

    V_new = np.array(V_list, dtype=float)
    E_new = np.array(E_new, dtype=int)

    # 5) Output periodic pairs explicitly (1-based)
    periodic_pairs_new = []
    seen = set()
    for i, j in p_map.items():
        if (i, j) in seen or (j, i) in seen:
            continue
        seen.add((i, j))
        # write as (bottom, top) by y
        if V_new[i, 1] <= V_new[j, 1]:
            periodic_pairs_new.append([i + 1, j + 1])
        else:
            periodic_pairs_new.append([j + 1, i + 1])

    # sort by x for readability
    periodic_pairs_new.sort(key=lambda p: V_new[p[0]-1, 0])

    return V_new, E_new, periodic_pairs_new

def plot_sizing_function_on_mesh(V, E, sizing_func, fname=None):
    """
    Optional plotting for visual verification.
    Visualize sizing function h(x,y) evaluated at triangle centroids.

    Parameters
    ----------
    V : (N,2) array of nodes
    E : (M,3) array of elements (0-based)
    sizing_func : instance of SizingFunction
    fname : optional output filename
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import numpy as np

    # Build triangulation
    tri = mtri.Triangulation(V[:, 0], V[:, 1], E)

    # Triangle centroids
    centroids = np.mean(V[E], axis=1)

    # Evaluate sizing function at centroids
    h_vals = np.array([sizing_func.get_h(c) for c in centroids])

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    tpc = ax.tripcolor(
        tri,
        facecolors=h_vals,
        shading="flat",
        cmap="turbo",
        vmin=0.4,
        vmax=1.6,
    )

    # Mesh outline
    ax.triplot(tri, color="k", lw=0.15, alpha=0.3)

    cbar = plt.colorbar(tpc, ax=ax)
    cbar.set_label("Target Element Size h(x,y)")

    ax.set_aspect("equal")
    ax.set_title("Sizing Function Field on Refined Mesh")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)

    plt.tight_layout()

    if fname:
        plt.savefig(fname, dpi=600)
        print(f"Saved {fname}")
    else:
        plt.show()

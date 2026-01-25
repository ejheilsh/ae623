# Pass down the Mesh class

# One of the files has how to get element to element distance

# Take the mesh objects (V, B, E) and then use the function in section 1.5 of the project specs to implement the refinement
# Note: examine edgehash() in E2N.py to fully understand it better

import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import CubicSpline
import numpy as np


class SizingFunction:
    def __init__(
        self,
        blade_coords,
        pitch,
        h_min=0.05,
        h_max=2.0,
        alpha=0.3,
        use_spline_distance=False,
        upper_path="data/bladeupper.txt",
        lower_path="data/bladelower.txt",
        le_te_sigma=None,
        blend_scale=None,
        refine_level=1.0,
    ):
        self.tree = KDTree(blade_coords)
        self.h_min = h_min
        self.h_mid = h_min * 4.0
        self.h_max = h_max
        self.alpha = alpha
        self._plotted_h_vs_d = False
        self.use_spline_distance = use_spline_distance
        self.le_te_sigma = le_te_sigma
        self.blend_scale = blend_scale
        self.refine_level = refine_level
        self._debug_plotted = False
        
        # Blade bounds
        self.x_min = np.min(blade_coords[:, 0])
        self.x_max = np.max(blade_coords[:, 0])
        self.chord = self.x_max - self.x_min

        # --- PERIODIC ADDITION ---
        self.pitch = pitch
        # -------------------------

        # Optional spline-based distance
        if self.use_spline_distance:
            from utils import config

            upper = np.loadtxt(upper_path)
            lower = np.loadtxt(lower_path)
            lower[:, 1] += config.H_box
            x_upper, y_upper = upper[:, 0], upper[:, 1]
            x_lower, y_lower = lower[:, 0], lower[:, 1]

            i_up = np.argsort(x_upper)
            i_lo = np.argsort(x_lower)
            x_upper, y_upper = x_upper[i_up], y_upper[i_up]
            x_lower, y_lower = x_lower[i_lo], y_lower[i_lo]

            self._spline_upper = CubicSpline(x_upper, y_upper, bc_type="natural")
            self._spline_lower = CubicSpline(x_lower, y_lower, bc_type="natural")
            self._x_upper_min = x_upper.min()
            self._x_upper_max = x_upper.max()
            self._x_lower_min = x_lower.min()
            self._x_lower_max = x_lower.max()
            self._le_te_points = np.array(
                [
                    [x_upper[0], y_upper[0]],
                    [x_upper[-1], y_upper[-1]],
                    [x_lower[0], y_lower[0]],
                    [x_lower[-1], y_lower[-1]],
                ],
                dtype=float,
            )
        else:
            self._le_te_points = None

    def get_h(self, point):
        x, y = point
        # Calculate distance to central blade, ghost blade above, and ghost blade below
        if self.use_spline_distance:
            d0 = self._spline_distance(x, y)
            d_up = self._spline_distance(x, y - self.pitch)
            d_down = self._spline_distance(x, y + self.pitch)
        else:
            d0, _ = self.tree.query([x, y])
            d_up, _ = self.tree.query([x, y - self.pitch])
            d_down, _ = self.tree.query([x, y + self.pitch])
        
        # Use the smallest of the three
        d = min(d0, d_up, d_down)
        # d = d0
        # d = d_down
        
        # Distance to nearest LE/TE point (consider periodic images in y)
        if self._le_te_points is not None:
            pts = self._le_te_points
            d0_le = np.linalg.norm(pts - np.array([x, y]), axis=1).min()
            d_up_le = np.linalg.norm(pts - np.array([x, y - self.pitch]), axis=1).min()
            d_down_le = np.linalg.norm(pts - np.array([x, y + self.pitch]), axis=1).min()
            d_le_te = min(d0_le, d_up_le, d_down_le)
            sigma = self.le_te_sigma
            if sigma is None or sigma <= 0.0:
                sigma = max(self.chord * 0.1, 1e-6)
            w = np.exp(-d_le_te / sigma)
            d = (1.0 - w) * d + w * d_le_te

        # Smooth blend to far-field: h_min at wall, h_max by ~pitch distance
        scale = self.blend_scale
        if scale is None or scale <= 0.0:
            scale = max(self.pitch / 3.0, 1e-6)
        blend = np.tanh(d / scale)
        h = self.h_min + (self.h_max - self.h_min) * blend
        h = h / max(self.refine_level, 1e-12)
        h = np.clip(h, self.h_min, self.h_max)

        return h

    def plot_debug(self):
        """
        Debug plots for sizing function (no arguments).
        Plots: h(d) curve, LE/TE weight vs distance, and h along chord midline.
        """
        if self._debug_plotted:
            return

        import matplotlib.pyplot as plt

        # 1) h(d) curve
        d_vals = np.linspace(0.0, max(self.pitch, self.chord, 1.0), 200)
        scale = self.blend_scale
        if scale is None or scale <= 0.0:
            scale = max(self.pitch / 3.0, 1e-6)
        h_curve = self.h_min + (self.h_max - self.h_min) * np.tanh(d_vals / scale)
        h_curve = h_curve / max(self.refine_level, 1e-12)
        h_curve = np.clip(h_curve, self.h_min, self.h_max)

        # 2) LE/TE weight curve
        sigma = self.le_te_sigma
        if sigma is None or sigma <= 0.0:
            sigma = max(self.chord * 0.1, 1e-6)
        d_le_vals = np.linspace(0.0, max(self.chord, 1.0), 200)
        w_vals = np.exp(-d_le_vals / sigma)

        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        ax[0].plot(d_vals, h_curve, "k-", lw=2)
        ax[0].set_title("h(d) blend")
        ax[0].set_xlabel("d")
        ax[0].set_ylabel("h")
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(d_le_vals, w_vals, "b-", lw=2)
        ax[1].set_title("LE/TE weight")
        ax[1].set_xlabel("d_le_te")
        ax[1].set_ylabel("w")
        ax[1].grid(True, alpha=0.3)

        # 3) h along chord midline (if splines available)
        if self.use_spline_distance:
            xs = np.linspace(self._x_upper_min, self._x_upper_max, 200)
            ys = 0.5 * (self._spline_upper(xs) + self._spline_lower(xs))
            h_line = []
            for x, y in zip(xs, ys):
                d = self._spline_distance(x, y)
                d_up = self._spline_distance(x, y - self.pitch)
                d_down = self._spline_distance(x, y + self.pitch)
                d = min(d, d_up, d_down)
                if self._le_te_points is not None:
                    pts = self._le_te_points
                    d0_le = np.linalg.norm(pts - np.array([x, y]), axis=1).min()
                    d_up_le = np.linalg.norm(pts - np.array([x, y - self.pitch]), axis=1).min()
                    d_down_le = np.linalg.norm(pts - np.array([x, y + self.pitch]), axis=1).min()
                    d_le_te = min(d0_le, d_up_le, d_down_le)
                    w = np.exp(-d_le_te / sigma)
                    d = (1.0 - w) * d + w * d_le_te
                h = self.h_min + (self.h_max - self.h_min) * np.tanh(d / scale)
                h = h / max(self.refine_level, 1e-12)
                h = np.clip(h, self.h_min, self.h_max)
                h_line.append(h)
            ax[2].plot(xs, h_line, "g-", lw=2)
            ax[2].set_title("h along midline")
            ax[2].set_xlabel("x")
            ax[2].set_ylabel("h")
            ax[2].grid(True, alpha=0.3)
        else:
            ax[2].axis("off")
            ax[2].text(0.5, 0.5, "use_spline_distance=False", ha="center", va="center")

        plt.tight_layout()
        plt.show()
        self._debug_plotted = True

    def _spline_distance(self, x, y):
        from scipy.optimize import minimize_scalar

        def project_to_spline(xc, yc, spline, x_bracket):
            def f(xi):
                yi = spline(xi)
                return (xi - xc) ** 2 + (yi - yc) ** 2

            res = minimize_scalar(f, bounds=x_bracket, method="bounded")
            return f(res.x)

        d2_u = project_to_spline(
            x, y, self._spline_upper, (self._x_upper_min, self._x_upper_max)
        )
        d2_l = project_to_spline(
            x, y, self._spline_lower, (self._x_lower_min, self._x_lower_max)
        )
        return np.sqrt(min(d2_u, d2_l))


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

    # 4) Rebuild triangles with 0/1/2/3-edge splits
    def angle(p, c, q):
        v1 = p - c
        v2 = q - c
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0.0:
            return 0.0
        cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
        return np.arccos(cosang)

    E_new = []
    for tri in E:
        a, b, c = tri
        edges = tri_edges(tri)  # e0=(b,c), e1=(c,a), e2=(a,b)
        split_flags = [e in split_edges for e in edges]
        nsplit = sum(split_flags)

        if nsplit == 0:
            E_new.append(tri.tolist())
            continue

        if nsplit == 3:
            m0 = edge_to_mid[edges[0]]  # (b,c)
            m1 = edge_to_mid[edges[1]]  # (c,a)
            m2 = edge_to_mid[edges[2]]  # (a,b)
            E_new.extend([
                [a, m2, m1],
                [b, m0, m2],
                [c, m1, m0],
                [m0, m1, m2],
            ])
            continue

        if nsplit == 1:
            if split_flags[0]:  # split (b,c)
                m0 = edge_to_mid[edges[0]]
                E_new.extend([[b, m0, a], [m0, c, a]])
            elif split_flags[1]:  # split (c,a)
                m1 = edge_to_mid[edges[1]]
                E_new.extend([[c, m1, b], [m1, a, b]])
            else:  # split (a,b)
                m2 = edge_to_mid[edges[2]]
                E_new.extend([[a, m2, c], [m2, b, c]])
            continue

        # nsplit == 2
        # Identify the uncut edge between v1 and v2, with v0 the shared vertex
        if not split_flags[0]:
            v1, v2, v0 = b, c, a
            m01 = edge_to_mid[edges[2]]  # (a,b)
            m02 = edge_to_mid[edges[1]]  # (c,a)
        elif not split_flags[1]:
            v1, v2, v0 = c, a, b
            m01 = edge_to_mid[edges[0]]  # (b,c)
            m02 = edge_to_mid[edges[2]]  # (a,b)
        else:
            v1, v2, v0 = a, b, c
            m01 = edge_to_mid[edges[1]]  # (c,a)
            m02 = edge_to_mid[edges[0]]  # (b,c)

        # Small triangle at the shared vertex
        E_new.append([v0, m01, m02])

        # Split the quad along the diagonal that cuts the larger angle
        ang_v1 = angle(V[v0], V[v1], V[v2])
        ang_v2 = angle(V[v0], V[v2], V[v1])
        if ang_v1 >= ang_v2:
            E_new.extend([[v1, v2, m02], [v1, m02, m01]])
        else:
            E_new.extend([[v1, v2, m01], [v2, m02, m01]])

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

def smooth_mesh_periodic_safe(
    V,
    E,
    periodic_pairs_nodes,
    iterations=3,
    omega=0.2,
    max_disp_factor=0.2,
):
    """
    Smooth mesh with periodic nodes allowed to move in x only.
    Non-periodic boundary nodes are fixed; their one-ring neighbors are frozen.
    """
    nnode = V.shape[0]
    neighbors = [set() for _ in range(nnode)]
    edge_counts = {}
    edge_lengths = {}
    for tri in E:
        a, b, c = tri
        for i, j in ((a, b), (b, c), (c, a)):
            neighbors[i].add(j)
            neighbors[j].add(i)
            e = tuple(sorted((i, j)))
            edge_counts[e] = edge_counts.get(e, 0) + 1
            if e not in edge_lengths:
                edge_lengths[e] = np.linalg.norm(V[e[0]] - V[e[1]])

    boundary_nodes = set()
    for (i, j), count in edge_counts.items():
        if count == 1:
            boundary_nodes.add(i)
            boundary_nodes.add(j)

    periodic_nodes = set()
    for p in periodic_pairs_nodes:
        periodic_nodes.add(p[0] - 1)
        periodic_nodes.add(p[1] - 1)

    nonperiodic_boundary = boundary_nodes - periodic_nodes
    frozen_nodes = set(nonperiodic_boundary)
    for i in nonperiodic_boundary:
        frozen_nodes.update(neighbors[i])

    V_new = V.copy()
    for _ in range(iterations):
        V_next = V_new.copy()
        for i in range(nnode):
            if i in frozen_nodes:
                continue
            nbrs = neighbors[i]
            if not nbrs:
                continue
            avg = np.mean(V_new[list(nbrs)], axis=0)
            proposal = (1.0 - omega) * V_new[i] + omega * avg
            V_next[i] = proposal

        # Enforce periodic pairing: average x, keep y fixed
        for p in periodic_pairs_nodes:
            i = p[0] - 1
            j = p[1] - 1
            x_avg = 0.5 * (V_next[i, 0] + V_next[j, 0])
            V_next[i, 0] = x_avg
            V_next[j, 0] = x_avg
            V_next[i, 1] = V_new[i, 1]
            V_next[j, 1] = V_new[j, 1]

        # Limit displacement (x-only for periodic nodes)
        for i in range(nnode):
            if i in frozen_nodes:
                V_next[i] = V_new[i]
                continue
            nbrs = neighbors[i]
            if not nbrs:
                continue
            local_edges = [tuple(sorted((i, j))) for j in nbrs]
            local_h = min(edge_lengths[e] for e in local_edges) if local_edges else 0.0
            max_disp = max_disp_factor * local_h
            if max_disp <= 0.0:
                continue
            disp = V_next[i] - V_new[i]
            if i in periodic_nodes:
                if abs(disp[0]) > max_disp:
                    disp[0] = np.sign(disp[0]) * max_disp
                disp[1] = 0.0
            else:
                disp_norm = np.linalg.norm(disp)
                if disp_norm > max_disp:
                    disp = disp * (max_disp / disp_norm)
            V_next[i] = V_new[i] + disp

        V_new = V_next

    return V_new

def get_wall_boundary_nodes(
    V,
    E,
    x_left,
    x_right,
    y_bottom_left,
    y_top_left,
    y_bottom_right,
    y_top_right,
    airfoil_x_min,
    airfoil_x_max,
    tol_x=0.5,
    tol_y=0.5,
):
    """
    Identify boundary nodes that lie on the wall (airfoil surface) only.
    """
    edge_counts = {}
    for tri in E:
        for i, j in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = tuple(sorted((i, j)))
            edge_counts[e] = edge_counts.get(e, 0) + 1

    wall_nodes = set()
    for (i, j), count in edge_counts.items():
        if count != 1:
            continue
        p1 = V[i]
        p2 = V[j]
        mid_x = 0.5 * (p1[0] + p2[0])
        mid_y = 0.5 * (p1[1] + p2[1])

        if abs(mid_x - x_left) < tol_x or abs(mid_x - x_right) < tol_x:
            continue  # inflow/outflow
        if (airfoil_x_min - tol_x) <= mid_x <= (airfoil_x_max + tol_x):
            wall_nodes.add(i)
            wall_nodes.add(j)
            continue
        if abs(mid_y - y_bottom_left) < tol_y or abs(mid_y - y_bottom_right) < tol_y:
            continue  # periodic bottom
        if abs(mid_y - y_top_left) < tol_y or abs(mid_y - y_top_right) < tol_y:
            continue  # periodic top

    return wall_nodes

def snap_wall_nodes_to_blade(
    V,
    wall_nodes,
    upper_path="data/bladeupper.txt",
    lower_path="data/bladelower.txt",
    le_te_exclude_tol=None,
):
    """
    Snap wall boundary nodes to the blade splines.
    """
    from scipy.optimize import minimize_scalar
    from utils import config

    upper = np.loadtxt(upper_path)
    lower = np.loadtxt(lower_path)
    lower[:, 1] += config.H_box
    x_upper, y_upper = upper[:, 0], upper[:, 1]
    x_lower, y_lower = lower[:, 0], lower[:, 1]

    i_up = np.argsort(x_upper)
    i_lo = np.argsort(x_lower)
    x_upper, y_upper = x_upper[i_up], y_upper[i_up]
    x_lower, y_lower = x_lower[i_lo], y_lower[i_lo]

    spline_upper = CubicSpline(x_upper, y_upper, bc_type="natural")
    spline_lower = CubicSpline(x_lower, y_lower, bc_type="natural")
    chord = x_upper.max() - x_upper.min()
    if le_te_exclude_tol is None:
        le_te_exclude_tol = max(0.05 * chord, 1e-6)

    le_te_pts = np.array(
        [
            [x_upper[0], y_upper[0]],
            [x_upper[-1], y_upper[-1]],
            [x_lower[0], y_lower[0]],
            [x_lower[-1], y_lower[-1]],
        ],
        dtype=float,
    )

    def project_point(xc, yc, spline, x_bracket):
        def f(x):
            y = spline(x)
            return (x - xc) ** 2 + (y - yc) ** 2
        res = minimize_scalar(f, bounds=x_bracket, method="bounded")
        return res.x, f(res.x)

    for i in wall_nodes:
        x, y = V[i]
        # Skip snapping outside blade x-extent (left/right of blade)
        if x < x_upper.min() or x > x_upper.max():
            continue
        # Avoid moving LE/TE endpoints to prevent periodic boundary artifacts
        if np.min(np.linalg.norm(le_te_pts - np.array([x, y]), axis=1)) <= le_te_exclude_tol:
            continue
        xu, d2_u = project_point(x, y, spline_upper, (x_upper.min(), x_upper.max()))
        xl, d2_l = project_point(x, y, spline_lower, (x_lower.min(), x_lower.max()))
        if d2_u <= d2_l:
            V[i, 0] = xu
            V[i, 1] = spline_upper(xu)
        else:
            V[i, 0] = xl
            V[i, 1] = spline_lower(xl)
    return V

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
        # vmin=0.4,
        # vmax=1.6,
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

def plot_mesh_edges_colored_by_h(V, E, sizing_func, fname=None):
    """
    Plot mesh edges with colors based on h evaluated at triangle centroids.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import numpy as np

    centroids = np.mean(V[E], axis=1)
    h_vals = np.array([sizing_func.get_h(c) for c in centroids])

    segments = []
    colors = []
    for tri, h in zip(E, h_vals):
        p0, p1, p2 = V[tri[0]], V[tri[1]], V[tri[2]]
        segments.extend([[p0, p1], [p1, p2], [p2, p0]])
        colors.extend([h, h, h])

    fig, ax = plt.subplots(figsize=(10, 5))
    lc = LineCollection(segments, array=np.array(colors), cmap="turbo", linewidths=0.4)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Mesh Edges Colored by Target Size h")
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label("Target Element Size h(x,y)")
    plt.tight_layout()

    if fname:
        plt.savefig(fname)
        print(f"Saved {fname}")
    else:
        plt.show()

def plot_refinement_edges(V, E, sizing_func, periodic_pairs_nodes, fname=None):
    """
    Plot mesh edges marked for refinement (green) and unmarked edges (red).
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import numpy as np

    marked_edges_list, _ = mark_edges_for_refinement(V, E, sizing_func, periodic_pairs_nodes)
    marked_set = {tuple(sorted(e)) for e in marked_edges_list}

    segments = []
    colors = []
    seen = set()
    for tri in E:
        edges = [(tri[1], tri[2]), (tri[2], tri[0]), (tri[0], tri[1])]
        for e in edges:
            e_key = tuple(sorted(e))
            if e_key in seen:
                continue
            seen.add(e_key)
            p0, p1 = V[e_key[0]], V[e_key[1]]
            segments.append([p0, p1])
            colors.append("green" if e_key in marked_set else "red")

    fig, ax = plt.subplots(figsize=(10, 5))
    lc = LineCollection(segments, colors=colors, linewidths=0.5)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Refinement Edges (green=refine, red=keep)")
    plt.tight_layout()

    if fname:
        plt.savefig(fname)
        print(f"Saved {fname}")
    else:
        plt.show()

def plot_sizing_function_with_centroid_projection(
    V,
    E,
    sizing_func,
    upper_path="data/bladeupper.txt",
    lower_path="data/bladelower.txt",
    fname=None,
):
    """
    Visualize sizing function h(x,y) at centroids and plot centroid projections
    onto the blade splines.
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from scipy.optimize import minimize_scalar
    from utils import config

    # Build triangulation
    tri = mtri.Triangulation(V[:, 0], V[:, 1], E)

    # Triangle centroids
    centroids = np.mean(V[E], axis=1)

    # Evaluate sizing function at centroids
    h_vals = np.array([sizing_func.get_h(c) for c in centroids])

    # Load blade surface points and build splines
    upper = np.loadtxt(upper_path)
    lower = np.loadtxt(lower_path)
    lower[:, 1] += config.H_box
    x_upper, y_upper = upper[:, 0], upper[:, 1]
    x_lower, y_lower = lower[:, 0], lower[:, 1]

    # Sort by x for spline construction
    i_up = np.argsort(x_upper)
    i_lo = np.argsort(x_lower)
    x_upper, y_upper = x_upper[i_up], y_upper[i_up]
    x_lower, y_lower = x_lower[i_lo], y_lower[i_lo]

    spline_upper = CubicSpline(x_upper, y_upper, bc_type="natural")
    spline_lower = CubicSpline(x_lower, y_lower, bc_type="natural")

    def project_point_to_spline(xc, yc, spline, x_bracket):
        def f(x):
            y = spline(x)
            return (x - xc) ** 2 + (y - yc) ** 2

        res = minimize_scalar(f, bounds=x_bracket, method="bounded")
        xb = res.x
        yb = spline(xb)
        return xb, yb, f(xb)

    projections = []
    segments = []

    for xc, yc in centroids:
        xb_u, yb_u, d2_u = project_point_to_spline(
            xc, yc, spline_upper, (x_upper.min(), x_upper.max())
        )
        xb_l, yb_l, d2_l = project_point_to_spline(
            xc, yc, spline_lower, (x_lower.min(), x_lower.max())
        )

        if d2_u <= d2_l:
            xb, yb = xb_u, yb_u
        else:
            xb, yb = xb_l, yb_l

        projections.append([xb, yb])
        segments.append([[xc, yc], [xb, yb]])

    projections = np.asarray(projections)

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

    # Blade splines
    xs_up = np.linspace(x_upper.min(), x_upper.max(), 500)
    xs_lo = np.linspace(x_lower.min(), x_lower.max(), 500)
    ax.plot(xs_up, spline_upper(xs_up), "k-", lw=1.0)
    ax.plot(xs_lo, spline_lower(xs_lo), "k-", lw=1.0)

    # Centroids and projection segments
    ax.scatter(centroids[:, 0], centroids[:, 1], s=4, c="r", label="Centroids")
    ax.scatter(projections[:, 0], projections[:, 1], s=6, c="b", label="Projection")
    from matplotlib.collections import LineCollection

    lc = LineCollection(segments, colors="r", linewidths=0.5, linestyles="dashed")
    ax.add_collection(lc)

    cbar = plt.colorbar(tpc, ax=ax)
    cbar.set_label("Target Element Size h(x,y)")

    ax.set_aspect("equal")
    ax.set_title("Sizing Function + Centroid Projections")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    ax.grid(False)

    plt.tight_layout()

    if fname:
        plt.savefig(fname, dpi=600)
        print(f"Saved {fname}")
    else:
        plt.show()

def plot_kdtree_points_on_surface(
    blade_coords,
    upper_path="data/bladeupper.txt",
    lower_path="data/bladelower.txt",
    fname=None,
):
    """
    Plot the KDTree point cloud on top of the blade spline surfaces.
    """
    import matplotlib.pyplot as plt
    from utils import config

    upper = np.loadtxt(upper_path)
    lower = np.loadtxt(lower_path)
    lower[:, 1] += config.H_box
    x_upper, y_upper = upper[:, 0], upper[:, 1]
    x_lower, y_lower = lower[:, 0], lower[:, 1]

    i_up = np.argsort(x_upper)
    i_lo = np.argsort(x_lower)
    x_upper, y_upper = x_upper[i_up], y_upper[i_up]
    x_lower, y_lower = x_lower[i_lo], y_lower[i_lo]

    spline_upper = CubicSpline(x_upper, y_upper, bc_type="natural")
    spline_lower = CubicSpline(x_lower, y_lower, bc_type="natural")

    fig, ax = plt.subplots(figsize=(10, 5))
    xs_up = np.linspace(x_upper.min(), x_upper.max(), 500)
    xs_lo = np.linspace(x_lower.min(), x_lower.max(), 500)
    ax.plot(xs_up, spline_upper(xs_up), "k-", lw=1.5, label="Upper spline")
    ax.plot(xs_lo, spline_lower(xs_lo), "k-", lw=1.5, label="Lower spline")

    ax.scatter(
        blade_coords[:, 0],
        blade_coords[:, 1],
        s=8,
        c="tab:orange",
        alpha=0.7,
        label="KDTree points",
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("KDTree Points on Blade Surfaces")
    ax.legend(loc="upper right")
    ax.grid(False)
    plt.tight_layout()

    if fname:
        plt.savefig(fname, dpi=600)
        print(f"Saved {fname}")
    else:
        plt.show()

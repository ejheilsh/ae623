import numpy as np
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.interpolate import CubicSpline
from scipy.spatial import Delaunay
from scipy.special import expit
from scipy.sparse import lil_matrix
from pathlib import Path
from scipy.optimize import minimize_scalar
import seaborn as sns


dir_base = Path(__file__).resolve().parent


@dataclass
class mesh_class():
    label: str = ""
    domain_width: float = 17
    domain_height: float = 18
    h_bounds: tuple = (0.2, 3)


    def __post_init__(self):
        self.read_blade_surfaces()
        self.generate_cubic_splines()
        self.init_domain_pts()
        self.mesh_from_points()
        self.edgehash()
        self.correct_edgehash_for_periodic_boundaries(plot_mode="revised")
        self.populate_geom_matrices()
        self.mesh_verification()
        # self.plot_by_distance()
        self.flag_cells_for_refinement()


        # self.write_gri()


    def read_blade_surfaces(self):
        # store original blade coords in 2D arrays, (x, y)
        self.surface_upper_orig = np.loadtxt("data/bladeupper.txt")
        self.surface_lower_orig = np.loadtxt("data/bladelower.txt")

        idx_shift = np.argmin(self.surface_upper_orig[:, 0]) # coord of LE for upper foil
        shift_x = self.surface_upper_orig[idx_shift, 0]
        shift_y = self.surface_upper_orig[idx_shift, 1]
        self.surface_upper_orig[:, 0] -= shift_x
        self.surface_upper_orig[:, 1] -= shift_y
        self.surface_lower_orig[:, 0] -= shift_x
        self.surface_lower_orig[:, 1] -= shift_y

        # modify for final blade placement
        self.surface_upper = self.surface_upper_orig
        self.surface_lower = self.surface_lower_orig + np.array([0.0, self.domain_height])

    def generate_cubic_splines(self):
        u = self.surface_upper
        l = self.surface_lower
        u_sorted = u[np.argsort(u[:, 0])]
        l_sorted = l[np.argsort(l[:, 0])]

        # upper splines, also shifted
        self.cs_upper = CubicSpline(u_sorted[:, 0], u_sorted[:, 1], extrapolate=False, bc_type="natural")
        self.cs_upper_up = CubicSpline(u_sorted[:, 0], u_sorted[:, 1] + self.domain_height, extrapolate=False, bc_type="natural")

        # lower splines, also shifted
        self.cs_lower = CubicSpline(l_sorted[:, 0], l_sorted[:, 1], extrapolate=False, bc_type="natural")
        self.cs_lower_down = CubicSpline(l_sorted[:, 0], l_sorted[:, 1] - self.domain_height, extrapolate=False, bc_type="natural")

        # spline bounds for minimization
        xs = np.hstack([u[:, 0], l[:, 0]])
        self.spline_bounds = (np.min(xs), np.max(xs))


    def init_domain_pts(self):
        # parameters from project description
        width = self.domain_width
        height = self.domain_height
        u = self.surface_upper
        l = self.surface_lower
        # n = 15 # arbitrary, controls fineness of initial unrefined mesh
        n = 5 # arbitrary, controls fineness of initial unrefined mesh

        # array of points to the left of the LE
        idx_LE = np.argmin(u[:, 0])
        x_LE, y_LE = u[idx_LE]
        self.x_left = x_LE - width
        xs = np.linspace(self.x_left, x_LE, n)
        ys = np.linspace(y_LE, y_LE + height, n)
        X, Y = np.meshgrid(xs, ys)
        pts_left = np.column_stack([X.ravel(), Y.ravel()])  # (N, 2)

        # array of points to the right of the TE
        idx_right = np.argmax(u[:, 0])
        x_TE, y_TE = u[idx_right]
        self.x_right = x_TE + width
        xs = np.linspace(x_TE, self.x_right, n)
        ys = np.linspace(y_TE, y_TE + height, n)
        X, Y = np.meshgrid(xs, ys)
        pts_right = np.column_stack([X.ravel(), Y.ravel()])

        # store LE and TE points for use w/ sizing function
        self.x_LE = x_LE
        self.y_LE = y_LE
        self.x_TE = x_TE
        self.y_TE = y_TE

        # points in between the blade surfaces
        y_max, y_min = np.max(l[:, 1]), y_TE #TODO change ordeR?
        xs = np.linspace(x_LE, x_TE, int((x_TE-x_LE)/(width/n)))[1:-1]
        ys = np.linspace(y_min, y_max, int((y_max - y_min) / (height/n)))
        X, Y = np.meshgrid(xs, ys)
        pts_blades = np.column_stack([X.ravel(), Y.ravel()])
        # mask w/ splines of foils

        u_on_mesh = np.column_stack([xs, self.cs_upper(xs)])
        l_on_mesh = np.column_stack([xs, self.cs_lower(xs)])

        y_upper = self.cs_upper(pts_blades[:, 0])
        y_lower = self.cs_lower(pts_blades[:, 0])
        mask = (pts_blades[:, 1] >= y_upper) & (pts_blades[:, 1] <= y_lower)
        pts_blades = pts_blades[mask]

        pts = np.vstack([pts_left, pts_blades, pts_right, u_on_mesh, l_on_mesh])
        # pts = np.vstack([pts_left, pts_blades, pts_right, u, l])

        # quick plot to check coords
        # plt.plot(u[:, 0], u[:, 1])
        # plt.plot(l[:, 0], l[:, 1])
        # plt.scatter(pts[:, 0], pts[:, 1])
        # plt.axis("equal")
        # plt.show()
        
        # store min and max y coords for plotting
        self.y_min = self.y_TE
        def f(x):
                return -self.cs_upper(x)
        res = minimize_scalar(f, bounds=(self.x_LE, self.x_TE), method="bounded")
        self.y_max = self.cs_lower(res.x)

        self.pts = pts

    def mesh_from_points(self):
        tri = Delaunay(self.pts)

        # filter triangles to keep only those inside the domain
        u = self.surface_upper
        l = self.surface_lower
        u_sorted = u[np.argsort(u[:, 0])]
        l_sorted = l[np.argsort(l[:, 0])]
        cs_upper = CubicSpline(u_sorted[:, 0], u_sorted[:, 1], extrapolate=True)
        cs_lower = CubicSpline(l_sorted[:, 0], l_sorted[:, 1], extrapolate=True)

        idx_LE = np.argmin(u[:, 0])
        idx_TE = np.argmax(u[:, 0])
        x_LE, y_LE = u[idx_LE]
        x_TE, y_TE = u[idx_TE]
        y_top = np.max(l[:, 1])

        tri_pts = self.pts[tri.simplices]
        centroids = tri_pts.mean(axis=1)

        left_block = (centroids[:, 0] <= x_LE) & (centroids[:, 1] >= y_LE) & (centroids[:, 1] <= y_LE + self.domain_height)
        right_block = (centroids[:, 0] >= x_TE) & (centroids[:, 1] >= y_TE) & (centroids[:, 1] <= y_TE + self.domain_height)

        mid_x = (centroids[:, 0] >= x_LE) & (centroids[:, 0] <= x_TE)
        y_upper = cs_upper(centroids[:, 0])
        y_lower = cs_lower(centroids[:, 0])
        mid_block = mid_x & (centroids[:, 1] >= y_upper) & (centroids[:, 1] <= y_lower)

        keep = left_block | right_block | mid_block
        self.E2N = tri.simplices[keep] + 1 # NEED TO KEEP E2N 1-INDEXED
        # self.E2N = tri.simplices
        self.V = self.pts

        # plot initial coarse mesh 
        # plt.triplot(self.V[:,0], self.V[:,1], self.E2N-1) # -1 is because E2N is 1-indexed
        # plt.plot(self.surface_upper[:, 0], self.surface_upper[:, 1])
        # plt.plot(self.surface_lower[:, 0], self.surface_lower[:, 1])
        # plt.show()

        
    def edgehash(self, plor_edges=None):
        """
        Identify interior and boundary edges from an element-to-node array.
        Returns:
            IE: (niedge, 4) array [n1, n2, elem1, elem2]
            BE: (nbedge, 3) array [n1, n2, elem]
        """
        # from kfid matlab code
        E2N = self.E2N

        nelem = E2N.shape[0]
        nnode = int(np.max(E2N))  # nodes are 1-based
        H = lil_matrix((nnode + 1, nnode + 1), dtype=int)
        IE = np.zeros((int(np.ceil(nelem * 3 / 2)), 4), dtype=int)
        niedge = 0

        for elem in range(1, nelem + 1):
            nv = E2N[elem - 1, 0:3]
            for edge in range(1, 4):
                n1 = nv[(edge) % 3]
                n2 = nv[(edge + 1) % 3]
                if H[n1, n2] == 0:
                    H[n1, n2] = elem
                    H[n2, n1] = elem
                else:
                    oldelem = H[n1, n2]
                    if oldelem < 0:
                        raise ValueError("Mesh input error: non-manifold edge")
                    niedge += 1
                    IE[niedge - 1, :] = [n1, n2, oldelem, elem]
                    H[n1, n2] = -1
                    H[n2, n1] = -1

        IE = IE[0:niedge, :]

        H_coo = H.tocoo()
        I, J = [], []
        for i, j, v in zip(H_coo.row, H_coo.col, H_coo.data):
            if i < j and v > 0:
                I.append(i)
                J.append(j)

        BE = np.zeros((len(I), 3), dtype=int)
        for b in range(len(I)):
            BE[b, :] = [I[b], J[b], H[I[b], J[b]]]

        # make sure the edge points are in the order that they appear in E2N
        def _order_edge(n1, n2, elem):
            nv = E2N[elem - 1]
            idx1 = np.where(nv == n1)[0]
            idx2 = np.where(nv == n2)[0]
            if idx1.size == 0 or idx2.size == 0:
                return n1, n2
            i = idx1[0]
            j = idx2[0]
            if j == (i + 1) % 3:
                return n1, n2
            if i == (j + 1) % 3:
                return n2, n1
            return n1, n2

        # BE: use its adjacent element
        for i in range(BE.shape[0]):
            n1, n2, elem = BE[i]
            n1, n2 = _order_edge(n1, n2, elem)
            BE[i, 0] = n1
            BE[i, 1] = n2

        # IE: use elemL for consistent ordering
        for i in range(IE.shape[0]):
            n1, n2, e1, e2 = IE[i]
            elem = min(e1, e2)
            n1, n2 = _order_edge(n1, n2, elem)
            IE[i, 0] = n1
            IE[i, 1] = n2

        self.IE = IE
        self.BE = BE

    def correct_edgehash_for_periodic_boundaries(self, plot_mode="pairs"):
        # look for boundary edges that have both nodes at the periodic y coordinates
        _, y_LE_upper = self.surface_upper[np.argmin(self.surface_upper[:, 0])]
        _, y_TE_upper = self.surface_upper[np.argmax(self.surface_upper[:, 0])]
        _, y_LE_lower = self.surface_lower[np.argmin(self.surface_lower[:, 0])]
        _, y_TE_lower = self.surface_lower[np.argmax(self.surface_lower[:, 0])]
        tol = 1e-8
        x_n1, y_n1 = self.V[self.BE[:, 0] - 1].T  # .T to unpack node coords for all node
        x_n2, y_n2 = self.V[self.BE[:, 1] - 1].T
        mask_LE_upper = (np.abs(y_n1 - y_LE_upper) < tol) & (np.abs(y_n2 - y_LE_upper) < tol)
        mask_TE_upper = (np.abs(y_n1 - y_TE_upper) < tol) & (np.abs(y_n2 - y_TE_upper) < tol)
        mask_LE_lower = (np.abs(y_n1 - y_LE_lower) < tol) & (np.abs(y_n2 - y_LE_lower) < tol)
        mask_TE_lower = (np.abs(y_n1 - y_TE_lower) < tol) & (np.abs(y_n2 - y_TE_lower) < tol)
        mask_periodic = (mask_LE_upper | mask_TE_upper | mask_LE_lower | mask_TE_lower)

        # collect unique periodic node pairs (top/bottom with same x)
        mask_bottom = mask_LE_upper | mask_TE_upper
        mask_top = mask_LE_lower | mask_TE_lower

        bottom_nodes = np.unique(self.BE[mask_bottom, :2])
        top_nodes = np.unique(self.BE[mask_top, :2])

        periodic_node_pairs = []
        for n_bot in bottom_nodes:
            x_bot = self.V[n_bot - 1, 0]
            matches = top_nodes[np.abs(self.V[top_nodes - 1, 0] - x_bot) < tol]
            if matches.size > 0:
                periodic_node_pairs.append([n_bot, int(matches[0])])

        periodic_node_pairs = np.asarray(periodic_node_pairs, dtype=int)
        # sort by x for readability
        if periodic_node_pairs.size > 0:
            order = np.argsort(self.V[periodic_node_pairs[:, 0] - 1, 0])
            periodic_node_pairs = periodic_node_pairs[order]
        self.periodic_node_pairs = periodic_node_pairs

        # now remove periodic boundaries from BE and add to IE
        # first, find corresponding elements
        periodic_BE = self.BE[mask_periodic]
        x1_periodic = x_n1[mask_periodic]
        x2_periodic = x_n2[mask_periodic]

        # pair periodic edges by matching endpoint x-coordinates (order-independent)
        x_pair = np.sort(np.column_stack([x1_periodic, x2_periodic]), axis=1)
        x_key = np.round(x_pair / tol).astype(int)
        periodic_idx = np.where(mask_periodic)[0]
        pairs = []
        used = np.zeros(len(periodic_BE), dtype=bool)
        for i in range(len(periodic_BE)):
            if used[i]:
                continue
            same_key = np.all(x_key == x_key[i], axis=1)
            idx = np.where(same_key)[0]
            if len(idx) == 2:
                pairs.append((idx[0], idx[1]))
                used[idx] = True
            else:
                print("unpaired periodic edge with x-pair:", x_pair[i], "count:", len(idx))

        # move paired periodic edges from BE to IE
        V = self.V
        paired_periodic_idx = periodic_idx[used]
        if paired_periodic_idx.size > 0:
            keep_mask = np.ones(self.BE.shape[0], dtype=bool)
            keep_mask[paired_periodic_idx] = False
            self.BE = self.BE[keep_mask]

            new_IE = []
            for i, j in pairs:
                n1, n2, elem1 = periodic_BE[i]
                n3, n4, elem2 = periodic_BE[j]
                new_IE.append([n1, n2, elem1, elem2])

                # x-coordinates for the two edges
                x1, x2 = V[n1 - 1, 0], V[n2 - 1, 0]
                x3, x4 = V[n3 - 1, 0], V[n4 - 1, 0]


            periodic_node_pairs = np.sort(periodic_node_pairs, axis=1)
            periodic_node_pairs = np.unique(periodic_node_pairs, axis=0)
            self.periodic_node_pairs = periodic_node_pairs

            if new_IE:
                self.IE = np.vstack([self.IE, np.asarray(new_IE, dtype=int)])
                # pass

        # if plot_mode:
        #     fig, ax = plt.subplots()
        #     for n1, n2, _, _ in self.IE:
        #         ax.plot([self.V[n1-1, 0], self.V[n2-1, 0]],
        #                 [self.V[n1-1, 1], self.V[n2-1, 1]],
        #                 color="tab:blue", linewidth=0.8)

        #     # ax.triplot(self.V[:, 0], self.V[:, 1], self.E2N-1, color="k")
        #     # for n1, n2, _, _ in new_IE:
        #     #     ax.plot([self.V[n1 - 1, 0], self.V[n2 - 1, 0]],
        #     #             [self.V[n1 - 1, 1], self.V[n2 - 1, 1]],
        #     #             color="tab:green", linewidth=0.8)

        #     if plot_mode == "pairs":
        #         # plot boundary edges; periodic pairs share a color
        #         cmap = plt.get_cmap("tab20")
        #         for n1, n2, _ in self.BE:
        #             ax.plot([self.V[n1 - 1, 0], self.V[n2 - 1, 0]],
        #                     [self.V[n1 - 1, 1], self.V[n2 - 1, 1]],
        #                     color="tab:red", linewidth=1.2)
        #         for k, (i, j) in enumerate(pairs):
        #             color = cmap(k % cmap.N)
        #             for idx in (i, j):
        #                 n1, n2, _ = periodic_BE[idx]
        #                 ax.plot([self.V[n1 - 1, 0], self.V[n2 - 1, 0]],
        #                         [self.V[n1 - 1, 1], self.V[n2 - 1, 1]],
        #                         color=color, linewidth=2.0)
        #         ax.set_title("Interior (blue), Boundary (red), Periodic pairs (same color)")
        #     elif plot_mode == "revised":
        #         # plot revised IE/BE after periodic pairing
        #         for n1, n2, _ in self.BE:
        #             ax.plot([self.V[n1 - 1, 0], self.V[n2 - 1, 0]],
        #                     [self.V[n1 - 1, 1], self.V[n2 - 1, 1]],
        #                     color="tab:red", linewidth=1.2)
        #         ax.set_title("Revised Interior (blue) and Boundary (red) edges")
        #     else:
        #         raise ValueError("plot_mode must be 'pairs', 'revised', or None")

        #     ax.set_aspect("equal", adjustable="box")
        #     plt.show()

    def populate_geom_matrices(self):
        # V,    vertices: node xy coords (x, y) x n_nodes
        # E2N,  element to node: nodes in elements, ccw ordering (n1, n2, n3) x n_elems
        # IE,   interior edges: interior nodes and corresponding neighbor cells (n1, n2, elemL, elemR) x n_int_edges
        # BE,   boundary edges: boundary nodes and corresponding cell (n1, n2, elem) x n_bound_edges
        # I2E,  interior edges to elements: elements and edges for interior edges (elemL, edgeL, elemR, edgeR) X n_int_edges, edgeL is local edge, corresponds to opposite node
        # B2E,  boundary edges to elements and boundary groups: (elemL, edgeL, bgroup) x n_bound_edges
        # In,   normal vectors for interior faces: (x_n, y_n) x n_int_edges, unit normals
        # Bn,   normal vectors for boundary faces: (x_n, y_n) x n_bound_edges, unit normals
        # Area, area of each element: (A) x n_elems
        # (elemL is the element with the smaller index)

        # V is made in init_mesh_from_pts() and updated during meshing
        # E2N is made in init_mesh_from_pts() and updated during meshing
        # IE and BE are made in edgehash() and corrected in correct_edgehash...()

        # I2E construction (elemL, edgeL, elemR, edgeR) x n_int_edges
        IE = self.IE
        V = self.V
        E2N = self.E2N
        I2E = np.zeros_like(self.IE)
        for i in range(len(IE)):
            # store elems
            elemL = np.min(IE[i, 2:])
            elemR = np.max(IE[i, 2:])
            ns_edge = IE[i, :2]
            I2E[i, 0] = elemL
            I2E[i, 2] = elemR

            # store nodes corresponding to edges (opposite)
            nsL = E2N[elemL-1] # elemL and elemR are one-based in IE
            nsR = E2N[elemR-1]
            isnt_in_L = ~np.isin(nsL, ns_edge)
            isnt_in_R = ~np.isin(nsR, ns_edge)

            if np.all(isnt_in_L == True):
                # periodic boundary, need to use nodes from other element
                # nsL are the nodes in the periodic cell, which does not include the nodes from the edge in IE
                # want nsL node with dissimilar y value
                ys = V[nsL-1, 1]
                counts = np.array([np.sum(ys == y) for y in ys])
                idx = np.where(counts == 1)[0][0]
                isnt_in_L = [i == idx for i in range(3)]

            if np.all(isnt_in_R == True):
                # periodic boundary, need to use nodes from other element
                ys = V[nsR-1, 1]
                counts = np.array([np.sum(ys == y) for y in ys])
                idx = np.where(counts == 1)[0][0]
                isnt_in_R = [i == idx for i in range(3)]

            # global node num for debugging
            # I2E[i, 1] = E2N[elemL, np.argwhere(isnt_in_L)[0][0]] # assuming there is only one match
            # I2E[i, 3] = E2N[elemR, np.argwhere(isnt_in_R)[0][0]] # assuming there is only one match

            # local node num for deliverable
            I2E[i, 1] = np.argwhere(isnt_in_L)[0, 0] + 1 # assuming there is only one match
            I2E[i, 3] = np.argwhere(isnt_in_R)[0, 0] + 1
        self.I2E = I2E

        # BE: (n1, n2, elem) x n_bound_edges
        # B2E: (elemL, edgeL, bgroup) x n_bound_edges
        BE = self.BE
        B2E = np.empty_like(BE, dtype=object)
        for i in range(len(BE)):
            # place adjacent element in B2E 
            elem = BE[i, 2]
            B2E[i, 0] = elem

            # other node
            ns_edge = BE[i, :2]
            ns_elem = E2N[elem - 1]
            isnt_in_E = ~np.isin(ns_elem, ns_edge)
            B2E[i, 1] = np.argwhere(isnt_in_E==True)[0, 0] + 1

            # grouping (inflow, wall, outflow)
            tol = 1e-8
            xs = V[ns_edge-1, 0]
            if np.all(np.abs(xs - self.x_left) < tol):
                B2E[i, 2] = "inflow"
            elif np.all(np.abs(xs - self.x_right) < tol):
                B2E[i, 2] = "outflow"
            else:
                B2E[i, 2] = "wall" #TODO perhaps split into upper and lower in the future for identifying spline? TBD...
        self.B2E = B2E
    
        # In, normal vectors for interior faces: (x_n, y_n) x n_int_edges, unit normals
        n1 = IE[:, 0] - 1
        n2 = IE[:, 1] - 1
        p1 = V[n1]            # (n_int, 2)
        p2 = V[n2]
        edge = p2 - p1        # (n_int, 2)
        normals = np.column_stack([-edge[:, 1], edge[:, 0]]) # rotate 90° CW: (x,y) -> (y, -x)
        self.IE_lens = np.linalg.norm(normals, axis=1, keepdims=True) # normalize, store for mesh check
        In = normals / self.IE_lens
        self.In = In 

        # Bn, boundary normals for boundary edges
        n1 = BE[:, 0] - 1
        n2 = BE[:, 1] - 1
        elem = BE[:, 2] - 1
        p1 = V[n1]
        p2 = V[n2]
        edge = p2 - p1
        normals = np.column_stack([edge[:, 1], -edge[:, 0]]) # rotate 90° CW: (x,y) -> (y, -x)
        self.BE_lens = np.linalg.norm(normals, axis=1, keepdims=True) # normalize, keepdims to make broadcasting easier
        Bn = normals / self.BE_lens
        self.Bn = Bn

        # Area, areas per cell
        e = E2N - 1
        p0 = V[e[:, 0]]
        p1 = V[e[:, 1]]
        p2 = V[e[:, 2]]
        area2 = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
        Area = 0.5 * np.abs(area2)   # (nelem,)
        self.Area = Area

        # # plot normals for IE and BE
        # mid_in = (V[IE[:, 0] - 1] + V[IE[:, 1] - 1]) * 0.5
        # mid_be = (V[BE[:, 0] - 1] + V[BE[:, 1] - 1]) * 0.5

        # fig, ax = plt.subplots()
        # ax.quiver(mid_in[:, 0], mid_in[:, 1], In[:, 0], In[:, 1],
        #           color="tab:blue", angles="xy", scale_units="xy", scale=1, width=0.002)
        # ax.quiver(mid_be[:, 0], mid_be[:, 1], Bn[:, 0], Bn[:, 1],
        #           color="tab:red", angles="xy", scale_units="xy", scale=1, width=0.002)
        # ax.set_aspect("equal", adjustable="box")
        # ax.set_title("Interior normals (blue) and Boundary normals (red)")
        # ax.triplot(V[:,0], V[:, 1], E2N - 1)
        # # ax.tripcolor(V[:, 0], V[:, 1], E2N - 1, facecolors=Area, edgecolors='k', linewidth=0.2)
        # # plt.colorbar(ax.collections[0], ax=ax)
        # for i, ns in enumerate(E2N):
        #     n1, n2, n3 = ns
        #     v1, v2, v3 = V[n1-1], V[n2-1], V[n3-1]
        #     ax.text(v1[0], v1[1], f"{n1}")
        #     ax.text(v2[0], v2[1], f"{n2}")
        #     ax.text(v3[0], v3[1], f"{n3}")
        #     cent = (1/3) * (v1 + v2 + v3)
        #     ax.text(cent[0], cent[1], f"{i+1}", color="lime")

    def mesh_verification(self):
        # verify per-element sum of outward normals * edge length is 0.
        V = self.V
        E2N = self.E2N
        IE = self.IE
        BE = self.BE
        In = self.In
        Bn = self.Bn
        IE_lens = self.IE_lens.reshape(-1)
        BE_lens = self.BE_lens.reshape(-1)

        # build edge lookup tables (unordered node pairs).
        ie_map = {}
        for i, (n1, n2, _, _) in enumerate(IE):
            a = int(n1)
            b = int(n2)
            key = (a, b) if a < b else (b, a)
            ie_map[key] = i
        be_map = {}
        for i, (n1, n2, _) in enumerate(BE):
            a = int(n1)
            b = int(n2)
            key = (a, b) if a < b else (b, a)
            be_map[key] = i

        n_elem = E2N.shape[0]
        E_totals = np.zeros((n_elem, 2), dtype=float)

        missing = 0
        for e in range(n_elem):
            n1, n2, n3 = E2N[e]
            edges = [(n1, n2), (n2, n3), (n3, n1)]
            centerE = np.mean(V[[n1 - 1, n2 - 1, n3 - 1]], axis=0)

            for a, b in edges:
                a_i = int(a)
                b_i = int(b)
                key = (a_i, b_i) if a_i < b_i else (b_i, a_i)
                p1 = V[a - 1]
                p2 = V[b - 1]
                mid_edge = 0.5 * (p1 + p2)
                toOutside = mid_edge - centerE

                if key in be_map:
                    idx = be_map[key]
                    normal = Bn[idx].copy()
                    length = BE_lens[idx]
                elif key in ie_map:
                    idx = ie_map[key]
                    normal = In[idx].copy()
                    length = IE_lens[idx]
                else:
                    # compute from geometry if edge isn't in IE/BE maps.
                    missing += 1
                    edge_vec = p2 - p1
                    length = np.linalg.norm(edge_vec)
                    if length == 0.0:
                        continue
                    normal = np.array([-edge_vec[1], edge_vec[0]], dtype=float)
                    normal /= np.linalg.norm(normal)

                if np.dot(normal, toOutside) < 0:
                    normal = -normal

                E_totals[e] += normal * length

        magnitudes = np.linalg.norm(E_totals, axis=1)
        max_error = float(np.max(magnitudes)) if magnitudes.size else 0.0
        mean_error = float(np.mean(magnitudes)) if magnitudes.size else 0.0
        sum_error = np.sum(magnitudes)

        print("Element totals after verification (should be close to zero):")
        # print(E_totals)
        print(f"Maximum error magnitude: {max_error:.6e}")
        print(f"Mean error magnitude:    {mean_error:.6e}")
        print(f"Sum of error magnitude:    {sum_error:.6e}")
        print(f"Number of elements:      {n_elem}")
        if missing:
            print(f"Warning: {missing} edges were not found in IE/BE and used geometric fallback.")
        return E_totals, max_error, mean_error

        # plt.show()

    def write_gri(self):
        n_node = len(self.V)
        n_elem = len(self.E2N)
        n_dims = self.V.shape[1]

        fname_gri = dir_base.joinpath(f"output/{self.label}.gri")
        fname_gri.parent.mkdir(parents=True, exist_ok=True)

        with open(fname_gri, "w") as f:
            # f.write(f"{n_node} {n_elem} {self.V.shape[1]}\n")
            # f.write(f"{self.V}\n".replace("[", "").replace("]", ""))
            # f.write(f"{len(np.unique(self.B2E[:, 2]))}\n")

            f.write(f"{n_node} {n_elem} {n_dims}\n")
            np.savetxt(f, self.V, fmt="%10.06f")  # or "%.8e" etc.

            # boundary groups
            boundaries = self.B2E[:, 2]
            f.write(f"{len(np.unique(boundaries))}\n")
            
            # get number of each boundaries
            boundary_groups = ["inflow", "wall", "outflow"]
            for bg in boundary_groups:
                mask = boundaries == bg
                n = np.sum(mask)
                f.write(f"{n} {n_dims} {bg}\n")
                bnodes = self.B2E[mask][:, :2]
                for bnode_pair in bnodes:
                    f.write(" ".join(map(str, bnode_pair)) + "\n")

            # elements
            f.write(f"{n_elem} 1 TriLagrange\n")
            np.savetxt(f, self.E2N, fmt="%i")

            # now periodic groups
            f.write("1 PeriodicGroup\n")
            f.write(f"{len(self.periodic_node_pairs)} Translational\n")
            for pair in self.periodic_node_pairs:
                f.write(" ".join(map(str, pair)) + "\n")



            print()

    def distance_to_blade_surface(self, point):
        x, y = point[0], point[1]

        def project_to_spline(xc, yc, spline):
            def f(xi):
                yi = spline(xi)
                return (xi - xc)**2 + (yi - yc)**2

            res = minimize_scalar(f, bounds=self.spline_bounds, method="bounded")
            return f(res.x)

        d_u = project_to_spline(x, y, self.cs_upper)
        d_u_u = project_to_spline(x, y, self.cs_upper_up)

        d_l = project_to_spline(x, y, self.cs_lower)
        d_l_d = project_to_spline(x, y, self.cs_lower_down)

        # check on this moore... maybe plot positions where it does matter
        # if np.sqrt(min(d_u_u, d_l_d)) < np.sqrt(min(d_u, d_l)):
        #     print("the projections matter...")

        return np.sqrt(min(d_u, d_l, d_u_u, d_l_d))

    def project_point_to_blade(self, point, plot=False):
        x, y = point[0], point[1]

        def project_to_spline(xc, yc, spline):
            def f(xi):
                yi = spline(xi)
                return (xi - xc) ** 2 + (yi - yc) ** 2

            res = minimize_scalar(f, bounds=self.spline_bounds, method="bounded")
            xi = res.x
            return np.array([xi, spline(xi)]), f(xi)

        candidates = []
        p_u, d_u = project_to_spline(x, y, self.cs_upper)
        candidates.append((d_u, p_u, "upper"))
        p_u_u, d_u_u = project_to_spline(x, y, self.cs_upper_up)
        candidates.append((d_u_u, p_u_u, "upper_up"))
        p_l, d_l = project_to_spline(x, y, self.cs_lower)
        candidates.append((d_l, p_l, "lower"))
        p_l_d, d_l_d = project_to_spline(x, y, self.cs_lower_down)
        candidates.append((d_l_d, p_l_d, "lower_down"))

        candidates.sort(key=lambda t: t[0])
        best = candidates[0]

        # plot = True
        if plot:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(7, 4))
            plt.plot(self.surface_upper[:, 0], self.surface_upper[:, 1], "k-", lw=1.0, label="upper")
            plt.plot(self.surface_lower[:, 0], self.surface_lower[:, 1], "k--", lw=1.0, label="lower")
            plt.plot(self.surface_upper[:, 0], self.surface_upper[:, 1] + self.domain_height, "0.5", lw=0.8, label="upper_up")
            plt.plot(self.surface_lower[:, 0], self.surface_lower[:, 1] - self.domain_height, "0.5", lw=0.8, label="lower_down")
            plt.scatter([x], [y], c="red", s=30, zorder=5, label="query")
            for d, p, name in candidates:
                plt.scatter([p[0]], [p[1]], s=20, label=f"{name} (d={np.sqrt(d):.3g})")
            plt.scatter([best[1][0]], [best[1][1]], c="gold", edgecolor="k", s=60, zorder=6, label="chosen")
            plt.axis("equal")
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()
            # plt.show()

        return best[1]

    def sizing_function(self, point):
        d_surf = self.distance_to_blade_surface(point)
        x_proj = self.project_point_to_blade(point)[0]
        x, y = point[0], point[1]

        # # compute min distance from LE and TE
        # d_LE_u = np.sqrt((x - self.x_LE)**2 + (y - self.y_LE)**2)
        # d_TE_u = np.sqrt((x - self.x_TE)**2 + (y - self.y_TE)**2)
        # d_LE_d = np.sqrt((x - self.x_LE)**2 + (y - (self.y_LE + self.domain_height))**2)
        # d_TE_d = np.sqrt((x - self.x_TE)**2 + (y - (self.y_TE + self.domain_height))**2)
        # d_le_te = min(d_LE_u, d_TE_u, d_LE_d, d_TE_d)

        # d = (1/2) * (d_surf + d_le_te)
        # d = d_surf


        # use dist to blade and x loc or proj point on bladex_blade = self.x_TE - self.x_LE
        x_proj = x
        a = d_surf
        dx_LE_TE = np.min([np.abs(x_proj - self.x_LE), np.abs(x_proj-self.x_TE)])
        b = 8 * np.abs(np.tanh(dx_LE_TE / 4))
        w = 3/4 # weight given to distance vs x position
        d = w * a + (1-w) * b

        h_min = self.h_bounds[0]
        h_max = self.h_bounds[1]
        h_range = h_max - h_min

        return h_min + h_range * np.tanh(d / 15)


    def plot_by_distance(self, color_by_edge_length=False, vmin=None, vmax=None):
        from matplotlib.collections import LineCollection
        import matplotlib.tri as mtri

        # Build triangulation (E2N is 1-based)
        tri = mtri.Triangulation(self.V[:, 0], self.V[:, 1], self.E2N - 1)

        # Per-edge value from midpoint
        edges = {}
        for tri_nodes in (self.E2N - 1):
            a, b, c = tri_nodes
            for i, j in ((a, b), (b, c), (c, a)):
                e = (i, j) if i < j else (j, i)
                if e in edges:
                    continue
                p0, p1 = self.V[e[0]], self.V[e[1]]
                mid = 0.5 * (p0 + p1)
                if color_by_edge_length:
                    edges[e] = np.linalg.norm(p1 - p0)
                else:
                    edges[e] = self.sizing_function(mid)

        segments = []
        colors = []
        for (i, j), d in edges.items():
            segments.append([self.V[i], self.V[j]])
            colors.append(d)

        fig, ax = plt.subplots(figsize=(12, 5))
        values = np.array(colors)
        lc = LineCollection(
            segments,
            array=values,
            cmap="turbo",
            linewidths=0.5,
        )
        if vmin is not None or vmax is not None:
            lc.set_clim(vmin=vmin, vmax=vmax)
        ax.add_collection(lc)
        # ax.autoscale()
        ax.set_aspect("equal")

        ax.plot(self.surface_upper[:, 0], self.surface_upper[:, 1], "k-", lw=1.0)
        ax.plot(self.surface_lower[:, 0], self.surface_lower[:, 1], "k-", lw=1.0)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Edge Length" if color_by_edge_length else "Sizing Function Target Edge Size")
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label("Edge length" if color_by_edge_length else "Sizing function target edge size")

        out_path = dir_base / "output" / "plot_by_distance.svg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        # plt.savefig(out_path, format="svg")
        # print(f"Saved {out_path}")
        # plt.show()


    def flag_cells_for_refinement(self):
        # use IE and BE

        # flag IE edges for refinement
        IE_n1_x = self.V[self.IE[:, 0]-1, 0]
        IE_n2_x = self.V[self.IE[:, 1]-1, 0]
        IE_n1_y = self.V[self.IE[:, 0]-1, 1]
        IE_n2_y = self.V[self.IE[:, 1]-1, 1]
        IE_midpoints = 0.5 * np.column_stack([
            IE_n1_x + IE_n2_x,
            IE_n1_y + IE_n2_y,
        ])
        IE_lengths = np.sqrt((IE_n2_x - IE_n1_x)**2 + (IE_n2_y - IE_n1_y)**2)
        IE_requested_size = np.array([self.sizing_function(p) for p in IE_midpoints])
        IE_mask = IE_lengths > IE_requested_size

        # flag BE edges for refinement
        BE_n1_x = self.V[self.BE[:, 0]-1, 0]
        BE_n2_x = self.V[self.BE[:, 1]-1, 0]
        BE_n1_y = self.V[self.BE[:, 0]-1, 1]
        BE_n2_y = self.V[self.BE[:, 1]-1, 1]
        BE_midpoints = 0.5 * np.column_stack([
            BE_n1_x + BE_n2_x,
            BE_n1_y + BE_n2_y,
        ])
        BE_lengths = np.sqrt((BE_n2_x - BE_n1_x)**2 + (BE_n2_y - BE_n1_y)**2)
        BE_requested_size = np.array([self.sizing_function(p) for p in BE_midpoints])
        BE_mask = BE_lengths > BE_requested_size

        # store info for local refinement
        self.IE_midpoints = IE_midpoints
        self.IE_mask = IE_mask
        self.BE_midpoints = BE_midpoints
        self.BE_mask = BE_mask

        # plot
        plot_refinement_edges = False
        if plot_refinement_edges:
            from matplotlib.collections import LineCollection

            ie_segments = []
            ie_colors = []
            for (n1, n2), is_marked in zip(self.IE[:, :2], IE_mask):
                p0 = self.V[n1 - 1]
                p1 = self.V[n2 - 1]
                ie_segments.append([p0, p1])
                ie_colors.append("green" if is_marked else "red")

            be_segments = []
            be_colors = []
            for (n1, n2), is_marked in zip(self.BE[:, :2], BE_mask):
                p0 = self.V[n1 - 1]
                p1 = self.V[n2 - 1]
                be_segments.append([p0, p1])
                be_colors.append("green" if is_marked else "red")

            fig, ax = plt.subplots(figsize=(12, 5))
            lc_ie = LineCollection(ie_segments, colors=ie_colors, linewidths=0.8)
            ax.add_collection(lc_ie)
            if be_segments:
                lc_be = LineCollection(be_segments, colors=be_colors, linewidths=0.8)
                ax.add_collection(lc_be)
            ax.autoscale()
            ax.set_aspect("equal")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Refinement Flags (green=refine, red=keep)")
            plt.tight_layout()
            # plt.show()

    def smooth_cells(self, iterations=10, omega=0.5, freeze_boundary=True, freeze_periodic=True, tol=1e-8):
        nnode = self.V.shape[0]
        neighbors = [set() for _ in range(nnode)]
        for n1, n2, _, _ in self.IE:
            i = n1 - 1
            j = n2 - 1
            neighbors[i].add(j)
            neighbors[j].add(i)

        frozen = set()
        if freeze_boundary and self.BE.size:
            bnodes = np.unique(self.BE[:, :2]) - 1
            frozen.update(bnodes.tolist())
        if freeze_periodic:
            y = self.V[:, 1]
            yvals = np.array(
                [
                    self.y_LE,
                    self.y_TE,
                    self.y_LE + self.domain_height,
                    self.y_TE + self.domain_height,
                ],
                dtype=float,
            )
            mask = np.any(np.isclose(y[:, None], yvals[None, :], atol=tol), axis=1)
            frozen.update(np.where(mask)[0].tolist())

        V_new = self.V.copy()
        for _ in range(iterations):
            V_next = V_new.copy()
            for i in range(nnode):
                if i in frozen:
                    continue
                nbrs = neighbors[i]
                if not nbrs:
                    continue
                avg = np.mean(V_new[list(nbrs)], axis=0)
                V_next[i] = (1.0 - omega) * V_new[i] + omega * avg
            V_new = V_next

        self.V = V_new
        self.pts = V_new
        return V_new


    def refinement_local(self):
        # add midpoints of edges to be refined
        new_nodes_IE = self.IE_midpoints[self.IE_mask]
        new_nodes_BE = self.BE_midpoints[self.BE_mask]

        # midpoint addition for periodic boundaries on top (since they are not stored in IE)
        ns_periodic = []
        for n in new_nodes_IE:
            y = n[1]
            if (y == self.y_LE) or (y == self.y_TE):
                ns_periodic.append(np.array([n[0], y + self.domain_height]))
        if ns_periodic:
            new_nodes_IE = np.vstack([new_nodes_IE, np.array(ns_periodic)])

        # snapping to blade spline for wall cells in BE
        if new_nodes_BE.size:
            be_idx = np.where(self.BE_mask)[0]
            wall_mask = self.B2E[:, 2] == "wall"
            for k, i_be in enumerate(be_idx):
                if wall_mask[i_be]:
                    new_nodes_BE[k] = self.project_point_to_blade(new_nodes_BE[k])

        self.pts = np.vstack([
            self.V,
            new_nodes_IE,
            new_nodes_BE
        ])

        self.mesh_from_points()
        self.edgehash()
        self.correct_edgehash_for_periodic_boundaries(plot_mode="revised")
        self.smooth_cells(iterations=1000, omega=0.1)


        self.populate_geom_matrices()
        print(f"n_cells after refinement {len(self.E2N)}")
        self.flag_cells_for_refinement()

    def refinement_global(self):
        V = self.V
        E = self.E2N

        # map boundary edge -> boundary type for snapping wall midpoints
        boundary_type = {}
        if getattr(self, "BE", None) is not None and getattr(self, "B2E", None) is not None:
            for (n1, n2, _), b in zip(self.BE, self.B2E[:, 2]):
                key = (n1, n2) if n1 < n2 else (n2, n1)
                boundary_type[key] = b

        edge_mid = {}
        V_list = V.tolist()

        def midpoint_index(n1, n2):
            key = (n1, n2) if n1 < n2 else (n2, n1)
            if key in edge_mid:
                return edge_mid[key]
            p0 = V[n1 - 1]
            p1 = V[n2 - 1]
            mid = 0.5 * (p0 + p1)
            if boundary_type.get(key) == "wall":
                mid = self.project_point_to_blade(mid)
            V_list.append([float(mid[0]), float(mid[1])])
            idx = len(V_list)  # 1-based index
            edge_mid[key] = idx
            return idx

        new_E = []
        for a, b, c in E:
            mab = midpoint_index(a, b)
            mbc = midpoint_index(b, c)
            mca = midpoint_index(c, a)
            new_E.append([a, mab, mca])
            new_E.append([mab, b, mbc])
            new_E.append([mca, mbc, c])
            new_E.append([mab, mbc, mca])

        self.V = np.asarray(V_list, dtype=float)
        self.E2N = np.asarray(new_E, dtype=int)
        self.pts = self.V

        self.edgehash()
        self.correct_edgehash_for_periodic_boundaries(plot_mode="revised")
        # self.smooth_cells(iterations=5, omega=0.5)
        self.populate_geom_matrices()
        print(f"n_cells after global refinement {len(self.E2N)}")
        # self.flag_cells_for_refinement()

    def mesh_check(self):
        # check for each element that the sum of the normal vectors is 0
        tot = 0
            
        for idx_elem, _ in enumerate(self.E2N):
            # first check to see if there are any interior edges
            es = self.I2E[self.I2E[:, ]]


    def visual_mesh(self, fname="plot", color_by=None):
        # should the triplot be plotted?
        # should the cells be colored?
        # what level of refinement should it be?

        # load in data
        V = self.V
        E2N = self.E2N
        import matplotlib.tri as mtri

        sns.set_context("paper")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.despine()

        blade_color = "red" if color_by is None or color_by is "projections" else "k"
        

        # plot spline of self.surface_upper and self.surface_lower as red curves
        ax.plot(self.surface_upper[:, 0], self.surface_upper[:, 1], color=blade_color, linewidth=1.0, zorder=3, label="Blade surface")
        ax.plot(self.surface_lower[:, 0], self.surface_lower[:, 1], color=blade_color, linewidth=1.0, zorder=3)

        tri = mtri.Triangulation(V[:, 0], V[:, 1], E2N - 1)

        if color_by is None:
            ax.triplot(tri, color="k", linewidth=0.2, label="Mesh")
        elif color_by == "edge_length":
            from matplotlib.collections import LineCollection

            edges = {}
            for tri_nodes in (E2N - 1):
                a, b, c = tri_nodes
                for i, j in ((a, b), (b, c), (c, a)):
                    e = (i, j) if i < j else (j, i)
                    if e in edges:
                        continue
                    p0, p1 = V[e[0]], V[e[1]]
                    edges[e] = np.linalg.norm(p1 - p0)

            segments = []
            colors = []
            for (i, j), d in edges.items():
                segments.append([V[i], V[j]])
                colors.append(d)

            lc = LineCollection(segments, array=np.array(colors), cmap="turbo", linewidths=0.2)
            ax.add_collection(lc)
            cbar = fig.colorbar(lc, ax=ax, label="Edge length")
            cbar.set_label("Edge length", fontsize=30)
            cbar.ax.tick_params(labelsize=24)
        elif color_by == "target_size":
            centroids = (V[E2N[:, 0] - 1] + V[E2N[:, 1] - 1] + V[E2N[:, 2] - 1]) / 3.0
            vals = np.array([self.sizing_function(c) for c in centroids])
            tpc = ax.tripcolor(tri, facecolors=vals, cmap="turbo", linewidth=None)
            cbar = plt.colorbar(tpc, ax=ax)
            cbar.set_label("Target size", fontsize=30)
            cbar.ax.tick_params(labelsize=24)
        elif color_by == "projections":
            ax.triplot(tri, color="k", linewidth=0.2, label="Mesh")

            edges = {}
            for tri_nodes in (E2N - 1):
                a, b, c = tri_nodes
                for i, j in ((a, b), (b, c), (c, a)):
                    e = (i, j) if i < j else (j, i)
                    if e in edges:
                        continue
                    edges[e] = True

            for (i, j) in edges.keys():
                p0, p1 = V[i], V[j]
                mid = 0.5 * (p0 + p1)
                proj = self.project_point_to_blade(mid)
                ax.plot([mid[0], proj[0]], [mid[1], proj[1]], color="blue", linewidth=0.15, alpha=0.7)

            # Single legend entry for projections
            from matplotlib.collections import LineCollection

            proj_proxy = LineCollection([], colors="blue", linewidths=0.3, label="Projections")
            ax.add_collection(proj_proxy)
        else:
            raise ValueError("color_by must be None, 'edge_length', 'target_size', or 'projections'")

        ax.set_xlim(1.01 * self.x_left, 1.01 * self.x_right)
        ax.set_ylim(1.01 * self.y_min, 1.01 * self.y_max)
        ax.legend(frameon=False, loc="lower left", fontsize=30)
        ax.axis("off")
        ax.set_aspect("equal", adjustable="box")
        # ax.set_aspect("equal")
        # leave default layout to avoid awkward padding
        fig.tight_layout()

        if color_by == "target_size":
            extension = "png"
        else:
            # extension = "svg"
            extension = "png"
        out_path = dir_base / "output" / f"{fname}.{extension}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, format=extension)
        print(f"Saved {out_path}")
        # plt.show()
        plt.close(fig)






        




def main():
    t_start = time.perf_counter()

    # initialize
    m = mesh_class(label="base", h_bounds=(0.67, 4.15))
    print(f"\nThere are {len(m.E2N)} cells in the base mesh")

    m.write_gri()
    m.label = "2k"

    # m.mesh_verification()

    # m.visual_mesh(fname="mesh0")

    # refine locally
    for _ in range(7): # 7 for report
        m.refinement_local()
        m.plot_by_distance(vmin=0, vmax=2)
    print(f"There are {len(m.E2N)} cells in the mesh after local refinement")
    t_end = time.perf_counter()
    print(f"Time to generate locally refined base mesh: {t_end - t_start:.3f} seconds")
    # m.visual_mesh(fname="mesh1")
    # m.visual_mesh(fname="mesh1_edge_length", color_by="edge_length")
    # m.plot_by_distance()
    # m.visual_mesh(fname="mesh1_projection", color_by="projections")
    print(f"\n local refinement: {len(m.E2N)} cells")
    # m.mesh_verification()
    m.write_gri()
    m.label = "8k"

    # # refine globally afterwards
    for i in range(3): # 3
        m.refinement_global()
        # m.plot_by_distance()
        t_end = time.perf_counter()
        print(f"Time to get to this point: {t_end - t_start:.3f} seconds")
        m.write_gri()
        if i == 0:
            m.label = "32k"
        if i == 1:
            m.label = "128k"
        # m.visual_mesh(fname=f"mesh{2 + i}")
        # m.visual_mesh(fname=f"mesh{2+i}_edge_length", color_by="edge_length")
        # if i == 2: # 2
        #     m.visual_mesh(fname=f"mesh{2+i}_target", color_by="target_size")

        print(f"\n Global refinement {i+1}: {len(m.E2N)} cells")


        # m.mesh_verification()




if __name__ == "__main__":
    main()

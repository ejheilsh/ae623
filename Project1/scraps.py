
def plot_mesh_with_centroids(nodes, elements, boundary_groups, periodic_pairs):
    """Plot the mesh with boundary groups colored differently."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # find nearest boundary node to centroid
    wall_edge_nodes = np.unique(boundary_groups["Wall"])
    n_xs = [nodes[n][0] for n in wall_edge_nodes]
    n_ys = [nodes[n][1] for n in wall_edge_nodes]
    
    # Plot all elements (light gray)
    for elem in elements:
        pts = nodes[elem - 1, :]  # convert to 0-based
        triangle = plt.Polygon(pts, fill=False, edgecolor='lightgray', linewidth=0.3)
        ax.add_patch(triangle)

        x_c, y_c = centroid(pts)
        
        dists = [np.sqrt((x_c - n_xs[i])**2 + (y_c - n_ys[i])**2) for i in range(len(wall_edge_nodes))]
        idxs_min = np.argsort(dists)[:2]
        target = {idxs_min[0], idxs_min[1]}
        edge = None
        for e in boundary_groups["Wall"]:
            if target.issubset(e):
                edge = e
                break


        plt.scatter([x_c], [y_c], color="r", s=1)

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
        nnode = int(np.max(E2N)) + 1  # 0-based nodes
        H = lil_matrix((nnode, nnode), dtype=int)
        IE = np.zeros((int(np.ceil(nelem * 3 / 2)), 4), dtype=int)
        niedge = 0

        for elem in range(nelem):
            nv = E2N[elem, 0:3]
            for edge in range(3):
                n1 = nv[edge % 3]
                n2 = nv[(edge + 1) % 3]
                if H[n1, n2] == 0:
                    H[n1, n2] = elem + 1
                    H[n2, n1] = elem + 1
                else:
                    oldelem = H[n1, n2]
                    if oldelem < 0:
                        raise ValueError("Mesh input error: non-manifold edge")
                    niedge += 1
                    IE[niedge - 1, :] = [n1, n2, oldelem - 1, elem]
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
            BE[b, :] = [I[b], J[b], H[I[b], J[b]] - 1]

        # # need to make one-indexed for nodes, 
        # IE[:, 0:2] += 1
        # BE[:, 0:2] += 1

        self.IE = IE
        self.BE = BE



    def correct_edgehash_for_periodic_boundaries(self, plot_mode="pairs"):
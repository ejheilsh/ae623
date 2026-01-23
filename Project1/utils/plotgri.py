"""
Verify the .gri file by reading it and visualizing the boundary groups.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from utils import config


from scipy.interpolate import CubicSpline
from scipy.optimize import newton, minimize_scalar
from scipy.spatial import KDTree


def read_gri_file(filename):
    """Read a .gri file and return mesh data."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    idx = 0
    nnode, nelem, dim = map(int, lines[idx].split())
    idx += 1
    
    # Read nodes
    nodes = np.zeros((nnode, 2))
    for i in range(nnode):
        nodes[i, :] = list(map(float, lines[idx].split()))
        idx += 1
    
    # Read boundary groups
    nbgroups = int(lines[idx].strip())
    idx += 1
    
    boundary_groups = {}
    for _ in range(nbgroups):
        parts = lines[idx].strip().split()
        nfaces = int(parts[0])
        bname = parts[2]
        idx += 1
        
        edges = np.zeros((nfaces, 2), dtype=int)
        for i in range(nfaces):
            edges[i, :] = list(map(int, lines[idx].split()))
            idx += 1
        
        boundary_groups[bname] = edges
    
    # Read element groups
    parts = lines[idx].strip().split()
    negroups = int(parts[0])
    idx += 1
    
    # Read all element groups
    all_elements = []
    for _ in range(negroups):
        # Element type line
        etype_parts = lines[idx].strip().split()
        idx += 1
        
        # Read elements (until we hit a non-numeric line or end of elements)
        while idx < len(lines):
            line = lines[idx].strip()
            if not line:
                idx += 1
                continue
            
            # Check if line starts with a number (element data)
            first_word = line.split()[0]
            try:
                int(first_word)
                # It's an element line
                elem = list(map(int, line.split()))
                if len(elem) == 3:
                    all_elements.append(elem)
                idx += 1
            except ValueError:
                # Hit a non-numeric line (e.g., "PeriodicGroup")
                break
    
    elements = np.array(all_elements, dtype=int)
    
    # Read periodic groups if present
    periodic_pairs = []
    if idx < len(lines):
        try:
            nPG = int(lines[idx].strip().split()[0])
            idx += 1
            # Skip type line
            idx += 1
            # Read pairs
            while idx < len(lines):
                parts = lines[idx].strip().split()
                if len(parts) == 2:
                    periodic_pairs.append(list(map(int, parts)))
                idx += 1
        except:
            pass

    return nodes, elements, boundary_groups, periodic_pairs

def centroid(nodes_6):
    # nodes_6 is 2D array: [[x1,y1], [x2,y2], [x3,y3]]
    n1 = nodes_6[0]
    n2 = nodes_6[1]
    n3 = nodes_6[2]

    x_c = (n1[0] + n2[0] + n3[0]) / 3
    y_c = (n1[1] + n2[1] + n3[1]) / 3

    return x_c, y_c


def plot_mesh_with_boundaries(nodes, elements, boundary_groups, periodic_pairs):
    """Plot the mesh with boundary groups colored differently."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Plot all elements (light gray)
    for elem in elements:
        pts = nodes[elem - 1, :]  # convert to 0-based
        triangle = plt.Polygon(pts, fill=False, edgecolor='lightgray', linewidth=0.3)
        ax.add_patch(triangle)
    
    # Define colors for each boundary type
    colors = {
        'Wall': 'black',
        'Inflow': 'blue',
        'Outflow': 'red',
        'PeriodicBottom': 'green',
        'PeriodicTop': 'orange'
    }
    
    # Plot boundary edges
    for bname, edges in boundary_groups.items():
        color = colors.get(bname, 'gray')
        for edge in edges:
            p1 = nodes[edge[0] - 1, :]  # convert to 0-based
            p2 = nodes[edge[1] - 1, :]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2, label=bname)
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Mesh with Boundary Groups')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/blade_mesh_boundaries.png', dpi=600)
    print("Saved data/blade_mesh_boundaries.png")
    
    # Plot periodic pairs
    if periodic_pairs:
        fig2, ax2 = plt.subplots(1, 1, figsize=(14, 6))
        
        # Plot mesh outline
        for bname, edges in boundary_groups.items():
            for edge in edges:
                p1 = nodes[edge[0] - 1, :]
                p2 = nodes[edge[1] - 1, :]
                ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=1)
        
        # Plot periodic pairs as vertical lines
        for pair in periodic_pairs:
            p_bottom = nodes[pair[0] - 1, :]
            p_top = nodes[pair[1] - 1, :]
            ax2.plot([p_bottom[0], p_top[0]], [p_bottom[1], p_top[1]], 
                    'r--', linewidth=1, alpha=0.5)
            ax2.plot(p_bottom[0], p_bottom[1], 'go', markersize=4)
            ax2.plot(p_top[0], p_top[1], 'bo', markersize=4)
        
        ax2.set_aspect('equal')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(f'Periodic Node Pairs ({len(periodic_pairs)} pairs)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/blade_mesh_periodic.png', dpi=600)
        print("Saved data/blade_mesh_periodic.png")
    
    #plt.show()


def plot_mesh_with_centroids(nodes, elements, boundary_groups, periodic_pairs,
                             upper_path="data/bladeupper.txt",
                             lower_path="data/bladelower.txt"):
    """
    Plot the mesh with centroids and their projections onto cubic spline surfaces
    built from blade upper/lower coordinate files.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # ---- load blade surface points and build splines ----
    upper = np.loadtxt(upper_path)
    lower = np.loadtxt(lower_path)
    lower[:, 1] += config.H_box
    x_upper, y_upper = upper[:, 0], upper[:, 1]
    x_lower, y_lower = lower[:, 0], lower[:, 1]

    # sort by x for spline construction
    i_up = np.argsort(x_upper)
    i_lo = np.argsort(x_lower)
    x_upper, y_upper = x_upper[i_up], y_upper[i_up]
    x_lower, y_lower = x_lower[i_lo], y_lower[i_lo]

    spline_upper = CubicSpline(x_upper, y_upper, bc_type="natural")
    spline_lower = CubicSpline(x_lower, y_lower, bc_type="natural")

    # plot splines
    xs_up = np.linspace(x_upper.min(), x_upper.max(), 500)
    xs_lo = np.linspace(x_lower.min(), x_lower.max(), 500)
    ax.plot(xs_up, spline_upper(xs_up), "k-", lw=1.5, label="Upper spline")
    ax.plot(xs_lo, spline_lower(xs_lo), "k-", lw=1.5, label="Lower spline")

    from scipy.optimize import minimize_scalar

    def centroid(tri_pts):
        return tri_pts.mean(axis=0)

    def project_point_to_spline(xc, yc, spline, x_bracket):
        def f(x):
            y = spline(x)
            return (x - xc) ** 2 + (y - yc) ** 2

        res = minimize_scalar(f, bounds=x_bracket, method="bounded")
        xb = res.x
        yb = spline(xb)
        return xb, yb, f(xb)

    # ---- collect centroids and projections ----
    centroids = []
    projections = []
    segments = []

    for elem in elements:
        pts = nodes[elem - 1, :]
        xc, yc = centroid(pts)
        centroids.append([xc, yc])

        # project to upper and lower splines and pick the closer one
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

    centroids = np.asarray(centroids)
    projections = np.asarray(projections)

    # ---- plot mesh ----
    for elem in elements:
        pts = nodes[elem - 1, :]
        ax.add_patch(plt.Polygon(pts, fill=False, edgecolor='lightgray', linewidth=0.3))

    # ---- plot centroids + projections + segments ----
    ax.scatter(centroids[:, 0], centroids[:, 1], s=3, c="r", label="Centroids")
    ax.scatter(projections[:, 0], projections[:, 1], s=6, c="b", label="Projection")

    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, colors="r", linewidths=0.5, linestyles="dashed")
    ax.add_collection(lc)

    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pygmsh as pg
from utils import config, plotgri, E2N, normals_area, write_gri
from utils import plotgri
"""
We have two txt files of blade upper and lower in data form in xy coordinate format
We want to first read these files and then plot them to see the blade shape.
Then we want to create connectivity of these using Delaunay triangulation and plot the mesh.
 to save the mesh in a .gri file format. 
This is a new line
"""

def read_xy_file(fname):
    """
    Read XY coordinates from txt files and return as numpy array
    """
    data = np.loadtxt(fname)
    return data

def resample_curve_uniform_x(xy_curve, n_points):
    """Resample a curve to have exactly n_points at uniform x-spacing."""
    xy_sorted = xy_curve[np.argsort(xy_curve[:, 0])]
    x_min, x_max = xy_sorted[0, 0], xy_sorted[-1, 0]
    x_new = np.linspace(x_min, x_max, n_points)
    y_new = np.interp(x_new, xy_sorted[:, 0], xy_sorted[:, 1])
    return np.column_stack([x_new, y_new])


def mesh_with_pygmsh(poly_xy, mesh_size, periodic_pairs: list = None):
    """Generate an unstructured triangular mesh inside poly_xy using pygmsh/Gmsh.
    
    For periodic boundaries, provide periodic_pairs as list of (idx1, idx2) tuples
    indicating which boundary segments are periodic.
    """
    
    # Remove consecutive duplicate points (gmsh hangs on duplicates)
    tol = 1e-10
    unique_pts = [poly_xy[0]]
    for i in range(1, len(poly_xy)):
        if np.linalg.norm(poly_xy[i] - unique_pts[-1]) > tol:
            unique_pts.append(poly_xy[i])
    # Check if last point duplicates first
    if np.linalg.norm(unique_pts[-1] - unique_pts[0]) < tol:
        unique_pts = unique_pts[:-1]
    
    poly_clean = np.array(unique_pts)
    print(f"Cleaned polygon: {len(poly_xy)} -> {len(poly_clean)} points")

    with pg.geo.Geometry() as geom:
        geom.characteristic_length_min = mesh_size
        geom.characteristic_length_max = mesh_size

        pts = [geom.add_point([x, y, 0.0], mesh_size=mesh_size) for x, y in poly_clean]
        lines = [geom.add_line(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        
        # Apply periodic constraints if provided
        if periodic_pairs:
            for idx1, idx2 in periodic_pairs:
                if idx1 < len(lines) and idx2 < len(lines):
                    try:
                        geom.set_periodic(lines[idx2], lines[idx1])
                        print(f"Set periodic constraint: line {idx1} <-> line {idx2}")
                    except Exception as e:
                        print(f"Warning: Could not set periodic constraint {idx1}<->{idx2}: {e}")
        
        loop = geom.add_curve_loop(lines)
        geom.add_plane_surface(loop)
        print("Generating mesh with pygmsh (this may take a moment)...")
        mesh = geom.generate_mesh(dim=2, verbose=True)

    V = mesh.points[:, :2]
    E = np.asarray(mesh.cells_dict["triangle"], dtype=int)

    return V, E

class Mesh:
    def __init__(self, V, E, B, Bname):
        self.V = V          # Vertices
        self.E = E          # Elements
        self.B = B          # Boundaries
        self.Bname = Bname  # Boundary names


def initialize_xy_splines():
    """
    Need to get XY points -> splines for the GMSH software to read and mesh
    1. Read the upper and lower blade files
    2. Get the leftmost and rightmost points of the upper blade
    3. Create a box around the airfoil using L_box and H_box from config.py
    4. Resample the upper and lower surfaces to have n_wall_pts each
    5. Return the polygon points for meshing
    """

    xy_data_upper = read_xy_file('data/bladeupper.txt')
    xy_data_lower = read_xy_file('data/bladelower.txt')
    
    x_left_upper_airfoil = xy_data_upper[np.argmin(xy_data_upper[:,0]), 0]
    y_left_upper_airfoil = xy_data_upper[np.argmin(xy_data_upper[:,0]), 1]
    print("Leftmost point of upper airfoil: (", x_left_upper_airfoil, ", ", y_left_upper_airfoil, ")")
    x_right_upper_airfoil = xy_data_upper[np.argmax(xy_data_upper[:,0]), 0]
    y_right_upper_airfoil = xy_data_upper[np.argmax(xy_data_upper[:,0]), 1]
    print("Rightmost point of upper airfoil: (", x_right_upper_airfoil, ", ", y_right_upper_airfoil, ")")
    
    # CREATE LEFT BOX (USE THE LEFT HAND SIDE OF ARIFOIL AT (X_MIN, Y_MIN) TO THE L=17mm, H=20mm)
    L_box = config.L_box
    H_box = config.H_box
    n_wall_pts = config.n_wall_pts  # Number of points on each airfoil surface
    
    # Transpose the lower surface up H_box to create a gap between upper and lower surfaces
    xy_data_lower[:,1] += H_box

    # Build a closed boundary polygon for the passage and let Gmsh do the meshing.
    # IMPORTANT: Resample walls to have matching x-coordinates for periodic BCs
    bottom = resample_curve_uniform_x(xy_data_upper, n_wall_pts)
    top = resample_curve_uniform_x(xy_data_lower, n_wall_pts)

    xL = x_left_upper_airfoil - L_box
    xR = x_right_upper_airfoil + L_box
    yL0 = y_left_upper_airfoil
    yL1 = y_left_upper_airfoil + H_box
    yR0 = y_right_upper_airfoil
    yR1 = y_right_upper_airfoil + H_box

    poly = []
    poly.append([xL, yL0])
    poly.append([x_left_upper_airfoil, yL0])
    poly.extend(bottom.tolist())
    poly.append([x_right_upper_airfoil, yR0])
    poly.append([xR, yR0])
    poly.append([xR, yR1])
    poly.append([x_right_upper_airfoil, yR1])
    poly.extend(top[::-1].tolist())
    poly.append([x_left_upper_airfoil, yL1])
    poly.append([xL, yL1])

    poly_xy = np.asarray(poly, dtype=float)
    print("Polygon for meshing has", poly_xy.shape[0], "points.")
    print(f"Bottom wall: {len(bottom)} points, Top wall: {len(top)} points (MATCHED)")
    
    # Return geometry info for boundary classification
    geom_info = {
        'x_left_upper_airfoil': x_left_upper_airfoil,
        'y_left_upper_airfoil': y_left_upper_airfoil,
        'x_right_upper_airfoil': x_right_upper_airfoil,
        'y_right_upper_airfoil': y_right_upper_airfoil,
        'x_left_box': xL,
        'x_right_box': xR,
        'y_bottom_left_box': yL0,
        'y_top_left_box': yL1,
        'y_bottom_right_box': yR0,
        'y_top_right_box': yR1,
        'L_box': L_box,
        'H_box': H_box
    }
    return poly_xy, geom_info

def mesh_verification(N_interior, N_boundary, Area, I2E_matrix, B2E_matrix, Mesh_obj):
    """
    As part of this project, you will be computing normal vectors and lengths for interior and boundary
    faces. Implement the following mesh veriﬁcation test of these quantities:
     Loop over all interior and boundary faces in your mesh.
     On each face, calculate the normal vector, ~n. Make this vector point out of the domain for
    boundary faces and from a consistently-chosen L to R element pair for interior faces.
     On each face, multiply the normal by the face length l and add/subtract the result to/from
    a running total on the adjacent elements. The goal is by the end of your calculation to have
    computed;
    For each element e, compute:
    E_e = sum_{i=1}^{3} n_{ei}^{outward} * l_{ei}
    """
    E_totals = np.zeros((Mesh_obj.E.shape[0], 2), dtype=float)
    # 1. Loop over all interior and boundary faces in your mesh.
    for i in range(I2E_matrix.shape[0]):
        n1, n2, elemL, elemR = I2E_matrix[i, :]
        p1 = Mesh_obj.V[n1 - 1, :]  # Convert back to 0-based
        p2 = Mesh_obj.V[n2 - 1, :]
        edge_vec = p2 - p1
        edge_length = np.linalg.norm(edge_vec)
        normal = np.array([-edge_vec[1], edge_vec[0]])
        normal /= np.linalg.norm(normal)
        
        # Ensure normal points from L to R element
        centerL = np.mean(Mesh_obj.V[Mesh_obj.E[elemL - 1, :], :], axis=0)
        centerR = np.mean(Mesh_obj.V[Mesh_obj.E[elemR - 1, :], :], axis=0)
        toR = centerR - centerL
        if np.dot(normal, toR) < 0:
            normal = -normal
        
        # Multiply normal by edge length
        contribution = normal * edge_length
        # Add to running total for elemL and subtract for elemR
        # For elemL, this is an OUTWARD normal, so ADD
        E_totals[elemL - 1, :] += contribution
        
        # For elemR, this is an INWARD normal, so SUBTRACT
        E_totals[elemR - 1, :] -= contribution
    
    for i in range(B2E_matrix.shape[0]):
        n1, n2, elem = B2E_matrix[i, :]
        p1 = Mesh_obj.V[n1 - 1, :]
        p2 = Mesh_obj.V[n2 - 1, :]
        edge_vec = p2 - p1
        edge_length = np.linalg.norm(edge_vec)
        normal = np.array([-edge_vec[1], edge_vec[0]])
        normal /= np.linalg.norm(normal)
        
        # Ensure normal points outward from the element
        centerE = np.mean(Mesh_obj.V[Mesh_obj.E[elem - 1, :], :], axis=0)
        mid_edge = (p1 + p2) / 2
        toOutside = mid_edge - centerE
        if np.dot(normal, toOutside) < 0:
            normal = -normal
        
        # Multiply normal by edge length
        contribution = normal * edge_length
        # Add to running total for elem (boundary is always outward)
        E_totals[elem - 1, :] += contribution
    print("Element totals after verification (should be close to zero):")
    print(E_totals)
    # Compute magnitude of error for each element
    E_magnitudes = np.linalg.norm(E_totals, axis=1)
    
    max_error = np.max(E_magnitudes)
    mean_error = np.mean(E_magnitudes)
    
    print(f"Maximum error magnitude: {max_error:.6e}")
    print(f"Mean error magnitude:    {mean_error:.6e}")
    print(f"Number of elements:      {Mesh_obj.E.shape[0]}")



#-----------------------------------------------------------
def main():
    # GATHER V, E DATA FROM MSH FILE
    poly_xy, geom_info = initialize_xy_splines()
    V, E = mesh_with_pygmsh(poly_xy, mesh_size=config.initial_mesh_size)
    
    B = ['Periodic', 'Wall']  # Boundary data can be added if needed
    Bname = []  # Boundary names can be added if needed
    Mesh_obj = Mesh(V, E, B, Bname)
    print("Vertices (V):"   , Mesh_obj.V.shape)
    print("Elements (E):"   , Mesh_obj.E.shape)
    # print the faces to elements mapping
    [I2E_matrix, B2E_matrix] = E2N.edgehash(Mesh_obj.E + 1)  # +1 for 1-based indexing
    [N_interior, N_boundary, Area] = normals_area.compute_normals_and_areas(Mesh_obj.V, Mesh_obj.E)
    print("Interior Normals:" , N_interior.shape)
    print("Boundary Normals:" , N_boundary.shape)
    print("Element Areas:"    , Area.shape)
    print("I2E Matrix:"     , I2E_matrix.shape)
    print("B2E Matrix:"     , B2E_matrix.shape)
    # Plot the mesh
    Mesh_dict = {'V': Mesh_obj.V, 'E': Mesh_obj.E, 'B': Mesh_obj.B, 'Bname': Mesh_obj.Bname}
    # Verify the normals and areas
    mesh_verification(N_interior, N_boundary, Area, I2E_matrix, B2E_matrix, Mesh_obj)
    
    # Save to GRI file with proper boundary classification
    write_gri.save_mesh_from_coarse_mesh(
        Mesh_obj, I2E_matrix, B2E_matrix,
        y_bottom_left=geom_info['y_bottom_left_box'],
        y_top_left=geom_info['y_top_left_box'],
        y_bottom_right=geom_info['y_bottom_right_box'],
        y_top_right=geom_info['y_top_right_box'],
        x_left=geom_info['x_left_box'],
        x_right=geom_info['x_right_box'],
        airfoil_x_min=geom_info['x_left_upper_airfoil'],
        airfoil_x_max=geom_info['x_right_upper_airfoil'],
        filename='data/coarse_blade_mesh.gri'
    )
    # Plot mesh in GRI
    nodes, elements, boundary_groups, periodic_pairs = plotgri.read_gri_file('data/coarse_blade_mesh.gri')
    # plotgri.plot_mesh_with_boundaries(nodes, elements, boundary_groups, periodic_pairs)
    plotgri.plot_mesh_with_centroids(nodes, elements, boundary_groups, periodic_pairs)

if __name__ == "__main__":
    main()

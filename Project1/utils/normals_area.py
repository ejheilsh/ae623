import numpy as np
from utils.E2N import edgehash
"""
In: normal vectors for interior faces. The number of rows is the number of interior faces.
Each row contains two floating-point numbers: the x and y components of the normal
that points from the L to the R element.
 Bn: normal vectors for boundary faces. The number of rows is the total number of
boundary faces. Each row contains two floating-point numbers: the x and y components
of the normal that points outward from the adjacent element.
 Area: the area of each element. The number of rows is the total number of elements,
and each row contains one number: the area of the element.
"""

def compute_normals_and_areas(V, E):
    """
    Compute normal vectors for interior and boundary faces,
    and area of each triangular element.
    
    Inputs:
    V : [nnode x 2] array of vertex coordinates
    E : [nelem x 3] array of element-to-node connectivity (0-based)
    
    Outputs:
    N_interior : [niedge x 2] array of interior face normals
    N_boundary : [nbedge x 2] array of boundary face normals
    Area : [nelem x 1] array of element areas
    """
    nelem = E.shape[0]
    nnode = V.shape[0]
    
    # SHOELACE FORMULA FOR AREA OF TRIANGLE
    Area = np.zeros(nelem)
    for elem in range(nelem):
        nv = E[elem, :]
        x1, y1 = V[nv[0], :]
        x2, y2 = V[nv[1], :]
        x3, y3 = V[nv[2], :]
        Area[elem] = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    
    IE, BE = edgehash(E + 1)  # Convert to 1-based for edgehash function
    
    # INT. FACE NORMALS
    N_interior = np.zeros((IE.shape[0], 2))
    for i in range(IE.shape[0]):
        n1, n2, elemL, elemR = IE[i, :]
        p1 = V[n1 - 1, :]  # Convert back to 0-based
        p2 = V[n2 - 1, :]
        edge_vec = p2 - p1
        normal = np.array([-edge_vec[1], edge_vec[0]])
        normal /= np.linalg.norm(normal)
        
        # Ensure normal points from L to R element
        centerL = np.mean(V[E[elemL - 1, :] - 1, :], axis=0)
        centerR = np.mean(V[E[elemR - 1, :] - 1, :], axis=0)
        toR = centerR - centerL
        if np.dot(normal, toR) < 0:
            normal = -normal
        N_interior[i, :] = normal
        
    # BOUNDARY FACE NORMALS
    N_boundary = np.zeros((BE.shape[0], 2))
    for i in range(BE.shape[0]):
        n1, n2, elem = BE[i, :]
        p1 = V[n1 - 1, :]
        p2 = V[n2 - 1, :]
        edge_vec = p2 - p1
        normal = np.array([-edge_vec[1], edge_vec[0]])
        normal /= np.linalg.norm(normal)
        
        # Ensure normal points outward from the adjacent element
        center = np.mean(V[E[elem - 1, :] - 1, :], axis=0)
        to_out = (p1 + p2) / 2 - center
        if np.dot(normal, to_out) < 0:
            normal = -normal
        N_boundary[i, :] = normal

    return N_interior, N_boundary, Area
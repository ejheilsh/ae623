"""
An "element-to-node" array,
E2N of size [nelem × 3], for a
triangular mesh
"""
import numpy as np
from scipy.sparse import lil_matrix

def edgehash(E2N):
    # This function identifies interior and boundary edges, and their
    # connectivities, in a triangular mesh given an element-to-node array.
    #
    # INPUT : E2N = [nelem x 3] array mapping triangles to nodes (1-based)
    # OUTPUT: IE = [niedge x 4] array giving (n1, n2, elem1, elem2)
    #         information for each interior edge
    #         BE = [nbedge x 3] array giving (n1, n2, elem)
    #         information for each boundary edge
    
    nelem = E2N.shape[0]  # number of elements
    nnode = np.max(E2N)   # number of nodes
    H = lil_matrix((nnode+1, nnode+1), dtype=int)  # Create a hash list to identify edges
    IE = np.zeros((int(np.ceil(nelem*3/2)), 4), dtype=int)  # (over) allocate interior edge array
    niedge = 0  # number of interior edges (running total)
    
    # Loop over elements and identify all edges
    for elem in range(1, nelem+1):
        nv = E2N[elem-1, 0:3]  # Python 0-based array indexing, but node values are 1-based
        for edge in range(1, 4):
            n1 = nv[(edge) % 3]
            n2 = nv[(edge+1) % 3]
            if H[n1, n2] == 0:  # edge hit for the first time
                # could be a boundary or interior; assume boundary
                H[n1, n2] = elem
                H[n2, n1] = elem
            else:  # this is an interior edge, hit for the second time
                oldelem = H[n1, n2]
                if oldelem < 0:
                    raise ValueError('Mesh input error')
                niedge = niedge + 1
                IE[niedge-1, :] = [n1, n2, oldelem, elem]  # Python 0-based array indexing
                H[n1, n2] = -1
                H[n2, n1] = -1
    
    IE = IE[0:niedge, :]  # clip IE
    
    # find boundary edges
    H_coo = H.tocoo()
    I, J = [], []
    for i, j, v in zip(H_coo.row, H_coo.col, H_coo.data):
        if i < j and v > 0:  # upper triangle
            I.append(i)
            J.append(j)
    
    BE = np.zeros((len(I), 3), dtype=int)
    for b in range(len(I)):
        BE[b, :] = [I[b], J[b], H[I[b], J[b]]]
    
    return IE, BE

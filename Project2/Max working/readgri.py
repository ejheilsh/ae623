import numpy as np
from scipy import sparse
from pathlib import Path
"""
Max Mah 
2/11/26

Functions for mesh preprocessing. Together, they read in .gri file and get all data structures 
- edgehash() and readgri() functions adapted from kfid provided readgri.py file 
- added get_normals() function and get_areas_centroids() function to be called inside readgri() to get additional Mesh data structures 

Inputs:
-------
gristr: str
    string for the mesh .gri file name 
    
Outputs: 
-------
Mesh: dict
    Mesh = {'V':V, 'E':E, 'IE':IE, 'BE':BE, 'Bname':Bname, 
            'Centroid': Centroid, 
            'Area': Area, 
            'In': In, 
            'Bn': Bn}
            
            Dim ;       Row values
V =         [Nn x 2];   (x, y)
E =         [Ne x 3];   (n1, n2, n3)
IE =        [NIe x 4];  (n1, n2, elemL, elemR)
BE =        [NBe x 4];  (n1, n2, elem, bindex)
Bname =     [1 x 3];    (bname1, bname2, ..., bnamei)
Centroid =  [Ne x 2];   (xc, yc)
Area =      [Ne x 1];   (A,)
In =        [NIe x 3];  (nx, ny, length)
Bn =        [NBe x 3];  (nx, ny, length)
    
NOTE  
Eric's .gri file has boundary edges in format (elem index, local face index) instead of expected (node1, node2)
so readgri() function was modified to get it in appropriate form 
"""
#-----------------------------------------------------------
# Identifies interior and boundary edges given element-to-node
# IE contains (n1, n2, elem1, elem2) for each interior edge
# BE contains (n1, n2, elem) for each boundary edge
def edgehash(E, B):
    Ne = E.shape[0]; Nn = np.amax(E)+1
    H = sparse.lil_matrix((Nn, Nn), dtype=int)
    IE = np.zeros([int(np.ceil(Ne*1.5)),4], dtype=int)
    ni = 0
    for e in range(Ne):
        for i in range(3):
            n1, n2 = E[e,i], E[e,(i+1)%3]
            if (H[n2,n1] == 0):
                H[n1,n2] = e+1
            else:
                eR = H[n2,n1]-1
                IE[ni,:] = n1, n2, e, eR
                H[n2,n1] = 0
                ni += 1
    IE = IE[0:ni,:]
    # boundaries
    nb0 = nb = 0
    for g in range(len(B)): nb0 += B[g].shape[0]
    BE = np.zeros([nb0,4], dtype=int)
    for g in range(len(B)):
        Bi = B[g]
        for b in range(Bi.shape[0]):
            n1, n2 = Bi[b,0], Bi[b,1]
            if (H[n1,n2] == 0): n1,n2 = n2,n1 # NOTE if the stored direction is reversed, it swaps n1, n2 so that +90 deg rotation produces outward normal for domain
            BE[nb,:] = n1, n2, H[n1,n2]-1, g  
            nb += 1
    return IE, BE

#-----------------------------------------------------------
def _parse_periodic_groups(f):
    """
    Parse optional trailing periodic-group section in a .gri file.
    Expected format (1-based node ids):
        <Ngroups> PeriodicGroup
        <Np> <ptype>
        n1 n2
        ...
    Returns a list of [Np x 2] integer arrays in 0-based node indexing.
    """
    remainder = [ln.strip() for ln in f.readlines() if ln.strip()]
    if len(remainder) == 0:
        return []

    head = remainder[0].split()
    if len(head) < 2 or head[1] != 'PeriodicGroup':
        return []

    ngroups = int(head[0])
    groups = []
    cursor = 1
    for _ in range(ngroups):
        if cursor >= len(remainder):
            break
        s = remainder[cursor].split()
        cursor += 1
        if len(s) < 1:
            continue
        npairs = int(s[0])
        if npairs <= 0:
            groups.append(np.zeros((0, 2), dtype=int))
            continue
        if cursor + npairs > len(remainder):
            raise ValueError("Malformed PeriodicGroup section: insufficient node-pair rows")
        pairs = np.array(
            [[int(t) - 1 for t in remainder[cursor + i].split()] for i in range(npairs)],
            dtype=int,
        )
        cursor += npairs
        groups.append(pairs)
    return groups

def _append_periodic_to_ie(V, E, IE, BE, periodic_groups):
    """
    Build periodic interior faces from node correspondences and append to IE.
    Also removes periodic faces from BE so they are not treated as physical boundaries.
    """
    if len(periodic_groups) == 0:
        return IE, BE, 0, 0

    # map unordered edge -> adjacent element ids from connectivity
    edge_to_elems = {}
    edge_oriented = {}
    for elem in range(E.shape[0]):
        n0, n1, n2 = [int(x) for x in E[elem, :3]]
        for a, b in ((n0, n1), (n1, n2), (n2, n0)):
            key = tuple(sorted((a, b)))
            edge_to_elems.setdefault(key, []).append(elem)
            if key not in edge_oriented:
                edge_oriented[key] = (a, b, elem)

    # existing IE edges (unordered) to avoid duplicate insertion
    ie_keys = set()
    for k in range(IE.shape[0]):
        key = tuple(sorted((int(IE[k, 0]), int(IE[k, 1]))))
        ie_keys.add(key)

    # boundary edge lookup (for optional periodic-face removal)
    be_key_to_idx = {}
    for k in range(BE.shape[0]):
        key = tuple(sorted((int(BE[k, 0]), int(BE[k, 1]))))
        be_key_to_idx.setdefault(key, []).append(k)

    remove_idx = set()
    new_ie = []
    unresolved = []
    expected_added = 0
    tol = 1e-8

    for gidx, node_pairs in enumerate(periodic_groups):
        if node_pairs.shape[0] < 2:
            continue

        bottom_nodes = set(int(n) for n in node_pairs[:, 0])
        top_nodes = set(int(n) for n in node_pairs[:, 1])

        # Build periodic boundary-edge pool for this group from connectivity (not from BE).
        periodic_edges = []
        for key, elems in edge_to_elems.items():
            if len(elems) != 1:
                continue
            a, b = key
            in_bottom = (a in bottom_nodes) and (b in bottom_nodes)
            in_top = (a in top_nodes) and (b in top_nodes)
            if not (in_bottom or in_top):
                continue
            ao, bo, elem = edge_oriented[key]
            x1 = V[ao, 0]
            x2 = V[bo, 0]
            periodic_edges.append((ao, bo, int(elem), in_bottom, np.sort([x1, x2])))

        if len(periodic_edges) == 0:
            continue

        # Pair periodic edges by matching x endpoints (v2.py logic).
        x_pair = np.array([pe[4] for pe in periodic_edges], dtype=float)
        x_key = np.round(x_pair / tol).astype(int)
        used = np.zeros(len(periodic_edges), dtype=bool)
        pairs = []
        for i in range(len(periodic_edges)):
            if used[i]:
                continue
            same_key = np.all(x_key == x_key[i], axis=1)
            idx = np.where(same_key)[0]
            if len(idx) == 2:
                pairs.append((int(idx[0]), int(idx[1])))
                used[idx] = True
            else:
                unresolved.append((gidx, i, x_pair[i].tolist(), int(len(idx))))
        expected_added += len(pairs)

        # Move paired periodic edges into IE.
        for i, j in pairs:
            e1 = periodic_edges[i]
            e2 = periodic_edges[j]

            # choose bottom edge as IE edge geometry
            if e1[3] and (not e2[3]):
                eb, et = e1, e2
            elif e2[3] and (not e1[3]):
                eb, et = e2, e1
            else:
                # fallback if "bottom/top" classification is ambiguous
                eb, et = e1, e2

            edge_bottom = (int(eb[0]), int(eb[1]))
            key_bottom = tuple(sorted(edge_bottom))
            key_top = tuple(sorted((int(et[0]), int(et[1]))))

            if key_bottom not in ie_keys:
                n1b, n2b = int(edge_bottom[0]), int(edge_bottom[1])
                # Enforce canonical orientation: first node is the "left" node.
                # This also gives ny < 0 for these near-horizontal periodic edges
                # because get_normals uses n = [y2 - y1, x1 - x2].
                x1b, x2b = V[n1b, 0], V[n2b, 0]
                y1b, y2b = V[n1b, 1], V[n2b, 1]
                if (x1b > x2b) or (np.isclose(x1b, x2b) and y1b > y2b):
                    n1b, n2b = n2b, n1b
                new_ie.append([n1b, n2b, int(eb[2]), int(et[2])])
                ie_keys.add(key_bottom)

            # remove periodic sides from BE when present (some files may still include them)
            for k in be_key_to_idx.get(key_bottom, []):
                remove_idx.add(k)
            for k in be_key_to_idx.get(key_top, []):
                remove_idx.add(k)

    added_count = len(new_ie)
    if len(unresolved) > 0:
        samples = unresolved[:5]
        print(
            f"WARNING: Unpaired periodic boundary-edge candidates in group matching. "
            f"Count={len(unresolved)}. Samples={samples}"
        )

    if len(new_ie) == 0:
        return IE, BE, added_count, expected_added

    IE_out = np.vstack([IE, np.asarray(new_ie, dtype=int)])
    keep_mask = np.ones(BE.shape[0], dtype=bool)
    keep_mask[list(remove_idx)] = False
    BE_out = BE[keep_mask]
    return IE_out, BE_out, added_count, expected_added

#-----------------------------------------------------------
def readgri(fname):
    with open(fname, 'r') as f:
        Nn, Ne, dim = [int(s) for s in f.readline().split()]
        # read vertices
        V = np.array([[float(s) for s in f.readline().split()] for n in range(Nn)])
        # read boundaries
        NB = int(f.readline())
        Braw = []; Bname = []; Bnnode = [] # changes boundary storage 
        for i in range(NB):
            s = f.readline().split(); Nb = int(s[0]); Bnnode.append(int(s[1])); Bname.append(s[2]) # stores boundary rows without -1 
            Bi = np.array([[int(t) for t in f.readline().split()] for n in range(Nb)], dtype=int)
            Braw.append(Bi)
        # read elements
        Ne0 = 0; E = []
        while (Ne0 < Ne):
            s = f.readline().split(); ne = int(s[0])
            Ei = np.array([[int(s)-1 for s in f.readline().split()] for n in range(ne)]) # element node IDs are 1-based, np arrays are 0-based, so subtract 1
            E = Ei if (Ne0==0) else np.concatenate((E,Ei), axis=0)
            Ne0 += ne
        periodic_groups = _parse_periodic_groups(f)
    # convert boundaries to node-pair format (0-based).
    # Some .gri files store boundary faces as (elem, local_face) instead of (n1, n2).
    B = []
    for i in range(NB):
        Bi = Braw[i]
        if Bi.size == 0:
            B.append(Bi)
            continue
        is_elem_face = (
            Bi.shape[1] == 2
            and Bnnode[i] == 2
            and np.max(Bi[:, 1]) <= 3
            and np.max(Bi[:, 0]) <= Ne
        )
        if is_elem_face: # for (elem, local_face) format, maps face id to edge nodes 
            elem_idx = Bi[:, 0] - 1 # convert element IDs from 1-based to 0-based 
            face_idx = Bi[:, 1] - 1
            # In this .gri variant, local face id is opposite local node id:
            # face 1 -> edge (2,3), face 2 -> edge (3,1), face 3 -> edge (1,2).
            n1 = E[elem_idx, (face_idx + 1) % 3]
            n2 = E[elem_idx, (face_idx + 2) % 3]
            B.append(np.column_stack((n1, n2)).astype(int))
        else:
            B.append(Bi - 1) 
    # make IE, BE structures
    IE, BE = edgehash(E, B)
    IE, BE, periodic_ie_added, periodic_ie_expected = _append_periodic_to_ie(V, E, IE, BE, periodic_groups)
    
    # NOTE added normals, centroids, areas 
    In, Bn = get_normals(V, IE, BE) # computes normals and edge lengths
    Centroid, Area = get_centroids_areas(V, E) # computes element centroids and areas 
    
    periodic_pairs = (
        np.vstack(periodic_groups).astype(int)
        if len(periodic_groups) > 0
        else np.zeros((0, 2), dtype=int)
    )

    Mesh = {'V':V, 'E':E, 'IE':IE, 'BE':BE, 'Bname':Bname, 
            'Centroid': Centroid, 
            'Area': Area, 
            'In': In, 
            'Bn': Bn,
            'PeriodicGroups': periodic_groups,
            'PeriodicPairs': periodic_pairs,
            'PeriodicIEAdded': periodic_ie_added,
            'PeriodicIEExpected': periodic_ie_expected}
    return Mesh

def get_normals(V, IE, BE):
    # compute normals of interior edges 
    n1i, n2i = IE[:, 0], IE[:, 1]
    x1i, y1i = V[n1i, 0], V[n1i, 1]
    x2i, y2i = V[n2i, 0], V[n2i, 1]
    inormal = np.column_stack([y2i-y1i, x1i-x2i])
    ilength = np.linalg.norm(inormal, axis=1)
    # ilength = np.sqrt(inormal[:,0]**2 + inormal[:,1]**2)
    inormal /= ilength[:, None] 
    # NOTE do i need to normalize to normal and then multiply by length later or can i just keep the length of the normal? 
    
    # compute normals of boundary edges 
    n1b, n2b = BE[:, 0], BE[:, 1]
    x1b, y1b = V[n1b, 0], V[n1b, 1]
    x2b, y2b = V[n2b, 0], V[n2b, 1]
    bnormal = np.column_stack([y2b - y1b, x1b - x2b])
    blength = np.linalg.norm(bnormal, axis=1)
    bnormal /= blength[:, None] 
    
    In = np.column_stack([inormal, ilength])
    Bn = np.column_stack([bnormal, blength])
    
    return In, Bn

def get_centroids_areas(V, E):
    # compute areas and centroids 
    n1, n2, n3 = E[:,0], E[:,1], E[:,2]
    x1, y1 = V[n1, 0], V[n1, 1]
    x2, y2 = V[n2, 0], V[n2, 1]
    x3, y3 = V[n3, 0], V[n3, 1]
    
    xc = (x1 + x2 + x3)/3
    yc = (y1 + y2 + y3)/3
    
    Centroid = np.column_stack([xc, yc])
    Area = 0.5*np.abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
    return Centroid, Area 
    
#-----------------------------------------------------------
def main():
    meshstr = '2k.gri'

    dir_base = Path(__file__).resolve().parent
    grifile = dir_base.joinpath(meshstr)
    Mesh = readgri(grifile)
    print(Mesh) 
    

if __name__ == "__main__":
    main()

import os
import struct

import numpy as np


GAMMA = 1.4

_GRI_TO_SHAPE = {
    1: [0, 1, 2],
    2: [0, 2, 5, 4, 3, 1],
    3: [0, 3, 9, 6, 8, 7, 4, 1, 2, 5],
}

_POLYGON_ORDER = {
    1: [0, 1, 2],
    2: [0, 1, 2, 4, 5, 3],
    3: [0, 1, 2, 3, 6, 8, 9, 7, 4],
}

_REF_SUBMESH_CACHE = {}


def infer_dg_filename(results_filename):
    if results_filename.endswith(".bin"):
        return results_filename[:-4] + "_dg.bin"
    return results_filename + "_dg"


def read_cell_results(filename):
    with open(filename, "rb") as f:
        ne = struct.unpack("i", f.read(4))[0]
        data = struct.unpack("d" * 4 * ne, f.read(8 * 4 * ne))
    return np.array(data).reshape(ne, 4)


def read_dg_results(filename):
    with open(filename, "rb") as f:
        ne = struct.unpack("i", f.read(4))[0]
        p_order = struct.unpack("i", f.read(4))[0]
        ndof = struct.unpack("i", f.read(4))[0]
        data = struct.unpack("d" * 4 * ne * ndof, f.read(8 * 4 * ne * ndof))
    return np.array(data).reshape(ne, ndof, 4), p_order, ndof


def maybe_read_dg_results(results_filename):
    dg_filename = infer_dg_filename(results_filename)
    if os.path.exists(dg_filename):
        U_dg, p_order, ndof = read_dg_results(dg_filename)
        return dg_filename, U_dg, p_order, ndof
    return None, None, None, None


def read_gri_mesh(filename):
    with open(filename, "r") as f:
        nn, ne, dim = map(int, f.readline().split())
        nodes = np.array([[float(s) for s in f.readline().split()] for _ in range(nn)])

        nb = int(f.readline())
        boundary_groups = {}
        for _ in range(nb):
            parts = f.readline().split()
            nedge = int(parts[0])
            name = parts[2] if len(parts) > 2 else f"boundary_{len(boundary_groups)}"
            edges = []
            for _ in range(nedge):
                a, b = map(int, f.readline().split()[:2])
                edges.append((a - 1, b - 1))
            boundary_groups[name] = edges

        elements = []
        ne_total = 0
        while ne_total < ne:
            line = f.readline().split()
            if not line:
                break
            nei = int(line[0])
            q = int(line[1])
            for _ in range(nei):
                row = [int(s) - 1 for s in f.readline().split()]
                corners = [row[0], row[q], row[-1]]
                elements.append(
                    {
                        "q": q,
                        "row": row,
                        "corners": corners,
                    }
                )
            ne_total += nei

    return {
        "nodes": nodes,
        "elements": elements,
        "boundary_groups": boundary_groups,
    }


def evaluate_basis(xi, eta, p):
    if p == 0:
        return np.array([1.0])
    if p == 1:
        return np.array([1.0 - xi - eta, xi, eta])
    if p == 2:
        return np.array(
            [
                1.0 - 3.0 * xi - 3.0 * eta + 2.0 * xi * xi + 4.0 * xi * eta + 2.0 * eta * eta,
                -xi + 2.0 * xi * xi,
                -eta + 2.0 * eta * eta,
                4.0 * xi * eta,
                4.0 * eta - 4.0 * xi * eta - 4.0 * eta * eta,
                4.0 * xi - 4.0 * xi * xi - 4.0 * xi * eta,
            ]
        )
    if p == 3:
        return np.array(
            [
                1.0
                - 11.0 / 2.0 * xi
                - 11.0 / 2.0 * eta
                + 9.0 * xi * xi
                + 18.0 * xi * eta
                + 9.0 * eta * eta
                - 9.0 / 2.0 * xi * xi * xi
                - 27.0 / 2.0 * xi * xi * eta
                - 27.0 / 2.0 * xi * eta * eta
                - 9.0 / 2.0 * eta * eta * eta,
                xi - 9.0 / 2.0 * xi * xi + 9.0 / 2.0 * xi * xi * xi,
                eta - 9.0 / 2.0 * eta * eta + 9.0 / 2.0 * eta * eta * eta,
                -9.0 / 2.0 * xi * eta + 27.0 / 2.0 * xi * xi * eta,
                -9.0 / 2.0 * xi * eta + 27.0 / 2.0 * xi * eta * eta,
                -9.0 / 2.0 * eta
                + 9.0 / 2.0 * xi * eta
                + 18.0 * eta * eta
                - 27.0 / 2.0 * xi * eta * eta
                - 27.0 / 2.0 * eta * eta * eta,
                9.0 * eta
                - 45.0 / 2.0 * xi * eta
                - 45.0 / 2.0 * eta * eta
                + 27.0 / 2.0 * xi * xi * eta
                + 27.0 * xi * eta * eta
                + 27.0 / 2.0 * eta * eta * eta,
                9.0 * xi
                - 45.0 / 2.0 * xi * xi
                - 45.0 / 2.0 * xi * eta
                + 27.0 / 2.0 * xi * xi * xi
                + 27.0 * xi * xi * eta
                + 27.0 / 2.0 * xi * eta * eta,
                -9.0 / 2.0 * xi
                + 18.0 * xi * xi
                + 9.0 / 2.0 * xi * eta
                - 27.0 / 2.0 * xi * xi * xi
                - 27.0 / 2.0 * xi * xi * eta,
                27.0 * xi * eta - 27.0 * xi * xi * eta - 27.0 * xi * eta * eta,
            ]
        )
    raise ValueError(f"Basis not implemented for p={p}")


def evaluate_basis_grad(xi, eta, p):
    ndof = (p + 1) * (p + 2) // 2
    g = np.zeros((2, ndof))
    if p == 0:
        return g
    if p == 1:
        g[0] = [-1.0, 1.0, 0.0]
        g[1] = [-1.0, 0.0, 1.0]
        return g
    if p == 2:
        g[0] = [
            -3.0 + 4.0 * xi + 4.0 * eta,
            -1.0 + 4.0 * xi,
            0.0,
            4.0 * eta,
            -4.0 * eta,
            4.0 - 8.0 * xi - 4.0 * eta,
        ]
        g[1] = [
            -3.0 + 4.0 * xi + 4.0 * eta,
            0.0,
            -1.0 + 4.0 * eta,
            4.0 * xi,
            4.0 - 4.0 * xi - 8.0 * eta,
            -4.0 * xi,
        ]
        return g
    if p == 3:
        g[0] = [
            -11.0 / 2.0 + 18.0 * xi + 18.0 * eta - 27.0 / 2.0 * xi * xi - 27.0 * xi * eta - 27.0 / 2.0 * eta * eta,
            1.0 - 9.0 * xi + 27.0 / 2.0 * xi * xi,
            0.0,
            -9.0 / 2.0 * eta + 27.0 * xi * eta,
            -9.0 / 2.0 * eta + 27.0 / 2.0 * eta * eta,
            9.0 / 2.0 * eta,
            -45.0 / 2.0 * eta + 27.0 * xi * eta + 27.0 * eta * eta,
            9.0 - 45.0 * xi - 45.0 / 2.0 * eta + 81.0 / 2.0 * xi * xi + 54.0 * xi * eta + 27.0 / 2.0 * eta * eta,
            -9.0 / 2.0 + 36.0 * xi + 9.0 / 2.0 * eta - 81.0 / 2.0 * xi * xi - 27.0 * xi * eta,
            27.0 * eta - 54.0 * xi * eta - 27.0 * eta * eta,
        ]
        g[1] = [
            -11.0 / 2.0 + 18.0 * xi + 18.0 * eta - 27.0 / 2.0 * xi * xi - 27.0 * xi * eta - 27.0 / 2.0 * eta * eta,
            0.0,
            1.0 - 9.0 * eta + 27.0 / 2.0 * eta * eta,
            -9.0 / 2.0 * xi + 27.0 / 2.0 * xi * xi,
            -9.0 / 2.0 * xi + 27.0 * xi * eta,
            -9.0 / 2.0 + 9.0 / 2.0 * xi + 36.0 * eta - 27.0 * xi * eta - 81.0 / 2.0 * eta * eta,
            9.0 - 45.0 / 2.0 * xi - 45.0 * eta + 27.0 / 2.0 * xi * xi + 54.0 * xi * eta + 81.0 / 2.0 * eta * eta,
            -45.0 / 2.0 * xi + 27.0 * xi * xi + 27.0 * xi * eta,
            9.0 / 2.0 * xi - 27.0 / 2.0 * xi * xi,
            27.0 * xi - 27.0 * xi * xi - 54.0 * xi * eta,
        ]
        return g
    raise ValueError(f"Basis gradient not implemented for p={p}")


def shape_order_node_ids(element):
    perm = _GRI_TO_SHAPE[element["q"]]
    return [element["row"][idx] for idx in perm]


def geometry_nodes(element, nodes):
    return nodes[shape_order_node_ids(element)]


def map_to_physical(element, nodes, xi, eta):
    xnode = geometry_nodes(element, nodes)
    phi = evaluate_basis(xi, eta, element["q"])
    xy = phi @ xnode
    return xy


def map_jacobian(element, nodes, xi, eta):
    xnode = geometry_nodes(element, nodes)
    gphi = evaluate_basis_grad(xi, eta, element["q"])
    dx_dxi = np.dot(gphi[0], xnode[:, 0])
    dx_deta = np.dot(gphi[1], xnode[:, 0])
    dy_dxi = np.dot(gphi[0], xnode[:, 1])
    dy_deta = np.dot(gphi[1], xnode[:, 1])
    return np.array([[dx_dxi, dx_deta], [dy_dxi, dy_deta]])


def element_polygon(element, nodes):
    order = _POLYGON_ORDER.get(element["q"], [0, element["q"], len(element["row"]) - 1])
    return nodes[[element["row"][idx] for idx in order]]


def reconstruct_state(U_elem, p_order, xi, eta):
    phi = evaluate_basis(xi, eta, p_order)
    return phi @ U_elem


def primitive_from_state(U):
    rho = U[..., 0]
    rhou = U[..., 1]
    rhov = U[..., 2]
    rhoe = U[..., 3]
    vel_u = rhou / rho
    vel_v = rhov / rho
    qsq = vel_u * vel_u + vel_v * vel_v
    p = (GAMMA - 1.0) * (rhoe - 0.5 * rho * qsq)
    a = np.sqrt(np.maximum(GAMMA * p / rho, 1e-16))
    mach = np.sqrt(np.maximum(qsq, 0.0)) / a
    entropy = np.log(np.maximum(p / np.power(rho, GAMMA), 1e-16))
    return {
        "rho": rho,
        "u": vel_u,
        "v": vel_v,
        "p": p,
        "mach": mach,
        "entropy": entropy,
    }


def build_reference_submesh(level):
    if level in _REF_SUBMESH_CACHE:
        return _REF_SUBMESH_CACHE[level]

    coords = []
    index = {}
    for i in range(level + 1):
        for j in range(level + 1 - i):
            index[(i, j)] = len(coords)
            coords.append((i / level, j / level))

    tris = []
    for i in range(level):
        for j in range(level - i):
            a = index[(i, j)]
            b = index[(i + 1, j)]
            c = index[(i, j + 1)]
            tris.append((a, b, c))
            if j < level - i - 1:
                d = index[(i + 1, j + 1)]
                tris.append((b, d, c))

    ref_xy = np.array(coords)
    ref_tris = np.array(tris, dtype=int)
    _REF_SUBMESH_CACHE[level] = (ref_xy, ref_tris)
    return ref_xy, ref_tris


def build_dg_triangulation(mesh, U_dg, p_order, field_name, refine_level=None):
    if refine_level is None:
        refine_level = max(4, 2 * p_order + 2)

    ref_xy, ref_tris = build_reference_submesh(refine_level)
    nodes = mesh["nodes"]

    x_all = []
    y_all = []
    field_all = []
    tri_all = []
    node_offset = 0

    for elem_idx, element in enumerate(mesh["elements"]):
        xnode = geometry_nodes(element, nodes)
        U_elem = U_dg[elem_idx]

        elem_xy = []
        elem_field = []
        for xi, eta in ref_xy:
            phi_geom = evaluate_basis(xi, eta, element["q"])
            xy = phi_geom @ xnode
            state = reconstruct_state(U_elem, p_order, xi, eta)
            prim = primitive_from_state(state)
            elem_xy.append(xy)
            elem_field.append(prim[field_name])

        elem_xy = np.array(elem_xy)
        elem_field = np.array(elem_field)

        x_all.extend(elem_xy[:, 0].tolist())
        y_all.extend(elem_xy[:, 1].tolist())
        field_all.extend(elem_field.tolist())
        tri_all.extend((ref_tris + node_offset).tolist())
        node_offset += len(ref_xy)

    return np.array(x_all), np.array(y_all), np.array(tri_all, dtype=int), np.array(field_all)


def local_edge_reference_data(element, edge):
    v0, v1, v2 = element["corners"]
    n1, n2 = edge
    if {n1, n2} == {v0, v1}:
        return (np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    if {n1, n2} == {v1, v2}:
        return (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    if {n1, n2} == {v2, v0}:
        return (np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    return None


def wall_edge_samples(mesh, U_dg, p_order, num_samples=5):
    wall_edges = None
    for name, edges in mesh["boundary_groups"].items():
        if name.lower() == "wall":
            wall_edges = edges
            break
    if wall_edges is None:
        return np.array([]), np.array([])

    edge_to_elem = {}
    for elem_idx, element in enumerate(mesh["elements"]):
        c = element["corners"]
        for edge in [(c[0], c[1]), (c[1], c[2]), (c[2], c[0])]:
            edge_to_elem[tuple(sorted(edge))] = elem_idx

    x_vals = []
    cp_vals = []
    rho0 = 1.0
    a0 = 1.0
    p0 = a0 * a0 * rho0 / GAMMA
    p_out = 0.7 * p0
    pressure_ratio = p0 / p_out
    M_out_sq = (2.0 / (GAMMA - 1.0)) * (pressure_ratio ** ((GAMMA - 1.0) / GAMMA) - 1.0)
    q_out = 0.5 * GAMMA * p_out * M_out_sq

    sample_t = np.linspace(0.0, 1.0, num_samples)
    for edge in wall_edges:
        elem_idx = edge_to_elem.get(tuple(sorted(edge)))
        if elem_idx is None:
            continue
        element = mesh["elements"][elem_idx]
        endpoints = local_edge_reference_data(element, edge)
        if endpoints is None:
            continue
        ref0, ref1 = endpoints
        for t in sample_t:
            xi_eta = (1.0 - t) * ref0 + t * ref1
            xi, eta = xi_eta
            xy = map_to_physical(element, mesh["nodes"], xi, eta)
            state = reconstruct_state(U_dg[elem_idx], p_order, xi, eta)
            p = primitive_from_state(state)["p"]
            cp = (p - p_out) / q_out
            x_vals.append(xy[0])
            cp_vals.append(cp)

    order = np.argsort(x_vals)
    return np.array(x_vals)[order], np.array(cp_vals)[order]


def integrate_wall_forces(mesh, U_dg, p_order):
    wall_edges = None
    for name, edges in mesh["boundary_groups"].items():
        if name.lower() == "wall":
            wall_edges = edges
            break
    if wall_edges is None:
        return 0.0, 0.0

    edge_to_elem = {}
    for elem_idx, element in enumerate(mesh["elements"]):
        c = element["corners"]
        for edge in [(c[0], c[1]), (c[1], c[2]), (c[2], c[0])]:
            edge_to_elem[tuple(sorted(edge))] = elem_idx

    rho0 = 1.0
    a0 = 1.0
    p0 = a0 * a0 * rho0 / GAMMA
    p_out = 0.7 * p0
    pressure_ratio = p0 / p_out
    M_out_sq = (2.0 / (GAMMA - 1.0)) * (pressure_ratio ** ((GAMMA - 1.0) / GAMMA) - 1.0)
    q_out = 0.5 * GAMMA * p_out * M_out_sq

    gp = np.array(
        [
            0.069431844202974,
            0.330009478207572,
            0.669990521792428,
            0.930568155797026,
        ]
    )
    gw = np.array(
        [
            0.173927422568727,
            0.326072577431273,
            0.326072577431273,
            0.173927422568727,
        ]
    )

    Fx = 0.0
    Fy = 0.0
    for edge in wall_edges:
        elem_idx = edge_to_elem.get(tuple(sorted(edge)))
        if elem_idx is None:
            continue
        element = mesh["elements"][elem_idx]
        endpoints = local_edge_reference_data(element, edge)
        if endpoints is None:
            continue
        ref0, ref1 = endpoints
        centroid = element_polygon(element, mesh["nodes"]).mean(axis=0)
        for t, w in zip(gp, gw):
            xi_eta = (1.0 - t) * ref0 + t * ref1
            xi, eta = xi_eta
            state = reconstruct_state(U_dg[elem_idx], p_order, xi, eta)
            p = primitive_from_state(state)["p"]
            J = map_jacobian(element, mesh["nodes"], xi, eta)
            dref_dt = ref1 - ref0
            tangent = J @ dref_dt
            ds_dt = np.linalg.norm(tangent)
            normal = np.array([tangent[1], -tangent[0]])
            normal /= np.linalg.norm(normal)
            xy = map_to_physical(element, mesh["nodes"], xi, eta)
            if np.dot(normal, xy - centroid) < 0.0:
                normal = -normal
            Fx += p * normal[0] * ds_dt * w
            Fy += p * normal[1] * ds_dt * w

    chord = 18.804
    return Fx / (q_out * chord), Fy / (q_out * chord)

import numpy as np
import matplotlib as plt

from plotting import *
from calculations import *
from flux import *

def load_nodes(blade_num):
    node_file = "meshes/blade" + str(blade_num) + ".node"
    elem_file = "meshes/blade" + str(blade_num) + ".elem"
    connect_file = "meshes/blade" + str(blade_num) + ".connect"
    coordinates = np.loadtxt(node_file, skiprows=1)
    node_connectivity = np.loadtxt(elem_file, skiprows=1, dtype=int) - 1
    cell_connectivity = np.loadtxt(connect_file, skiprows=8, dtype=int)
    cell_connectivity[cell_connectivity >= 0] -= 1
    if blade_num == -1:
        coordinates, node_connectivity, cell_connectivity = load_unit_nodes()  
    return coordinates, node_connectivity, cell_connectivity


def split_edges(coordinates, node_connectivity, cell_connectivity):
    unique_interior_edge_info = set()
    interior_edges = []
    exterior_edges = []
    max_edges = node_connectivity.shape[1]
    for cell_number in range(len(cell_connectivity)):
        edge_num = 0
        right_cell_index = 2
        for i_start_node in range(max_edges):
            if edge_num < max_edges - 1:
                i_end_node = i_start_node + 1
            else:
                i_end_node = 0
            start_node = node_connectivity[cell_number][i_start_node]
            end_node = node_connectivity[cell_number][i_end_node]
            left_cell = cell_number
            right_cell = cell_connectivity[cell_number][right_cell_index]

            x1 = coordinates[start_node][0]
            x2 = coordinates[end_node][0]
            y1 = coordinates[start_node][1]
            y2 = coordinates[end_node][1]
            edge_length = np.power(np.power(x2 - x1, 2) + np.power(y2 - y1, 2), 1 / 2)

            unit_normal_x = (y2 - y1) / edge_length
            unit_normal_y = (x1 - x2) / edge_length

            edge_center_x = (x1 + x2) / 2
            edge_center_y = (y1 + y2) / 2

            edge_info = [start_node, end_node, left_cell, right_cell, edge_length, unit_normal_x, unit_normal_y, edge_center_x, edge_center_y]
            if right_cell >= 0:
                edge_info_for_unique_set = (start_node, end_node, left_cell, right_cell)
                edge_info_check = (end_node, start_node, right_cell, left_cell)
                if edge_info_check not in unique_interior_edge_info:
                    unique_interior_edge_info.add(edge_info_for_unique_set)
                    interior_edges.append(edge_info)
            elif right_cell < 0:
                exterior_edges.append(edge_info)
            else:
                print(f"right cell error? :(")
            edge_num += 1
            right_cell_index += 1
            if right_cell_index > max_edges - 1:
                right_cell_index = 0
    return np.array(interior_edges), np.array(exterior_edges)


def cell_edges(coordinates, node_connectivity, cell_connectivity):
    n_cells = len(cell_connectivity)
    n_edges_per_cell = node_connectivity.shape[1]
    n_values_per_edge = 9
    edges = np.zeros((n_cells, n_edges_per_cell, n_values_per_edge))
    for cell_number in range(n_cells):
        edge_num = 0
        right_cell_index = 2
        for i_start_node in range(n_edges_per_cell):
            if edge_num < n_edges_per_cell - 1:
                i_end_node = i_start_node + 1
            else:
                i_end_node = 0
            start_node = node_connectivity[cell_number][i_start_node]
            end_node = node_connectivity[cell_number][i_end_node]
            left_cell = cell_number
            right_cell = cell_connectivity[cell_number][right_cell_index]

            x1 = coordinates[start_node][0]
            x2 = coordinates[end_node][0]
            y1 = coordinates[start_node][1]
            y2 = coordinates[end_node][1]
            edge_length = np.power(np.power(x2 - x1, 2) + np.power(y2 - y1, 2), 1 / 2)

            unit_normal_x = (y2 - y1) / edge_length
            unit_normal_y = (x1 - x2) / edge_length

            edge_center_x = (x1 + x2) / 2
            edge_center_y = (y1 + y2) / 2

            edge_info = [start_node, end_node, left_cell, right_cell, edge_length, unit_normal_x, unit_normal_y,
                         edge_center_x, edge_center_y]
            edges[cell_number][edge_num] = edge_info
            edge_num += 1
            right_cell_index += 1
            if right_cell_index > n_edges_per_cell - 1:
                right_cell_index = 0
    return np.array(edges)


def load_unit_nodes():
    coordinates = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    node_connectivity = np.array([[0, 1, 3], [1, 2, 3]])
    cell_connectivity = np.array([[2, -1, -3], [-4, 1, -2]])
    cell_connectivity[cell_connectivity >= 0] -= 1
    return coordinates, node_connectivity, cell_connectivity


def initialize_cell_states(node_connectivity):
    cell_states = np.zeros([len(node_connectivity), 4])
    cell_states[:] = calc_initial_cell_state()
    return cell_states


def freestream_test(initial_cell_states, edges, areas, perimeters, hydrodynamic_diameters, n_iters):
    free_stream_state = np.copy(initial_cell_states[0])
    u0 = np.copy(free_stream_state)
    u1 = np.copy(free_stream_state)
    gamma = 1.4
    CFL = 1
    n_cells = len(initial_cell_states)
    n_edges = 3
    dts = np.zeros(n_cells)
    weighted_avg_wave_speed = np.zeros(n_cells)
    residuals = np.zeros([n_cells, 4])
    cell_states = np.copy(initial_cell_states)
    cell_states_history = [initial_cell_states]
    L1_norm_history = []
    for i in range(n_iters):
        print(f"Iteration {i}")
        for cell_num in range(n_cells):
            weighted_avg_wave_speed[cell_num] = 0
            residuals[cell_num] = 0
            for edge_num in range(n_edges):
                edge = edges[cell_num][edge_num]
                right_cell = int(edge[3])
                u_left = cell_states[cell_num]
                unit_normal = [edge[5], edge[6]]
                edge_length = edge[4]
                u_right = np.copy(free_stream_state)
                flux, smag = FluxFunction(u_left, u_right, gamma, unit_normal)
                weighted_avg_wave_speed[cell_num] += smag * edge_length
                residuals[cell_num] += flux * edge_length
            weighted_avg_wave_speed[cell_num] = weighted_avg_wave_speed[cell_num] / perimeters[cell_num]
            dts[cell_num] = hydrodynamic_diameters[cell_num] * CFL / weighted_avg_wave_speed[cell_num]
        for cell_num in range(n_cells):
            for u_i in range(4):
                cell_states[cell_num][u_i] = cell_states[cell_num][u_i] - dts[cell_num] * residuals[cell_num][u_i] / areas[cell_num]
        cell_states_history.append(cell_states)

        L1_norm = 0
        for i in range(4):
            L1_norm += np.linalg.norm(residuals[:][i])
        L1_norm_history.append(L1_norm)


    return cell_states, cell_states_history, L1_norm_history


def fvsolve(initial_cell_states, edges, areas, perimeters, hydrodynamic_diameters, n_iters):
    # edge[cell][edge] = [start_node(0), end_node(1), left_cell(2), right_cell(3), edge_length(4),
    #               unit_normal_x(5), unit_normal_y(6), edge_center_x(7), edge_center_y(8)]
    tol = 1e-5
    gamma = 1.4
    CFL = 1
    n_cells = len(initial_cell_states)
    n_edges = 3
    dts = np.zeros(n_cells)
    weighted_avg_wave_speed = np.zeros(n_cells)
    residuals = np.zeros([n_cells, 4])
    cell_states = np.copy(initial_cell_states)
    cell_states_history = [initial_cell_states]
    L1_norm_history = []
    for i in range(n_iters):
        print(f"Iteration {i + 1}")
        for cell_num in range(n_cells):
            weighted_avg_wave_speed[cell_num] = 0
            residuals[cell_num] = 0
            for edge_num in range(n_edges):
                edge = edges[cell_num][edge_num]
                right_cell = int(edge[3])
                u_left = cell_states[cell_num]
                unit_normal = [edge[5], edge[6]]
                edge_length = edge[4]
                if right_cell >= 0:
                    u_right = cell_states[right_cell]
                    flux, smag = FluxFunction(u_left, u_right, gamma, unit_normal)
                elif right_cell == -1:
                    flux, smag = calc_flux_inflow(u_left, edge)
                elif right_cell == -2:
                    flux, smag = calc_flux_outflow(u_left, edge)
                elif right_cell == -3 or right_cell == -4:
                    flux, smag = calc_flux_wall(u_left, edge)
                else:
                    print(f"bad right cell...")
                weighted_avg_wave_speed[cell_num] += smag * edge_length
                residuals[cell_num] += flux * edge_length
            weighted_avg_wave_speed[cell_num] = weighted_avg_wave_speed[cell_num] / perimeters[cell_num]
            dts[cell_num] = hydrodynamic_diameters[cell_num] * CFL / weighted_avg_wave_speed[cell_num]
        for cell_num in range(n_cells):
            for u_i in range(4):
                cell_states[cell_num][u_i] = cell_states[cell_num][u_i] - dts[cell_num] * residuals[cell_num][u_i] / areas[cell_num]
        cell_states_history.append(cell_states)

        L1_norm = 0
        for i in range(4):
            L1_norm += np.linalg.norm(residuals[:][i])
        L1_norm_history.append(L1_norm)

        if L1_norm < tol:
            return cell_states, cell_states_history, L1_norm_history

    return cell_states, cell_states_history, L1_norm_history


def initialize_blade(blade_num):
    coordinates, node_connectivity, cell_connectivity = load_nodes(blade_num)
    centroids = calc_cell_centroids(coordinates, node_connectivity)
    areas = calc_cell_areas(coordinates, node_connectivity)
    edges = cell_edges(coordinates, node_connectivity, cell_connectivity)
    perimeters, hydrodynamic_diameters = calc_cell_perimeter_and_hydrodynamic_diameter(areas, edges)
    initial_cell_states = initialize_cell_states(node_connectivity)

    return (coordinates, node_connectivity, cell_connectivity, centroids, areas, edges,
            perimeters, hydrodynamic_diameters, initial_cell_states)

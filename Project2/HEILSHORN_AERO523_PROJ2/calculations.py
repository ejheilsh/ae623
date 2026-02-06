import numpy as np
import cmath as cm
from flux import *

def calc_cell_areas(coordinates, node_connectivity):
    n_cells = node_connectivity.shape[0]
    areas = np.zeros(n_cells)
    node1 = node_connectivity[:, 0]
    node2 = node_connectivity[:, 1]
    node3 = node_connectivity[:, 2]
    x1 = coordinates[node1, 0]
    x2 = coordinates[node2, 0]
    x3 = coordinates[node3, 0]
    y1 = coordinates[node1, 1]
    y2 = coordinates[node2, 1]
    y3 = coordinates[node3, 1]
    areas[:] = np.abs(.5 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))
    return areas


def calc_cell_centroids(coordinates, node_connectivity):
    n_cells = node_connectivity.shape[0]
    centroids = np.zeros((n_cells, 2))
    node1 = node_connectivity[:, 0]
    node2 = node_connectivity[:, 1]
    node3 = node_connectivity[:, 2]
    x1 = coordinates[node1, 0]
    x2 = coordinates[node2, 0]
    x3 = coordinates[node3, 0]
    y1 = coordinates[node1, 1]
    y2 = coordinates[node2, 1]
    y3 = coordinates[node3, 1]
    centroids[:, 0] = (x1 + x2 + x3) / 3
    centroids[:, 1] = (y1 + y2 + y3) / 3
    return centroids


def calc_initial_cell_state():
    alpha_deg = 52.4                                                        # angle of attack in degrees
    gamma = 1.4                                                             # ratio of specific heats for air
    R_air = 287                                                             # specific gas constant for air

    rho_t = 1                                                               # inlet stagnation density
    c_t = 1                                                                 # inlet stagnation speed of sound, a_0
    p_t = c_t ** 2 * rho_t / gamma                                          # inlet stagnation pressure
    T_t = p_t / (rho_t * R_air)                                             # inlet stagnation temperature

    M_plus = .1                                                             # assumed mach number of cell
    T_plus = T_t / (1 + .5 * (gamma - 1) * M_plus**2)                       # static temperature in cell
    p_plus = p_t * np.power(T_plus / T_t, gamma / (gamma - 1))              # pressure in cell
    rho_plus = p_plus / (R_air * T_plus)                                    # density in cell
    c_plus = np.sqrt(gamma * p_plus / rho_plus)                             # speed of sound in cell
    mag_vel_plus = M_plus * c_plus                                          # magnitude of velocity in cell
    vel_x_plus = mag_vel_plus * np.sin(np.radians(alpha_deg))               # x comp of vel in cell
    vel_y_plus = mag_vel_plus * np.cos(np.radians(alpha_deg))               # y comp of vel in cell
    rho_E_plus = p_plus / (gamma - 1) + .5 * rho_plus * mag_vel_plus**2     # total energy in cell

    cell_state = [rho_plus, rho_plus * vel_x_plus, rho_plus * vel_y_plus, rho_E_plus]
    return cell_state


def calc_flux_inflow(left_cell_state, edge_info):
    # edge_info = [start_node, end_node, left_cell, right_cell, edge_length, unit_normal_x, unit_normal_y,
    # edge_center_x, edge_center_y]

    alpha_deg = 52.4                    # inflow angle in degrees
    alpha_rad = np.radians(alpha_deg)   # inflow angle in radians
    gamma = 1.4                         # ratio of specific heats for air
    R_air = 287                         # specific gas constant for air
    rho_t_b = 1                         # inlet stagnation density
    c_t_b = 1                           # inlet stagnation speed of sound
    p_t_b = c_t_b**2 * rho_t_b / gamma      # inlet stagnation pressure
    T_t_b = p_t_b / (rho_t_b * R_air)       # inlet stagnation temperature
    normal_in = [np.cos(alpha_rad), np.sin(alpha_rad)]

    rho_plus = left_cell_state[0]
    vel_x_plus = left_cell_state[1] / rho_plus
    vel_y_plus = left_cell_state[2] / rho_plus
    vel_plus = np.array([vel_x_plus, vel_y_plus])
    E_plus = left_cell_state[3] / rho_plus
    unit_normal_x = edge_info[5]
    unit_normal_y = edge_info[6]
    edge_normal_vect = [unit_normal_x, unit_normal_y]

    mag_vel_plus = np.sqrt(vel_x_plus**2 + vel_y_plus**2)
    p_plus = rho_plus * (E_plus - .5 * mag_vel_plus**2) * (gamma - 1)
    d_n = np.dot(edge_normal_vect, normal_in)
    c_plus = np.sqrt(gamma * p_plus / rho_plus)
    normal_vel_plus = np.dot(vel_plus, edge_normal_vect)
    J_plus = normal_vel_plus + 2 * c_plus / (gamma - 1)

    a = (gamma * R_air * T_t_b * d_n**2 - .5 * (gamma - 1) * J_plus**2)
    b = 4 * gamma * R_air * T_t_b * d_n / (gamma - 1)
    c = 4 * gamma * R_air * T_t_b / (gamma - 1)**2 - J_plus**2
    d = b**2 - 4 * a * c

    M_b1 = (-b + cm.sqrt(d)) / (2 * a)
    M_b2 = (-b - cm.sqrt(d)) / (2 * a)

    if M_b1 > 0 and M_b2 < 0:
        M_b = M_b1
    elif M_b1 < 0 and M_b2 > 0:
        M_b = M_b2
    elif M_b1 > 0 and M_b2 > 0:
        M_b = min(M_b1, M_b2)
    elif M_b1 < 0 and M_b2 < 0:
        if M_b1 > M_b2:
            M_b = M_b1
        elif M_b2 > M_b1:
            M_b = M_b2
        else:
            print(f"something weird happened with M_b")
    else:
        print(f"weird M_b case... M_b1 = {M_b1}, M_b2 = {M_b2}")
        M_b = np.abs(M_b1)

    T_b = T_t_b / (1 + .5 * (gamma - 1) * M_b**2)
    p_b = p_t_b * np.power(T_b / T_t_b, gamma / (gamma - 1))
    rho_b = p_b / (R_air * T_b)
    c_b = np.sqrt(gamma * p_b / rho_b)
    mag_vel_b = M_b * c_b
    vel_x_b = mag_vel_b * np.sin(alpha_rad)
    vel_y_b = mag_vel_b * np.cos(alpha_rad)
    rho_E_b = p_b / (gamma - 1) + .5 * rho_b * mag_vel_b**2
    rho_H_b = rho_E_b + p_b

    flux_x = [rho_b * vel_x_b, rho_b * vel_x_b**2 + p_b, rho_b * vel_x_b * vel_y_b, rho_H_b * vel_x_b]
    flux_y = [rho_b * vel_y_b, rho_b * vel_y_b * vel_x_b, rho_b * vel_y_b**2 + p_b, rho_H_b * vel_y_b]

    F_b = np.zeros(4)
    F_b[0] = np.dot([flux_x[0], flux_y[0]], edge_normal_vect)
    F_b[1] = np.dot([flux_x[1], flux_y[1]], edge_normal_vect)
    F_b[2] = np.dot([flux_x[2], flux_y[2]], edge_normal_vect)
    F_b[3] = np.dot([flux_x[3], flux_y[3]], edge_normal_vect)

    smag = mag_vel_b + c_b

    return F_b, smag


def calc_flux_outflow(left_cell_state, edge_info):
    gamma = 1.4
    pressure_ratio = .7
    rho_0 = 1
    c_t_0 = 1
    p_0 = rho_0 * c_t_0**2 / gamma
    p_out = pressure_ratio * p_0

    rho_plus = left_cell_state[0]
    vel_x_plus = left_cell_state[1] / rho_plus
    vel_y_plus = left_cell_state[2] / rho_plus
    vel_plus = np.array([vel_x_plus, vel_y_plus])
    E_plus = left_cell_state[3] / rho_plus
    unit_normal_x = edge_info[5]
    unit_normal_y = edge_info[6]
    edge_normal_vect = np.array([unit_normal_x, unit_normal_y])
    mag_vel_plus = np.sqrt(vel_x_plus ** 2 + vel_y_plus ** 2)
    p_plus = rho_plus * (E_plus - .5 * mag_vel_plus ** 2) * (gamma - 1)

    S_plus = p_plus / np.power(rho_plus, gamma)
    p_b = p_out
    rho_b = np.power(p_b / S_plus, 1 / gamma)
    c_b = np.sqrt(gamma * p_b / rho_b)

    normal_vel_plus = np.dot(vel_plus, edge_normal_vect)
    c_plus = np.sqrt(gamma * p_plus / rho_plus)
    J_plus = normal_vel_plus + 2 * c_plus / (gamma - 1)

    mag_normal_vel_b = J_plus - 2 * c_b / (gamma - 1)
    normal_vel_b = np.array(mag_normal_vel_b * edge_normal_vect)
    tang_vel_plus = np.array(vel_plus) - np.dot(vel_plus, edge_normal_vect) * np.array(edge_normal_vect)
    vel_b = tang_vel_plus + normal_vel_b
    vel_x_b = vel_b[0]
    vel_y_b = vel_b[1]
    mag_vel_b = np.sqrt(vel_x_b**2 + vel_y_b**2)

    rho_E_b = p_b / (gamma - 1) + .5 * rho_b * mag_vel_b**2
    rho_H_b = rho_E_b + p_b

    flux_x = [rho_b * vel_x_b, rho_b * vel_x_b ** 2 + p_b, rho_b * vel_x_b * vel_y_b, rho_H_b * vel_x_b]
    flux_y = [rho_b * vel_y_b, rho_b * vel_y_b * vel_x_b, rho_b * vel_y_b ** 2 + p_b, rho_H_b * vel_y_b]

    F_b = np.zeros(4)
    F_b[0] = np.dot([flux_x[0], flux_y[0]], edge_normal_vect)
    F_b[1] = np.dot([flux_x[1], flux_y[1]], edge_normal_vect)
    F_b[2] = np.dot([flux_x[2], flux_y[2]], edge_normal_vect)
    F_b[3] = np.dot([flux_x[3], flux_y[3]], edge_normal_vect)

    smag = mag_vel_b + c_b

    return F_b, smag


def calc_flux_wall(left_cell_state, edge_info):
    gamma = 1.4

    rho_plus = left_cell_state[0]
    vel_x_plus = left_cell_state[1] / rho_plus
    vel_y_plus = left_cell_state[2] / rho_plus
    vel_plus = [vel_x_plus, vel_y_plus]
    rho_E_plus = left_cell_state[3]
    E_plus = rho_E_plus / rho_plus
    unit_normal_x = edge_info[5]
    unit_normal_y = edge_info[6]
    edge_normal_vect = [unit_normal_x, unit_normal_y]
    mag_vel_plus = np.sqrt(vel_x_plus ** 2 + vel_y_plus ** 2)

    tang_vel_plus = np.array(vel_plus) - np.dot(vel_plus, edge_normal_vect) * np.array(edge_normal_vect)
    vel_b = tang_vel_plus
    vel_x_b = vel_b[0]
    vel_y_b = vel_b[1]
    mag_vel_b = np.sqrt(vel_x_b ** 2 + vel_y_b ** 2)

    p_b = (gamma - 1) * (rho_E_plus - .5 * rho_plus * mag_vel_b**2)

    F_b = np.zeros(4)
    F_b[0] = 0
    F_b[1] = p_b * unit_normal_x
    F_b[2] = p_b * unit_normal_y
    F_b[3] = 0

    p_plus = rho_plus * (E_plus - .5 * mag_vel_plus ** 2) * (gamma - 1)
    c_plus = np.sqrt(gamma * p_plus / rho_plus)
    J_plus = 2 * c_plus / (gamma - 1)   # since no normal velocity

    c_b = (gamma - 1) * J_plus / 2

    smag = mag_vel_b + c_b

    return F_b, smag


def calc_cell_perimeter_and_hydrodynamic_diameter(areas, edges):
    n_cells = len(areas)
    hydrodynamic_diameters = np.zeros(n_cells)
    perimeters = np.zeros(n_cells)
    for i in range(n_cells):
        perimeter = 0
        for j in range(3):
            perimeter += edges[i][j][4]
        hydrodynamic_diameters[i] = 2 * areas[i] / perimeter
        perimeters[i] = perimeter
    return perimeters, hydrodynamic_diameters


def calc_mach_numbers(cell_states):
    n_cells = len(cell_states)
    mach_numbers = np.zeros(n_cells)
    for cell_num in range(n_cells):
        gamma = 1.4
        cell_state = cell_states[cell_num]
        rho_plus = cell_state[0]
        vel_x_plus = cell_state[1] / rho_plus
        vel_y_plus = cell_state[2] / rho_plus
        E_plus = cell_state[3] / rho_plus
        mag_vel_plus = np.sqrt(vel_x_plus ** 2 + vel_y_plus ** 2)
        p_plus = rho_plus * (E_plus - .5 * mag_vel_plus ** 2) * (gamma - 1)
        c_plus = np.sqrt(gamma * p_plus / rho_plus)
        M_plus = mag_vel_plus / c_plus
        mach_numbers[cell_num] = M_plus
    return mach_numbers


def calc_wall_pressure(left_cell_state, edge_info):
    gamma = 1.4
    rho_plus = left_cell_state[0]
    vel_x_plus = left_cell_state[1] / rho_plus
    vel_y_plus = left_cell_state[2] / rho_plus
    vel_plus = [vel_x_plus, vel_y_plus]
    rho_E_plus = left_cell_state[3]
    E_plus = rho_E_plus / rho_plus
    unit_normal_x = edge_info[5]
    unit_normal_y = edge_info[6]
    edge_normal_vect = [unit_normal_x, unit_normal_y]
    mag_vel_plus = np.sqrt(vel_x_plus ** 2 + vel_y_plus ** 2)

    tang_vel_plus = np.array(vel_plus) - np.dot(vel_plus, edge_normal_vect) * np.array(edge_normal_vect)
    vel_b = tang_vel_plus
    vel_x_b = vel_b[0]
    vel_y_b = vel_b[1]
    mag_vel_b = np.sqrt(vel_x_b ** 2 + vel_y_b ** 2)

    p_b = (gamma - 1) * (rho_E_plus - .5 * rho_plus * mag_vel_b ** 2)

    return p_b


def calc_pressure_coeff(cell_states, edges):
    # edge_info = [start_node(0), end_node(1), left_cell(2), right_cell(3), edge_length(4),
    #               unit_normal_x(5), unit_normal_y(6), edge_center_x(7), edge_center_y(8)]

    gamma = 1.4
    pressure_ratio = .7
    rho_0 = 1
    c_t_0 = 1
    p_0 = rho_0 * c_t_0 ** 2 / gamma
    p_out = pressure_ratio * p_0

    M_out_squared = (2 / (gamma - 1)) * (np.power(p_0 / p_out, (gamma - 1) / gamma) - 1)
    q_out = .5 * gamma * p_out * M_out_squared

    n_cells = len(cell_states)
    x_upper_surface = []
    x_lower_surface = []
    c_p_upper_surface = []
    c_p_lower_surface = []
    for cell_num in range(n_cells):
        for edge_num in range(3):
            edge = edges[cell_num][edge_num]
            right_cell = edge[3]
            if right_cell == -3 or right_cell == -4:
                left_cell_state = cell_states[cell_num]
                p_b = calc_wall_pressure(left_cell_state, edge)
                c_p = (p_b - p_out) / q_out
                edge_midpoint_x = edge[7]
                if right_cell == -3:
                    c_p_upper_surface.append(c_p)
                    x_upper_surface.append(edge_midpoint_x)
                elif right_cell == -4:
                    c_p_lower_surface.append(c_p)
                    x_lower_surface.append(edge_midpoint_x)
                else:
                    print(f"something weird happened")
    upper_array = np.column_stack((x_upper_surface, c_p_upper_surface))
    lower_array = np.column_stack((x_lower_surface, c_p_lower_surface))
    upper_array = upper_array[np.argsort(upper_array[:, 0])]
    lower_array = lower_array[np.argsort(lower_array[:, 0])]
    return upper_array, lower_array


def calc_force_coeff(cell_states, edges):
    # edge_info = [start_node(0), end_node(1), left_cell(2), right_cell(3), edge_length(4),
    #               unit_normal_x(5), unit_normal_y(6), edge_center_x(7), edge_center_y(8)]

    c_ref = 18.804
    gamma = 1.4
    pressure_ratio = .7
    rho_0 = 1
    c_t_0 = 1
    p_0 = rho_0 * c_t_0 ** 2 / gamma
    p_out = pressure_ratio * p_0

    M_out_squared = (2 / (gamma - 1)) * (np.power(p_0 / p_out, (gamma - 1) / gamma) - 1)
    q_out = .5 * gamma * p_out * M_out_squared

    n_cells = len(cell_states)
    F_prime_x_tot = 0
    F_prime_y_tot = 0

    for cell_num in range(n_cells):
        for edge_num in range(3):
            edge = edges[cell_num][edge_num]
            right_cell = edge[3]
            unit_normal_x = edge[5]
            unit_normal_y = edge[6]
            if right_cell == -3 or right_cell == -4:
                left_cell_state = cell_states[cell_num]
                p_b = calc_wall_pressure(left_cell_state, edge)
                F_prime_mag = p_b * edge[4]
                F_prime_x = F_prime_mag * unit_normal_x
                F_prime_y = F_prime_mag * unit_normal_y
                F_prime_x_tot += F_prime_x
                F_prime_y_tot += F_prime_y

    C_x = F_prime_x_tot / (q_out * c_ref)
    C_y = F_prime_y_tot / (q_out * c_ref)

    return C_x, C_y


import numpy as np
import matplotlib.pyplot as plt
# import niceplots

def plot_nodes(coordinates):
    plt.plot(coordinates[:, 0], coordinates[:, 1], "o", clip_on=False, label="Nodes")
    plt.legend()
    plt.show()


def plot_cells(coordinates, node_connectivity):
    plt.triplot(coordinates[:, 0], coordinates[:, 1], node_connectivity, color="k", clip_on=False)
    plt.show()


def plot_cell_areas(coordinates, node_connectivity, areas):
    tripcolor_plot = plt.tripcolor(coordinates[:, 0], coordinates[:, 1], node_connectivity, areas, cmap="viridis")
    plt.triplot(coordinates[:, 0], coordinates[:, 1], node_connectivity, color="k", clip_on=False, linewidth=.25)
    colorbar = plt.colorbar(tripcolor_plot)
    colorbar.set_label('Cell Area')
    plt.xlabel("X Position (mm)")
    plt.ylabel("Y Position (mm)")
    plt.show()


def plot_connectivity(coordinates, node_connectivity, cell_connectivity, centroids):
    plt.triplot(coordinates[:, 0], coordinates[:, 1], node_connectivity, color="k", clip_on=False)
    for ii in range(1):
        index = np.random.randint(0, cell_connectivity.shape[0])
        centroid = centroids[index]
        for jj in range(3):
            neighbor_index = cell_connectivity[index, jj]
            if neighbor_index >= 0:
                neighbor_centroid = centroids[neighbor_index]
                plt.plot([centroid[0], neighbor_centroid[0]], [centroid[1], neighbor_centroid[1]], "r-o")
    plt.show()


def plot_cell_numbers(coordinates, node_connectivity, centroids):
    plt.triplot(coordinates[:, 0], coordinates[:, 1], node_connectivity, color="k", clip_on=False)
    for i in range(len(centroids)):
        plt.annotate(str(i), (centroids[i, 0], centroids[i, 1]), color="b", fontsize=10, ha='center', va='center')
    plt.show()


def plot_edge(coordinates, edge):
    interior_color = 'gray'
    inflow_color = 'green'
    outflow_color = 'red'
    lower_surface_color = 'blue'
    upper_surface_color = 'aqua'
    x1 = coordinates[int(edge[0])][0]
    x2 = coordinates[int(edge[1])][0]
    y1 = coordinates[int(edge[0])][1]
    y2 = coordinates[int(edge[1])][1]
    if edge[3] >= 0:
        plt.plot([x1, x2], [y1, y2], color=interior_color)
    elif edge[3] == -1:
        plt.plot([x1, x2], [y1, y2], color=inflow_color)
    elif edge[3] == -2:
        plt.plot([x1, x2], [y1, y2], color=outflow_color)
    elif edge[3] == -3:
        plt.plot([x1, x2], [y1, y2], color=upper_surface_color)
    elif edge[3] == -4:
        plt.plot([x1, x2], [y1, y2], color=lower_surface_color)


def color_edges(coordinates, node_connectivity, edges):
    for i_cell in range(len(node_connectivity)):
        for i_edge in range(node_connectivity.shape[1]):
            plot_edge(coordinates, edges[i_cell][i_edge])
    plt.show()
    print(f"edges colored")


def plot_normals(coordinates, node_connectivity, edges):
    # edge[cell][edge] = [start_node(0), end_node(1), left_cell(2), right_cell(3), edge_length(4),
    #               unit_normal_x(5), unit_normal_y(6), edge_center_x(7), edge_center_y(8)]
    arrow_sf = 1
    plt.triplot(coordinates[:, 0], coordinates[:, 1], node_connectivity, color="k", clip_on=False)
    for i_cell in range(len(node_connectivity)):
        for i_edge in range(node_connectivity.shape[1]):
            edge = edges[i_cell][i_edge]
            if edge[3] >= 0:
                plt.arrow(edge[7], edge[8], arrow_sf * edge[5], arrow_sf * edge[6],
                        head_width=(arrow_sf / 2), head_length=(arrow_sf / 2), fc='r', ec='r')
            elif edge[3] < 0:
                plt.arrow(edge[7], edge[8], arrow_sf * edge[5], arrow_sf * edge[6],
                          head_width=(arrow_sf / 2), head_length=(arrow_sf / 2), fc='g', ec='g')
            else:
                print(f"Bad right cell definition?")
    plt.axis('equal')
    plt.show()
    print(f"normals drawn")


def color_edges_split(coordinates, interior_edges, exterior_edges):
    for i in range(len(interior_edges)):
        plot_edge(coordinates, interior_edges[i])
    for i in range(len(exterior_edges)):
        plot_edge(coordinates, exterior_edges[i])
    plt.show()
    print(f"edges colored")


def plot_normals_split(coordinates, node_connectivity, interior_edges, exterior_edges):
    # edge_info = [start_node(0), end_node(1), left_cell(2), right_cell(3), edge_length(4),
    #               unit_normal_x(5), unit_normal_y(6), edge_center_x(7), edge_center_y(8)]
    arrow_sf = 1
    plt.triplot(coordinates[:, 0], coordinates[:, 1], node_connectivity, color="k", clip_on=False)
    for i in range(len(interior_edges)):
        edge = interior_edges[i]
        plt.arrow(edge[7], edge[8], arrow_sf * edge[5], arrow_sf * edge[6],
                  head_width=(arrow_sf / 2), head_length=(arrow_sf / 2), fc='r', ec='r')
    for i in range(len(exterior_edges)):
        edge = exterior_edges[i]
        plt.arrow(edge[7], edge[8], arrow_sf * edge[5], arrow_sf * edge[6],
                  head_width=(arrow_sf / 2), head_length=(arrow_sf / 2), fc='g', ec='g')
    plt.axis('equal')
    plt.show()


def plot_cell_histories(cell_state_histories, n_iters, initial_cell_states):
    cell_state_histories = np.array(cell_state_histories)
    cell_num = np.random.randint(0, len(initial_cell_states))
    iters = np.arange(n_iters + 1)
    plt.plot(iters, cell_state_histories[:, cell_num, 1])
    plt.show()


def plot_final_state(coordinates, node_connectivity, final_cell_states):
    for i in range(4):
        tripcolor_plot = plt.tripcolor(coordinates[:, 0], coordinates[:, 1], node_connectivity, final_cell_states[:, i], cmap="viridis")
        plt.triplot(coordinates[:, 0], coordinates[:, 1], node_connectivity, color="k", clip_on=False, linewidth=.25)
        colorbar = plt.colorbar(tripcolor_plot)
        if i == 0:
            colorbar.set_label('Density (normalized at inlet)')
        elif i == 1:
            colorbar.set_label('X Momentum')
        elif i == 2:
            colorbar.set_label('Y Momentum')
        elif i == 3:
            colorbar.set_label('Total Energy')
        plt.xlabel("X Position (mm)")
        plt.ylabel("Y Position (mm)")
        plt.show()


def plot_mach_numbers(coordinates, node_connectivity, mach_numbers, blade_num):
    tripcolor_plot = plt.tripcolor(coordinates[:, 0], coordinates[:, 1], node_connectivity, mach_numbers,
                                   cmap="viridis")
    plt.triplot(coordinates[:, 0], coordinates[:, 1], node_connectivity, color="k", clip_on=False, linewidth=.25)
    colorbar = plt.colorbar(tripcolor_plot)
    colorbar.set_label('Mach Number')
    plt.title(f"Mach Number Contour for Blade {blade_num}")
    plt.xlabel("X Position (mm)")
    plt.ylabel("Y Position (mm)")
    plt.show()


def plot_pressure_coeff(upper_arrays, lower_arrays, blade_nums):
    plt.axhline(0, color='black', linestyle='-', linewidth=2)
    plt.axvline(0, color='black', linestyle='-', linewidth=2)
    for i in range(len(blade_nums)):
        data_upper = np.array(upper_arrays[i])
        data_lower = np.array(lower_arrays[i])

        x_upper_surface = data_upper[:, 0]
        x_lower_surface = data_lower[:, 0]
        c_p_upper_surface = data_upper[:, 1]
        c_p_lower_surface = data_lower[:, 1]

        linestyle = '--' if i == 0 else '-.' if i == 1 else '-'
        upper_color = 'lightblue' if i == 0 else 'blue' if i == 1 else 'darkblue'
        lower_color = 'lightgreen' if i == 0 else 'green' if i == 1 else 'darkgreen'

        plt.plot(x_upper_surface, c_p_upper_surface, label=f"Bl {blade_nums[i]}, Up Srf",
                 linestyle=linestyle, color=upper_color)
        plt.plot(x_lower_surface, c_p_lower_surface, label=f"Bl {blade_nums[i]}, Up Srf",
                 linestyle=linestyle, color=lower_color)
    plt.grid()
    plt.xlabel('Position (mm)')
    plt.ylabel('c$_p$')
    plt.title("c$_p$ for Blades 0, 1, and 2")
    plt.legend()
    plt.show()


def plot_norm_history(L1_norm_history):
    iters = range(len(L1_norm_history))
    plt.plot(iters, L1_norm_history)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('L$_1$ Norm')
    plt.title('L$_1$ Norm Convergence for Blade 0')
    plt.grid()
    plt.show()

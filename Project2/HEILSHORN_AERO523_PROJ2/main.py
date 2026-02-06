from functions import *

# CHANGING THESE VALUES MEANS THAT THE PORTION OF THE CODE WILL BE EXECUTED
nItersMax = 5000
partA = 0
partB = 0
partC = 1; partCi = 1; partCii = 0
partD = 0
partE = 0
partF = 0

if partA == 1:
    bladeNum = 0
    # bladeNum = 0
    (coordinates, nodeConnectivity, cellConnectivity, centroids, areas, edges,
        perimeters, hydrodynamicDiameters, initialCellStates) = initialize_blade(bladeNum)

    plot_cell_numbers(coordinates, nodeConnectivity, centroids)
    color_edges(coordinates, nodeConnectivity, edges)
    plot_normals(coordinates, nodeConnectivity, edges)
    plot_cell_areas(coordinates, nodeConnectivity, areas)
    finalCellStates, cellStatesHistory, L1NormHistory = fvsolve(initialCellStates, edges, areas,
                                                                perimeters,
                                                                hydrodynamicDiameters, nItersMax)
    plot_final_state(coordinates, nodeConnectivity, finalCellStates)

if partB == 1:
    bladeNum = 0
    (coordinates, nodeConnectivity, cellConnectivity, centroids, areas, edges,
        perimeters, hydrodynamicDiameters, initialCellStates) = initialize_blade(bladeNum)
    finalCellStates, cellStatesHistory, L1NormHistory = fvsolve(initialCellStates, edges, areas,
                                                                                 perimeters,
                                                                                 hydrodynamicDiameters, nItersMax)
    plot_norm_history(L1NormHistory)

if partC == 1:
    bladeNum = 0
    if partCi == 1:
        nItersMax = 1
        (coordinates, nodeConnectivity, cellConnectivity, centroids, areas, edges,
            perimeters, hydrodynamicDiameters, initialCellStates) = initialize_blade(bladeNum)
        finalCellStates, cellStatesHistory, L1NormHistory = freestream_test(initialCellStates, edges, areas, perimeters,
                                                                hydrodynamicDiameters, nItersMax)
        print(f"Norm of residual for freestream test is: {L1NormHistory}")
    if partCii == 1:
        nItersMax = 5000
        (coordinates, nodeConnectivity, cellConnectivity, centroids, areas, edges,
            perimeters, hydrodynamicDiameters, initialCellStates) = initialize_blade(bladeNum)
        finalCellStates, cellStatesHistory, L1NormHistory = freestream_test(initialCellStates, edges, areas, perimeters,
                                                                hydrodynamicDiameters, nItersMax)
        print(f"Norm of residual for freestream preservation test is: {L1NormHistory[-1]}")

if partD == 1:
    bladeNums = [0, 1, 2]
    for bladeNum in bladeNums:
        (coordinates, nodeConnectivity, cellConnectivity, centroids, areas, edges,
         perimeters, hydrodynamicDiameters, initialCellStates) = initialize_blade(bladeNum)

        finalCellStates, cellStatesHistory, L1NormHistory = fvsolve(initialCellStates, edges, areas, perimeters,
                                                                    hydrodynamicDiameters, nItersMax)
        machNumbers = calc_mach_numbers(finalCellStates)
        plot_mach_numbers(coordinates, nodeConnectivity, machNumbers, bladeNum)

if partE == 1:
    c_pUppers = []
    c_pLowers = []
    bladeNums = [0, 1, 2]
    for bladeNum in bladeNums:
        (coordinates, nodeConnectivity, cellConnectivity, centroids, areas, edges,
         perimeters, hydrodynamicDiameters, initialCellStates) = initialize_blade(bladeNum)

        finalCellStates, cellStatesHistory, L1NormHistory = fvsolve(initialCellStates, edges, areas, perimeters,
                                                                                     hydrodynamicDiameters, nItersMax)

        c_pUpperSurface, c_pLowerSurface = calc_pressure_coeff(finalCellStates, edges)
        c_pUppers.append(c_pUpperSurface.tolist())
        c_pLowers.append(c_pLowerSurface.tolist())

    plot_pressure_coeff(c_pUppers, c_pLowers, bladeNums)

if partF == 1:
    C_xs = []
    C_ys = []
    bladeNums = [0, 1, 2]
    for bladeNum in bladeNums:
        (coordinates, nodeConnectivity, cellConnectivity, centroids, areas, edges,
         perimeters, hydrodynamicDiameters, initialCellStates) = initialize_blade(bladeNum)

        finalCellStates, cellStatesHistory, L1NormHistory = fvsolve(initialCellStates, edges, areas, perimeters,
                                                                                     hydrodynamicDiameters, nItersMax)

        C_x, C_y = calc_force_coeff(finalCellStates, edges)
        C_xs.append(C_x)
        C_ys.append(C_y)
    print(f"C_x's are: {C_xs}")
    print(f"C_y's are: {C_ys}")

print(f"done!")


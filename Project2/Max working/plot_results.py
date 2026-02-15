import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import struct
import sys

def read_results(filename):
    with open(filename, 'rb') as f:
        ne = struct.unpack('i', f.read(4))[0]
        data = struct.unpack('d' * 4 * ne, f.read(8 * 4 * ne))
        return np.array(data).reshape(ne, 4)

def read_mesh(filename):
    with open(filename, 'r') as f:
        nn, ne, dim = map(int, f.readline().split())
        v = np.array([[float(s) for s in f.readline().split()] for _ in range(nn)])
        f.readline() # NB
        # Skip boundaries for this simple plotter
        # We just need vertices and elements
        # This is a simplified parser
        return v, None # Placeholder for elements

def get_elements_from_gri(filename):
    with open(filename, 'r') as f:
        nn, ne, dim = map(int, f.readline().split())
        for _ in range(nn): f.readline()
        nb = int(f.readline())
        for _ in range(nb):
            line = f.readline().split()
            nb_in = int(line[0])
            for _ in range(nb_in): f.readline()
        
        elements = []
        ne_total = 0
        while ne_total < ne:
            line = f.readline().split()
            if not line: break
            nei = int(line[0])
            for _ in range(nei):
                elements.append([int(s)-1 for s in f.readline().split()])
            ne_total += nei
        return np.array(elements)

def plot_results(meshfile, resultsfile):
    u = read_results(resultsfile)
    v, _ = read_mesh(meshfile)
    e = get_elements_from_gri(meshfile)
    
    # Calculate Primitive variables
    gamma = 1.4
    rho = u[:, 0]
    rhou = u[:, 1]
    rhov = u[:, 2]
    rhoe = u[:, 3]
    
    vel_u = rhou / rho
    vel_v = rhov / rho
    qsq = vel_u**2 + vel_v**2
    p = (gamma - 1) * (rhoe - 0.5 * rho * qsq)
    mach = np.sqrt(qsq) / np.sqrt(gamma * p / rho)
    
    # Create triangles for plotting
    verts = v[e]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    pc = PolyCollection(verts, cmap='jet')
    pc.set_array(mach)
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect('equal')
    plt.colorbar(pc, label='Mach Number')
    plt.title(f'Mach Number Distribution - {meshfile}')
    plt.savefig('mach_distribution.png')
    print("Saved mach_distribution.png")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_results.py <meshfile> <results.bin>")
    else:
        plot_results(sys.argv[1], sys.argv[2])

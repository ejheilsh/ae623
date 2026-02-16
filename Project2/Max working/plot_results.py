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
        return v, None

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

def read_residual(filename):
    try:
        with open(filename, 'rb') as f:
            nit = struct.unpack('i', f.read(4))[0]
            data = struct.unpack('d' * nit, f.read(8 * nit))
            return np.array(data)
    except FileNotFoundError:
        return None

def read_cell_residual(filename):
    try:
        with open(filename, 'rb') as f:
            ne = struct.unpack('i', f.read(4))[0]
            data = struct.unpack('d' * ne, f.read(8 * ne))
            return np.array(data)
    except FileNotFoundError:
        return None

def plot_results(meshfile, resultsfile, residualfile, cellresfile):
    u = read_results(resultsfile)
    v, _ = read_mesh(meshfile)
    e = get_elements_from_gri(meshfile)
    res_hist = read_residual(residualfile)
    cell_res = read_cell_residual(cellresfile)
    
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
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
    
    # Plot Mach Number
    pc = PolyCollection(verts, cmap='jet')
    pc.set_array(mach)
    ax1.add_collection(pc)
    ax1.autoscale()
    ax1.set_aspect('equal')
    fig.colorbar(pc, ax=ax1, label='Mach Number')
    ax1.set_title(f'Mach Number Distribution')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # Plot Spatial Residuals
    if cell_res is not None:
        pc2 = PolyCollection(verts, cmap='magma')
        # Log scale for better visibility of residual distribution
        pc2.set_array(np.log10(np.maximum(cell_res, 1e-16)))
        ax2.add_collection(pc2)
        ax2.autoscale()
        ax2.set_aspect('equal')
        fig.colorbar(pc2, ax=ax2, label='Log10(Residual Norm)')
        ax2.set_title(f'Spatial Residual Distribution')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
    else:
        ax2.text(0.5, 0.5, 'Cell residuals not found', ha='center', va='center')

    # Plot Residual History
    if res_hist is not None:
        ax3.plot(res_hist)
        ax3.set_yscale('log')
        ax3.set_title('Convergence History')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Residual')
        ax3.grid(True, which="both", ls="-", alpha=0.5)
    else:
        ax3.text(0.5, 0.5, 'Residual history not found', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('solution_summary.png')
    print("Saved solution_summary.png")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_results.py <meshfile> <results.bin> [residual.bin] [cell_res.bin]")
    else:
        mesh = sys.argv[1]
        results = sys.argv[2]
        residual = sys.argv[3] if len(sys.argv) > 3 else "residual.bin"
        cellres = sys.argv[4] if len(sys.argv) > 4 else "cell_res.bin"
        plot_results(mesh, results, residual, cellres)

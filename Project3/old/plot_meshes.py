import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from readgri import readgri
import os

def plot_mesh_on_ax(ax, mesh_file, title):
    if not os.path.exists(mesh_file):
        ax.text(0.5, 0.5, f"File not found:\n{mesh_file}", 
                ha='center', va='center')
        ax.set_title(title)
        return

    print(f"Reading {mesh_file}...")
    mesh = readgri(mesh_file)
    V = mesh['V']
    E = mesh['E']
    
    tri = Triangulation(V[:, 0], V[:, 1], E)
    ax.triplot(tri, 'b-', lw=0.5)
    ax.set_aspect('equal')
    ax.set_title(f"{title}\n({len(E)} elements)")
    ax.set_xlabel('x')
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel('y')

def main():
    meshes = [
        ('2k.gri', '2k Mesh'),
        ('8k.gri', '8k Mesh'),
        ('32k.gri', '32k Mesh'),
        ('128k.gri', '128k Mesh')
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    
    for i, (fname, title) in enumerate(meshes):
        plot_mesh_on_ax(axes[i], fname, title)
    
    plt.suptitle("AE623 Project 2: Mesh Comparison", fontsize=16)
    
    output_png = 'mesh_comparison.png'
    plt.savefig(output_png, dpi=300)
    print(f"Plot saved to {output_png}")
    plt.show()

if __name__ == "__main__":
    main()

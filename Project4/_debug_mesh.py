"""Parse base.gri following the GRI format used by C++ Mesh::readGRI."""
f = open('grids/base.gri')
line1 = f.readline().split()
nv, ne = int(line1[0]), int(line1[1])
V = []
for i in range(nv):
    parts = f.readline().split()
    V.append((float(parts[0]), float(parts[1])))

# Boundary groups
NB = int(f.readline().strip())
print(f"Nboundary_groups={NB}")
for g in range(NB):
    info = f.readline().split()
    n_edges = int(info[0])
    bnnode = int(info[1])
    name = info[2]
    for j in range(n_edges):
        parts = f.readline().split()
        if name == 'wall':
            v0, v1 = int(parts[0])-1, int(parts[1])-1
            dx = V[v1][0] - V[v0][0]
            dy = V[v1][1] - V[v0][1]
            length = (dx**2 + dy**2)**0.5
            print(f"  wall edge {j}: ({V[v0][0]:.3f},{V[v0][1]:.3f}) -> ({V[v1][0]:.3f},{V[v1][1]:.3f}), len={length:.3f}")
    if name != 'wall':
        print(f"  {name}: {n_edges} edges")
f.close()

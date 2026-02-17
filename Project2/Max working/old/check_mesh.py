import numpy as np
from readgri import readgri

def check_mesh(filename):
    print(f"Checking {filename}...")
    m = readgri(filename)
    V = m['V']
    E = m['E']
    areas = []
    for e in E:
        v = V[e]
        a = 0.5 * abs((v[1,0]-v[0,0])*(v[2,1]-v[0,1]) - (v[2,0]-v[0,0])*(v[1,1]-v[0,1]))
        areas.append(a)
    
    print(f"Min area: {min(areas):.2e}")
    print(f"Max area: {max(areas):.2e}")
    print(f"Count < 1e-12: {sum(1 for a in areas if a < 1e-12)}")
    
    # Check periodic connectivity
    if m['PeriodicGroups']:
        pg = m['PeriodicGroups'][0]
        # pg is usually a numpy array of shape (Np, 2)
        bot_nodes = set(pg[:, 0])
        top_nodes = set(pg[:, 1])
        
        bot_edges = []
        for e in E:
            for i in range(3):
                n1, n2 = e[i], e[(i+1)%3]
                if n1 in bot_nodes and n2 in bot_nodes:
                    bot_edges.append(tuple(sorted((n1, n2))))
        
        bot_edges = list(set(bot_edges))
        edge_nodes = set()
        for n1, n2 in bot_edges:
            edge_nodes.add(n1); edge_nodes.add(n2)
        
        missing_nodes = bot_nodes - edge_nodes
        print(f"Periodic nodes (bot): {len(bot_nodes)}, edges found: {len(bot_edges)}")
        if missing_nodes:
            print(f"Nodes in group but NO periodic edge found: {len(missing_nodes)}")
            
            # Check if used in elements
            all_nodes_in_E = set(E.flatten())
            never_used = missing_nodes - all_nodes_in_E
            if never_used:
                print(f"Nodes in group but NEVER used in any cell: {len(never_used)}")
            else:
                print(f"All missing nodes ARE used in elements!")
            BE_arr = m['BE']
            for bidx, bname in enumerate(m['Bname']):
                b_nodes = set()
                # BE is [NBe x 4]: (n1, n2, elem, bindex)
                mask = BE_arr[:, 3] == bidx
                group_edges = BE_arr[mask]
                for row in group_edges:
                    b_nodes.add(row[0])
                    b_nodes.add(row[1])
                overlap = b_nodes & bot_nodes
                if overlap:
                    print(f"Boundary group '{bname}' (idx {bidx}) contains {len(overlap)} periodic bot nodes!")

if __name__ == "__main__":
    check_mesh('2k.gri')
    check_mesh('8k.gri')
    check_mesh('32k.gri')

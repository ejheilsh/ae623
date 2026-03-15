#ifndef MESH_HPP
#define MESH_HPP

#include "Vector.hpp"
#include <map>
#include <set>
#include <string>
#include <vector>

struct Element {
  int v[3]; // corner vertex indices (always 3 for triangles)
  int q_order = 1;  // geometric degree of this element (1 = straight, 2/3 = curved)
  // For q>1, high-order nodes beyond the 3 corners are stored in ho_nodes[].
  // p=2 triangle: 3 corner + 3 edge-midpoint = 6 nodes  -> ho_nodes has 3 entries
  // p=3 triangle: 3 corner + 6 edge + 1 interior  = 10  -> ho_nodes has 7 entries
  std::vector<int> ho_nodes; // indices into Mesh::V of the extra high-order nodes
};

struct Edge {
  int v[2];    // vertex indices for elemL's side of the edge
  int vR[2];   // vertex indices for elemR's side (same as v[] for regular edges,
               // different top-periodic nodes for periodic edges)
  int elemL, elemR;
};

struct BoundaryEdge {
  int v[2]; // vertex indices
  int elemL;
  int bIndex; // boundary group index
};

struct PeriodicGroup {
  int nPairs;
  std::string type;
  std::vector<std::pair<int, int>> pairs;
};

class Mesh {
public:
  std::vector<Vec2> V;
  std::vector<Element> E;
  std::vector<Edge> IE;
  std::vector<BoundaryEdge> BE;
  std::vector<std::string> Bname;
  std::vector<PeriodicGroup> periodicGroups;

  int q_order_global = 1;  // geometric order read from the GRI file
  bool has_curved_elements = false;  // true when q_order_global > 1

  std::vector<Vec2> centroids;
  std::vector<double> areas;
  std::vector<Vec2> inormals;   // interior edge normals (normalized)
  std::vector<double> ilengths; // interior edge lengths
  std::vector<Vec2> bnormals;   // boundary edge normals (normalized)
  std::vector<double> blengths; // boundary edge lengths

  bool readGRI(const std::string &filename);
  void computeGeometry();

  // Inverse mapping: given a physical point xglob, find the reference coords
  // (xi, eta) for element e using Newton-Raphson (Listing 4.4.1 equivalent).
  // Returns true on convergence.  Works for both straight (q=1) and curved (q>1) elements.
  bool globalToReference(int e, const Vec2 &xglob, double &xi, double &eta,
                         double tol = 1e-10, int maxIter = 100) const;

private:
  void edgeHash(const std::vector<std::vector<std::vector<int>>> &B);
  void appendPeriodicToIE();
};

#endif

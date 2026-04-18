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
  // For q=1, ho_nodes is empty (corners are in v[]).
  // For q>1, ho_nodes stores ALL geometry node indices in GRI row-by-row order
  // (including corners).  v[0..2] hold the 3 corner indices extracted from
  // GRI positions 0, q, npe-1.  The globalToReference function permutes
  // ho_nodes from GRI order to shape.c order for the isoparametric mapping.
  std::vector<int> ho_nodes;
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

struct ElementGeomEval {
  Vec2 x;
  double dx_dxi;
  double dx_deta;
  double dy_dxi;
  double dy_deta;
  double detJ;
};

struct EdgeGeomEval {
  Vec2 x;
  Vec2 tangent;
  Vec2 normal;
  double ds_dt;
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
  std::vector<int> ie_face_colors;
  std::vector<std::vector<int>> ie_faces_by_color;

  void appendPeriodicToIE(); // Used by MeshReader and MeshRefinement decorators

  bool readGRI(const std::string &filename);
  void computeGeometry();

  // Inverse mapping: given a physical point xglob, find the reference coords
  // (xi, eta) for element e using Newton-Raphson (Listing 4.4.1 equivalent).
  // Returns true on convergence.  Works for both straight (q=1) and curved (q>1) elements.
  bool globalToReference(int e, const Vec2 &xglob, double &xi, double &eta,
                         double tol = 1e-10, int maxIter = 100) const;
  ElementGeomEval evaluateElementGeometry(int e, double xi, double eta) const;
  EdgeGeomEval evaluateEdgeGeometry(int e, int va, int vb, double t) const;

private:
  void edgeHash(const std::vector<std::vector<std::vector<int>>> &B);
  void buildInteriorFaceColors();
};

#endif

#ifndef MESH_HPP
#define MESH_HPP

#include "Vector.hpp"
#include <map>
#include <string>
#include <vector>

struct Element {
  int v[3]; // vertex indices
};

struct Edge {
  int v[2]; // vertex indices
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

  std::vector<Vec2> centroids;
  std::vector<double> areas;
  std::vector<Vec2> inormals;   // interior edge normals (normalized)
  std::vector<double> ilengths; // interior edge lengths
  std::vector<Vec2> bnormals;   // boundary edge normals (normalized)
  std::vector<double> blengths; // boundary edge lengths

  bool readGRI(const std::string &filename);
  void computeGeometry();

private:
  void edgeHash(const std::vector<std::vector<std::vector<int>>> &B);
  void appendPeriodicToIE();
};

#endif

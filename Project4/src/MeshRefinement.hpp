#ifndef MESHREFINEMENT_HPP
#define MESHREFINEMENT_HPP

#include "Mesh.hpp"
#include "Vector.hpp"
#include <vector>

// Maps new (refined) mesh entities back to the old mesh for solution transfer.
struct RefinementMap {
  // child_to_parent[new_elem] = old_elem that was split to create it
  std::vector<int> child_to_parent;
  // For each new vertex added at an edge midpoint:
  //   new_vertex_edges[i] = {v0, v1}  the two OLD vertices whose midpoint it is
  std::vector<std::pair<int,int>> new_vertex_edges;
};

// Given element-wise error indicators, mark the top `fraction` for refinement.
std::vector<bool> markByIndicator(const std::vector<double> &eps,
                                  double fraction = 0.25);

// Mark triangles whose aspect ratio exceeds `max_aspect_ratio`.
// Aspect ratio is defined as longest_edge / shortest_altitude.
std::vector<bool> markByAspectRatio(const Mesh &mesh,
                                    double max_aspect_ratio);

// Configure the post-refinement smoother.
void setMeshSmoothingIterations(int iterations);
void setWallGeometryTolerance(double tolerance);

// Bisect each marked triangle at its longest edge.
// Modifies `mesh` in place and returns the parent-child mapping.
RefinementMap bisectMarkedElements(Mesh &mesh, const std::vector<bool> &marked_in);

// Transfer the DG solution from the old mesh to the refined mesh.
// For an element that was not refined, the DOFs are copied directly.
// For a child of a split element, the parent DOFs are copied (constant injection).
std::vector<std::vector<Vec4>> interpolateSolution(
    const std::vector<std::vector<Vec4>> &U_old,
    const RefinementMap &rmap,
    int ndof_per_elem);

#endif

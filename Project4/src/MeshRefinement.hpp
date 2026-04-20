#ifndef MESHREFINEMENT_HPP
#define MESHREFINEMENT_HPP

#include "Mesh.hpp"
#include "Vector.hpp"
#include <set>
#include <vector>

// Maps new (refined) mesh entities back to the old mesh for solution transfer.
struct RefinementMap {
  // child_to_parent[new_elem] = old_elem that was split to create it
  std::vector<int> child_to_parent;
  // For each new vertex added at an edge midpoint:
  //   new_vertex_edges[i] = {v0, v1}  the two OLD vertices whose midpoint it is
  std::vector<std::pair<int,int>> new_vertex_edges;
};

struct CurvedElementDetJMinimum {
  bool exact = false;
  double detJ = 0.0;
  Vec2 ref{0.0, 0.0};
  Vec2 x{0.0, 0.0};
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

// Same as above, but with in-loop fallback: if fewer than target_adj_splits
// adjoint-marked cells are refined (due to geometry rejections), candidates from
// fallback_priority (sorted by descending indicator) are tried automatically.
// Only one call to the connectivity rebuild ever happens.
RefinementMap bisectMarkedElements(Mesh &mesh, const std::vector<bool> &marked_in,
                                   const std::vector<int> &fallback_priority,
                                   int target_adj_splits);

// Print centroid/status diagnostics for wall-adjacent elements in the current mesh.
// Intended for debugging why large blade-surface cells are not being split.
void printWallRefinementDiagnostics(const Mesh &mesh,
                                    const std::vector<bool> &marked_in);

// Compute the exact minimum Jacobian determinant for a q=2 curved triangle.
// Returns false for non-q2 elements.
bool exactQ2DetJMinimum(const Mesh &mesh, int elem_idx,
                        CurvedElementDetJMinimum &out);

// Try a localized high-order patch repair around an invalid curved element.
// Intended to rescue a small q2 wall-adjacent region before rejecting a pass.
bool repairInvalidCurvedPatch(Mesh &mesh, int elem_idx);

// Try a localized patch-quality repair around a valid but poorly conditioned
// element. Intended for thin first-layer patches that pass detJ checks but
// still destabilize the next primal solve.
bool repairLowQualityCurvedPatch(Mesh &mesh, int elem_idx);

// Transfer the DG solution from the old mesh to the refined mesh.
// For an element that was not refined, the DOFs are copied directly.
// For a child of a split element, the parent DOFs are copied (constant injection).
std::vector<std::vector<Vec4>> interpolateSolution(
    const std::vector<std::vector<Vec4>> &U_old,
    const RefinementMap &rmap,
    int ndof_per_elem);

#endif

#include "MeshRefinement.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <vector>

std::vector<bool> markByIndicator(const std::vector<double> &eps,
                                  double fraction) {
  int Ne = eps.size();
  std::vector<bool> marked(Ne, false);

  std::vector<double> sorted_eps = eps;
  std::sort(sorted_eps.rbegin(), sorted_eps.rend());  // descending
  int n_mark = std::max(1, (int)(fraction * Ne));
  double threshold = sorted_eps[n_mark - 1];

  for (int e = 0; e < Ne; ++e)
    if (eps[e] >= threshold) marked[e] = true;

  return marked;
}

RefinementMap bisectMarkedElements(Mesh &mesh, const std::vector<bool> &marked_in) {
  RefinementMap rmap;
  std::vector<bool> marked = marked_in;  // mutable local copy

  // ── Phase 1: Propagate marks across shared edges for conformity ──
  //    If a marked element shares an internal edge with an unmarked one,
  //    mark the neighbor too.  Repeat until stable.
  bool changed = true;
  while (changed) {
    changed = false;
    for (int i = 0; i < (int)mesh.IE.size(); ++i) {
      int eL = mesh.IE[i].elemL;
      int eR = mesh.IE[i].elemR;
      if (marked[eL] && !marked[eR]) { marked[eR] = true; changed = true; }
      if (marked[eR] && !marked[eL]) { marked[eL] = true; changed = true; }
    }
  }

  // ── Phase 2: Split marked elements ──
  //    Record old boundary edges so we can reassign groups after splitting.
  //    Key = sorted vertex pair → boundary group index.
  std::map<std::pair<int,int>, int> old_boundary_edge;
  for (int i = 0; i < (int)mesh.BE.size(); ++i) {
    int a = mesh.BE[i].v[0], b = mesh.BE[i].v[1];
    old_boundary_edge[{std::min(a,b), std::max(a,b)}] = mesh.BE[i].bIndex;
  }

  int old_Ne = mesh.E.size();
  rmap.child_to_parent.resize(old_Ne);
  for (int e = 0; e < old_Ne; ++e) rmap.child_to_parent[e] = e;

  std::map<std::pair<int,int>, int> edge_midpoint;  // sorted edge → midpoint vertex

  for (int e = 0; e < old_Ne; ++e) {
    if (!marked[e]) continue;

    // Find the longest edge of triangle e
    int v0 = mesh.E[e].v[0], v1 = mesh.E[e].v[1], v2 = mesh.E[e].v[2];
    double len01 = (mesh.V[v1] - mesh.V[v0]).norm();
    double len12 = (mesh.V[v2] - mesh.V[v1]).norm();
    double len20 = (mesh.V[v0] - mesh.V[v2]).norm();

    int va, vb, vc;  // va-vb = longest edge, vc = opposite vertex
    if (len01 >= len12 && len01 >= len20) { va = v0; vb = v1; vc = v2; }
    else if (len12 >= len01 && len12 >= len20) { va = v1; vb = v2; vc = v0; }
    else                                       { va = v2; vb = v0; vc = v1; }

    // Get or create midpoint vertex on the longest edge
    auto edge_key = std::make_pair(std::min(va, vb), std::max(va, vb));
    int vm;
    if (edge_midpoint.count(edge_key)) {
      vm = edge_midpoint[edge_key];
    } else {
      vm = (int)mesh.V.size();
      mesh.V.push_back((mesh.V[va] + mesh.V[vb]) * 0.5);
      edge_midpoint[edge_key] = vm;
      rmap.new_vertex_edges.push_back({va, vb});

      // If the split edge was a boundary edge, register both halves
      if (old_boundary_edge.count(edge_key)) {
        int g = old_boundary_edge[edge_key];
        auto k1 = std::make_pair(std::min(va, vm), std::max(va, vm));
        auto k2 = std::make_pair(std::min(vm, vb), std::max(vm, vb));
        old_boundary_edge[k1] = g;
        old_boundary_edge[k2] = g;
      }
    }

    // Replace element e with child 1: (vc, va, vm)
    mesh.E[e].v[0] = vc; mesh.E[e].v[1] = va; mesh.E[e].v[2] = vm;
    mesh.E[e].q_order = 1;
    mesh.E[e].ho_nodes.clear();

    // Append child 2: (vc, vm, vb)
    Element child2;
    child2.v[0] = vc; child2.v[1] = vm; child2.v[2] = vb;
    child2.q_order = 1;
    mesh.E.push_back(child2);
    rmap.child_to_parent.push_back(e);
  }

  // ── Phase 3: Rebuild IE and BE from scratch ──
  //    Hash every edge of every element.  Edges shared by two elements
  //    become interior edges (IE).  Unmatched edges that appear in
  //    old_boundary_edge become boundary edges (BE).
  int new_Ne = (int)mesh.E.size();
  std::map<std::pair<int,int>, int> H;  // directed edge (n1,n2) → elem+1

  mesh.IE.clear();
  mesh.BE.clear();

  for (int e = 0; e < new_Ne; ++e) {
    for (int i = 0; i < 3; ++i) {
      int n1 = mesh.E[e].v[i];
      int n2 = mesh.E[e].v[(i + 1) % 3];
      // Check if the reverse edge was already registered (= shared edge)
      if (H.count({n2, n1})) {
        int eL = H[{n2, n1}] - 1;
        int eR = e;
        Edge ie;
        ie.v[0] = n2; ie.v[1] = n1;
        ie.vR[0] = n2; ie.vR[1] = n1;
        ie.elemL = eL; ie.elemR = eR;
        mesh.IE.push_back(ie);
        H.erase({n2, n1});
      } else {
        H[{n1, n2}] = e + 1;
      }
    }
  }

  // Remaining unmatched edges are boundary edges
  for (auto &[key, val] : H) {
    int n1 = key.first, n2 = key.second;
    int elem = val - 1;
    auto skey = std::make_pair(std::min(n1, n2), std::max(n1, n2));
    int bIdx = 0;
    if (old_boundary_edge.count(skey))
      bIdx = old_boundary_edge[skey];
    BoundaryEdge be;
    be.v[0] = n1; be.v[1] = n2;
    be.elemL = elem;
    be.bIndex = bIdx;
    mesh.BE.push_back(be);
  }

  // Recompute geometric data (centroids, areas, normals, lengths)
  mesh.has_curved_elements = false;
  mesh.q_order_global = 1;
  mesh.computeGeometry();

  std::cerr << "bisectMarkedElements: " << old_Ne << " -> " << new_Ne
            << " elements (" << (new_Ne - old_Ne) << " added)" << std::endl;

  return rmap;
}

std::vector<std::vector<Vec4>> interpolateSolution(
    const std::vector<std::vector<Vec4>> &U_old,
    const RefinementMap &rmap,
    int ndof_per_elem) {
  int new_Ne = rmap.child_to_parent.size();
  std::vector<std::vector<Vec4>> U_new(new_Ne, std::vector<Vec4>(ndof_per_elem, Vec4{0,0,0,0}));

  for (int e_new = 0; e_new < new_Ne; ++e_new) {
    int e_old = rmap.child_to_parent[e_new];
    for (int j = 0; j < ndof_per_elem; ++j)
      U_new[e_new][j] = U_old[e_old][j];
  }

  return U_new;
}

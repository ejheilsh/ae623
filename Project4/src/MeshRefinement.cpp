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

  // ── Phase 1: Propagate marks for conformity (longest-edge bisection) ──
  //    A marked element e is bisected along its longest edge.  If that edge
  //    is shared with an unmarked neighbor n, then n also needs bisection
  //    to avoid a hanging node — BUT only if the shared edge is ALSO the
  //    longest edge of n (otherwise the neighbor's bisection would split a
  //    different edge and the hanging node persists).  Propagate until stable.
  //
  //    Helper: check if a given edge (va,vb) is the longest edge of element e.
  auto isLongestEdge = [&](int e, int va, int vb) -> bool {
    int v0 = mesh.E[e].v[0], v1 = mesh.E[e].v[1], v2 = mesh.E[e].v[2];
    double len01 = (mesh.V[v1] - mesh.V[v0]).norm();
    double len12 = (mesh.V[v2] - mesh.V[v1]).norm();
    double len20 = (mesh.V[v0] - mesh.V[v2]).norm();
    double longestLen = std::max({len01, len12, len20});
    double edgeLen = (mesh.V[vb] - mesh.V[va]).norm();
    return edgeLen >= longestLen - 1e-14;
  };

  auto longestEdgeOf = [&](int e) -> std::pair<int,int> {
    int v0 = mesh.E[e].v[0], v1 = mesh.E[e].v[1], v2 = mesh.E[e].v[2];
    double len01 = (mesh.V[v1] - mesh.V[v0]).norm();
    double len12 = (mesh.V[v2] - mesh.V[v1]).norm();
    double len20 = (mesh.V[v0] - mesh.V[v2]).norm();
    int va, vb;
    if (len01 >= len12 && len01 >= len20) { va = v0; vb = v1; }
    else if (len12 >= len01 && len12 >= len20) { va = v1; vb = v2; }
    else { va = v2; vb = v0; }
    return {std::min(va,vb), std::max(va,vb)};
  };

  bool changed = true;
  while (changed) {
    changed = false;
    for (int i = 0; i < (int)mesh.IE.size(); ++i) {
      int eL = mesh.IE[i].elemL;
      int eR = mesh.IE[i].elemR;
      int sv0 = mesh.IE[i].v[0], sv1 = mesh.IE[i].v[1];
      // If eL is marked and eR is not: mark eR only if the shared edge
      // is the longest edge of eL (so eL WILL be bisected along it)
      // AND the shared edge is also the longest edge of eR.
      if (marked[eL] && !marked[eR]) {
        auto leL = longestEdgeOf(eL);
        auto sharedKey = std::make_pair(std::min(sv0,sv1), std::max(sv0,sv1));
        if (leL == sharedKey && isLongestEdge(eR, sv0, sv1)) {
          marked[eR] = true; changed = true;
        }
      }
      if (marked[eR] && !marked[eL]) {
        auto leR = longestEdgeOf(eR);
        auto sharedKey = std::make_pair(std::min(sv0,sv1), std::max(sv0,sv1));
        if (leR == sharedKey && isLongestEdge(eL, sv0, sv1)) {
          marked[eL] = true; changed = true;
        }
      }
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

  // ── Pre-pass: collect curved edge midpoints from all q=2 elements ───────
  // For a q=2 element with ho_nodes in GRI order, the curved midpoints are:
  //   edge (v[0],v[1]) → V[ho_nodes[1]]
  //   edge (v[1],v[2]) → V[ho_nodes[4]]
  //   edge (v[0],v[2]) → V[ho_nodes[3]]
  // Storing these before bisection lets us place new bisection vertices ON the
  // actual curved boundary instead of at the straight arithmetic average.
  std::map<std::pair<int,int>, Vec2> curved_edge_midpoint;
  for (int e = 0; e < old_Ne; ++e) {
    if (mesh.E[e].q_order < 2 || mesh.E[e].ho_nodes.size() < 6) continue;
    int v0 = mesh.E[e].v[0], v1 = mesh.E[e].v[1], v2 = mesh.E[e].v[2];
    const auto &hn = mesh.E[e].ho_nodes;
    auto add = [&](int a, int b, int mid_hn) {
      auto key = std::make_pair(std::min(a,b), std::max(a,b));
      if (!curved_edge_midpoint.count(key))
        curved_edge_midpoint[key] = mesh.V[hn[mid_hn]];
    };
    add(v0, v1, 1);   // midpoint of edge v[0]–v[1] lives at ho_nodes[1]
    add(v1, v2, 4);   // midpoint of edge v[1]–v[2] lives at ho_nodes[4]
    add(v0, v2, 3);   // midpoint of edge v[0]–v[2] lives at ho_nodes[3]
  }

  std::map<std::pair<int,int>, int> edge_midpoint;  // sorted edge → midpoint vertex

  // ── Geometry midpoint tracking for q=2 preservation ──
  std::map<std::pair<int,int>, int> geo_midpoint;  // sorted edge → q=2 geometry midpoint vertex
  for (int e = 0; e < old_Ne; ++e) {
    if (mesh.E[e].q_order < 2 || mesh.E[e].ho_nodes.size() < 6) continue;
    int gv0 = mesh.E[e].v[0], gv1 = mesh.E[e].v[1], gv2 = mesh.E[e].v[2];
    const auto &ghn = mesh.E[e].ho_nodes;
    auto addGeo = [&](int a, int b, int hn_idx) {
      auto key = std::make_pair(std::min(a,b), std::max(a,b));
      if (!geo_midpoint.count(key))
        geo_midpoint[key] = ghn[hn_idx];
    };
    addGeo(gv0, gv1, 1);
    addGeo(gv1, gv2, 4);
    addGeo(gv0, gv2, 3);
  }
  auto getOrCreateGeoMid = [&](int a, int b, Vec2 pos) -> int {
    auto key = std::make_pair(std::min(a,b), std::max(a,b));
    if (geo_midpoint.count(key)) return geo_midpoint[key];
    int idx = (int)mesh.V.size();
    mesh.V.push_back(pos);
    geo_midpoint[key] = idx;
    return idx;
  };

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
      // Use the actual curved midpoint if available (from a q=2 parent element),
      // otherwise fall back to the straight arithmetic average.
      if (curved_edge_midpoint.count(edge_key))
        mesh.V.push_back(curved_edge_midpoint[edge_key]);
      else
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

    // Create children, preserving q=2 geometry when parent is curved
    std::cerr << "  DBG bisect e=" << e << " q=" << mesh.E[e].q_order
              << " ho=" << mesh.E[e].ho_nodes.size() << std::endl;
    if (mesh.E[e].q_order >= 2 && mesh.E[e].ho_nodes.size() >= 6) {
      // Cache parent data before modifying element e
      std::vector<int> phn = mesh.E[e].ho_nodes;
      int pv0 = mesh.E[e].v[0], pv1 = mesh.E[e].v[1], pv2 = mesh.E[e].v[2];
      // ho_nodes GRI order: [v0, mid01, v1, mid02, mid12, v2]
      int mid01 = phn[1], mid02 = phn[3], mid12 = phn[4];

      // Identify midpoint vertex indices for inherited and split edges
      int mid_split = -1, mid_vc_va = -1, mid_vc_vb = -1;
      struct EM { int a, b, mid; };
      EM ems[3] = {{pv0,pv1,mid01}, {pv1,pv2,mid12}, {pv0,pv2,mid02}};
      for (auto &em : ems) {
        if ((em.a==va && em.b==vb) || (em.a==vb && em.b==va)) mid_split = em.mid;
        if ((em.a==vc && em.b==va) || (em.a==va && em.b==vc)) mid_vc_va = em.mid;
        if ((em.a==vc && em.b==vb) || (em.a==vb && em.b==vc)) mid_vc_vb = em.mid;
      }

      // Quadratic subdivision: evaluate original curve at t=0.25 and t=0.75
      // Lagrange basis on (va, mid_split, vb) at t=0,0.5,1:
      //   N0(0.25)=0.375, N1(0.25)=0.75,  N2(0.25)=-0.125
      //   N0(0.75)=-0.125, N1(0.75)=0.75, N2(0.75)=0.375
      Vec2 qpt  = mesh.V[va]*0.375 + mesh.V[mid_split]*0.75 + mesh.V[vb]*(-0.125);
      Vec2 tqpt = mesh.V[va]*(-0.125) + mesh.V[mid_split]*0.75 + mesh.V[vb]*0.375;

      // Get or create geometry midpoint vertices for new child edges
      int gm_vc_vm = getOrCreateGeoMid(vc, vm, (mesh.V[vc]+mesh.V[vm])*0.5);
      int gm_va_vm = getOrCreateGeoMid(va, vm, qpt);
      int gm_vm_vb = getOrCreateGeoMid(vm, vb, tqpt);

      // Child 1: (vc, va, vm)
      // ho_nodes GRI order: [c0, mid(c0,c1), c1, mid(c0,c2), mid(c1,c2), c2]
      mesh.E[e].v[0] = vc; mesh.E[e].v[1] = va; mesh.E[e].v[2] = vm;
      mesh.E[e].q_order = 2;
      mesh.E[e].ho_nodes = {vc, mid_vc_va, va, gm_vc_vm, gm_va_vm, vm};

      // Child 2: (vc, vm, vb)
      Element child2;
      child2.v[0] = vc; child2.v[1] = vm; child2.v[2] = vb;
      child2.q_order = 2;
      child2.ho_nodes = {vc, gm_vc_vm, vm, mid_vc_vb, gm_vm_vb, vb};
      mesh.E.push_back(child2);
      rmap.child_to_parent.push_back(e);
    } else {
      // q=1 parent: keep q=1 children
      mesh.E[e].v[0] = vc; mesh.E[e].v[1] = va; mesh.E[e].v[2] = vm;
      mesh.E[e].q_order = 1;
      mesh.E[e].ho_nodes.clear();

      Element child2;
      child2.v[0] = vc; child2.v[1] = vm; child2.v[2] = vb;
      child2.q_order = 1;
      mesh.E.push_back(child2);
      rmap.child_to_parent.push_back(e);
    }
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

  // Detect actual geometry order after refinement
  mesh.has_curved_elements = false;
  mesh.q_order_global = 1;
  for (int e = 0; e < (int)mesh.E.size(); ++e) {
    if (mesh.E[e].q_order > 1) {
      mesh.has_curved_elements = true;
      mesh.q_order_global = std::max(mesh.q_order_global, mesh.E[e].q_order);
    }
  }
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

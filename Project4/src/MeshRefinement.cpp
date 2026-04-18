#include "MeshRefinement.hpp"
#include "spline.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <vector>
#include <string>

namespace {

std::string lowerCopy(std::string s) {
  for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

double goldenSectionArgmin(const std::function<double(double)> &f,
                           double a, double b, int max_iter = 80,
                           double tol = 1e-10) {
  const double phi = 0.5 * (1.0 + std::sqrt(5.0));
  const double invphi = 1.0 / phi;

  double c = b - invphi * (b - a);
  double d = a + invphi * (b - a);
  double fc = f(c);
  double fd = f(d);

  for (int iter = 0; iter < max_iter && std::abs(b - a) > tol; ++iter) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - invphi * (b - a);
      fc = f(c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + invphi * (b - a);
      fd = f(d);
    }
  }
  return (fc < fd) ? c : d;
}

struct BladeSplineReference {
  bool attempted = false;
  bool loaded = false;
  double xmin = 0.0;
  double xmax = 0.0;
  double s_total_upper = 0.0;
  double s_total_lower = 0.0;
  double snap_distance_tol = 50.0;
  tk::spline upper_x;
  tk::spline upper_y;
  tk::spline lower_x;
  tk::spline lower_y;
};

BladeSplineReference &bladeSplineReference() {
  static BladeSplineReference ref;
  if (ref.attempted) return ref;
  ref.attempted = true;

  std::vector<std::pair<std::string, std::string>> candidates;
  candidates.push_back({"grids/bladeupper.txt", "grids/bladelower.txt"});
  candidates.push_back({"../Project1/data/bladeupper.txt",
                        "../Project1/data/bladelower.txt"});
  candidates.push_back({"../../Project1/data/bladeupper.txt",
                        "../../Project1/data/bladelower.txt"});
  
  double domain_height = 18.0;
  if (const char *shift_env = std::getenv("BLADE_SURFACE_YSHIFT")) {
    if (*shift_env) domain_height = std::atof(shift_env);
  }
  double snap_distance_tol = 50.0;
  if (const char *snap_env = std::getenv("BLADE_SNAP_MAX_DIST")) {
    if (*snap_env) snap_distance_tol = std::atof(snap_env);
  }

  for (const auto &paths : candidates) {
    std::ifstream fup(paths.first), flo(paths.second);
    if (!fup.good() || !flo.good()) continue;

    std::vector<double> up_x, up_y, lo_x, lo_y;
    double x = 0.0, y = 0.0;
    while (fup >> x >> y) { up_x.push_back(x); up_y.push_back(y); }
    while (flo >> x >> y) { lo_x.push_back(x); lo_y.push_back(y); }
    if (up_x.size() < 2 || lo_x.size() < 2) continue;

    // Shift Geometry
    auto it_le = std::min_element(up_x.begin(), up_x.end());
    double shift_x = *it_le;
    double shift_y = up_y[std::distance(up_x.begin(), it_le)];
    for (size_t i = 0; i < up_x.size(); ++i) {
      up_x[i] -= shift_x;
      up_y[i] -= shift_y;
    }
    for (size_t i = 0; i < lo_x.size(); ++i) {
      lo_x[i] -= shift_x;
      lo_y[i] -= shift_y;
      lo_y[i] += domain_height;
    }

    auto paramArcLength = [](const std::vector<double>& x, const std::vector<double>& y, 
                             std::vector<double>& s) -> double {
      s.resize(x.size());
      s[0] = 0.0;
      for(size_t i=1; i<x.size(); ++i) {
        double dx = x[i] - x[i-1];
        double dy = y[i] - y[i-1];
        s[i] = s[i-1] + std::sqrt(dx*dx + dy*dy);
      }
      return s.back();
    };

    std::vector<double> up_s, lo_s;
    ref.s_total_upper = paramArcLength(up_x, up_y, up_s);
    ref.s_total_lower = paramArcLength(lo_x, lo_y, lo_s);

    ref.upper_x.set_points(up_s, up_x);
    ref.upper_y.set_points(up_s, up_y);
    ref.lower_x.set_points(lo_s, lo_x);
    ref.lower_y.set_points(lo_s, lo_y);

    ref.loaded = true;
    ref.xmin = -1.0; 
    ref.xmax = 10.0; 
    ref.snap_distance_tol = snap_distance_tol;
    std::cerr << "Loaded analytic tk::spline geometry reference from " << paths.first
              << " & " << paths.second << std::endl;
    return ref;
  }

  return ref;
}

Vec2 projectToBladeSpline(const Vec2 &p) {
  BladeSplineReference &ref = bladeSplineReference();
  if (!ref.loaded) return p;

  auto objectiveFor = [&](const tk::spline &sx, const tk::spline &sy) {
    return [&](double s) {
      double dx = sx(s) - p.x;
      double dy = sy(s) - p.y;
      return dx * dx + dy * dy;
    };
  };

  struct Candidate {
    double d2;
    Vec2 pt;
  };
  std::vector<Candidate> candidates;
  auto addCandidate = [&](const tk::spline &sx, const tk::spline &sy, double s_total) {
    auto f = objectiveFor(sx, sy);
    int n_search = 20;
    double best_s = 0;
    double best_obj = 1e300;
    for (int i=0; i<=n_search; i++) {
        double s = s_total * (i / (double)n_search);
        double obj = f(s);
        if (obj < best_obj) { best_obj = obj; best_s = s; }
    }
    double lower = std::max(0.0, best_s - s_total/n_search);
    double upper = std::min(s_total, best_s + s_total/n_search);

    double sbest = goldenSectionArgmin(f, lower, upper);
    candidates.push_back({f(sbest), {sx(sbest), sy(sbest)}});
  };
  
  addCandidate(ref.upper_x, ref.upper_y, ref.s_total_upper);
  addCandidate(ref.lower_x, ref.lower_y, ref.s_total_lower);
  
  auto checkOffset = [&](const tk::spline &sx, const tk::spline &sy, double s_total, double offsetY) {
    auto f = [&](double s) {
      double dx = sx(s) - p.x;
      double dy = (sy(s) + offsetY) - p.y;
      return dx*dx + dy*dy;
    };
    int n_search = 20;
    double best_s = 0;
    double best_obj = 1e300;
    for (int i=0; i<=n_search; i++) {
        double s = s_total * (i / (double)n_search);
        double obj = f(s);
        if (obj < best_obj) { best_obj = obj; best_s = s; }
    }
    double lower = std::max(0.0, best_s - s_total/n_search);
    double upper = std::min(s_total, best_s + s_total/n_search);
    double sbest = goldenSectionArgmin(f, lower, upper);
    candidates.push_back({f(sbest), {sx(sbest), sy(sbest) + offsetY}});
  };

  checkOffset(ref.upper_x, ref.upper_y, ref.s_total_upper, 18.0);
  checkOffset(ref.lower_x, ref.lower_y, ref.s_total_lower, -18.0);

  auto best = std::min_element(
      candidates.begin(), candidates.end(),
      [](const Candidate &a, const Candidate &b) { return a.d2 < b.d2; });
  double best_d = std::sqrt(best->d2);
  if (best_d > ref.snap_distance_tol) return p;
  return best->pt;
}

} // namespace

std::vector<bool> markByIndicator(const std::vector<double> &eps,
                                  double fraction) {
  int Ne = eps.size();
  std::vector<bool> marked(Ne, false);

  if (fraction <= 0.0) return marked;

  std::vector<double> sorted_eps = eps;
  std::sort(sorted_eps.rbegin(), sorted_eps.rend());  // descending
  int n_mark = std::max(1, (int)(fraction * Ne));
  if (n_mark > Ne) n_mark = Ne;
  double threshold = sorted_eps[n_mark - 1];

  for (int e = 0; e < Ne; ++e)
    if (eps[e] >= threshold) marked[e] = true;

  return marked;
}

RefinementMap bisectMarkedElements(Mesh &mesh, const std::vector<bool> &marked_in) {
  RefinementMap rmap;
  std::vector<bool> marked = marked_in;  // mutable local copy

  // Map Periodic Nodes for twin synchronization
  struct PNodeData { int twin; int g; bool is_bottom; };
  std::map<int, PNodeData> ptwin;
  for (int g = 0; g < (int)mesh.periodicGroups.size(); ++g) {
      for (const auto& pair : mesh.periodicGroups[g].pairs) {
          ptwin[pair.first] = {pair.second, g, true};
          ptwin[pair.second] = {pair.first, g, false};
      }
  }

  // ── Phase 1: Red-Green Splitting Logic ──
  // Track which edges are flagged for splitting to guarantee topological conformity.
  std::map<std::pair<int,int>, bool> edge_split;
  int old_Ne = mesh.E.size();
  for (int e = 0; e < old_Ne; ++e) {
    int v0 = mesh.E[e].v[0], v1 = mesh.E[e].v[1], v2 = mesh.E[e].v[2];
    edge_split[{std::min(v0,v1), std::max(v0,v1)}] = false;
    edge_split[{std::min(v1,v2), std::max(v1,v2)}] = false;
    edge_split[{std::min(v2,v0), std::max(v2,v0)}] = false;
  }

  // Mark the longest edge of every element explicitly targeted by error indicators (Red Split)
  for (int e = 0; e < old_Ne; ++e) {
    if (marked[e]) {
      int v0 = mesh.E[e].v[0], v1 = mesh.E[e].v[1], v2 = mesh.E[e].v[2];
      double len01 = (mesh.V[v1] - mesh.V[v0]).norm();
      double len12 = (mesh.V[v2] - mesh.V[v1]).norm();
      double len20 = (mesh.V[v0] - mesh.V[v2]).norm();
      if (len01 >= len12 && len01 >= len20) edge_split[{std::min(v0,v1), std::max(v0,v1)}] = true;
      else if (len12 >= len01 && len12 >= len20) edge_split[{std::min(v1,v2), std::max(v1,v2)}] = true;
      else edge_split[{std::min(v2,v0), std::max(v2,v0)}] = true;
    }
  }

  // Build boundary-edge lookup early so the propagation loop can use it.
  std::map<std::pair<int,int>, int> old_boundary_edge;
  for (int i = 0; i < (int)mesh.BE.size(); ++i) {
    int a = mesh.BE[i].v[0], b = mesh.BE[i].v[1];
    old_boundary_edge[{std::min(a,b), std::max(a,b)}] = mesh.BE[i].bIndex;
  }

  // Propagate to prevent hanging nodes:
  // If an element is subjected to >= 2 split edges from neighbors, it upgrades to a full
  // isotropic Quartering (all 3 edges split). This stops massive chains of low-quality slivers.
  // Additionally, any element that is going to be refined (>= 1 split edge) and touches the
  // blade wall is always upgraded to a full red split so the wall edge gets a new on-surface
  // vertex and snapping to the blade spline occurs.
  bool changed = true;
  while (changed) {
    changed = false;
    for (int e = 0; e < old_Ne; ++e) {
      int v0 = mesh.E[e].v[0], v1 = mesh.E[e].v[1], v2 = mesh.E[e].v[2];
      auto e0 = std::make_pair(std::min(v0,v1), std::max(v0,v1));
      auto e1 = std::make_pair(std::min(v1,v2), std::max(v1,v2));
      auto e2 = std::make_pair(std::min(v2,v0), std::max(v2,v0));
      int counts = edge_split[e0] + edge_split[e1] + edge_split[e2];

      if (counts == 2) {
        if (!edge_split[e0]) { edge_split[e0] = true; changed = true; }
        if (!edge_split[e1]) { edge_split[e1] = true; changed = true; }
        if (!edge_split[e2]) { edge_split[e2] = true; changed = true; }
      }

      // Wall-adjacent upgrade: any element being refined that touches a blade wall
      // is forced to red so all three edges (including the wall edge) are split.
      if (counts >= 1 && counts < 3) {
        bool has_wall = false;
        for (const auto& key : {e0, e1, e2}) {
          auto it = old_boundary_edge.find(key);
          if (it != old_boundary_edge.end()) {
            int g = it->second;
            if (g >= 0 && g < (int)mesh.Bname.size() &&
                lowerCopy(mesh.Bname[g]) == "wall") {
              has_wall = true;
              break;
            }
          }
        }
        if (has_wall) {
          if (!edge_split[e0]) { edge_split[e0] = true; changed = true; }
          if (!edge_split[e1]) { edge_split[e1] = true; changed = true; }
          if (!edge_split[e2]) { edge_split[e2] = true; changed = true; }
        }
      }
    }

    // Synchronize periodic boundaries geometrically via their graph mappings!
    for (auto const& [edge, is_split] : edge_split) {
        if (is_split) {
            int va = edge.first, vb = edge.second;
            if (ptwin.count(va) && ptwin.count(vb) && ptwin[va].g == ptwin[vb].g) {
                auto twin_edge = std::make_pair(std::min(ptwin[va].twin, ptwin[vb].twin),
                                                std::max(ptwin[va].twin, ptwin[vb].twin));
                if (edge_split.count(twin_edge) && !edge_split[twin_edge]) {
                    edge_split[twin_edge] = true;
                    changed = true;
                }
            }
        }
    }
  }

  // ── Phase 2: Red-Green Sub-Division Execution ──

  std::map<std::pair<int,int>, int> edge_midpoint;
  std::vector<Element> next_E;
  std::vector<int> next_child_to_parent;

  auto getMidpoint = [&](int a, int b) {
    auto key = std::make_pair(std::min(a,b), std::max(a,b));
    if (edge_midpoint.count(key)) return edge_midpoint[key];
    
    int v_idx = (int)mesh.V.size();
    Vec2 midpoint = (mesh.V[a] + mesh.V[b]) * 0.5;
    
    // AMR debugging: spline snapping restored for aerodynamic continuity!
    bool is_wall_edge = false;
    if (old_boundary_edge.count(key)) {
      int g = old_boundary_edge[key];
      if (g >= 0 && g < (int)mesh.Bname.size())
        is_wall_edge = (lowerCopy(mesh.Bname[g]) == "wall");
    }
    if (is_wall_edge) midpoint = projectToBladeSpline(midpoint);
    
    mesh.V.push_back(midpoint);
    edge_midpoint[key] = v_idx;
    rmap.new_vertex_edges.push_back({a, b});
    
    // If the split edge was a boundary edge, register both halves back to the group
    if (old_boundary_edge.count(key)) {
      int g = old_boundary_edge[key];
      old_boundary_edge[{std::min(a, v_idx), std::max(a, v_idx)}] = g;
      old_boundary_edge[{std::min(b, v_idx), std::max(b, v_idx)}] = g;
    }
    return v_idx;
  };

  for (int e = 0; e < old_Ne; ++e) {
    int v[3] = {mesh.E[e].v[0], mesh.E[e].v[1], mesh.E[e].v[2]};
    bool s[3];
    s[0] = edge_split[{std::min(v[0],v[1]), std::max(v[0],v[1])}];
    s[1] = edge_split[{std::min(v[1],v[2]), std::max(v[1],v[2])}];
    s[2] = edge_split[{std::min(v[2],v[0]), std::max(v[2],v[0])}];

    int split_count = s[0] + s[1] + s[2];

    Element child_template;
    child_template.q_order = 1;

    if (split_count == 0) {
      // Kept elements
      child_template.v[0] = v[0]; child_template.v[1] = v[1]; child_template.v[2] = v[2];
      next_E.push_back(child_template);
      next_child_to_parent.push_back(e);
    } 
    else if (split_count == 1) {
      // Green Refinement: EXACTLY one edge split. Bisection to seal hanging nodes.
      int split_idx = 0;
      if (s[1]) split_idx = 1;
      if (s[2]) split_idx = 2;
      
      int va = v[split_idx];
      int vb = v[(split_idx+1)%3];
      int vc = v[(split_idx+2)%3]; // opposite vertex
      int vm = getMidpoint(va, vb);

      // Child 1: vc, va, vm
      child_template.v[0] = vc; child_template.v[1] = va; child_template.v[2] = vm;
      next_E.push_back(child_template);
      next_child_to_parent.push_back(e);

      // Child 2: vc, vm, vb
      child_template.v[0] = vc; child_template.v[1] = vm; child_template.v[2] = vb;
      next_E.push_back(child_template);
      next_child_to_parent.push_back(e);
    } 
    else if (split_count == 3) {
      // Red Refinement: Isotropic Quartering (all 3 edges symmetrically split)
      int m0 = getMidpoint(v[0], v[1]);
      int m1 = getMidpoint(v[1], v[2]);
      int m2 = getMidpoint(v[2], v[0]);

      // Corner 0: v[0], m0, m2
      child_template.v[0] = v[0]; child_template.v[1] = m0; child_template.v[2] = m2;
      next_E.push_back(child_template);
      next_child_to_parent.push_back(e);

      // Corner 1: v[1], m1, m0
      child_template.v[0] = v[1]; child_template.v[1] = m1; child_template.v[2] = m0;
      next_E.push_back(child_template);
      next_child_to_parent.push_back(e);

      // Corner 2: v[2], m2, m1
      child_template.v[0] = v[2]; child_template.v[1] = m2; child_template.v[2] = m1;
      next_E.push_back(child_template);
      next_child_to_parent.push_back(e);

      // Center Core Triangle: m0, m1, m2
      child_template.v[0] = m0; child_template.v[1] = m1; child_template.v[2] = m2;
      next_E.push_back(child_template);
      next_child_to_parent.push_back(e);
    }
  }

  // Load newly constructed structurally conforming arrays
  mesh.E = next_E;
  rmap.child_to_parent = next_child_to_parent;

  // ── Phase 2.5: Synchronize midpoints into Periodic Groups ──
  for (auto const& [edge, is_split] : edge_split) {
      if (is_split) {
          int va = edge.first, vb = edge.second;
          if (ptwin.count(va) && ptwin.count(vb) && ptwin[va].g == ptwin[vb].g) {
              auto twin_edge = std::make_pair(std::min(ptwin[va].twin, ptwin[vb].twin),
                                              std::max(ptwin[va].twin, ptwin[vb].twin));
              if (edge_midpoint.count(edge) && edge_midpoint.count(twin_edge)) {
                  // Process exactly once per reciprocal set
                  if (edge < twin_edge) {
                      int v_m1 = edge_midpoint[edge];
                      int v_m2 = edge_midpoint[twin_edge];
                      if (mesh.V[v_m1].y > mesh.V[v_m2].y) std::swap(v_m1, v_m2);
                      mesh.periodicGroups[ptwin[va].g].pairs.push_back({v_m1, v_m2});
                  }
              }
          }
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

  // Restore explicit periodic pairings that were dumped into BE by the geometric hash!
  mesh.appendPeriodicToIE();

  // ── Phase 4: Angle-Guided Smart Laplacian Smoothing ──
  std::vector<bool> is_bdry(mesh.V.size(), false);
  for (const auto& be : mesh.BE) {
      is_bdry[be.v[0]] = true;
      is_bdry[be.v[1]] = true;
  }
  // Periodic IE nodes were removed from BE by appendPeriodicToIE but must
  // not be smoothed — they sit exactly on the periodic boundary surface.
  // Periodic IEs are identified by v[] != vR[] (regular IEs have v==vR).
  for (const auto& ie : mesh.IE) {
      if (ie.v[0] != ie.vR[0] || ie.v[1] != ie.vR[1]) {
          is_bdry[ie.v[0]]  = true;
          is_bdry[ie.v[1]]  = true;
          is_bdry[ie.vR[0]] = true;
          is_bdry[ie.vR[1]] = true;
      }
  }
  std::vector<std::vector<int>> adj(mesh.V.size());
  for (const auto& ie : mesh.IE) {
      adj[ie.v[0]].push_back(ie.v[1]);
      adj[ie.v[1]].push_back(ie.v[0]);
  }
  for (const auto& be : mesh.BE) {
      adj[be.v[0]].push_back(be.v[1]);
      adj[be.v[1]].push_back(be.v[0]);
  }
  std::vector<std::vector<int>> node_elem(mesh.V.size());
  for (int e = 0; e < new_Ne; ++e) {
      node_elem[mesh.E[e].v[0]].push_back(e);
      node_elem[mesh.E[e].v[1]].push_back(e);
      node_elem[mesh.E[e].v[2]].push_back(e);
  }

  auto elemQuality = [&](int e, const std::vector<Vec2>& V) -> double {
      int v0 = mesh.E[e].v[0], v1 = mesh.E[e].v[1], v2 = mesh.E[e].v[2];
      Vec2 a = V[v0], b = V[v1], c = V[v2];
      double ab_x = b.x - a.x, ab_y = b.y - a.y;
      double bc_x = c.x - b.x, bc_y = c.y - b.y;
      double ca_x = a.x - c.x, ca_y = a.y - c.y;
      double L2 = (ab_x*ab_x + ab_y*ab_y) + (bc_x*bc_x + bc_y*bc_y) + (ca_x*ca_x + ca_y*ca_y);
      if (L2 < 1e-16) return -1.0;
      double area = 0.5 * (ab_x * (-ca_y) - ab_y * (-ca_x));
      return 4.0 * 1.7320508075688772 * area / L2;
  };

  int num_smoothing_passes = 200;
  for (int pass = 0; pass < num_smoothing_passes; ++pass) {
      int num_moved = 0;
      for (size_t i = 0; i < mesh.V.size(); ++i) {
          if (is_bdry[i] || adj[i].empty()) continue;

          double orig_min_qual = 1e300;
          for (int e : node_elem[i]) {
              orig_min_qual = std::min(orig_min_qual, elemQuality(e, mesh.V));
          }

          Vec2 avg{0.0, 0.0};
          for (int nbr : adj[i]) avg = avg + mesh.V[nbr];
          avg = avg * (1.0 / adj[i].size());

          // Under-relax the pull slightly to maintain structural continuity
          Vec2 candidate = mesh.V[i] * 0.2 + avg * 0.8;

          std::vector<Vec2> V_test = mesh.V;
          V_test[i] = candidate;

          double new_min_qual = 1e300;
          for (int e : node_elem[i]) {
              new_min_qual = std::min(new_min_qual, elemQuality(e, V_test));
          }

          // Relax the absolute optimization lock to allow escaping local geometric wells!
          if (new_min_qual > orig_min_qual - 1e-3) {
              mesh.V[i] = candidate;
              num_moved++;
          }
      }
      if (num_moved == 0) break;
  }

  // ── Phase 5: keep refined mesh straight-sided (q=1) for AMR debugging ──
  // This avoids introducing a q=1 -> q=3 geometry jump after cycle 0, which
  // has been destabilizing the refined primal solves. Leave every child
  // element as q=1 for now and recompute geometry on the straight mesh.
  //
  // Original q=3 wall-elevation logic retained below for later restoration:
  //
  // std::map<std::pair<int,int>, std::pair<int,int>> e3nodes;
  // auto get_or_create_e3_nodes = [&](int a, int b, bool is_wall) -> std::pair<int,int> {
  //     int s_min = std::min(a,b), s_max = std::max(a,b);
  //     auto skey = std::make_pair(s_min, s_max);
  //     if (!e3nodes.count(skey)) {
  //         Vec2 p1 = (mesh.V[s_min]*2.0 + mesh.V[s_max]*1.0)/3.0;
  //         Vec2 p2 = (mesh.V[s_min]*1.0 + mesh.V[s_max]*2.0)/3.0;
  //         if (is_wall) {
  //             p1 = projectToBladeSpline(p1);
  //             p2 = projectToBladeSpline(p2);
  //         }
  //         int i1 = mesh.V.size(); mesh.V.push_back(p1);
  //         int i2 = mesh.V.size(); mesh.V.push_back(p2);
  //         e3nodes[skey] = {i1, i2};
  //     }
  //     auto pair = e3nodes[skey];
  //     if (a == s_min) return pair;
  //     return {pair.second, pair.first};
  // };
  mesh.has_curved_elements = false;
  mesh.q_order_global = 1;
  for (int e = 0; e < new_Ne; ++e) {
      mesh.E[e].q_order = 1;
      mesh.E[e].ho_nodes.clear();
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

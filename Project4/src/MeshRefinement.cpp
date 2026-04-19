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
#include <functional>

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
    if (fc < fd) { b = d; d = c; fd = fc; c = b - invphi * (b - a); fc = f(c); }
    else { a = c; c = d; fc = fd; d = a + invphi * (b - a); fd = f(d); }
  }
  return (fc < fd) ? c : d;
}

struct BladeSplineReference {
  bool attempted = false;
  bool loaded = false;
  double s_total_upper = 0.0;
  double s_total_lower = 0.0;
  double snap_distance_tol = 50.0;
  tk::spline upper_x;
  tk::spline upper_y;
  tk::spline lower_x;
  tk::spline lower_y;
};

struct MeshSmoothingConfig {
  int iterations = 120;
  double wall_geom_tol = 0.15;
};

MeshSmoothingConfig &meshSmoothingConfig() {
  static MeshSmoothingConfig cfg;
  return cfg;
}

BladeSplineReference &bladeSplineReference() {
  static BladeSplineReference ref;
  if (ref.attempted) return ref;
  ref.attempted = true;
  std::vector<std::pair<std::string, std::string>> candidates = {{"grids/bladeupper.txt", "grids/bladelower.txt"}, {"../Project1/data/bladeupper.txt", "../Project1/data/bladelower.txt"}};
  double domain_height = 18.0;
  if (const char *env = std::getenv("BLADE_SURFACE_YSHIFT")) domain_height = std::atof(env);
  if (const char *env = std::getenv("BLADE_SNAP_MAX_DIST")) ref.snap_distance_tol = std::atof(env);
  for (const auto &p : candidates) {
    std::ifstream fup(p.first), flo(p.second);
    if (!fup.good() || !flo.good()) continue;
    std::vector<double> up_x, up_y, lo_x, lo_y;
    double x, y;
    while (fup >> x >> y) { up_x.push_back(x); up_y.push_back(y); }
    while (flo >> x >> y) { lo_x.push_back(x); lo_y.push_back(y); }
    if (up_x.size() < 2 || lo_x.size() < 2) continue;
    auto it = std::min_element(up_x.begin(), up_x.end());
    double sx = *it, sy = up_y[std::distance(up_x.begin(), it)];
    for (size_t i=0; i<up_x.size(); i++) { up_x[i]-=sx; up_y[i]-=sy; }
    for (size_t i=0; i<lo_x.size(); i++) { lo_x[i]-=sx; lo_y[i]-=sy; lo_y[i]+=domain_height; }
    auto pal = [](const std::vector<double>& x, const std::vector<double>& y, std::vector<double>& s) {
      s.resize(x.size()); s[0]=0; for(size_t i=1; i<x.size(); i++) { double dx=x[i]-x[i-1], dy=y[i]-y[i-1]; s[i]=s[i-1]+sqrt(dx*dx+dy*dy); }
      return s.back();
    };
    std::vector<double> us, ls;
    ref.s_total_upper = pal(up_x, up_y, us);
    ref.s_total_lower = pal(lo_x, lo_y, ls);
    ref.upper_x.set_points(us, up_x); ref.upper_y.set_points(us, up_y);
    ref.lower_x.set_points(ls, lo_x); ref.lower_y.set_points(ls, lo_y);
    ref.loaded = true;
    return ref;
  }
  return ref;
}

enum class BladeBranch {
  upper,
  lower,
  upper_shifted,
  lower_shifted,
  none,
};

struct BladeProjection {
  bool valid = false;
  double d2 = std::numeric_limits<double>::infinity();
  double s = 0.0;
  double s_total = 0.0;
  double offset_y = 0.0;
  Vec2 pt{0.0, 0.0};
  BladeBranch branch = BladeBranch::none;
};

Vec2 evaluateBladeBranch(BladeBranch branch, double s) {
  BladeSplineReference &ref = bladeSplineReference();
  switch (branch) {
    case BladeBranch::upper: return {ref.upper_x(s), ref.upper_y(s)};
    case BladeBranch::lower: return {ref.lower_x(s), ref.lower_y(s)};
    case BladeBranch::upper_shifted: return {ref.upper_x(s), ref.upper_y(s) + 18.0};
    case BladeBranch::lower_shifted: return {ref.lower_x(s), ref.lower_y(s) - 18.0};
    case BladeBranch::none: return {0.0, 0.0};
  }
  return {0.0, 0.0};
}

double bladeBranchArcLength(BladeBranch branch) {
  BladeSplineReference &ref = bladeSplineReference();
  switch (branch) {
    case BladeBranch::upper:
    case BladeBranch::upper_shifted:
      return ref.s_total_upper;
    case BladeBranch::lower:
    case BladeBranch::lower_shifted:
      return ref.s_total_lower;
    case BladeBranch::none:
      return 0.0;
  }
  return 0.0;
}

double bladeBranchOffsetY(BladeBranch branch) {
  switch (branch) {
    case BladeBranch::upper: return 0.0;
    case BladeBranch::lower: return 0.0;
    case BladeBranch::upper_shifted: return 18.0;
    case BladeBranch::lower_shifted: return -18.0;
    case BladeBranch::none: return 0.0;
  }
  return 0.0;
}

const tk::spline &bladeBranchSplineX(BladeBranch branch) {
  BladeSplineReference &ref = bladeSplineReference();
  switch (branch) {
    case BladeBranch::upper:
    case BladeBranch::upper_shifted:
      return ref.upper_x;
    case BladeBranch::lower:
    case BladeBranch::lower_shifted:
      return ref.lower_x;
    case BladeBranch::none:
      return ref.upper_x;
  }
  return ref.upper_x;
}

const tk::spline &bladeBranchSplineY(BladeBranch branch) {
  BladeSplineReference &ref = bladeSplineReference();
  switch (branch) {
    case BladeBranch::upper:
    case BladeBranch::upper_shifted:
      return ref.upper_y;
    case BladeBranch::lower:
    case BladeBranch::lower_shifted:
      return ref.lower_y;
    case BladeBranch::none:
      return ref.upper_y;
  }
  return ref.upper_y;
}

BladeProjection projectToBladeBranchDetailed(const Vec2 &p, BladeBranch branch) {
  BladeSplineReference &ref = bladeSplineReference();
  if (!ref.loaded || branch == BladeBranch::none) return {};

  const tk::spline &sx = bladeBranchSplineX(branch);
  const tk::spline &sy = bladeBranchSplineY(branch);
  const double s_total = bladeBranchArcLength(branch);
  const double offset_y = bladeBranchOffsetY(branch);
  auto f = [&](double s) {
    const double dx = sx(s) - p.x;
    const double dy = (sy(s) + offset_y) - p.y;
    return dx * dx + dy * dy;
  };

  int ns = 20;
  double best_s = 0.0;
  double best_obj = std::numeric_limits<double>::infinity();
  for (int i = 0; i <= ns; ++i) {
    const double s = s_total * i / ns;
    const double obj = f(s);
    if (obj < best_obj) {
      best_obj = obj;
      best_s = s;
    }
  }
  const double ds = (ns > 0) ? (s_total / ns) : s_total;
  best_s = goldenSectionArgmin(f, std::max(0.0, best_s - ds), std::min(s_total, best_s + ds));
  BladeProjection proj;
  proj.valid = std::sqrt(f(best_s)) <= ref.snap_distance_tol;
  proj.d2 = f(best_s);
  proj.s = best_s;
  proj.s_total = s_total;
  proj.offset_y = offset_y;
  proj.pt = {sx(best_s), sy(best_s) + offset_y};
  proj.branch = branch;
  return proj.valid ? proj : BladeProjection{};
}

BladeProjection projectToBladeSplineDetailed(const Vec2 &p) {
  BladeSplineReference &ref = bladeSplineReference();
  if (!ref.loaded) return {};
  std::vector<BladeProjection> cands;
  for (BladeBranch branch : {BladeBranch::upper, BladeBranch::lower,
                             BladeBranch::upper_shifted, BladeBranch::lower_shifted}) {
    BladeProjection proj = projectToBladeBranchDetailed(p, branch);
    if (proj.valid) cands.push_back(proj);
  }
  auto b = std::min_element(cands.begin(), cands.end(), [](const BladeProjection &a, const BladeProjection &b){return a.d2 < b.d2;});
  if (b == cands.end() || std::sqrt(b->d2) > ref.snap_distance_tol) return {};
  return *b;
}

Vec2 projectToBladeSpline(const Vec2 &p) {
  BladeProjection proj = projectToBladeSplineDetailed(p);
  return proj.valid ? proj.pt : p;
}

double triangleQuality(const Element &elem, const std::vector<Vec2> &verts) {
  const Vec2 &a = verts[elem.v[0]];
  const Vec2 &b = verts[elem.v[1]];
  const Vec2 &c = verts[elem.v[2]];
  const Vec2 ab = b - a;
  const Vec2 bc = c - b;
  const Vec2 ca = a - c;
  const double sum_len2 = ab.normSq() + bc.normSq() + ca.normSq();
  if (sum_len2 <= 1e-16) return -1.0;
  const double twice_area = ab.x * (-ca.y) - ab.y * (-ca.x);
  return 2.0 * std::sqrt(3.0) * twice_area / sum_len2;
}

double triangleSignedArea(const Element &elem, const std::vector<Vec2> &verts) {
  const Vec2 &a = verts[elem.v[0]];
  const Vec2 &b = verts[elem.v[1]];
  const Vec2 &c = verts[elem.v[2]];
  return 0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
}

double angleDegBetween(const Vec2 &u, const Vec2 &v) {
  const double nu = u.norm();
  const double nv = v.norm();
  if (nu <= 0.0 || nv <= 0.0) return 0.0;
  double c = u.dot(v) / (nu * nv);
  c = std::max(-1.0, std::min(1.0, c));
  return std::acos(c) * 180.0 / M_PI;
}

double triangleMinAngleDeg(const Element &elem, const std::vector<Vec2> &verts) {
  const Vec2 &a = verts[elem.v[0]];
  const Vec2 &b = verts[elem.v[1]];
  const Vec2 &c = verts[elem.v[2]];
  const double ang_a = angleDegBetween(b - a, c - a);
  const double ang_b = angleDegBetween(a - b, c - b);
  const double ang_c = angleDegBetween(a - c, b - c);
  return std::min(ang_a, std::min(ang_b, ang_c));
}

Vec2 branchTangent(BladeBranch branch, double s) {
  const double s_total = bladeBranchArcLength(branch);
  if (s_total <= 0.0) return {1.0, 0.0};
  const double ds = std::max(1e-6, 1e-3 * s_total);
  const double s0 = std::max(0.0, s - ds);
  const double s1 = std::min(s_total, s + ds);
  Vec2 x0 = evaluateBladeBranch(branch, s0);
  Vec2 x1 = evaluateBladeBranch(branch, s1);
  Vec2 t = x1 - x0;
  const double tn = t.norm();
  if (tn <= 1e-14) return {1.0, 0.0};
  return t / tn;
}

bool branchFluidIsAbove(BladeBranch branch) {
  switch (branch) {
    case BladeBranch::upper:
    case BladeBranch::upper_shifted:
      return true;
    case BladeBranch::lower:
    case BladeBranch::lower_shifted:
      return false;
    case BladeBranch::none:
      return true;
  }
  return true;
}

Vec2 branchFluidNormal(BladeBranch branch, double s) {
  Vec2 t_hat = branchTangent(branch, s);
  Vec2 n_hat{-t_hat.y, t_hat.x};
  const double desired_sign_y = branchFluidIsAbove(branch) ? 1.0 : -1.0;
  if (n_hat.y * desired_sign_y < 0.0) n_hat = -1.0 * n_hat;
  const double nn = n_hat.norm();
  if (nn <= 1e-14) return {0.0, desired_sign_y};
  return n_hat / nn;
}

bool pointIsOnFluidSide(const Vec2 &p, const BladeProjection &proj, double tol = 1e-10) {
  if (!proj.valid) return true;
  if (branchFluidIsAbove(proj.branch)) return p.y >= proj.pt.y - tol;
  return p.y <= proj.pt.y + tol;
}

Vec2 movePointToFluidSide(const Vec2 &p, const BladeProjection &proj, double push_dist) {
  if (!proj.valid) return p;
  Vec2 n_hat = branchFluidNormal(proj.branch, proj.s);
  return proj.pt + std::max(push_dist, 1e-6) * n_hat;
}

double orient2d(const Vec2 &a, const Vec2 &b, const Vec2 &c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool onSegment(const Vec2 &a, const Vec2 &b, const Vec2 &p, double tol = 1e-12) {
  return std::min(a.x, b.x) - tol <= p.x && p.x <= std::max(a.x, b.x) + tol &&
         std::min(a.y, b.y) - tol <= p.y && p.y <= std::max(a.y, b.y) + tol;
}

bool segmentsIntersect(const Vec2 &a, const Vec2 &b,
                       const Vec2 &c, const Vec2 &d,
                       double tol = 1e-12) {
  const double o1 = orient2d(a, b, c);
  const double o2 = orient2d(a, b, d);
  const double o3 = orient2d(c, d, a);
  const double o4 = orient2d(c, d, b);

  const bool proper = ((o1 > tol && o2 < -tol) || (o1 < -tol && o2 > tol)) &&
                      ((o3 > tol && o4 < -tol) || (o3 < -tol && o4 > tol));
  if (proper) return true;

  if (std::abs(o1) <= tol && onSegment(a, b, c, tol)) return true;
  if (std::abs(o2) <= tol && onSegment(a, b, d, tol)) return true;
  if (std::abs(o3) <= tol && onSegment(c, d, a, tol)) return true;
  if (std::abs(o4) <= tol && onSegment(c, d, b, tol)) return true;
  return false;
}

bool localEdgesSelfIntersect(const Mesh &mesh,
                             int vid,
                             const std::vector<std::vector<int>> &node_to_elem,
                             const std::vector<Vec2> &verts) {
  std::set<int> local_nodes;
  std::set<int> local_elems;
  local_nodes.insert(vid);
  for (int e : node_to_elem[vid]) {
    local_elems.insert(e);
    local_nodes.insert(mesh.E[e].v[0]);
    local_nodes.insert(mesh.E[e].v[1]);
    local_nodes.insert(mesh.E[e].v[2]);
  }
  std::vector<int> ring_nodes(local_nodes.begin(), local_nodes.end());
  for (int n : ring_nodes) {
    for (int e : node_to_elem[n]) local_elems.insert(e);
  }

  std::set<std::pair<int, int>> edge_set;
  for (int e : local_elems) {
    const auto &elem = mesh.E[e];
    edge_set.insert({std::min(elem.v[0], elem.v[1]), std::max(elem.v[0], elem.v[1])});
    edge_set.insert({std::min(elem.v[1], elem.v[2]), std::max(elem.v[1], elem.v[2])});
    edge_set.insert({std::min(elem.v[2], elem.v[0]), std::max(elem.v[2], elem.v[0])});
  }

  std::vector<std::pair<int, int>> edges(edge_set.begin(), edge_set.end());
  for (size_t i = 0; i < edges.size(); ++i) {
    const int a0 = edges[i].first;
    const int a1 = edges[i].second;
    for (size_t j = i + 1; j < edges.size(); ++j) {
      const int b0 = edges[j].first;
      const int b1 = edges[j].second;
      if (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1) continue;
      if (segmentsIntersect(verts[a0], verts[a1], verts[b0], verts[b1])) return true;
    }
  }
  return false;
}

bool localMovePreservesMesh(const Mesh &mesh,
                            int vid,
                            const std::vector<std::vector<int>> &node_to_elem,
                            const std::vector<Vec2> &verts_before,
                            std::vector<Vec2> &verts_trial,
                            const Vec2 &candidate) {
  double orig_min_area = std::numeric_limits<double>::infinity();
  double orig_min_angle = std::numeric_limits<double>::infinity();
  for (int e : node_to_elem[vid]) {
    orig_min_area = std::min(orig_min_area, triangleSignedArea(mesh.E[e], verts_before));
    orig_min_angle = std::min(orig_min_angle, triangleMinAngleDeg(mesh.E[e], verts_before));
  }

  verts_trial[vid] = candidate;
  double new_min_area = std::numeric_limits<double>::infinity();
  double new_min_angle = std::numeric_limits<double>::infinity();
  for (int e : node_to_elem[vid]) {
    const double area = triangleSignedArea(mesh.E[e], verts_trial);
    if (!(area > 1e-12)) return false;
    new_min_area = std::min(new_min_area, area);
    new_min_angle = std::min(new_min_angle, triangleMinAngleDeg(mesh.E[e], verts_trial));
  }

  if (new_min_area < 0.2 * orig_min_area) return false;
  if (new_min_angle < 2.0) return false;
  if (new_min_angle < std::min(orig_min_angle - 0.25, 0.75 * orig_min_angle)) return false;
  if (localEdgesSelfIntersect(mesh, vid, node_to_elem, verts_trial)) return false;
  return true;
}

std::vector<bool> snapWallVerticesToBladeSpline(Mesh &mesh) {
  std::set<int> wall_vertices;
  for (const auto &be : mesh.BE) {
    if (be.bIndex < 0 || be.bIndex >= static_cast<int>(mesh.Bname.size())) continue;
    if (lowerCopy(mesh.Bname[be.bIndex]) != "wall") continue;
    wall_vertices.insert(be.v[0]);
    wall_vertices.insert(be.v[1]);
  }

  std::vector<std::vector<int>> node_to_elem(mesh.V.size());
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    node_to_elem[mesh.E[e].v[0]].push_back(e);
    node_to_elem[mesh.E[e].v[1]].push_back(e);
    node_to_elem[mesh.E[e].v[2]].push_back(e);
  }

  int rejected = 0;
  int accepted = 0;
  std::vector<bool> rejected_elem_marks(mesh.E.size(), false);
  std::vector<Vec2> trial = mesh.V;
  for (int vid : wall_vertices) {
    const Vec2 original = mesh.V[vid];
    const Vec2 snapped = projectToBladeSpline(original);
    if ((snapped - original).normSq() <= 1e-24) continue;

    bool moved = false;
    for (int backtrack = 0; backtrack < 8 && !moved; ++backtrack) {
      const double step = std::ldexp(1.0, -backtrack);
      const Vec2 candidate = original + step * (snapped - original);
      if (localMovePreservesMesh(mesh, vid, node_to_elem, mesh.V, trial, candidate)) {
        mesh.V[vid] = candidate;
        trial[vid] = candidate;
        accepted++;
        moved = true;
      }
    }
    if (!moved) {
      trial[vid] = mesh.V[vid];
      rejected++;
      for (int e : node_to_elem[vid]) {
        rejected_elem_marks[e] = true;
        const auto &elem = mesh.E[e];
        for (int n : {elem.v[0], elem.v[1], elem.v[2]}) {
          for (int ee : node_to_elem[n]) rejected_elem_marks[ee] = true;
        }
      }
    }
  }

  if (rejected > 0) {
    std::cerr << "    wall snapping accepted " << accepted
              << " moves, rejected " << rejected
              << " due to local consistency checks" << std::endl;
  }
  return rejected_elem_marks;
}

void snapWallVerticesToBladeSplineConsistent(Mesh &mesh) {
  std::vector<std::set<int>> wall_adj(mesh.V.size());
  for (const auto &be : mesh.BE) {
    if (be.bIndex < 0 || be.bIndex >= static_cast<int>(mesh.Bname.size())) continue;
    if (lowerCopy(mesh.Bname[be.bIndex]) != "wall") continue;
    wall_adj[be.v[0]].insert(be.v[1]);
    wall_adj[be.v[1]].insert(be.v[0]);
  }

  std::vector<std::vector<int>> node_to_elem(mesh.V.size());
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    node_to_elem[mesh.E[e].v[0]].push_back(e);
    node_to_elem[mesh.E[e].v[1]].push_back(e);
    node_to_elem[mesh.E[e].v[2]].push_back(e);
  }

  int accepted = 0;
  int rejected = 0;
  std::vector<bool> visited(mesh.V.size(), false);
  std::vector<Vec2> trial = mesh.V;
  for (int start = 0; start < static_cast<int>(mesh.V.size()); ++start) {
    if (visited[start] || wall_adj[start].empty()) continue;

    std::vector<int> component;
    std::vector<int> stack{start};
    visited[start] = true;
    while (!stack.empty()) {
      int v = stack.back();
      stack.pop_back();
      component.push_back(v);
      for (int nbr : wall_adj[v]) {
        if (!visited[nbr]) {
          visited[nbr] = true;
          stack.push_back(nbr);
        }
      }
    }

    BladeBranch best_branch = BladeBranch::none;
    double best_cost = std::numeric_limits<double>::infinity();
    std::map<int, BladeProjection> best_proj_by_vid;
    for (BladeBranch branch : {BladeBranch::upper, BladeBranch::lower,
                               BladeBranch::upper_shifted, BladeBranch::lower_shifted}) {
      double total_cost = 0.0;
      std::map<int, BladeProjection> proj_by_vid;
      bool ok = true;
      for (int vid : component) {
        BladeProjection proj = projectToBladeBranchDetailed(mesh.V[vid], branch);
        if (!proj.valid) {
          ok = false;
          break;
        }
        total_cost += proj.d2;
        proj_by_vid[vid] = proj;
      }
      if (ok && total_cost < best_cost) {
        best_cost = total_cost;
        best_branch = branch;
        best_proj_by_vid = std::move(proj_by_vid);
      }
    }
    if (best_branch == BladeBranch::none) continue;

    for (int vid : component) {
      const Vec2 original = mesh.V[vid];
      const Vec2 snapped = best_proj_by_vid[vid].pt;
      if ((snapped - original).normSq() <= 1e-24) continue;

      bool moved = false;
      for (int backtrack = 0; backtrack < 8 && !moved; ++backtrack) {
        const double step = std::ldexp(1.0, -backtrack);
        const Vec2 candidate = original + step * (snapped - original);
        if (localMovePreservesMesh(mesh, vid, node_to_elem, mesh.V, trial, candidate)) {
          mesh.V[vid] = candidate;
          trial[vid] = candidate;
          accepted++;
          moved = true;
        }
      }
      if (!moved) {
        trial[vid] = mesh.V[vid];
        rejected++;
      }
    }
  }

  if (accepted > 0 || rejected > 0) {
    std::cerr << "    consistent wall snapping accepted " << accepted
              << " moves, rejected " << rejected
              << " due to local consistency checks" << std::endl;
  }
}

void redistributeWallVerticesAlongBlade(Mesh &mesh, int iterations = 20, double omega = 0.35) {
  std::vector<std::set<int>> wall_adj(mesh.V.size());
  for (const auto &be : mesh.BE) {
    if (be.bIndex < 0 || be.bIndex >= static_cast<int>(mesh.Bname.size())) continue;
    if (lowerCopy(mesh.Bname[be.bIndex]) != "wall") continue;
    wall_adj[be.v[0]].insert(be.v[1]);
    wall_adj[be.v[1]].insert(be.v[0]);
  }

  std::vector<std::vector<int>> node_to_elem(mesh.V.size());
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    node_to_elem[mesh.E[e].v[0]].push_back(e);
    node_to_elem[mesh.E[e].v[1]].push_back(e);
    node_to_elem[mesh.E[e].v[2]].push_back(e);
  }

  std::vector<bool> visited(mesh.V.size(), false);
  for (int start = 0; start < static_cast<int>(mesh.V.size()); ++start) {
    if (visited[start] || wall_adj[start].empty()) continue;

    std::vector<int> component;
    std::vector<int> stack{start};
    visited[start] = true;
    while (!stack.empty()) {
      int v = stack.back();
      stack.pop_back();
      component.push_back(v);
      for (int nbr : wall_adj[v]) {
        if (!visited[nbr]) {
          visited[nbr] = true;
          stack.push_back(nbr);
        }
      }
    }

    int path_start = component.front();
    for (int v : component) {
      if (wall_adj[v].size() == 1) {
        path_start = v;
        break;
      }
    }

    std::vector<int> path;
    std::set<int> seen;
    int prev = -1;
    int cur = path_start;
    while (cur >= 0 && !seen.count(cur)) {
      path.push_back(cur);
      seen.insert(cur);
      int next = -1;
      for (int nbr : wall_adj[cur]) {
        if (nbr != prev) {
          next = nbr;
          break;
        }
      }
      prev = cur;
      cur = next;
    }
    if (path.size() < 3) continue;

    std::vector<BladeProjection> proj(path.size());
    bool compatible = true;
    for (size_t i = 0; i < path.size(); ++i) {
      proj[i] = projectToBladeSplineDetailed(mesh.V[path[i]]);
      if (!proj[i].valid) { compatible = false; break; }
      if (i > 0 && proj[i].branch != proj[0].branch) { compatible = false; break; }
    }
    if (!compatible) continue;

    const double sign = (proj.back().s >= proj.front().s) ? 1.0 : -1.0;
    std::vector<double> u(path.size());
    for (size_t i = 0; i < path.size(); ++i) u[i] = sign * proj[i].s;
    const double min_sep = 0.05 * std::abs(u.back() - u.front()) / std::max<int>(1, path.size() - 1);

    std::vector<Vec2> trial = mesh.V;
    for (int iter = 0; iter < iterations; ++iter) {
      int moved = 0;
      for (size_t i = 1; i + 1 < path.size(); ++i) {
        const double target_u = (1.0 - omega) * u[i] + 0.5 * omega * (u[i - 1] + u[i + 1]);
        const double lo = u[i - 1] + min_sep;
        const double hi = u[i + 1] - min_sep;
        if (lo >= hi) continue;
        const double new_u = std::max(lo, std::min(hi, target_u));
        const double new_s = sign * new_u;
        const Vec2 candidate = evaluateBladeBranch(proj[i].branch, new_s);
        if (localMovePreservesMesh(mesh, path[i], node_to_elem, mesh.V, trial, candidate)) {
          mesh.V[path[i]] = candidate;
          trial[path[i]] = candidate;
          u[i] = new_u;
          moved++;
        }
      }
      if (moved == 0) break;
    }
  }
}

void relocateSkinnyTriangleApexes(Mesh &mesh,
                                  double quality_trigger = 0.10,
                                  double omega = 0.35,
                                  int passes = 2) {
  if (mesh.E.empty()) return;

  std::vector<bool> frozen(mesh.V.size(), false);
  for (const auto &be : mesh.BE) {
    frozen[be.v[0]] = true;
    frozen[be.v[1]] = true;
  }
  for (const auto &pg : mesh.periodicGroups) {
    for (const auto &pair : pg.pairs) {
      frozen[pair.first] = true;
      frozen[pair.second] = true;
    }
  }

  std::vector<std::vector<int>> node_to_elem(mesh.V.size());
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    node_to_elem[mesh.E[e].v[0]].push_back(e);
    node_to_elem[mesh.E[e].v[1]].push_back(e);
    node_to_elem[mesh.E[e].v[2]].push_back(e);
  }

  std::vector<Vec2> trial = mesh.V;
  for (int pass = 0; pass < passes; ++pass) {
    int moved = 0;
    for (const auto &elem : mesh.E) {
      const double q = triangleQuality(elem, mesh.V);
      if (q >= quality_trigger) continue;

      const int v[3] = {elem.v[0], elem.v[1], elem.v[2]};
      const double l2[3] = {
          (mesh.V[v[1]] - mesh.V[v[0]]).normSq(),
          (mesh.V[v[2]] - mesh.V[v[1]]).normSq(),
          (mesh.V[v[0]] - mesh.V[v[2]]).normSq(),
      };
      int shortest = 0;
      if (l2[1] < l2[shortest]) shortest = 1;
      if (l2[2] < l2[shortest]) shortest = 2;

      const int apex = v[(shortest + 2) % 3];
      const int b0 = v[shortest];
      const int b1 = v[(shortest + 1) % 3];
      if (frozen[apex]) continue;

      const Vec2 base_mid = 0.5 * (mesh.V[b0] + mesh.V[b1]);
      const Vec2 base = mesh.V[b1] - mesh.V[b0];
      const double base_len = base.norm();
      if (base_len <= 1e-12) continue;

      Vec2 n{-base.y / base_len, base.x / base_len};
      const double orient = orient2d(mesh.V[b0], mesh.V[b1], mesh.V[apex]);
      if (orient < 0.0) n = -1.0 * n;

      const Vec2 ideal = base_mid + n * (0.8660254037844386 * base_len);
      const Vec2 candidate = (1.0 - omega) * mesh.V[apex] + omega * ideal;
      if (localMovePreservesMesh(mesh, apex, node_to_elem, mesh.V, trial, candidate)) {
        mesh.V[apex] = candidate;
        trial[apex] = candidate;
        moved++;
      }
    }
    if (moved == 0) break;
  }
}

void smoothInteriorVertices(Mesh &mesh, int iterations,
                            double blend_old = 0.2, double blend_avg = 0.8,
                            double quality_slack = 1e-3) {
  if (mesh.V.empty() || mesh.E.empty() || iterations <= 0) return;

  std::vector<std::set<int>> neighbors(mesh.V.size());
  std::vector<std::vector<int>> node_to_elem(mesh.V.size());
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    const auto &elem = mesh.E[e];
    const int v0 = elem.v[0];
    const int v1 = elem.v[1];
    const int v2 = elem.v[2];
    neighbors[v0].insert(v1); neighbors[v0].insert(v2);
    neighbors[v1].insert(v0); neighbors[v1].insert(v2);
    neighbors[v2].insert(v0); neighbors[v2].insert(v1);
    node_to_elem[v0].push_back(e);
    node_to_elem[v1].push_back(e);
    node_to_elem[v2].push_back(e);
  }

  std::vector<bool> frozen(mesh.V.size(), false);
  for (const auto &be : mesh.BE) {
    frozen[be.v[0]] = true;
    frozen[be.v[1]] = true;
  }
  for (const auto &pg : mesh.periodicGroups) {
    for (const auto &pair : pg.pairs) {
      frozen[pair.first] = true;
      frozen[pair.second] = true;
    }
  }

  for (int iter = 0; iter < iterations; ++iter) {
    int moved = 0;
    int inversion_rejects = 0;
    int angle_rejects = 0;
    for (int i = 0; i < static_cast<int>(mesh.V.size()); ++i) {
      if (frozen[i] || neighbors[i].empty()) continue;

      double orig_min_qual = std::numeric_limits<double>::infinity();
      double orig_min_area = std::numeric_limits<double>::infinity();
      double orig_min_angle = std::numeric_limits<double>::infinity();
      double min_incident_edge = std::numeric_limits<double>::infinity();
      for (int e : node_to_elem[i]) {
        orig_min_qual = std::min(orig_min_qual, triangleQuality(mesh.E[e], mesh.V));
        orig_min_area = std::min(orig_min_area, triangleSignedArea(mesh.E[e], mesh.V));
        orig_min_angle = std::min(orig_min_angle, triangleMinAngleDeg(mesh.E[e], mesh.V));
      }
      for (int j : neighbors[i]) {
        min_incident_edge = std::min(min_incident_edge, (mesh.V[j] - mesh.V[i]).norm());
      }

      Vec2 avg{0.0, 0.0};
      for (int j : neighbors[i]) avg += mesh.V[j];
      avg = avg / static_cast<double>(neighbors[i].size());
      Vec2 target = blend_old * mesh.V[i] + blend_avg * avg;
      Vec2 raw_step = target - mesh.V[i];
      const double raw_norm = raw_step.norm();
      const double max_step = 0.35 * std::max(min_incident_edge, 1e-12);
      if (raw_norm > max_step) {
        target = mesh.V[i] + (max_step / raw_norm) * raw_step;
      }

      bool accepted = false;
      std::vector<Vec2> trial = mesh.V;
      for (int backtrack = 0; backtrack < 8 && !accepted; ++backtrack) {
        const double step = std::ldexp(1.0, -backtrack);
        const Vec2 candidate = mesh.V[i] + step * (target - mesh.V[i]);
        trial[i] = candidate;
        double new_min_qual = std::numeric_limits<double>::infinity();
        double new_min_area = std::numeric_limits<double>::infinity();
        double new_min_angle = std::numeric_limits<double>::infinity();
        bool valid = true;
        for (int e : node_to_elem[i]) {
          const double area = triangleSignedArea(mesh.E[e], trial);
          new_min_area = std::min(new_min_area, area);
          if (!(area > 1e-12)) {
            valid = false;
            inversion_rejects++;
            break;
          }
          new_min_qual = std::min(new_min_qual, triangleQuality(mesh.E[e], trial));
          new_min_angle = std::min(new_min_angle, triangleMinAngleDeg(mesh.E[e], trial));
        }

        if (valid &&
            (new_min_angle < 2.0 ||
             new_min_angle < std::min(orig_min_angle - 0.25, 0.75 * orig_min_angle))) {
          valid = false;
          angle_rejects++;
        }

        if (valid &&
            new_min_area >= 0.25 * orig_min_area &&
            new_min_qual > orig_min_qual - quality_slack) {
          mesh.V[i] = candidate;
          moved++;
          accepted = true;
        }
      }
    }
    if (moved == 0) break;
    if ((inversion_rejects > 0 || angle_rejects > 0) &&
        (iter == 0 || (iter + 1) % 25 == 0)) {
      std::cerr << "    smoothing iter " << (iter + 1)
                << ": rejected " << inversion_rejects
                << " candidate moves due to local inversion/tiny area, "
                << angle_rejects << " due to angle collapse" << std::endl;
    }
  }
}

RefinementMap bisectMarkedElementsImpl(Mesh &mesh, const std::vector<bool> &marked_in,
                                       int retry_depth) {
  RefinementMap rmap;
  int old_Ne = mesh.E.size();
  std::map<std::pair<int,int>, int> edge_midpoint;
  std::map<std::pair<int,int>, int> old_boundary_edge;
  for (const auto &be : mesh.BE) old_boundary_edge[{std::min(be.v[0],be.v[1]), std::max(be.v[0],be.v[1])}] = be.bIndex;
  std::vector<bool> marked = marked_in;

  std::map<std::pair<int,int>, std::vector<int>> edge_to_elem;
  for (int e = 0; e < old_Ne; ++e) {
    const int *v = mesh.E[e].v;
    edge_to_elem[{std::min(v[0], v[1]), std::max(v[0], v[1])}].push_back(e);
    edge_to_elem[{std::min(v[1], v[2]), std::max(v[1], v[2])}].push_back(e);
    edge_to_elem[{std::min(v[2], v[0]), std::max(v[2], v[0])}].push_back(e);
  }

  auto getLE = [&](int e) -> std::pair<int,int> {
    int v[3] = {mesh.E[e].v[0], mesh.E[e].v[1], mesh.E[e].v[2]};
    double l[3] = {(mesh.V[v[1]]-mesh.V[v[0]]).norm(), (mesh.V[v[2]]-mesh.V[v[1]]).norm(), (mesh.V[v[0]]-mesh.V[v[2]]).norm()};
    int i = (l[0] >= l[1] && l[0] >= l[2]) ? 0 : (l[1] >= l[0] && l[1] >= l[2] ? 1 : 2);
    return {std::min(v[i], v[(i+1)%3]), std::max(v[i], v[(i+1)%3])};
  };

  auto edgeLength2 = [&](int a, int b) {
    Vec2 d = mesh.V[b] - mesh.V[a];
    return d.x * d.x + d.y * d.y;
  };

  auto isWallEdge = [&](const std::pair<int, int> &key) {
    auto it = old_boundary_edge.find(key);
    if (it == old_boundary_edge.end()) return false;
    int g = it->second;
    return g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
           lowerCopy(mesh.Bname[g]) == "wall";
  };

  std::set<std::pair<int,int>> e2b;
  int expanded_wall_neighbors = 0;
  for (int e = 0; e < old_Ne; ++e) {
    if (!marked[e]) continue;
    const int *v = mesh.E[e].v;
    std::pair<int,int> ee[3] = {
      {std::min(v[0], v[1]), std::max(v[0], v[1])},
      {std::min(v[1], v[2]), std::max(v[1], v[2])},
      {std::min(v[2], v[0]), std::max(v[2], v[0])},
    };
    bool touches_wall = false;
    for (int i = 0; i < 3; ++i) touches_wall = touches_wall || isWallEdge(ee[i]);
    if (!touches_wall) continue;
    for (int i = 0; i < 3; ++i) {
      if (isWallEdge(ee[i])) continue;
      auto it = edge_to_elem.find(ee[i]);
      if (it == edge_to_elem.end()) continue;
      for (int nbr : it->second) {
        if (nbr == e || marked[nbr]) continue;
        marked[nbr] = true;
        expanded_wall_neighbors++;
      }
    }
  }
  if (expanded_wall_neighbors > 0) {
    std::cerr << "    expanded wall-adjacent refinement by "
              << expanded_wall_neighbors
              << " neighboring elements" << std::endl;
  }
  for(int e=0; e<old_Ne; e++) if(marked[e]) e2b.insert(getLE(e));

  if (meshSmoothingConfig().wall_geom_tol > 0.0) {
    int forced_wall_edges = 0;
    for (const auto &be : mesh.BE) {
      auto key = std::make_pair(std::min(be.v[0], be.v[1]), std::max(be.v[0], be.v[1]));
      if (!isWallEdge(key)) continue;
      Vec2 mid = 0.5 * (mesh.V[be.v[0]] + mesh.V[be.v[1]]);
      Vec2 snapped_mid = projectToBladeSpline(mid);
      double geom_err = (snapped_mid - mid).norm();
      if (geom_err > meshSmoothingConfig().wall_geom_tol) {
        if (!e2b.count(key)) forced_wall_edges++;
        e2b.insert(key);
      }
    }
    if (forced_wall_edges > 0) {
      std::cerr << "    forcing " << forced_wall_edges
                << " wall-edge splits due to blade geometry under-resolution" << std::endl;
    }
  }

  std::map<std::pair<int,int>, std::pair<int,int>> ppartner;
  for (const auto &ie : mesh.IE) {
    auto kL = std::make_pair(std::min(ie.v[0], ie.v[1]), std::max(ie.v[0], ie.v[1]));
    auto kR = std::make_pair(std::min(ie.vR[0], ie.vR[1]), std::max(ie.vR[0], ie.vR[1]));
    if (kL != kR) { ppartner[kL] = kR; ppartner[kR] = kL; }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (auto const& edge : e2b) {
      if (ppartner.count(edge) && !e2b.count(ppartner[edge])) { e2b.insert(ppartner[edge]); changed = true; break; }
    }
  }

  rmap.child_to_parent.resize(old_Ne);
  for(int i=0; i<old_Ne; i++) rmap.child_to_parent[i] = i;

  auto getMid = [&](int va, int vb) {
    auto key = std::make_pair(std::min(va,vb), std::max(va,vb));
    if(edge_midpoint.count(key)) return edge_midpoint[key];
    int vm = (int)mesh.V.size(); Vec2 mid = (mesh.V[va] + mesh.V[vb]) * 0.5;
    if (old_boundary_edge.count(key)) {
      int g = old_boundary_edge[key];
      // Keep new wall midpoints straight initially. They are reprojected later
      // by the wall-only snap pass, which applies local validity checks and
      // backtracking before accepting any move toward the blade spline.
      old_boundary_edge[{std::min(va,vm), std::max(va,vm)}] = g;
      old_boundary_edge[{std::min(vb,vm), std::max(vb,vm)}] = g;
    }
    mesh.V.push_back(mid); edge_midpoint[key] = vm; rmap.new_vertex_edges.push_back({va, vb});
    return vm;
  };

  changed = true;
  while(changed) {
    changed = false;
    int cur_Ne = mesh.E.size();
    for(int e=0; e<cur_Ne; e++) {
      int v[3] = {mesh.E[e].v[0], mesh.E[e].v[1], mesh.E[e].v[2]};
      std::pair<int,int> ee[3] = {{std::min(v[0],v[1]), std::max(v[0],v[1])}, {std::min(v[1],v[2]), std::max(v[1],v[2])}, {std::min(v[2],v[0]), std::max(v[2],v[0])}};
      int t = -1;
      double best_len2 = -1.0;
      for (int i = 0; i < 3; i++) {
        if (!e2b.count(ee[i])) continue;
        double len2 = edgeLength2(v[i], v[(i + 1) % 3]);
        if (len2 > best_len2) {
          best_len2 = len2;
          t = i;
        }
      }
      if (t == -1) {
        for (int i = 0; i < 3; i++) {
          if (!edge_midpoint.count(ee[i])) continue;
          double len2 = edgeLength2(v[i], v[(i + 1) % 3]);
          if (len2 > best_len2) {
            best_len2 = len2;
            t = i;
          }
        }
      }
      if(t == -1) continue;

      int va = v[t], vb = v[(t+1)%3], vc = v[(t+2)%3], vm = getMid(va, vb);
      int wall_edge_idx = -1;
      for (int i = 0; i < 3; ++i) {
        if (isWallEdge(ee[i])) {
          wall_edge_idx = i;
          break;
        }
      }
      if (wall_edge_idx >= 0 && wall_edge_idx != t) {
        int w0 = v[wall_edge_idx];
        int w1 = v[(wall_edge_idx + 1) % 3];
        Vec2 wall_mid = 0.5 * (mesh.V[w0] + mesh.V[w1]);
        BladeProjection wall_proj = projectToBladeSplineDetailed(wall_mid);
        if (wall_proj.valid) {
          Vec2 candidate = mesh.V[vm];
          BladeProjection candidate_proj =
              projectToBladeBranchDetailed(candidate, wall_proj.branch);
          if (candidate_proj.valid && !pointIsOnFluidSide(candidate, candidate_proj)) {
            double eps_push = 0.05 * (mesh.V[w1] - mesh.V[w0]).norm();
            mesh.V[vm] = movePointToFluidSide(candidate, candidate_proj, eps_push);
          }
        }
      }
      int parent = rmap.child_to_parent[e];
      mesh.E[e].v[0] = vc; mesh.E[e].v[1] = va; mesh.E[e].v[2] = vm;
      Element c2; c2.v[0] = vc; c2.v[1] = vm; c2.v[2] = vb; c2.q_order = 1;
      mesh.E.push_back(c2); rmap.child_to_parent.push_back(parent);
      e2b.erase(ee[t]); changed = true;
    }
  }

  for (auto const& [edge, vm] : edge_midpoint) {
    if (ppartner.count(edge)) {
      auto p = ppartner[edge];
      if (edge_midpoint.count(p) && edge < p) {
        int v1 = vm, v2 = edge_midpoint[p];
        if (mesh.V[v1].y > mesh.V[v2].y) std::swap(v1, v2);
        mesh.periodicGroups[0].pairs.push_back({v1, v2});
        mesh.periodicGroups[0].nPairs++;
      }
    }
  }

  std::map<std::pair<int,int>, int> H;
  mesh.IE.clear(); mesh.BE.clear();
  for (int e=0; e<(int)mesh.E.size(); e++)
    for (int i=0; i<3; i++) {
      int n1=mesh.E[e].v[i], n2=mesh.E[e].v[(i+1)%3];
      if (H.count({n2, n1})) {
          Edge ie; ie.v[0]=n2; ie.v[1]=n1; ie.vR[0]=n2; ie.vR[1]=n1; ie.elemL=H[{n2, n1}]-1; ie.elemR=e;
          mesh.IE.push_back(ie); H.erase({n2, n1});
      }
      else H[{n1, n2}] = e + 1;
    }

  for (auto &[key, val] : H) {
    auto skey = std::make_pair(std::min(key.first,key.second), std::max(key.first,key.second));
    if (old_boundary_edge.count(skey)) {
        BoundaryEdge be; be.v[0]=key.first; be.v[1]=key.second; be.elemL=val-1; be.bIndex=old_boundary_edge[skey];
        mesh.BE.push_back(be);
    }
  }

  mesh.appendPeriodicToIE();
  snapWallVerticesToBladeSplineConsistent(mesh);
  mesh.has_curved_elements = false; mesh.q_order_global = 1;
  for (auto &e : mesh.E) { e.q_order = 1; e.ho_nodes.clear(); }
  mesh.computeGeometry();
  return rmap;
}

} // namespace

std::vector<bool> markByIndicator(const std::vector<double> &eps, double f) {
  int Ne = eps.size(); std::vector<bool> m(Ne, false); if (f <= 0.0) return m;
  std::vector<double> s = eps; std::sort(s.rbegin(), s.rend());
  double thr = s[std::clamp((int)(f * Ne), 1, Ne) - 1];
  for (int e = 0; e < Ne; e++) if (eps[e] >= thr) m[e] = true;
  return m;
}

std::vector<bool> markByAspectRatio(const Mesh &mesh, double max_aspect_ratio) {
  const int Ne = static_cast<int>(mesh.E.size());
  std::vector<bool> marked(Ne, false);
  if (max_aspect_ratio <= 0.0) return marked;

  for (int e = 0; e < Ne; ++e) {
    const auto &elem = mesh.E[e];
    const Vec2 &a = mesh.V[elem.v[0]];
    const Vec2 &b = mesh.V[elem.v[1]];
    const Vec2 &c = mesh.V[elem.v[2]];
    const double l0 = (b - a).norm();
    const double l1 = (c - b).norm();
    const double l2 = (a - c).norm();
    const double lmax = std::max({l0, l1, l2});
    const double twice_area = std::abs((b.x - a.x) * (c.y - a.y) -
                                       (b.y - a.y) * (c.x - a.x));
    if (twice_area <= 1e-14) {
      marked[e] = true;
      continue;
    }
    const double shortest_altitude = twice_area / std::max(lmax, 1e-14);
    const double aspect_ratio = lmax / std::max(shortest_altitude, 1e-14);
    if (aspect_ratio > max_aspect_ratio) marked[e] = true;
  }
  return marked;
}

void setMeshSmoothingIterations(int iterations) {
  meshSmoothingConfig().iterations = std::max(0, iterations);
}

void setWallGeometryTolerance(double tolerance) {
  meshSmoothingConfig().wall_geom_tol = std::max(0.0, tolerance);
}

RefinementMap bisectMarkedElements(Mesh &mesh, const std::vector<bool> &marked_in) {
  return bisectMarkedElementsImpl(mesh, marked_in, 2);
}

std::vector<std::vector<Vec4>> interpolateSolution(const std::vector<std::vector<Vec4>> &U_old, const RefinementMap &rmap, int ndof) {
  std::vector<std::vector<Vec4>> U_new(rmap.child_to_parent.size(), std::vector<Vec4>(ndof));
  for (int i=0; i<(int)U_new.size(); i++) U_new[i] = U_old[rmap.child_to_parent[i]];
  return U_new;
}

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

double triangleMinAngleDeg(const Element &elem, const std::vector<Vec2> &verts);
double triangleSignedAreaPts(const Vec2 &a, const Vec2 &b, const Vec2 &c);
double triangleMinAngleDegPts(const Vec2 &a, const Vec2 &b, const Vec2 &c);
bool segmentsIntersect(const Vec2 &a, const Vec2 &b,
                       const Vec2 &c, const Vec2 &d,
                       double tol = 1e-12);
bool pointIsOnFluidSide(const Vec2 &p, const BladeProjection &proj, double tol = 1e-10);

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
  return triangleSignedAreaPts(a, b, c);
}

double triangleSignedAreaPts(const Vec2 &a, const Vec2 &b, const Vec2 &c) {
  return 0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
}

Element makePositiveElement(int a, int b, int c, const std::vector<Vec2> &verts) {
  Element elem;
  elem.q_order = 1;
  elem.ho_nodes.clear();
  const double area2 = (verts[b].x - verts[a].x) * (verts[c].y - verts[a].y) -
                       (verts[b].y - verts[a].y) * (verts[c].x - verts[a].x);
  if (area2 > 0.0) {
    elem.v[0] = a; elem.v[1] = b; elem.v[2] = c;
  } else {
    elem.v[0] = a; elem.v[1] = c; elem.v[2] = b;
  }
  return elem;
}

void rebuildMeshEdgeConnectivity(
    Mesh &mesh,
    const std::map<std::pair<int, int>, int> &boundary_edge_groups) {
  std::map<std::pair<int, int>, int> H;
  mesh.IE.clear();
  mesh.BE.clear();
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    for (int i = 0; i < 3; ++i) {
      const int n1 = mesh.E[e].v[i];
      const int n2 = mesh.E[e].v[(i + 1) % 3];
      if (H.count({n2, n1})) {
        Edge ie;
        ie.v[0] = n2; ie.v[1] = n1;
        ie.vR[0] = n2; ie.vR[1] = n1;
        ie.elemL = H[{n2, n1}] - 1;
        ie.elemR = e;
        mesh.IE.push_back(ie);
        H.erase({n2, n1});
      } else {
        H[{n1, n2}] = e + 1;
      }
    }
  }

  for (const auto &[key, val] : H) {
    auto skey = std::make_pair(std::min(key.first, key.second),
                               std::max(key.first, key.second));
    auto it = boundary_edge_groups.find(skey);
    if (it == boundary_edge_groups.end()) continue;
    BoundaryEdge be;
    be.v[0] = key.first;
    be.v[1] = key.second;
    be.elemL = val - 1;
    be.bIndex = it->second;
    mesh.BE.push_back(be);
  }
}

bool localWallSplitPatchValid(const Mesh &mesh,
                              int va, int vb, int vopp,
                              const Vec2 &mid,
                              const Vec2 &vopp_pos,
                              const std::vector<std::vector<int>> &node_to_elem,
                              double area_tol = 1e-12,
                              double min_angle_tol = 8.0,
                              double min_quality_tol = 0.18,
                              double area_ratio_tol = 0.10) {
  std::vector<Vec2> verts = mesh.V;
  verts[vopp] = vopp_pos;

  std::set<int> local_elems;
  for (int e : node_to_elem[va]) local_elems.insert(e);
  for (int e : node_to_elem[vb]) local_elems.insert(e);
  for (int e : node_to_elem[vopp]) local_elems.insert(e);

  double orig_local_min_area = std::numeric_limits<double>::infinity();
  double orig_local_min_quality = std::numeric_limits<double>::infinity();
  double orig_local_min_angle = std::numeric_limits<double>::infinity();
  for (int e : local_elems) {
    orig_local_min_area = std::min(orig_local_min_area, triangleSignedArea(mesh.E[e], mesh.V));
    orig_local_min_quality = std::min(orig_local_min_quality, triangleQuality(mesh.E[e], mesh.V));
    orig_local_min_angle = std::min(orig_local_min_angle, triangleMinAngleDeg(mesh.E[e], mesh.V));
  }
  if (!std::isfinite(orig_local_min_area)) return false;

  const double a1 = triangleSignedAreaPts(vopp_pos, verts[va], mid);
  const double a2 = triangleSignedAreaPts(vopp_pos, mid, verts[vb]);
  if (!(a1 > area_tol) || !(a2 > area_tol)) return false;
  if (a1 < area_ratio_tol * orig_local_min_area || a2 < area_ratio_tol * orig_local_min_area) {
    return false;
  }

  const double ang1 = triangleMinAngleDegPts(vopp_pos, verts[va], mid);
  const double ang2 = triangleMinAngleDegPts(vopp_pos, mid, verts[vb]);
  if (ang1 < min_angle_tol || ang2 < min_angle_tol) return false;

  const double q1 = std::abs(2.0 * std::sqrt(3.0) *
                             ((verts[va] - vopp_pos).x * (mid - vopp_pos).y -
                              (verts[va] - vopp_pos).y * (mid - vopp_pos).x)) /
                    std::max((verts[va] - vopp_pos).normSq() +
                                 (mid - verts[va]).normSq() +
                                 (vopp_pos - mid).normSq(),
                             1e-16);
  const double q2 = std::abs(2.0 * std::sqrt(3.0) *
                             ((mid - vopp_pos).x * (verts[vb] - vopp_pos).y -
                              (mid - vopp_pos).y * (verts[vb] - vopp_pos).x)) /
                    std::max((mid - vopp_pos).normSq() +
                                 (verts[vb] - mid).normSq() +
                                 (vopp_pos - verts[vb]).normSq(),
                             1e-16);
  if (q1 < min_quality_tol || q2 < min_quality_tol) return false;

  for (int e : node_to_elem[vopp]) {
    const auto &elem = mesh.E[e];
    if ((elem.v[0] == va || elem.v[1] == va || elem.v[2] == va) &&
        (elem.v[0] == vb || elem.v[1] == vb || elem.v[2] == vb)) {
      continue;
    }
    if (!(triangleSignedArea(elem, verts) > area_tol)) return false;
    if (triangleMinAngleDeg(elem, verts) < min_angle_tol) return false;
    const double local_quality_floor =
        (orig_local_min_quality > 0.0)
            ? std::max(min_quality_tol, 0.5 * orig_local_min_quality)
            : min_quality_tol;
    if (triangleQuality(elem, verts) < local_quality_floor) {
      return false;
    }
  }

  std::set<std::pair<int, int>> local_edges;
  for (int e : local_elems) {
    const auto &elem = mesh.E[e];
    local_edges.insert({std::min(elem.v[0], elem.v[1]), std::max(elem.v[0], elem.v[1])});
    local_edges.insert({std::min(elem.v[1], elem.v[2]), std::max(elem.v[1], elem.v[2])});
    local_edges.insert({std::min(elem.v[2], elem.v[0]), std::max(elem.v[2], elem.v[0])});
  }
  const std::vector<std::pair<Vec2, Vec2>> new_edges = {
      {mid, verts[va]},
      {mid, verts[vb]},
      {mid, vopp_pos},
  };
  for (const auto &[e0, e1] : new_edges) {
    const Vec2 edge_mid = 0.5 * (e0 + e1);
    BladeProjection proj = projectToBladeSplineDetailed(edge_mid);
    if (proj.valid && !pointIsOnFluidSide(edge_mid, proj)) return false;
  }
  for (const auto &edge : local_edges) {
    const int x = edge.first;
    const int y = edge.second;
    if ((x == va && y == vb) || (x == vb && y == va)) continue;
    for (const auto &[e0, e1] : new_edges) {
      const bool shares_va = (verts[x] - e0).normSq() <= 1e-24 || (verts[y] - e0).normSq() <= 1e-24;
      const bool shares_vb = (verts[x] - e1).normSq() <= 1e-24 || (verts[y] - e1).normSq() <= 1e-24;
      if (shares_va || shares_vb) continue;
      if (segmentsIntersect(e0, e1, verts[x], verts[y])) return false;
    }
  }

  return true;
}
bool pointInTriangleStrict(const Vec2 &p, const Vec2 &a, const Vec2 &b, const Vec2 &c,
                           double tol = 1e-12) {
  const auto orient = [](const Vec2 &u, const Vec2 &v, const Vec2 &w) {
    return (v.x - u.x) * (w.y - u.y) - (v.y - u.y) * (w.x - u.x);
  };
  const double o1 = orient(a, b, p);
  const double o2 = orient(b, c, p);
  const double o3 = orient(c, a, p);
  const bool has_pos = (o1 > tol) || (o2 > tol) || (o3 > tol);
  const bool has_neg = (o1 < -tol) || (o2 < -tol) || (o3 < -tol);
  return !(has_pos && has_neg);
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
  return triangleMinAngleDegPts(a, b, c);
}

double triangleMinAngleDegPts(const Vec2 &a, const Vec2 &b, const Vec2 &c) {
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

bool pointIsOnFluidSide(const Vec2 &p, const BladeProjection &proj, double tol) {
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
                       double tol) {
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

RefinementMap bisectMarkedElementsImpl(Mesh &mesh, const std::vector<bool> &marked_in,
                                       int retry_depth) {
  RefinementMap rmap;
  int old_Ne = mesh.E.size();
  std::map<std::pair<int,int>, int> edge_midpoint;
  std::map<std::pair<int,int>, int> edge_geom_midpoint_q2;
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

  auto buildCurrentNodeToElem = [&]() {
    std::vector<std::vector<int>> cur_node_to_elem(mesh.V.size());
    for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
      cur_node_to_elem[mesh.E[e].v[0]].push_back(e);
      cur_node_to_elem[mesh.E[e].v[1]].push_back(e);
      cur_node_to_elem[mesh.E[e].v[2]].push_back(e);
    }
    return cur_node_to_elem;
  };

  auto buildCurrentVertexNeighbors = [&]() {
    std::vector<std::set<int>> cur_vertex_neighbors(mesh.V.size());
    for (const auto &elem : mesh.E) {
      cur_vertex_neighbors[elem.v[0]].insert(elem.v[1]);
      cur_vertex_neighbors[elem.v[0]].insert(elem.v[2]);
      cur_vertex_neighbors[elem.v[1]].insert(elem.v[0]);
      cur_vertex_neighbors[elem.v[1]].insert(elem.v[2]);
      cur_vertex_neighbors[elem.v[2]].insert(elem.v[0]);
      cur_vertex_neighbors[elem.v[2]].insert(elem.v[1]);
    }
    return cur_vertex_neighbors;
  };

  std::set<std::pair<int,int>> e2b;
  for(int e=0; e<old_Ne; e++) if(marked[e]) e2b.insert(getLE(e));

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

  auto getMid = [&](int va, int vb, bool &ok) {
    auto key = std::make_pair(std::min(va,vb), std::max(va,vb));
    if(edge_midpoint.count(key)) { ok = true; return edge_midpoint[key]; }
    int vm = (int)mesh.V.size();
    const Vec2 straight_mid = (mesh.V[va] + mesh.V[vb]) * 0.5;
    Vec2 mid = straight_mid;
    if (old_boundary_edge.count(key)) {
      int g = old_boundary_edge[key];
      if (g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
          lowerCopy(mesh.Bname[g]) == "wall") {
        const Vec2 snapped_mid = projectToBladeSpline(straight_mid);
        auto eit = edge_to_elem.find(key);
        const auto cur_node_to_elem = buildCurrentNodeToElem();
        const auto cur_vertex_neighbors = buildCurrentVertexNeighbors();
        bool accepted = true;
        std::map<int, Vec2> repaired_positions;

        if (eit != edge_to_elem.end()) {
          for (int eadj : eit->second) {
            const int *ev = mesh.E[eadj].v;
            int vopp = -1;
            for (int k = 0; k < 3; ++k) {
              if (ev[k] != va && ev[k] != vb) {
                vopp = ev[k];
                break;
              }
            }
            if (vopp < 0) continue;

            Vec2 vopp_pos = repaired_positions.count(vopp) ? repaired_positions[vopp] : mesh.V[vopp];
            if (localWallSplitPatchValid(mesh, va, vb, vopp, snapped_mid, vopp_pos,
                                         cur_node_to_elem)) {
              continue;
            }

            Vec2 avg{0.0, 0.0};
            int count = 0;
            for (int nbr : cur_vertex_neighbors[vopp]) {
              avg = avg + (repaired_positions.count(nbr) ? repaired_positions[nbr] : mesh.V[nbr]);
              count++;
            }
            if (count == 0) {
              accepted = false;
              break;
            }
            avg = avg / static_cast<double>(count);
            if (!localWallSplitPatchValid(mesh, va, vb, vopp, snapped_mid, avg,
                                          cur_node_to_elem)) {
              accepted = false;
              break;
            }
            repaired_positions[vopp] = avg;
          }
        }

        if (!accepted) {
          std::cerr << "    warning: skipped refinement on wall edge (" << va
                    << ", " << vb
                    << ") because snap and opposite-node averaging failed"
                    << std::endl;
          ok = false;
          return -1;
        }
        for (const auto &[vid, pos] : repaired_positions) mesh.V[vid] = pos;
        mid = snapped_mid;
      }
      old_boundary_edge[{std::min(va,vm), std::max(va,vm)}] = g;
      old_boundary_edge[{std::min(vb,vm), std::max(vb,vm)}] = g;
    } else {
      BladeProjection proj = projectToBladeSplineDetailed(straight_mid);
      if (proj.valid && !pointIsOnFluidSide(straight_mid, proj)) {
        std::cerr << "    warning: skipped refinement on interior edge (" << va
                  << ", " << vb
                  << ") because midpoint would lie on wrong side of blade"
                  << std::endl;
        ok = false;
        return -1;
      }
    }
    mesh.V.push_back(mid); edge_midpoint[key] = vm; rmap.new_vertex_edges.push_back({va, vb});
    ok = true;
    return vm;
  };

  auto getGeomMidQ2 = [&](int va, int vb) {
    auto key = std::make_pair(std::min(va, vb), std::max(va, vb));
    if (edge_geom_midpoint_q2.count(key)) return edge_geom_midpoint_q2[key];
    Vec2 mid = 0.5 * (mesh.V[va] + mesh.V[vb]);
    if (old_boundary_edge.count(key)) {
      int g = old_boundary_edge[key];
      if (g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
          lowerCopy(mesh.Bname[g]) == "wall") {
        mid = projectToBladeSpline(mid);
      }
    }
    int vm = static_cast<int>(mesh.V.size());
    mesh.V.push_back(mid);
    edge_geom_midpoint_q2[key] = vm;
    return vm;
  };

  auto assignQ2Child = [&](Element &child, int a, int b, int c) {
    child.v[0] = a;
    child.v[1] = b;
    child.v[2] = c;
    child.q_order = 2;
    child.ho_nodes = {
      a,
      getGeomMidQ2(a, b),
      b,
      getGeomMidQ2(c, a),
      getGeomMidQ2(b, c),
      c
    };
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

      bool mid_ok = false;
      int va = v[t], vb = v[(t+1)%3], vc = v[(t+2)%3], vm = getMid(va, vb, mid_ok);
      if (!mid_ok) {
        e2b.erase(ee[t]);
        continue;
      }
      int parent = rmap.child_to_parent[e];
      const int parent_q_order = mesh.E[e].q_order;
      if (parent_q_order == 2) {
        assignQ2Child(mesh.E[e], vc, va, vm);
      } else {
        mesh.E[e].v[0] = vc; mesh.E[e].v[1] = va; mesh.E[e].v[2] = vm;
        mesh.E[e].q_order = 1;
        mesh.E[e].ho_nodes.clear();
      }
      Element c2;
      if (parent_q_order == 2) {
        assignQ2Child(c2, vc, vm, vb);
      } else {
        c2.v[0] = vc; c2.v[1] = vm; c2.v[2] = vb; c2.q_order = 1;
      }
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

  rebuildMeshEdgeConnectivity(mesh, old_boundary_edge);
  mesh.appendPeriodicToIE();
  mesh.has_curved_elements = false;
  mesh.q_order_global = 1;
  for (const auto &e : mesh.E) {
    mesh.q_order_global = std::max(mesh.q_order_global, e.q_order);
    mesh.has_curved_elements = mesh.has_curved_elements || (e.q_order > 1);
  }
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

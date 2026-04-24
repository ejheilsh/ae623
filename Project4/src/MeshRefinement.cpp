#include "MeshRefinement.hpp"
#include "MeshRefinementQ1.hpp"
#include "MeshRefinementQ2.hpp"
#include "spline.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <deque>
#include <numeric>
#include <queue>
#include <set>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <functional>
#include <array>

int improvePatchByEdgeSwaps(
    Mesh &mesh,
    const std::set<int> &patch_elems,
    const std::map<std::pair<int, int>, int> &boundary_edge_groups,
    int max_swaps,
    const std::function<bool(int)> &is_wall_boundary_vertex,
    bool allow_periodic_vertices = false);

namespace {

enum class BladeBranch;

double oppositeAngleDeg(const Vec2 &a, const Vec2 &b, const Vec2 &p);
bool diametralLensEncroached(const Vec2 &a, const Vec2 &b, const Vec2 &p,
                             double min_angle_deg = 30.0);
bool branchFluidIsAbove(BladeBranch branch);

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

bool curvedRefinementSmoothingEnabled() {
  return meshSmoothingConfig().iterations > 0;
}

bool trueBladeWallSubdivisionEnabled() {
  const char *env = std::getenv("AMR_TRUE_BLADE_WALL_SUBDIVISION");
  return env != nullptr && std::string(env) != "0";
}

bool q1WallFanTrisectionEnabled() {
  const char *env = std::getenv("AMR_Q1_WALL_FAN_TRISECT");
  return env != nullptr && std::string(env) != "0";
}

bool q1WallOnlyRefinementEnabled() {
  const char *env = std::getenv("AMR_Q1_WALL_ONLY");
  return env != nullptr && std::string(env) != "0";
}

bool q1AllowBoundaryChildEdgeOutsideBlade() {
  const char *env = std::getenv("AMR_Q1_STRICT_WALL_CHILD_EDGE_CHECK");
  return !(env != nullptr && std::string(env) != "0");
}

bool q1ExtensionNeighborEdgeSwapsEnabled() {
  const char *env = std::getenv("AMR_Q1_EXTENSION_EDGE_SWAPS");
  return !(env != nullptr && std::string(env) == "0");
}

bool q1UniqueShortestEdgeVetoEnabled() {
  const char *env = std::getenv("AMR_Q1_UNIQUE_SHORTEST_VETO");
  return !(env != nullptr && std::string(env) == "0");
}

bool q1EndpointPolygonOverrideEnabled() {
  const char *env = std::getenv("AMR_Q1_ENDPOINT_POLYGON_OVERRIDE");
  return !(env != nullptr && std::string(env) == "0");
}

bool q1NormalBladeGuardEnabled() {
  const char *env = std::getenv("AMR_Q1_NORMAL_BLADE_GUARD");
  return env != nullptr && std::string(env) != "0";
}

bool q1PolygonBladeGuardEnabled() {
  const char *env = std::getenv("AMR_Q1_POLYGON_BLADE_GUARD");
  return env != nullptr && std::string(env) != "0";
}

const std::set<std::pair<int, int>> &q1BladeGuardDebugEdges() {
  static const std::set<std::pair<int, int>> edges = []() {
    std::set<std::pair<int, int>> parsed;
    const char *env = std::getenv("AMR_Q1_DEBUG_GUARD_EDGES");
    if (env == nullptr) return parsed;
    std::string spec(env);
    for (char &c : spec) {
      if (c == ';' || c == '|' || c == ':') c = ' ';
    }
    std::stringstream ss(spec);
    std::string token;
    while (ss >> token) {
      const size_t comma = token.find(',');
      if (comma == std::string::npos) continue;
      try {
        const int a = std::stoi(token.substr(0, comma));
        const int b = std::stoi(token.substr(comma + 1));
        if (a == b) continue;
        parsed.insert({std::min(a, b), std::max(a, b)});
      } catch (...) {
      }
    }
    return parsed;
  }();
  return edges;
}

int q1BladeGuardDebugMinElems() {
  const char *env = std::getenv("AMR_Q1_DEBUG_GUARD_MIN_ELEMS");
  if (env == nullptr) return std::numeric_limits<int>::max();
  try {
    return std::stoi(env);
  } catch (...) {
    return std::numeric_limits<int>::max();
  }
}

bool q1BladeGuardDebugEnabledForEdge(const std::pair<int, int> &edge,
                                     int current_elems) {
  const auto &debug_edges = q1BladeGuardDebugEdges();
  return !debug_edges.empty() &&
         current_elems >= q1BladeGuardDebugMinElems() &&
         debug_edges.count(edge) > 0;
}

double q1EndpointFanAreaRatioThreshold() {
  const char *env = std::getenv("AMR_Q1_ENDPOINT_FAN_AREA_RATIO");
  return env != nullptr ? std::atof(env) : 1.75;
}

int q1EndpointOverrideHopRadius() {
  const char *env = std::getenv("AMR_Q1_ENDPOINT_OVERRIDE_HOPS");
  if (env == nullptr) return 0;
  try {
    return std::max(0, std::stoi(env));
  } catch (...) {
    return 0;
  }
}

double q1ExtensionPatchMinAngleDeg() {
  const char *env = std::getenv("AMR_Q1_EXTENSION_PATCH_MIN_ANGLE");
  return env != nullptr ? std::atof(env) : 4.0;
}

double q1ExtensionPatchMinQuality() {
  const char *env = std::getenv("AMR_Q1_EXTENSION_PATCH_MIN_QUALITY");
  return env != nullptr ? std::atof(env) : 0.08;
}

double q1PatchRepairMinAngleDeg() {
  const char *env = std::getenv("AMR_Q1_PATCH_REPAIR_MIN_ANGLE");
  return env != nullptr ? std::atof(env) : 6.0;
}

double q1PatchRepairMinQuality() {
  const char *env = std::getenv("AMR_Q1_PATCH_REPAIR_MIN_QUALITY");
  return env != nullptr ? std::atof(env) : 0.14;
}

int q1PatchRepairMaxSwaps() {
  const char *env = std::getenv("AMR_Q1_PATCH_REPAIR_MAX_SWAPS");
  if (env == nullptr) return 16;
  try {
    return std::max(0, std::stoi(env));
  } catch (...) {
    return 16;
  }
}

int q1PatchRepairRings() {
  const char *env = std::getenv("AMR_Q1_PATCH_REPAIR_RINGS");
  if (env == nullptr) return 2;
  try {
    return std::max(1, std::stoi(env));
  } catch (...) {
    return 2;
  }
}

double q1EndpointOverrideMinAngleDeg() {
  const char *env = std::getenv("AMR_Q1_ENDPOINT_OVERRIDE_MIN_ANGLE");
  return env != nullptr ? std::atof(env) : 8.0;
}

double q1EndpointOverrideMinQuality() {
  const char *env = std::getenv("AMR_Q1_ENDPOINT_OVERRIDE_MIN_QUALITY");
  return env != nullptr ? std::atof(env) : 0.12;
}

double q1ClosureSplitMinAngleDeg() {
  const char *env = std::getenv("AMR_Q1_CLOSURE_SPLIT_MIN_ANGLE");
  return env != nullptr ? std::atof(env) : 2.0;
}

double q1ClosureSplitMinQuality() {
  const char *env = std::getenv("AMR_Q1_CLOSURE_SPLIT_MIN_QUALITY");
  return env != nullptr ? std::atof(env) : 0.04;
}

double q1LocalGradingAreaRatio() {
  const char *env = std::getenv("AMR_Q1_LOCAL_GRADING_AREA_RATIO");
  return env != nullptr ? std::atof(env) : 2.5;
}

int q1LocalGradingMaxSupports() {
  const char *env = std::getenv("AMR_Q1_LOCAL_GRADING_MAX_SUPPORTS");
  if (env == nullptr) return 2;
  try {
    return std::max(0, std::stoi(env));
  } catch (...) {
    return 2;
  }
}

int q2ConformityWallDistanceLimit() {
  const char *env = std::getenv("AMR_Q2_CONFORMITY_WALL_DISTANCE");
  if (env == nullptr) return -1;
  try {
    return std::stoi(env);
  } catch (...) {
    return -1;
  }
}

[[maybe_unused]] bool bezierWallPatchOptimizationEnabled() {
  const char *env = std::getenv("AMR_ENABLE_BEZIER_PATCH_OPT");
  return env != nullptr && std::string(env) != "0";
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

enum ParentSplitReason : unsigned char {
  kSplitNone = 0,
  kSplitMarked = 1,
  kSplitFallback = 2,
  kSplitConformity = 3,
  kSplitPeriodic = 4,
  kSplitWallFan = 5,
  kSplitWallCavity = 6,
  kSplitEndpointFan = 7,
  kSplitEndpointOverride = 8,
  kSplitGrading = 9,
};

const char *parentSplitReasonName(unsigned char reason) {
  switch (reason) {
    case kSplitMarked: return "marked";
    case kSplitFallback: return "fallback";
    case kSplitConformity: return "conformity";
    case kSplitPeriodic: return "periodic";
    case kSplitWallFan: return "wall_fan";
    case kSplitWallCavity: return "wall_cavity";
    case kSplitEndpointFan: return "endpoint_fan";
    case kSplitEndpointOverride: return "endpoint_override";
    case kSplitGrading: return "grading";
    default: return "none";
  }
}

enum class Q1FluidSideMode {
  legacy_y_only,
  env_selected,
  normal_half_space,
  polygon_solid,
};

const char *q1FluidSideModeName(Q1FluidSideMode mode) {
  switch (mode) {
    case Q1FluidSideMode::legacy_y_only: return "y_only";
    case Q1FluidSideMode::env_selected:
      if (q1PolygonBladeGuardEnabled()) return "polygon";
      if (q1NormalBladeGuardEnabled()) return "normal";
      return "y_only";
    case Q1FluidSideMode::normal_half_space: return "normal";
    case Q1FluidSideMode::polygon_solid: return "polygon";
  }
  return "y_only";
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

const char *bladeBranchName(BladeBranch branch) {
  switch (branch) {
    case BladeBranch::upper: return "upper";
    case BladeBranch::lower: return "lower";
    case BladeBranch::upper_shifted: return "upper_shifted";
    case BladeBranch::lower_shifted: return "lower_shifted";
    case BladeBranch::none: return "none";
  }
  return "none";
}

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

std::vector<Vec2> sampleBladeSolidLoop(double y_shift = 0.0) {
  BladeSplineReference &ref = bladeSplineReference();
  std::vector<Vec2> loop;
  if (!ref.loaded) return loop;
  const int ns = 240;
  loop.reserve(2 * (ns + 1));
  for (int i = 0; i <= ns; ++i) {
    const double s = ref.s_total_upper * static_cast<double>(i) /
                     static_cast<double>(ns);
    Vec2 pt = evaluateBladeBranch(BladeBranch::upper, s);
    pt.y += y_shift;
    loop.push_back(pt);
  }
  for (int i = ns; i >= 0; --i) {
    const double s = ref.s_total_lower * static_cast<double>(i) /
                     static_cast<double>(ns);
    Vec2 pt = evaluateBladeBranch(BladeBranch::lower, s);
    pt.y += y_shift;
    loop.push_back(pt);
  }
  return loop;
}

const std::vector<std::vector<Vec2>> &q1BladeSolidLoops() {
  static const std::vector<std::vector<Vec2>> loops = []() {
    std::vector<std::vector<Vec2>> built;
    const auto base = sampleBladeSolidLoop(0.0);
    if (base.empty()) return built;
    built.push_back(base);
    built.push_back(sampleBladeSolidLoop(18.0));
    built.push_back(sampleBladeSolidLoop(-18.0));
    return built;
  }();
  return loops;
}

bool pointInsidePolygon(const Vec2 &p, const std::vector<Vec2> &poly,
                        double tol = 1e-10) {
  if (poly.size() < 3) return false;
  auto pointOnEdge = [&](const Vec2 &a, const Vec2 &b) {
    const Vec2 ab = b - a;
    const Vec2 ap = p - a;
    const double cross = std::abs(ab.x * ap.y - ab.y * ap.x);
    if (cross > tol * std::max(1.0, ab.norm())) return false;
    return std::min(a.x, b.x) - tol <= p.x && p.x <= std::max(a.x, b.x) + tol &&
           std::min(a.y, b.y) - tol <= p.y && p.y <= std::max(a.y, b.y) + tol;
  };
  bool inside = false;
  for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
    const Vec2 &a = poly[j];
    const Vec2 &b = poly[i];
    if (pointOnEdge(a, b)) return false;
    const bool crosses = ((a.y > p.y) != (b.y > p.y));
    if (!crosses) continue;
    const double x_intersect =
        a.x + (b.x - a.x) * (p.y - a.y) / ((b.y - a.y) + 1e-300);
    if (x_intersect > p.x) inside = !inside;
  }
  return inside;
}

bool pointInsideBladeSolid(const Vec2 &p, double tol = 1e-10) {
  for (const auto &loop : q1BladeSolidLoops()) {
    if (pointInsidePolygon(p, loop, tol)) return true;
  }
  return false;
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

bool bladeEdgeArcMidpoint(const Vec2 &a, const Vec2 &b, Vec2 &mid) {
  BladeProjection pa = projectToBladeSplineDetailed(a);
  BladeProjection pb = projectToBladeSplineDetailed(b);
  if (!pa.valid || !pb.valid) return false;
  if (pa.branch != pb.branch) return false;

  const double s_mid = 0.5 * (pa.s + pb.s);
  mid = evaluateBladeBranch(pa.branch, s_mid);
  return true;
}

bool bladeEdgeArcFractionPoint(const Vec2 &a, const Vec2 &b, double frac, Vec2 &pt) {
  BladeProjection pa = projectToBladeSplineDetailed(a);
  BladeProjection pb = projectToBladeSplineDetailed(b);
  if (!pa.valid || !pb.valid) return false;
  if (pa.branch != pb.branch) return false;
  const double s = (1.0 - frac) * pa.s + frac * pb.s;
  pt = evaluateBladeBranch(pa.branch, s);
  return true;
}

bool curvedParentEdgeMidpoint(const Mesh &mesh,
                              const std::map<std::pair<int, int>, std::vector<int>> &edge_to_elem,
                              int va, int vb, Vec2 &mid) {
  const auto key = std::make_pair(std::min(va, vb), std::max(va, vb));
  auto eit = edge_to_elem.find(key);
  if (eit == edge_to_elem.end()) return false;

  const std::array<Vec2, 3> ref_corner = {{
      {0.0, 0.0},
      {1.0, 0.0},
      {0.0, 1.0},
  }};
  for (int eadj : eit->second) {
    if (eadj < 0 || eadj >= static_cast<int>(mesh.E.size())) continue;
    const auto &elem = mesh.E[eadj];
    if (elem.q_order <= 1) continue;

    int ia = -1;
    int ib = -1;
    for (int k = 0; k < 3; ++k) {
      if (elem.v[k] == va) ia = k;
      if (elem.v[k] == vb) ib = k;
    }
    if (ia < 0 || ib < 0) continue;

    const Vec2 rm = 0.5 * (ref_corner[ia] + ref_corner[ib]);
    mid = mesh.evaluateElementGeometry(eadj, rm.x, rm.y).x;
    return true;
  }
  return false;
}

double triangleMinAngleDeg(const Element &elem, const std::vector<Vec2> &verts);
double triangleSignedAreaPts(const Vec2 &a, const Vec2 &b, const Vec2 &c);
double triangleMinAngleDegPts(const Vec2 &a, const Vec2 &b, const Vec2 &c);
bool segmentsIntersect(const Vec2 &a, const Vec2 &b,
                       const Vec2 &c, const Vec2 &d,
                       double tol = 1e-12);
bool pointIsOnFluidSide(const Vec2 &p, const BladeProjection &proj, double tol = 1e-10);
double signedDistanceToFluidHalfSpace(const Vec2 &p, const BladeProjection &proj);
bool q1PointIsOnFluidSide(const Vec2 &p, const BladeProjection &proj,
                          Q1FluidSideMode mode,
                          double tol = 1e-10);
bool q1PointIsOnFluidSide(const Vec2 &p, const BladeProjection &proj,
                          double tol = 1e-10);

double triangleQualityPts(const Vec2 &a, const Vec2 &b, const Vec2 &c) {
  const Vec2 ab = b - a;
  const Vec2 bc = c - b;
  const Vec2 ca = a - c;
  const double sum_len2 = ab.normSq() + bc.normSq() + ca.normSq();
  if (sum_len2 <= 1e-16) return -1.0;
  const double twice_area =
      std::abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
  return 2.0 * std::sqrt(3.0) * twice_area / sum_len2;
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

std::string localWallSplitPatchFailureReason(const Mesh &mesh,
                                             int va, int vb, int vopp,
                                             const Vec2 &mid,
                                             const Vec2 &vopp_pos,
                                             const std::vector<std::vector<int>> &node_to_elem,
                                             double area_tol = 1e-12,
                                             double min_angle_tol = 8.0,
                                             double min_quality_tol = 0.18,
                                             double area_ratio_tol = 0.10) {
  (void)min_angle_tol;
  (void)min_quality_tol;
  (void)area_ratio_tol;
  std::vector<Vec2> verts = mesh.V;
  verts[vopp] = vopp_pos;

  std::set<int> local_elems;
  for (int e : node_to_elem[va]) local_elems.insert(e);
  for (int e : node_to_elem[vb]) local_elems.insert(e);
  for (int e : node_to_elem[vopp]) local_elems.insert(e);
  bool curved_wall_edge = false;
  for (int e : node_to_elem[va]) {
    const auto &elem = mesh.E[e];
    if ((elem.v[0] == va || elem.v[1] == va || elem.v[2] == va) &&
        (elem.v[0] == vb || elem.v[1] == vb || elem.v[2] == vb) &&
        elem.q_order > 1) {
      curved_wall_edge = true;
      break;
    }
  }
  if (local_elems.empty()) return "local patch unavailable";

  const double a1 = triangleSignedAreaPts(vopp_pos, verts[va], mid);
  const double a2 = triangleSignedAreaPts(vopp_pos, mid, verts[vb]);
  if (!(a1 > area_tol) || !(a2 > area_tol)) return "child area non-positive";

  for (int e : node_to_elem[vopp]) {
    const auto &elem = mesh.E[e];
    if ((elem.v[0] == va || elem.v[1] == va || elem.v[2] == va) &&
        (elem.v[0] == vb || elem.v[1] == vb || elem.v[2] == vb)) {
      continue;
    }
    if (!(triangleSignedArea(elem, verts) > area_tol)) return "neighbor area non-positive";
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
  for (size_t edge_idx = 0; edge_idx < new_edges.size(); ++edge_idx) {
    const auto &[e0, e1] = new_edges[edge_idx];
    const bool boundary_child_edge = edge_idx < 2;
    if ((curved_wall_edge && boundary_child_edge) ||
        (!curved_wall_edge && boundary_child_edge &&
         q1AllowBoundaryChildEdgeOutsideBlade())) {
      continue;
    }
    const Vec2 edge_mid = 0.5 * (e0 + e1);
    BladeProjection proj = projectToBladeSplineDetailed(edge_mid);
    if (proj.valid && !pointIsOnFluidSide(edge_mid, proj)) return "child-edge midpoint outside blade";
  }
  for (const auto &edge : local_edges) {
    const int x = edge.first;
    const int y = edge.second;
    if ((x == va && y == vb) || (x == vb && y == va)) continue;
    for (size_t edge_idx = 0; edge_idx < new_edges.size(); ++edge_idx) {
      const auto &[e0, e1] = new_edges[edge_idx];
      const bool boundary_child_edge = edge_idx < 2;
      if ((curved_wall_edge && boundary_child_edge) ||
          (!curved_wall_edge && boundary_child_edge &&
           q1AllowBoundaryChildEdgeOutsideBlade())) {
        continue;
      }
      const bool shares_va = (verts[x] - e0).normSq() <= 1e-24 || (verts[y] - e0).normSq() <= 1e-24;
      const bool shares_vb = (verts[x] - e1).normSq() <= 1e-24 || (verts[y] - e1).normSq() <= 1e-24;
      if (shares_va || shares_vb) continue;
      if (segmentsIntersect(e0, e1, verts[x], verts[y])) return "new edge intersects local edge";
    }
  }

  return {};
}

std::string q1InteriorSplitBladeGuardFailureReason(
    const Mesh &mesh,
    int va,
    int vb,
    const std::map<std::pair<int, int>, std::vector<int>> &edge_to_elem,
    Q1FluidSideMode mode = Q1FluidSideMode::env_selected) {
  const auto key = std::make_pair(std::min(va, vb), std::max(va, vb));
  const bool debug_guard =
      q1BladeGuardDebugEnabledForEdge(key, static_cast<int>(mesh.E.size()));
  auto guardPointAccepted = [&](const Vec2 &candidate,
                                const BladeProjection &proj) {
    return q1PointIsOnFluidSide(candidate, proj, mode);
  };
  const char *guard_mode = q1FluidSideModeName(mode);
  auto log_guard_failure = [&](const char *stage,
                               const Vec2 &candidate,
                               const BladeProjection &proj,
                               int eadj,
                               int local_opp,
                               const Vec2 &straight_mid) {
    if (!debug_guard) return;
    std::cerr << std::setprecision(17)
              << "    q1_guard_debug edge=(" << key.first << ", " << key.second
              << ") stage=" << stage
              << " elems=" << mesh.E.size()
              << " candidate=(" << candidate.x << ", " << candidate.y << ")"
              << " proj_valid=" << proj.valid
              << " proj_branch=" << bladeBranchName(proj.branch)
              << " proj_pt=(" << proj.pt.x << ", " << proj.pt.y << ")"
              << " sample_minus_proj_y=" << (candidate.y - proj.pt.y)
              << " normal_dot=" << signedDistanceToFluidHalfSpace(candidate, proj)
              << " guard_mode=" << guard_mode
              << " fluid_above=" << branchFluidIsAbove(proj.branch)
              << " fluid_ok=" << guardPointAccepted(candidate, proj)
              << " straight_mid=(" << straight_mid.x << ", "
              << straight_mid.y << ")";
    if (eadj >= 0) {
      std::cerr << " elem=" << eadj;
      const auto &elem = mesh.E[eadj];
      std::cerr << " elem_v=(" << elem.v[0] << ", " << elem.v[1]
                << ", " << elem.v[2] << ")";
    }
    if (local_opp >= 0) {
      std::cerr << " opp=" << local_opp
                << " opp_pt=(" << mesh.V[local_opp].x << ", "
                << mesh.V[local_opp].y << ")";
    }
    std::cerr << std::endl;
  };
  const Vec2 straight_mid = 0.5 * (mesh.V[va] + mesh.V[vb]);
  BladeProjection mid_proj = projectToBladeSplineDetailed(straight_mid);
  if (mid_proj.valid && !guardPointAccepted(straight_mid, mid_proj)) {
    log_guard_failure("split_midpoint", straight_mid, mid_proj, -1, -1,
                      straight_mid);
    return "split midpoint outside blade";
  }

  auto eit = edge_to_elem.find(key);
  if (eit == edge_to_elem.end()) return {};
  for (int eadj : eit->second) {
    if (eadj < 0 || eadj >= static_cast<int>(mesh.E.size())) continue;
    const int *ev = mesh.E[eadj].v;
    int local_opp = -1;
    for (int k = 0; k < 3; ++k) {
      if (ev[k] != va && ev[k] != vb) {
        local_opp = ev[k];
        break;
      }
    }
    if (local_opp < 0) continue;
    const Vec2 bisector_mid = 0.5 * (straight_mid + mesh.V[local_opp]);
    BladeProjection bisector_proj = projectToBladeSplineDetailed(bisector_mid);
    if (bisector_proj.valid && !guardPointAccepted(bisector_mid, bisector_proj)) {
      log_guard_failure("child_bisector", bisector_mid, bisector_proj, eadj,
                        local_opp, straight_mid);
      return "child bisector midpoint outside blade";
    }
  }
  return {};
}

std::string q1WallFanTrisectionFailureReason(
    const Mesh &mesh,
    int va,
    int vb,
    int vc,
    const Vec2 &vm1,
    const Vec2 &vm2,
    const std::vector<std::vector<int>> &node_to_elem) {
  const double area1 = triangleSignedAreaPts(mesh.V[vc], mesh.V[va], vm1);
  const double area2 = triangleSignedAreaPts(mesh.V[vc], vm1, vm2);
  const double area3 = triangleSignedAreaPts(mesh.V[vc], vm2, mesh.V[vb]);
  if (!(area1 > 1e-12) || !(area2 > 1e-12) || !(area3 > 1e-12)) {
    return "child area non-positive";
  }

  const std::vector<std::pair<Vec2, Vec2>> new_edges = {
      {mesh.V[vc], vm1},
      {mesh.V[vc], vm2},
  };
  for (const auto &[e0, e1] : new_edges) {
    const Vec2 edge_mid = 0.5 * (e0 + e1);
    BladeProjection proj = projectToBladeSplineDetailed(edge_mid);
    if (proj.valid && !pointIsOnFluidSide(edge_mid, proj)) {
      return "child-edge midpoint outside blade";
    }
  }

  std::set<int> local_elems;
  for (int e : node_to_elem[vc]) local_elems.insert(e);
  for (int e : node_to_elem[va]) local_elems.insert(e);
  for (int e : node_to_elem[vb]) local_elems.insert(e);

  std::set<std::pair<int, int>> local_edges;
  for (int e : local_elems) {
    const auto &elem = mesh.E[e];
    local_edges.insert({std::min(elem.v[0], elem.v[1]), std::max(elem.v[0], elem.v[1])});
    local_edges.insert({std::min(elem.v[1], elem.v[2]), std::max(elem.v[1], elem.v[2])});
    local_edges.insert({std::min(elem.v[2], elem.v[0]), std::max(elem.v[2], elem.v[0])});
  }
  for (const auto &edge : local_edges) {
    const int x = edge.first;
    const int y = edge.second;
    if ((x == va && y == vb) || (x == vb && y == va)) continue;
    for (const auto &[e0, e1] : new_edges) {
      const bool shares_e0 = (mesh.V[x] - e0).normSq() <= 1e-24 ||
                             (mesh.V[y] - e0).normSq() <= 1e-24;
      const bool shares_e1 = (mesh.V[x] - e1).normSq() <= 1e-24 ||
                             (mesh.V[y] - e1).normSq() <= 1e-24;
      if (shares_e0 || shares_e1) continue;
      if (segmentsIntersect(e0, e1, mesh.V[x], mesh.V[y])) {
        return "new edge intersects local edge";
      }
    }
  }
  return {};
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

double oppositeAngleDeg(const Vec2 &a, const Vec2 &b, const Vec2 &p) {
  return angleDegBetween(a - p, b - p);
}

bool diametralLensEncroached(const Vec2 &a, const Vec2 &b, const Vec2 &p,
                             double min_angle_deg) {
  const double threshold = 180.0 - 2.0 * min_angle_deg;
  return oppositeAngleDeg(a, b, p) >= threshold;
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

double signedDistanceToFluidHalfSpace(const Vec2 &p, const BladeProjection &proj) {
  if (!proj.valid) return std::numeric_limits<double>::infinity();
  const Vec2 n_hat = branchFluidNormal(proj.branch, proj.s);
  return (p - proj.pt).dot(n_hat);
}

bool q1PointIsOnFluidSide(const Vec2 &p, const BladeProjection &proj,
                          Q1FluidSideMode mode,
                          double tol) {
  if (!proj.valid) return true;
  switch (mode) {
    case Q1FluidSideMode::legacy_y_only:
      return pointIsOnFluidSide(p, proj, tol);
    case Q1FluidSideMode::env_selected:
      if (q1PolygonBladeGuardEnabled() && !q1BladeSolidLoops().empty()) {
        return !pointInsideBladeSolid(p, tol);
      }
      if (!q1NormalBladeGuardEnabled()) {
        return pointIsOnFluidSide(p, proj, tol);
      }
      return signedDistanceToFluidHalfSpace(p, proj) >= -tol;
    case Q1FluidSideMode::normal_half_space:
      return signedDistanceToFluidHalfSpace(p, proj) >= -tol;
    case Q1FluidSideMode::polygon_solid:
      if (!q1BladeSolidLoops().empty()) return !pointInsideBladeSolid(p, tol);
      return pointIsOnFluidSide(p, proj, tol);
  }
  return pointIsOnFluidSide(p, proj, tol);
}

bool q1PointIsOnFluidSide(const Vec2 &p, const BladeProjection &proj,
                          double tol) {
  return q1PointIsOnFluidSide(p, proj, Q1FluidSideMode::env_selected, tol);
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

std::set<int> collectPeriodicVertices(const Mesh &mesh) {
  std::set<int> periodic_vertices;
  for (const auto &group : mesh.periodicGroups) {
    for (const auto &[v0, v1] : group.pairs) {
      periodic_vertices.insert(v0);
      periodic_vertices.insert(v1);
    }
  }
  return periodic_vertices;
}

std::set<std::pair<int, int>> collectPeriodicEdgeKeys(const Mesh &mesh) {
  std::set<std::pair<int, int>> periodic_edges;
  for (const auto &ie : mesh.IE) {
    const auto key_l = std::make_pair(std::min(ie.v[0], ie.v[1]),
                                      std::max(ie.v[0], ie.v[1]));
    const auto key_r = std::make_pair(std::min(ie.vR[0], ie.vR[1]),
                                      std::max(ie.vR[0], ie.vR[1]));
    if (key_l != key_r) {
      periodic_edges.insert(key_l);
      periodic_edges.insert(key_r);
    }
  }
  return periodic_edges;
}

std::set<int> collectBoundaryVertices(const Mesh &mesh) {
  std::set<int> boundary_vertices;
  for (const auto &be : mesh.BE) {
    boundary_vertices.insert(be.v[0]);
    boundary_vertices.insert(be.v[1]);
  }
  return boundary_vertices;
}

int improveMeshByEdgeSwaps(Mesh &mesh,
                           const std::map<std::pair<int, int>, int> &boundary_edge_groups,
                           int max_swaps,
                           int first_new_vertex = -1,
                           int first_new_elem = -1) {
  if (max_swaps <= 0) return 0;

  const auto periodic_vertices = collectPeriodicVertices(mesh);
  const auto periodic_edges = collectPeriodicEdgeKeys(mesh);
  const auto boundary_vertices = collectBoundaryVertices(mesh);
  int swap_count = 0;

  for (int swap_iter = 0; swap_iter < max_swaps; ++swap_iter) {
    std::map<std::pair<int, int>, std::vector<int>> edge_to_elem;
    for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
      const auto &elem = mesh.E[e];
      if (elem.q_order != 1) continue;
      edge_to_elem[{std::min(elem.v[0], elem.v[1]), std::max(elem.v[0], elem.v[1])}].push_back(e);
      edge_to_elem[{std::min(elem.v[1], elem.v[2]), std::max(elem.v[1], elem.v[2])}].push_back(e);
      edge_to_elem[{std::min(elem.v[2], elem.v[0]), std::max(elem.v[2], elem.v[0])}].push_back(e);
    }

    bool changed = false;
    for (const auto &[edge, elems] : edge_to_elem) {
      if (elems.size() != 2) continue;
      if (boundary_edge_groups.count(edge)) continue;

      const int e0 = elems[0];
      const int e1 = elems[1];
      const auto &elem0 = mesh.E[e0];
      const auto &elem1 = mesh.E[e1];
      if (elem0.q_order != 1 || elem1.q_order != 1) continue;
      if (periodic_edges.count(edge)) continue;

      const int a = edge.first;
      const int b = edge.second;
      int c = -1;
      int d = -1;
      for (int k = 0; k < 3; ++k) {
        if (elem0.v[k] != a && elem0.v[k] != b) c = elem0.v[k];
        if (elem1.v[k] != a && elem1.v[k] != b) d = elem1.v[k];
      }
      if (c < 0 || d < 0 || c == d) continue;

      if (first_new_vertex >= 0 || first_new_elem >= 0) {
        const bool touches_new_vertex =
            (first_new_vertex >= 0) &&
            (a >= first_new_vertex || b >= first_new_vertex ||
             c >= first_new_vertex || d >= first_new_vertex);
        const bool touches_new_elem =
            (first_new_elem >= 0) && (e0 >= first_new_elem || e1 >= first_new_elem);
        if (!touches_new_vertex && !touches_new_elem) continue;
      }

      if (periodic_vertices.count(a) || periodic_vertices.count(b) ||
          periodic_vertices.count(c) || periodic_vertices.count(d) ||
          boundary_vertices.count(a) || boundary_vertices.count(b) ||
          boundary_vertices.count(c) || boundary_vertices.count(d)) {
        continue;
      }

      const auto new_edge = std::make_pair(std::min(c, d), std::max(c, d));
      if (boundary_edge_groups.count(new_edge)) continue;
      if (periodic_edges.count(new_edge)) continue;
      auto edge_it = edge_to_elem.find(new_edge);
      if (edge_it != edge_to_elem.end() && !edge_it->second.empty()) continue;
      if (!segmentsIntersect(mesh.V[a], mesh.V[b], mesh.V[c], mesh.V[d], 1e-12)) continue;

      Element swapped0 = makePositiveElement(c, d, a, mesh.V);
      Element swapped1 = makePositiveElement(d, c, b, mesh.V);
      const double area0 = triangleSignedArea(swapped0, mesh.V);
      const double area1 = triangleSignedArea(swapped1, mesh.V);
      if (!(area0 > 1e-12) || !(area1 > 1e-12)) continue;

      const double old_min_angle = std::min(triangleMinAngleDeg(elem0, mesh.V),
                                            triangleMinAngleDeg(elem1, mesh.V));
      const double new_min_angle = std::min(triangleMinAngleDeg(swapped0, mesh.V),
                                            triangleMinAngleDeg(swapped1, mesh.V));
      const double old_min_quality = std::min(triangleQuality(elem0, mesh.V),
                                              triangleQuality(elem1, mesh.V));
      const double new_min_quality = std::min(triangleQuality(swapped0, mesh.V),
                                              triangleQuality(swapped1, mesh.V));
      if (new_min_quality < 0.18) continue;
      if (new_min_angle < old_min_angle + 1.0 &&
          new_min_quality < old_min_quality + 0.03) {
        continue;
      }

      mesh.E[e0] = swapped0;
      mesh.E[e1] = swapped1;
      changed = true;
      swap_count++;
      break;
    }

    if (!changed) break;
  }

  return swap_count;
}

int smoothNewInteriorVertices(Mesh &mesh, int first_new_vertex,
                              std::set<int> *moved_patch_elems = nullptr) {
  if (first_new_vertex >= static_cast<int>(mesh.V.size())) return 0;

  const auto boundary_vertices = collectBoundaryVertices(mesh);
  const auto periodic_vertices = collectPeriodicVertices(mesh);
  std::map<std::pair<int, int>, int> boundary_edge_groups;
  for (const auto &be : mesh.BE) {
    boundary_edge_groups[{std::min(be.v[0], be.v[1]), std::max(be.v[0], be.v[1])}] = be.bIndex;
  }
  std::vector<std::vector<int>> node_to_elem(mesh.V.size());
  std::vector<std::set<int>> vertex_neighbors(mesh.V.size());
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    const auto &elem = mesh.E[e];
    node_to_elem[elem.v[0]].push_back(e);
    node_to_elem[elem.v[1]].push_back(e);
    node_to_elem[elem.v[2]].push_back(e);
    vertex_neighbors[elem.v[0]].insert(elem.v[1]);
    vertex_neighbors[elem.v[0]].insert(elem.v[2]);
    vertex_neighbors[elem.v[1]].insert(elem.v[0]);
    vertex_neighbors[elem.v[1]].insert(elem.v[2]);
    vertex_neighbors[elem.v[2]].insert(elem.v[0]);
    vertex_neighbors[elem.v[2]].insert(elem.v[1]);
  }

  auto elementTouchesWall = [&](int e) {
    const auto &elem = mesh.E[e];
    const std::pair<int, int> edges[3] = {
      {std::min(elem.v[0], elem.v[1]), std::max(elem.v[0], elem.v[1])},
      {std::min(elem.v[1], elem.v[2]), std::max(elem.v[1], elem.v[2])},
      {std::min(elem.v[2], elem.v[0]), std::max(elem.v[2], elem.v[0])}
    };
    for (const auto &edge : edges) {
      auto it = boundary_edge_groups.find(edge);
      if (it == boundary_edge_groups.end()) continue;
      int g = it->second;
      if (g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
          lowerCopy(mesh.Bname[g]) == "wall") {
        return true;
      }
    }
    return false;
  };

  std::set<int> candidates;
  for (int vid = first_new_vertex; vid < static_cast<int>(mesh.V.size()); ++vid) {
    if (boundary_vertices.count(vid) || periodic_vertices.count(vid)) continue;
    bool touches_wall_patch = false;
    for (int e : node_to_elem[vid]) {
      if (elementTouchesWall(e)) {
        touches_wall_patch = true;
        break;
      }
    }
    if (touches_wall_patch) continue;
    candidates.insert(vid);
  }
  if (candidates.empty()) return 0;

  const int n_iters = std::min(meshSmoothingConfig().iterations, 1);
  const double omega = 0.15;
  int move_count = 0;
  std::set<int> moved_vertices;

  for (int iter = 0; iter < n_iters; ++iter) {
    const std::vector<Vec2> verts_before = mesh.V;
    std::vector<Vec2> verts_trial = mesh.V;
    bool iter_changed = false;

    for (int vid : candidates) {
      const auto &nbrs = vertex_neighbors[vid];
      if (nbrs.size() < 2) continue;

      Vec2 avg{0.0, 0.0};
      for (int nbr : nbrs) avg = avg + verts_before[nbr];
      avg = avg / static_cast<double>(nbrs.size());
      Vec2 candidate = (1.0 - omega) * verts_before[vid] + omega * avg;

      BladeProjection proj = projectToBladeSplineDetailed(candidate);
      if (proj.valid && !pointIsOnFluidSide(candidate, proj, meshSmoothingConfig().wall_geom_tol)) {
        candidate = movePointToFluidSide(candidate, proj, meshSmoothingConfig().wall_geom_tol);
      }

      if ((candidate - verts_before[vid]).norm() <= 1e-10) continue;
      if (!localMovePreservesMesh(mesh, vid, node_to_elem, verts_before, verts_trial, candidate)) {
        continue;
      }
      verts_trial[vid] = candidate;
      iter_changed = true;
      move_count++;
      moved_vertices.insert(vid);
    }

    mesh.V.swap(verts_trial);
    if (!iter_changed) break;
  }

  if (moved_patch_elems != nullptr) {
    for (int vid : moved_vertices) {
      for (int e : node_to_elem[vid]) moved_patch_elems->insert(e);
    }
  }

  return move_count;
}

[[maybe_unused]] void refreshQ2GeometryNodes(
    Mesh &mesh,
    const std::map<std::pair<int, int>, int> &boundary_edge_groups) {
  std::set<int> updated_mid_nodes;
  auto edgeMidpoint = [&](int a, int b) {
    Vec2 mid = 0.5 * (mesh.V[a] + mesh.V[b]);
    auto it = boundary_edge_groups.find({std::min(a, b), std::max(a, b)});
    if (it != boundary_edge_groups.end()) {
      int g = it->second;
      if (g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
          lowerCopy(mesh.Bname[g]) == "wall") {
        Vec2 arc_mid = mid;
        if (!bladeEdgeArcMidpoint(mesh.V[a], mesh.V[b], arc_mid)) {
          arc_mid = projectToBladeSpline(mid);
        }
        mid = arc_mid;
      }
    }
    return mid;
  };

  for (auto &elem : mesh.E) {
    if (elem.q_order != 2 || elem.ho_nodes.size() != 6) continue;
    const int a = elem.v[0];
    const int b = elem.v[1];
    const int c = elem.v[2];

    const std::array<std::pair<int, std::pair<int, int>>, 3> mids = {{
        {elem.ho_nodes[1], {a, b}},
        {elem.ho_nodes[3], {c, a}},
        {elem.ho_nodes[4], {b, c}},
    }};
    for (const auto &[mid_idx, edge] : mids) {
      if (!updated_mid_nodes.insert(mid_idx).second) continue;
      mesh.V[mid_idx] = edgeMidpoint(edge.first, edge.second);
    }
  }
}

struct WallSplitAssessment {
  bool feasible = false;
  int failing_adjacent = 0;
  double min_patch_detJ = -std::numeric_limits<double>::infinity();
  double min_bezier_corner_area = -std::numeric_limits<double>::infinity();
  double min_child_angle = 0.0;
  double min_child_quality = 0.0;
  double score = -std::numeric_limits<double>::infinity();
};

void retunePatchQ2Geometry(
    Mesh &mesh,
    const std::set<int> &patch_elems,
    const std::map<std::pair<int, int>, int> &boundary_edge_groups) {
  std::set<int> updated_mid_nodes;
  auto edgeMidpoint = [&](int a, int b) {
    Vec2 mid = 0.5 * (mesh.V[a] + mesh.V[b]);
    auto it = boundary_edge_groups.find({std::min(a, b), std::max(a, b)});
    if (it != boundary_edge_groups.end()) {
      const int g = it->second;
      if (g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
          lowerCopy(mesh.Bname[g]) == "wall") {
        Vec2 arc_mid = mid;
        if (!bladeEdgeArcMidpoint(mesh.V[a], mesh.V[b], arc_mid)) {
          arc_mid = projectToBladeSpline(mid);
        }
        mid = arc_mid;
      }
    }
    return mid;
  };

  for (int e : patch_elems) {
    if (e < 0 || e >= static_cast<int>(mesh.E.size())) continue;
    auto &elem = mesh.E[e];
    if (elem.q_order != 2 || elem.ho_nodes.size() != 6) continue;

    const int a = elem.v[0];
    const int b = elem.v[1];
    const int c = elem.v[2];
    const std::array<std::pair<int, std::pair<int, int>>, 3> mids = {{
        {elem.ho_nodes[1], {a, b}},
        {elem.ho_nodes[3], {c, a}},
        {elem.ho_nodes[4], {b, c}},
    }};
    for (const auto &[mid_idx, edge] : mids) {
      if (!updated_mid_nodes.insert(mid_idx).second) continue;
      mesh.V[mid_idx] = edgeMidpoint(edge.first, edge.second);
    }
  }
}

[[maybe_unused]] void retuneWallPatchQ2Geometry(
    Mesh &mesh,
    const std::set<int> &patch_elems,
    const std::map<std::pair<int, int>, int> &boundary_edge_groups) {
  std::set<int> updated_mid_nodes;
  for (int e : patch_elems) {
    if (e < 0 || e >= static_cast<int>(mesh.E.size())) continue;
    auto &elem = mesh.E[e];
    if (elem.q_order != 2 || elem.ho_nodes.size() != 6) continue;

    const int a = elem.v[0];
    const int b = elem.v[1];
    const int c = elem.v[2];
    const std::array<std::pair<int, std::pair<int, int>>, 3> mids = {{
        {elem.ho_nodes[1], {a, b}},
        {elem.ho_nodes[3], {c, a}},
        {elem.ho_nodes[4], {b, c}},
    }};
    for (const auto &[mid_idx, edge] : mids) {
      if (!updated_mid_nodes.insert(mid_idx).second) continue;
      auto it = boundary_edge_groups.find({std::min(edge.first, edge.second),
                                           std::max(edge.first, edge.second)});
      if (it == boundary_edge_groups.end()) continue;
      const int g = it->second;
      if (g < 0 || g >= static_cast<int>(mesh.Bname.size()) ||
          lowerCopy(mesh.Bname[g]) != "wall") {
        continue;
      }
      Vec2 wall_mid = 0.5 * (mesh.V[edge.first] + mesh.V[edge.second]);
      if (!bladeEdgeArcMidpoint(mesh.V[edge.first], mesh.V[edge.second], wall_mid)) {
        wall_mid = projectToBladeSpline(wall_mid);
      }
      mesh.V[mid_idx] = wall_mid;
    }
  }
}

double patchMinimumExactDetJ(const Mesh &mesh, const std::set<int> &patch_elems) {
  double min_detJ = std::numeric_limits<double>::infinity();
  for (int e : patch_elems) {
    if (e < 0 || e >= static_cast<int>(mesh.E.size())) continue;
    const auto &elem = mesh.E[e];
    if (elem.q_order == 2) {
      CurvedElementDetJMinimum detJ_info;
      if (exactQ2DetJMinimum(mesh, e, detJ_info)) {
        min_detJ = std::min(min_detJ, detJ_info.detJ);
        continue;
      }
    }
    min_detJ = std::min(min_detJ, 2.0 * triangleSignedArea(elem, mesh.V));
  }
  if (!std::isfinite(min_detJ)) return 0.0;
  return min_detJ;
}

WallSplitAssessment assessWallSplitPatch(
    const Mesh &mesh,
    int va, int vb,
    const Vec2 &mid,
    const std::vector<int> &edge_adj_elems,
    const std::set<int> &patch_elems,
    const std::vector<std::vector<int>> &node_to_elem) {
  WallSplitAssessment out;
  out.feasible = true;
  out.min_patch_detJ = patchMinimumExactDetJ(mesh, patch_elems);
  out.min_bezier_corner_area = std::numeric_limits<double>::infinity();
  out.min_child_angle = std::numeric_limits<double>::infinity();
  out.min_child_quality = std::numeric_limits<double>::infinity();

  std::set<int> local_edge_patch_elems;
  for (int e : node_to_elem[va]) local_edge_patch_elems.insert(e);
  for (int e : node_to_elem[vb]) local_edge_patch_elems.insert(e);
  double orig_local_min_area = std::numeric_limits<double>::infinity();
  for (int e : local_edge_patch_elems) {
    orig_local_min_area = std::min(orig_local_min_area, triangleSignedArea(mesh.E[e], mesh.V));
  }
  orig_local_min_area = std::max(orig_local_min_area, 1e-12);

  for (int eadj : edge_adj_elems) {
    const int *ev = mesh.E[eadj].v;
    int vopp = -1;
    for (int k = 0; k < 3; ++k) {
      if (ev[k] != va && ev[k] != vb) {
        vopp = ev[k];
        break;
      }
    }
    if (vopp < 0) continue;

    std::string reason =
        localWallSplitPatchFailureReason(mesh, va, vb, vopp, mid, mesh.V[vopp], node_to_elem);
    if (!reason.empty()) {
      out.feasible = false;
      out.failing_adjacent++;
    }

    const Vec2 &vopp_pos = mesh.V[vopp];
    const double a1 = triangleSignedAreaPts(vopp_pos, mesh.V[va], mid);
    const double a2 = triangleSignedAreaPts(vopp_pos, mid, mesh.V[vb]);
    if (a1 > 1e-12 && a2 > 1e-12) {
      out.min_child_angle = std::min(
          out.min_child_angle,
          std::min(triangleMinAngleDegPts(vopp_pos, mesh.V[va], mid),
                   triangleMinAngleDegPts(vopp_pos, mid, mesh.V[vb])));
      const double q1 = std::abs(2.0 * std::sqrt(3.0) *
                                 ((mesh.V[va] - vopp_pos).x * (mid - vopp_pos).y -
                                  (mesh.V[va] - vopp_pos).y * (mid - vopp_pos).x)) /
                        std::max((mesh.V[va] - vopp_pos).normSq() +
                                     (mid - mesh.V[va]).normSq() +
                                     (vopp_pos - mid).normSq(),
                                 1e-16);
      const double q2 = std::abs(2.0 * std::sqrt(3.0) *
                                 ((mid - vopp_pos).x * (mesh.V[vb] - vopp_pos).y -
                                  (mid - vopp_pos).y * (mesh.V[vb] - vopp_pos).x)) /
                        std::max((mid - vopp_pos).normSq() +
                                     (mesh.V[vb] - mid).normSq() +
                                     (vopp_pos - mesh.V[vb]).normSq(),
                                 1e-16);
      out.min_child_quality = std::min(out.min_child_quality, std::min(q1, q2));
    } else {
      out.min_child_angle = -1.0;
      out.min_child_quality = -1.0;
    }
  }

  for (int e : patch_elems) {
    if (e < 0 || e >= static_cast<int>(mesh.E.size())) continue;
    if (mesh.E[e].q_order == 2) {
      out.min_bezier_corner_area =
          std::min(out.min_bezier_corner_area, q2BezierMinCornerArea(mesh, e));
    }
  }

  if (!std::isfinite(out.min_bezier_corner_area)) out.min_bezier_corner_area = 0.0;
  if (!std::isfinite(out.min_child_angle)) out.min_child_angle = -1.0;
  if (!std::isfinite(out.min_child_quality)) out.min_child_quality = -1.0;

  const double delta = (out.min_patch_detJ < 0.0)
                           ? std::sqrt(1e-8 + 0.04 * out.min_patch_detJ * out.min_patch_detJ)
                           : 1e-4;
  const double JR = 0.5 * (out.min_patch_detJ +
                           std::sqrt(out.min_patch_detJ * out.min_patch_detJ + 4.0 * delta * delta));
  out.score = (out.feasible ? 1e6 : 0.0)
              - 5000.0 * static_cast<double>(out.failing_adjacent)
              + 25.0 * out.min_child_angle
              + 400.0 * out.min_child_quality
              + 50.0 * out.min_bezier_corner_area
              + 10.0 * std::log(std::max(JR, 1e-12));
  return out;
}

[[maybe_unused]] bool optimizeWallPatchForSplit(
    Mesh &mesh,
    int va, int vb,
    const Vec2 &mid,
    const std::vector<std::vector<int>> &node_to_elem,
    const std::vector<std::set<int>> &vertex_neighbors,
    const std::map<std::pair<int, int>, int> &boundary_edge_groups,
    std::string &repair_summary) {
  std::vector<int> edge_adj_elems;
  for (int e : node_to_elem[va]) {
    const int *ev = mesh.E[e].v;
    if (ev[0] == vb || ev[1] == vb || ev[2] == vb) edge_adj_elems.push_back(e);
  }
  if (edge_adj_elems.empty()) return false;

  std::set<int> patch_elems(edge_adj_elems.begin(), edge_adj_elems.end());
  std::set<int> patch_nodes{va, vb};
  std::vector<int> local_opposites;
  for (int e : edge_adj_elems) {
    const int *ev = mesh.E[e].v;
    for (int k = 0; k < 3; ++k) {
      if (ev[k] != va && ev[k] != vb) {
        local_opposites.push_back(ev[k]);
        patch_nodes.insert(ev[k]);
      }
    }
  }
  for (int vid : local_opposites) {
    for (int e : node_to_elem[vid]) patch_elems.insert(e);
    for (int nbr : vertex_neighbors[vid]) patch_nodes.insert(nbr);
  }
  for (int vid : patch_nodes) {
    for (int e : node_to_elem[vid]) patch_elems.insert(e);
  }

  const auto periodic_vertices = collectPeriodicVertices(mesh);
  const auto boundary_vertices = collectBoundaryVertices(mesh);
  auto isWallBoundaryVertex = [&](int vid) {
    for (int nbr : vertex_neighbors[vid]) {
      const auto key = std::make_pair(std::min(vid, nbr), std::max(vid, nbr));
      auto it = boundary_edge_groups.find(key);
      if (it == boundary_edge_groups.end()) continue;
      const int g = it->second;
      if (g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
          lowerCopy(mesh.Bname[g]) == "wall") {
        return true;
      }
    }
    return false;
  };

  std::vector<int> movable_vertices = local_opposites;
  for (int vid : patch_nodes) {
    if (std::find(movable_vertices.begin(), movable_vertices.end(), vid) != movable_vertices.end()) {
      continue;
    }
    movable_vertices.push_back(vid);
  }
  movable_vertices.erase(
      std::remove_if(movable_vertices.begin(), movable_vertices.end(),
                     [&](int vid) {
                       if (periodic_vertices.count(vid)) return true;
                       if (boundary_vertices.count(vid) && !isWallBoundaryVertex(vid)) return true;
                       if (vid == va || vid == vb) return true;
                       return false;
                     }),
      movable_vertices.end());
  if (movable_vertices.empty()) return false;

  const std::vector<Vec2> original_vertices = mesh.V;
  retunePatchQ2Geometry(mesh, patch_elems, boundary_edge_groups);
  WallSplitAssessment best = assessWallSplitPatch(mesh, va, vb, mid, edge_adj_elems,
                                                  patch_elems, node_to_elem);
  if (best.feasible) {
    repair_summary = "local patch already feasible";
    return true;
  }

  bool any_change = false;
  for (int iter = 0; iter < 5; ++iter) {
    bool iter_changed = false;
    for (int vid : movable_vertices) {
      const Vec2 current = mesh.V[vid];
      std::vector<Vec2> candidates;

      if (!vertex_neighbors[vid].empty()) {
        Vec2 avg{0.0, 0.0};
        for (int nbr : vertex_neighbors[vid]) avg = avg + mesh.V[nbr];
        avg = avg / static_cast<double>(vertex_neighbors[vid].size());
        candidates.push_back(0.75 * current + 0.25 * avg);
        candidates.push_back(0.5 * current + 0.5 * avg);
      }

      BladeProjection proj = projectToBladeSplineDetailed(current);
      BladeProjection mid_proj = projectToBladeSplineDetailed(mid);
      const double edge_len = (mesh.V[vb] - mesh.V[va]).norm();
      if (mid_proj.valid) {
        const Vec2 t_hat = branchTangent(mid_proj.branch, mid_proj.s);
        const Vec2 n_hat = branchFluidNormal(mid_proj.branch, mid_proj.s);
        const Vec2 rel = current - mid_proj.pt;
        const double tangential = rel.dot(t_hat);
        for (double normal_scale : {0.20, 0.30, 0.40}) {
          candidates.push_back(mid_proj.pt + tangential * t_hat +
                               normal_scale * edge_len * n_hat);
        }
      }
      if (boundary_vertices.count(vid) && isWallBoundaryVertex(vid)) {
        Vec2 projected = current;
        if (proj.valid) projected = proj.pt;
        candidates.push_back(projected);
      }

      Vec2 best_pos = current;
      WallSplitAssessment best_local = best;
      for (const Vec2 &candidate : candidates) {
        std::vector<Vec2> verts_trial = mesh.V;
        if (!localMovePreservesMesh(mesh, vid, node_to_elem, mesh.V, verts_trial, candidate)) {
          continue;
        }
        mesh.V[vid] = candidate;
        retunePatchQ2Geometry(mesh, patch_elems, boundary_edge_groups);
        WallSplitAssessment trial = assessWallSplitPatch(
            mesh, va, vb, mid, edge_adj_elems, patch_elems, node_to_elem);
        if (trial.score > best_local.score + 1e-8) {
          best_local = trial;
          best_pos = candidate;
        }
        mesh.V[vid] = current;
        retunePatchQ2Geometry(mesh, patch_elems, boundary_edge_groups);
      }

      if ((best_pos - current).norm() > 1e-12) {
        mesh.V[vid] = best_pos;
        retunePatchQ2Geometry(mesh, patch_elems, boundary_edge_groups);
        best = best_local;
        iter_changed = true;
        any_change = true;
        if (best.feasible) {
          repair_summary = "Bezier/Jacobian local patch optimization succeeded";
          return true;
        }
      }
    }
    if (!iter_changed) break;
  }

  if (!best.feasible) {
    mesh.V = original_vertices;
    retunePatchQ2Geometry(mesh, patch_elems, boundary_edge_groups);
    repair_summary = "local Bezier/Jacobian patch optimization failed";
    return false;
  }

  repair_summary = "Bezier/Jacobian local patch optimization succeeded";
  return any_change;
}

[[maybe_unused]] void smoothRefinedCurvedMesh(
    Mesh &mesh,
    const std::map<std::pair<int, int>, int> &boundary_edge_groups,
    int first_new_vertex) {
  if (!curvedRefinementSmoothingEnabled()) return;

  bool has_curved = false;
  for (const auto &elem : mesh.E) {
    if (elem.q_order > 1) {
      has_curved = true;
      break;
    }
  }
  if (!has_curved) return;

  std::set<int> moved_patch_elems;
  const int move_count =
      smoothNewInteriorVertices(mesh, first_new_vertex, &moved_patch_elems);
  if (move_count <= 0) return;

  if (!moved_patch_elems.empty()) {
    retunePatchQ2Geometry(mesh, moved_patch_elems, boundary_edge_groups);
  }
  std::cerr << "    curved smoothing: moved_corner_nodes=" << move_count
            << std::endl;
}

[[maybe_unused]] void applyLocalEdgeSwapCleanup(
    Mesh &mesh,
    const std::map<std::pair<int, int>, int> &boundary_edge_groups,
    int first_new_vertex,
    int first_new_elem) {
  if (mesh.E.empty()) return;
  rebuildMeshEdgeConnectivity(mesh, boundary_edge_groups);
  mesh.appendPeriodicToIE();
  const int swap_count = improveMeshByEdgeSwaps(
      mesh, boundary_edge_groups, 8, first_new_vertex, first_new_elem);
  if (swap_count > 0) {
    rebuildMeshEdgeConnectivity(mesh, boundary_edge_groups);
    mesh.appendPeriodicToIE();
    std::cerr << "    local cleanup: swaps=" << swap_count << std::endl;
  }
}

struct SimplePatchQuality {
  double min_angle_deg = std::numeric_limits<double>::infinity();
  double min_quality = std::numeric_limits<double>::infinity();
};

struct Q1SplitPatchAssessment {
  bool feasible = false;
  double min_angle_deg = 0.0;
  double min_quality = 0.0;
  int adjacent_count = 0;
};

SimplePatchQuality assessSimplePatchQuality(const Mesh &mesh,
                                            const std::set<int> &patch_elems) {
  SimplePatchQuality out;
  for (int e : patch_elems) {
    if (e < 0 || e >= static_cast<int>(mesh.E.size())) continue;
    const auto &elem = mesh.E[e];
    if (elem.q_order != 1) continue;
    out.min_angle_deg =
        std::min(out.min_angle_deg, triangleMinAngleDeg(elem, mesh.V));
    out.min_quality =
        std::min(out.min_quality, triangleQuality(elem, mesh.V));
  }
  if (!std::isfinite(out.min_angle_deg)) out.min_angle_deg = 0.0;
  if (!std::isfinite(out.min_quality)) out.min_quality = 0.0;
  return out;
}

Q1SplitPatchAssessment assessQ1SplitPatch(
    const Mesh &mesh,
    int va,
    int vb,
    const std::map<std::pair<int, int>, std::vector<int>> &edge_to_elem) {
  Q1SplitPatchAssessment out;
  const auto key = std::make_pair(std::min(va, vb), std::max(va, vb));
  auto it = edge_to_elem.find(key);
  if (it == edge_to_elem.end()) return out;

  const Vec2 mid = 0.5 * (mesh.V[va] + mesh.V[vb]);
  out.feasible = true;
  out.min_angle_deg = std::numeric_limits<double>::infinity();
  out.min_quality = std::numeric_limits<double>::infinity();
  for (int eadj : it->second) {
    if (eadj < 0 || eadj >= static_cast<int>(mesh.E.size())) continue;
    const auto &elem = mesh.E[eadj];
    if (elem.q_order != 1) continue;
    int vopp = -1;
    for (int k = 0; k < 3; ++k) {
      if (elem.v[k] != va && elem.v[k] != vb) {
        vopp = elem.v[k];
        break;
      }
    }
    if (vopp < 0) continue;
    ++out.adjacent_count;
    const Vec2 &opp = mesh.V[vopp];
    const double a1 = std::abs(triangleSignedAreaPts(opp, mesh.V[va], mid));
    const double a2 = std::abs(triangleSignedAreaPts(opp, mid, mesh.V[vb]));
    if (!(a1 > 1e-12) || !(a2 > 1e-12)) {
      out.feasible = false;
      out.min_angle_deg = 0.0;
      out.min_quality = 0.0;
      return out;
    }
    out.min_angle_deg = std::min(
        out.min_angle_deg,
        std::min(triangleMinAngleDegPts(opp, mesh.V[va], mid),
                 triangleMinAngleDegPts(opp, mid, mesh.V[vb])));
    out.min_quality = std::min(
        out.min_quality,
        std::min(triangleQualityPts(opp, mesh.V[va], mid),
                 triangleQualityPts(opp, mid, mesh.V[vb])));
  }
  if (out.adjacent_count == 0) {
    out.feasible = false;
    out.min_angle_deg = 0.0;
    out.min_quality = 0.0;
    return out;
  }
  if (!std::isfinite(out.min_angle_deg)) out.min_angle_deg = 0.0;
  if (!std::isfinite(out.min_quality)) out.min_quality = 0.0;
  return out;
}

int smoothQ1PatchVertices(Mesh &mesh,
                          const std::set<int> &patch_elems,
                          double min_angle_floor,
                          double min_quality_floor) {
  if (patch_elems.empty()) return 0;

  std::map<std::pair<int, int>, int> boundary_edge_groups;
  for (const auto &be : mesh.BE) {
    boundary_edge_groups[{std::min(be.v[0], be.v[1]),
                          std::max(be.v[0], be.v[1])}] = be.bIndex;
  }

  const auto boundary_vertices = collectBoundaryVertices(mesh);
  const auto periodic_vertices = collectPeriodicVertices(mesh);
  std::vector<std::vector<int>> node_to_elem(mesh.V.size());
  std::vector<std::set<int>> vertex_neighbors(mesh.V.size());
  std::set<int> patch_nodes;
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    if (e < 0 || e >= static_cast<int>(mesh.E.size())) continue;
    const auto &elem = mesh.E[e];
    if (elem.q_order != 1) continue;
    for (int k = 0; k < 3; ++k) {
      node_to_elem[elem.v[k]].push_back(e);
      if (patch_elems.count(e)) patch_nodes.insert(elem.v[k]);
    }
    vertex_neighbors[elem.v[0]].insert(elem.v[1]);
    vertex_neighbors[elem.v[0]].insert(elem.v[2]);
    vertex_neighbors[elem.v[1]].insert(elem.v[0]);
    vertex_neighbors[elem.v[1]].insert(elem.v[2]);
    vertex_neighbors[elem.v[2]].insert(elem.v[0]);
    vertex_neighbors[elem.v[2]].insert(elem.v[1]);
  }

  auto isWallBoundaryVertex = [&](int vid) {
    for (int nbr : vertex_neighbors[vid]) {
      const auto key = std::make_pair(std::min(vid, nbr), std::max(vid, nbr));
      auto it = boundary_edge_groups.find(key);
      if (it == boundary_edge_groups.end()) continue;
      const int g = it->second;
      if (g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
          lowerCopy(mesh.Bname[g]) == "wall") {
        return true;
      }
    }
    return false;
  };

  auto wallNeighborVertices = [&](int vid) {
    std::vector<int> nbrs;
    for (int nbr : vertex_neighbors[vid]) {
      const auto key = std::make_pair(std::min(vid, nbr), std::max(vid, nbr));
      auto it = boundary_edge_groups.find(key);
      if (it == boundary_edge_groups.end()) continue;
      const int g = it->second;
      if (g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
          lowerCopy(mesh.Bname[g]) == "wall") {
        nbrs.push_back(nbr);
      }
    }
    return nbrs;
  };

  std::vector<int> movable_vertices;
  for (int vid : patch_nodes) {
    if (periodic_vertices.count(vid)) continue;
    if (boundary_vertices.count(vid) && !isWallBoundaryVertex(vid)) continue;
    movable_vertices.push_back(vid);
  }
  if (movable_vertices.empty()) return 0;

  int move_count = 0;
  for (int iter = 0; iter < 2; ++iter) {
    bool iter_changed = false;
    for (int vid : movable_vertices) {
      std::set<int> local_patch_elems;
      for (int e : node_to_elem[vid]) {
        if (patch_elems.count(e)) local_patch_elems.insert(e);
      }
      if (local_patch_elems.empty()) continue;
      const SimplePatchQuality before =
          assessSimplePatchQuality(mesh, local_patch_elems);
      if (before.min_angle_deg >= min_angle_floor &&
          before.min_quality >= min_quality_floor) {
        continue;
      }

      const auto &nbrs = vertex_neighbors[vid];
      if (nbrs.size() < 2) continue;

      const Vec2 current = mesh.V[vid];
      Vec2 avg{0.0, 0.0};
      double avg_len = 0.0;
      int avg_count = 0;
      for (int nbr : nbrs) {
        avg = avg + mesh.V[nbr];
        avg_len += (mesh.V[nbr] - current).norm();
        ++avg_count;
      }
      avg = avg / static_cast<double>(avg_count);
      avg_len = std::max(avg_len / static_cast<double>(avg_count), 1e-8);

      std::vector<Vec2> candidates;
      if (boundary_vertices.count(vid) && isWallBoundaryVertex(vid)) {
        const auto wn = wallNeighborVertices(vid);
        if (wn.size() >= 2) {
          Vec2 arc_mid = current;
          if (bladeEdgeArcMidpoint(mesh.V[wn[0]], mesh.V[wn[1]], arc_mid)) {
            candidates.push_back(0.85 * current + 0.15 * arc_mid);
            candidates.push_back(0.70 * current + 0.30 * arc_mid);
            candidates.push_back(0.50 * current + 0.50 * arc_mid);
            candidates.push_back(arc_mid);
          }
        }

        const BladeProjection old_proj = projectToBladeSplineDetailed(current);
        const BladeProjection avg_proj = projectToBladeSplineDetailed(avg);
        if (avg_proj.valid &&
            (!old_proj.valid || old_proj.branch == avg_proj.branch)) {
          candidates.push_back(0.85 * current + 0.15 * avg_proj.pt);
          candidates.push_back(0.70 * current + 0.30 * avg_proj.pt);
          candidates.push_back(0.50 * current + 0.50 * avg_proj.pt);
          candidates.push_back(avg_proj.pt);
        }
      } else {
        candidates.push_back(0.85 * current + 0.15 * avg);
        candidates.push_back(0.70 * current + 0.30 * avg);
        candidates.push_back(0.50 * current + 0.50 * avg);
        for (int nbr : nbrs) {
          Vec2 d = current - mesh.V[nbr];
          const double dn = d.norm();
          if (dn <= 1e-12) continue;
          const Vec2 n_hat{-d.y / dn, d.x / dn};
          for (double scale : {0.02, 0.05, 0.10}) {
            const double step = scale * std::max(dn, avg_len);
            candidates.push_back(current + step * n_hat);
            candidates.push_back(current - step * n_hat);
          }
        }
      }

      Vec2 best_pos = current;
      double best_score = 30.0 * before.min_angle_deg + 1200.0 * before.min_quality;
      bool found_better = false;
      for (Vec2 candidate : candidates) {
        BladeProjection proj = projectToBladeSplineDetailed(candidate);
        if (proj.valid && pointInsideBladeSolid(candidate, 1e-10)) {
          candidate = movePointToFluidSide(candidate, proj, 1e-4);
        }
        if ((candidate - current).norm() <= 1e-10) continue;
        std::vector<Vec2> verts_trial = mesh.V;
        if (!localMovePreservesMesh(mesh, vid, node_to_elem, mesh.V,
                                    verts_trial, candidate)) {
          continue;
        }
        const Vec2 saved = mesh.V[vid];
        mesh.V[vid] = candidate;
        const SimplePatchQuality after =
            assessSimplePatchQuality(mesh, local_patch_elems);
        mesh.V[vid] = saved;
        const double score =
            30.0 * after.min_angle_deg + 1200.0 * after.min_quality;
        if (score > best_score + 1e-8) {
          best_score = score;
          best_pos = candidate;
          found_better = true;
        }
      }

      if (found_better) {
        mesh.V[vid] = best_pos;
        ++move_count;
        iter_changed = true;
      }
    }
    if (!iter_changed) break;
  }

  return move_count;
}

[[maybe_unused]] void applySeedParentPatchEdgeSwapCleanup(
    Mesh &mesh,
    const std::map<std::pair<int, int>, int> &boundary_edge_groups,
    const RefinementMap &rmap,
    const std::set<int> &seed_parents,
    const char *label,
    double min_angle_floor,
    double min_quality_floor,
    bool allow_periodic_vertices) {
  if (!q1ExtensionNeighborEdgeSwapsEnabled()) return;
  if (mesh.q_order_global > 1 || seed_parents.empty()) return;
  if (static_cast<int>(rmap.child_to_parent.size()) != static_cast<int>(mesh.E.size())) return;

  std::set<int> seed_elems;
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    if (mesh.E[e].q_order != 1) continue;
    if (seed_parents.count(rmap.child_to_parent[e])) {
      seed_elems.insert(e);
    }
  }
  if (seed_elems.empty()) return;

  std::map<std::pair<int, int>, std::vector<int>> edge_to_elem;
  std::vector<std::set<int>> elem_neighbors(mesh.E.size());
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    const auto &elem = mesh.E[e];
    if (elem.q_order != 1) continue;
    edge_to_elem[{std::min(elem.v[0], elem.v[1]), std::max(elem.v[0], elem.v[1])}].push_back(e);
    edge_to_elem[{std::min(elem.v[1], elem.v[2]), std::max(elem.v[1], elem.v[2])}].push_back(e);
    edge_to_elem[{std::min(elem.v[2], elem.v[0]), std::max(elem.v[2], elem.v[0])}].push_back(e);
  }
  for (const auto &[edge, elems] : edge_to_elem) {
    (void)edge;
    if (elems.size() != 2) continue;
    elem_neighbors[elems[0]].insert(elems[1]);
    elem_neighbors[elems[1]].insert(elems[0]);
  }

  std::set<int> patch_elems = seed_elems;
  for (int e : seed_elems) {
    for (int nb : elem_neighbors[e]) patch_elems.insert(nb);
  }
  if (patch_elems.size() < 2) return;

  const SimplePatchQuality before = assessSimplePatchQuality(mesh, patch_elems);
  const bool patch_below_floor =
      before.min_angle_deg < min_angle_floor ||
      before.min_quality < min_quality_floor;
  if (!patch_below_floor) return;

  auto isWallBoundaryVertex = [&](int vid) {
    for (const auto &be : mesh.BE) {
      if (be.v[0] != vid && be.v[1] != vid) continue;
      const int g = be.bIndex;
      if (g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
          lowerCopy(mesh.Bname[g]) == "wall") {
        return true;
      }
    }
    return false;
  };

  rebuildMeshEdgeConnectivity(mesh, boundary_edge_groups);
  mesh.appendPeriodicToIE();
  const int swap_count = improvePatchByEdgeSwaps(
      mesh, patch_elems, boundary_edge_groups, 8,
      isWallBoundaryVertex,
      allow_periodic_vertices);
  if (swap_count > 0) {
    rebuildMeshEdgeConnectivity(mesh, boundary_edge_groups);
    mesh.appendPeriodicToIE();
  }

  SimplePatchQuality after = assessSimplePatchQuality(mesh, patch_elems);
  int smooth_count = 0;
  if (after.min_angle_deg < min_angle_floor ||
      after.min_quality < min_quality_floor) {
    smooth_count =
        smoothQ1PatchVertices(mesh, patch_elems, min_angle_floor,
                              min_quality_floor);
    after = assessSimplePatchQuality(mesh, patch_elems);
  }

  if (swap_count > 0 || smooth_count > 0) {
    std::cerr << "    " << label << ": swaps=" << swap_count
              << " smooth_moves=" << smooth_count
              << " seed_parents=" << seed_parents.size()
              << " patch_elems=" << patch_elems.size()
              << " | min_angle " << before.min_angle_deg << " -> "
              << after.min_angle_deg
              << " | min_quality " << before.min_quality << " -> "
              << after.min_quality
              << std::endl;
  }
  if (after.min_angle_deg < min_angle_floor ||
      after.min_quality < min_quality_floor) {
    std::cerr << "    warning: " << label
              << " remains below local quality floor"
              << " (min_angle=" << after.min_angle_deg
              << ", min_quality=" << after.min_quality << ")"
              << std::endl;
  }
}

[[maybe_unused]] void applyExtensionNeighborEdgeSwapCleanup(
    Mesh &mesh,
    const std::map<std::pair<int, int>, int> &boundary_edge_groups,
    const RefinementMap &rmap,
    const std::set<int> &extension_refined_parents) {
  applySeedParentPatchEdgeSwapCleanup(
      mesh, boundary_edge_groups, rmap, extension_refined_parents,
      "extension-neighbor cleanup", q1ExtensionPatchMinAngleDeg(),
      q1ExtensionPatchMinQuality(), false);
}

RefinementMap bisectMarkedElementsImpl(
    Mesh &mesh, const std::vector<bool> &marked_in,
    const std::vector<int> &fallback_priority = {}, int target_adj_splits = 0) {
  RefinementMap rmap;
  int old_Ne = mesh.E.size();
  int old_Nv = mesh.V.size();
  const bool use_minimal_q1_rules = mesh.q_order_global <= 1;
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

  auto edgeIsUniqueShortestInElement = [&](int e, const std::pair<int, int> &edge) {
    int v[3] = {mesh.E[e].v[0], mesh.E[e].v[1], mesh.E[e].v[2]};
    const double l2[3] = {
        edgeLength2(v[0], v[1]),
        edgeLength2(v[1], v[2]),
        edgeLength2(v[2], v[0]),
    };
    const double min_l2 = std::min({l2[0], l2[1], l2[2]});
    const double max_l2 = std::max({l2[0], l2[1], l2[2]});
    if (max_l2 <= 0.0) return false;
    if (std::abs(max_l2 - min_l2) <= 1e-12 * std::max(1.0, max_l2)) return false;
    for (int i = 0; i < 3; ++i) {
      const auto ei = std::make_pair(std::min(v[i], v[(i + 1) % 3]),
                                     std::max(v[i], v[(i + 1) % 3]));
      if (ei != edge) continue;
      return std::abs(l2[i] - min_l2) <= 1e-12 * std::max(1.0, max_l2);
    }
    return false;
  };
  auto edgeIsUniqueShortestInAllAdjacentQ1Elements =
      [&](const std::pair<int, int> &edge) {
        bool found_adjacent_q1 = false;
        for (const auto &elem : mesh.E) {
          if (elem.q_order != 1) continue;
          const int vloc[3] = {elem.v[0], elem.v[1], elem.v[2]};
          bool has_edge = false;
          for (int i = 0; i < 3; ++i) {
            const auto elem_edge =
                std::make_pair(std::min(vloc[i], vloc[(i + 1) % 3]),
                               std::max(vloc[i], vloc[(i + 1) % 3]));
            if (elem_edge == edge) {
              has_edge = true;
              break;
            }
          }
          if (!has_edge) continue;
          found_adjacent_q1 = true;
          const double l2[3] = {
              edgeLength2(vloc[0], vloc[1]),
              edgeLength2(vloc[1], vloc[2]),
              edgeLength2(vloc[2], vloc[0]),
          };
          const double min_l2 = std::min({l2[0], l2[1], l2[2]});
          const double max_l2 = std::max({l2[0], l2[1], l2[2]});
          if (max_l2 <= 0.0) return false;
          if (std::abs(max_l2 - min_l2) <= 1e-12 * std::max(1.0, max_l2)) {
            return false;
          }
          bool this_elem_unique_shortest = false;
          for (int i = 0; i < 3; ++i) {
            const auto elem_edge =
                std::make_pair(std::min(vloc[i], vloc[(i + 1) % 3]),
                               std::max(vloc[i], vloc[(i + 1) % 3]));
            if (elem_edge != edge) continue;
            this_elem_unique_shortest =
                std::abs(l2[i] - min_l2) <= 1e-12 * std::max(1.0, max_l2);
            break;
          }
          if (!this_elem_unique_shortest) return false;
        }
        return found_adjacent_q1;
      };

  auto getLongestWallEdge = [&](int e) -> std::pair<int, int> {
    int v[3] = {mesh.E[e].v[0], mesh.E[e].v[1], mesh.E[e].v[2]};
    std::pair<int, int> best{-1, -1};
    double best_len2 = -1.0;
    for (int i = 0; i < 3; ++i) {
      const int a = v[i];
      const int b = v[(i + 1) % 3];
      const auto edge = std::make_pair(std::min(a, b), std::max(a, b));
      auto it = old_boundary_edge.find(edge);
      if (it == old_boundary_edge.end()) continue;
      const int g = it->second;
      if (g < 0 || g >= static_cast<int>(mesh.Bname.size()) ||
          lowerCopy(mesh.Bname[g]) != "wall") {
        continue;
      }
      Vec2 d = mesh.V[b] - mesh.V[a];
      const double len2 = d.x * d.x + d.y * d.y;
      if (len2 > best_len2) {
        best = edge;
        best_len2 = len2;
      }
    }
    return best;
  };

  auto getEncroachedWallEdge = [&](int e) -> std::pair<int, int> {
    int v[3] = {mesh.E[e].v[0], mesh.E[e].v[1], mesh.E[e].v[2]};
    std::pair<int, int> best{-1, -1};
    double best_score = -1.0;
    for (int i = 0; i < 3; ++i) {
      const int a = v[i];
      const int b = v[(i + 1) % 3];
      const int p = v[(i + 2) % 3];
      const auto edge = std::make_pair(std::min(a, b), std::max(a, b));
      auto it = old_boundary_edge.find(edge);
      if (it == old_boundary_edge.end()) continue;
      const int g = it->second;
      if (g < 0 || g >= static_cast<int>(mesh.Bname.size()) ||
          lowerCopy(mesh.Bname[g]) != "wall") {
        continue;
      }
      if (!diametralLensEncroached(mesh.V[a], mesh.V[b], mesh.V[p])) continue;
      const double angle = oppositeAngleDeg(mesh.V[a], mesh.V[b], mesh.V[p]);
      const double score = 1e6 * angle + edgeLength2(a, b);
      if (score > best_score) {
        best = edge;
        best_score = score;
      }
    }
    return best;
  };

  auto getPreferredRefinementEdge = [&](int e) -> std::pair<int, int> {
    if (use_minimal_q1_rules) {
      // In q1 wall-only mode, we restrict which elements may seed refinement,
      // but we still want classic longest-edge selection inside those elements.
      return getLE(e);
    }
    const auto encroached_wall = getEncroachedWallEdge(e);
    if (encroached_wall.first >= 0) return encroached_wall;
    const auto wall_edge = getLongestWallEdge(e);
    if (wall_edge.first >= 0) return wall_edge;
    return getLE(e);
  };

  auto isWallEdge = [&](const std::pair<int, int> &edge) {
    auto it = old_boundary_edge.find(edge);
    if (it == old_boundary_edge.end()) return false;
    const int g = it->second;
    return g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
           lowerCopy(mesh.Bname[g]) == "wall";
  };
  auto oldElementTouchesWallSimple = [&](int e) {
    int v[3] = {mesh.E[e].v[0], mesh.E[e].v[1], mesh.E[e].v[2]};
    for (int i = 0; i < 3; ++i) {
      if (isWallEdge({std::min(v[i], v[(i + 1) % 3]),
                      std::max(v[i], v[(i + 1) % 3])})) {
        return true;
      }
    }
    return false;
  };

  if (use_minimal_q1_rules && q1WallOnlyRefinementEnabled()) {
    for (int e = 0; e < old_Ne; ++e) {
      if (marked[e] && !oldElementTouchesWallSimple(e)) {
        marked[e] = false;
      }
    }
  }

  auto buildCurrentNodeToElem = [&]() {
    std::vector<std::vector<int>> cur_node_to_elem(mesh.V.size());
    for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
      cur_node_to_elem[mesh.E[e].v[0]].push_back(e);
      cur_node_to_elem[mesh.E[e].v[1]].push_back(e);
      cur_node_to_elem[mesh.E[e].v[2]].push_back(e);
    }
    return cur_node_to_elem;
  };

  auto buildCurrentEdgeToElem = [&]() {
    std::map<std::pair<int, int>, std::vector<int>> cur_edge_to_elem;
    for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
      const auto &elem = mesh.E[e];
      if (elem.q_order != 1) continue;
      cur_edge_to_elem[{std::min(elem.v[0], elem.v[1]),
                        std::max(elem.v[0], elem.v[1])}]
          .push_back(e);
      cur_edge_to_elem[{std::min(elem.v[1], elem.v[2]),
                        std::max(elem.v[1], elem.v[2])}]
          .push_back(e);
      cur_edge_to_elem[{std::min(elem.v[2], elem.v[0]),
                        std::max(elem.v[2], elem.v[0])}]
          .push_back(e);
    }
    return cur_edge_to_elem;
  };

  auto interiorMidpointAllowed = [&](int va, int vb) {
    (void)va;
    (void)vb;
    return true;
  };

  std::set<std::pair<int,int>> e2b;
  for(int e=0; e<old_Ne; e++) if(marked[e]) e2b.insert(getPreferredRefinementEdge(e));

  std::map<std::pair<int,int>, std::pair<int,int>> ppartner;
  for (const auto &ie : mesh.IE) {
    auto kL = std::make_pair(std::min(ie.v[0], ie.v[1]), std::max(ie.v[0], ie.v[1]));
    auto kR = std::make_pair(std::min(ie.vR[0], ie.vR[1]), std::max(ie.vR[0], ie.vR[1]));
    if (kL != kR) { ppartner[kL] = kR; ppartner[kR] = kL; }
  }

  std::map<std::pair<int, int>, int> periodic_edge_group;
  for (int gidx = 0; gidx < static_cast<int>(mesh.periodicGroups.size()); ++gidx) {
    std::vector<std::pair<int, int>> pairs = mesh.periodicGroups[gidx].pairs;
    for (auto &p : pairs) {
      if (mesh.V[p.first].y > mesh.V[p.second].y) {
        std::swap(p.first, p.second);
      }
    }
    std::stable_sort(
        pairs.begin(), pairs.end(),
        [&](const std::pair<int, int> &a, const std::pair<int, int> &b) {
          if (mesh.V[a.first].x != mesh.V[b.first].x) {
            return mesh.V[a.first].x < mesh.V[b.first].x;
          }
          return mesh.V[a.first].y < mesh.V[b.first].y;
        });
    for (int i = 0; i + 1 < static_cast<int>(pairs.size()); ++i) {
      const auto edge_bottom =
          std::make_pair(std::min(pairs[i].first, pairs[i + 1].first),
                         std::max(pairs[i].first, pairs[i + 1].first));
      const auto edge_top =
          std::make_pair(std::min(pairs[i].second, pairs[i + 1].second),
                         std::max(pairs[i].second, pairs[i + 1].second));
      periodic_edge_group[edge_bottom] = gidx;
      periodic_edge_group[edge_top] = gidx;
    }
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
  rmap.parent_split_reason.assign(old_Ne, kSplitNone);
  std::set<int> blocked_wall_cleanup_parents;
  std::vector<std::pair<std::pair<int, int>, int>> latest_created_midpoints;
  auto getMid = [&](int va, int vb, bool &ok) {
    latest_created_midpoints.clear();
    auto key = std::make_pair(std::min(va,vb), std::max(va,vb));
    if(edge_midpoint.count(key)) { ok = true; return edge_midpoint[key]; }
    const Vec2 straight_mid = (mesh.V[va] + mesh.V[vb]) * 0.5;
    Vec2 mid = straight_mid;
    int boundary_group = -1;
    if (old_boundary_edge.count(key)) {
      boundary_group = old_boundary_edge[key];
      if (boundary_group >= 0 &&
          boundary_group < static_cast<int>(mesh.Bname.size()) &&
          lowerCopy(mesh.Bname[boundary_group]) == "wall") {
        Vec2 snapped_mid = projectToBladeSpline(straight_mid);
        std::string split_mid_source = "projected straight midpoint";
        Vec2 arc_mid = snapped_mid;
        if (bladeEdgeArcMidpoint(mesh.V[va], mesh.V[vb], arc_mid)) {
          snapped_mid = arc_mid;
          split_mid_source = "blade arc midpoint";
        } else {
          Vec2 curved_mid = snapped_mid;
          if (curvedParentEdgeMidpoint(mesh, edge_to_elem, va, vb, curved_mid)) {
            snapped_mid = curved_mid;
            split_mid_source = "parent curved map midpoint";
          }
        }
        const auto cur_node_to_elem = buildCurrentNodeToElem();
        bool accepted = true;
        std::string wall_failure_reason;
        auto eit = edge_to_elem.find(key);
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

            std::string base_reason = localWallSplitPatchFailureReason(
                mesh, va, vb, vopp, snapped_mid, mesh.V[vopp], cur_node_to_elem);
            if (base_reason.empty()) continue;

            // Heuristic opposite-node repair is disabled for now. Keep the
            // wall split purely geometry-driven: if the direct curved-wall
            // patch check fails, reject the split and move on.
            wall_failure_reason = split_mid_source + " failed: " + base_reason;
            accepted = false;
            break;
          }
        }
        if (!accepted) {
          std::map<std::pair<int, int>, std::vector<int>> current_edge_to_elem;
          for (int ecur = 0; ecur < static_cast<int>(mesh.E.size()); ++ecur) {
            const auto &elem = mesh.E[ecur];
            if (elem.q_order != 1) continue;
            current_edge_to_elem[{std::min(elem.v[0], elem.v[1]),
                                  std::max(elem.v[0], elem.v[1])}]
                .push_back(ecur);
            current_edge_to_elem[{std::min(elem.v[1], elem.v[2]),
                                  std::max(elem.v[1], elem.v[2])}]
                .push_back(ecur);
            current_edge_to_elem[{std::min(elem.v[2], elem.v[0]),
                                  std::max(elem.v[2], elem.v[0])}]
                .push_back(ecur);
          }
          auto cur_eit = current_edge_to_elem.find(key);
          if (cur_eit != current_edge_to_elem.end()) {
            for (int eadj : cur_eit->second) {
              if (eadj < 0 ||
                  eadj >= static_cast<int>(rmap.child_to_parent.size())) {
                continue;
              }
              const int parent_adj = rmap.child_to_parent[eadj];
              if (parent_adj >= 0 && parent_adj < old_Ne) {
                blocked_wall_cleanup_parents.insert(parent_adj);
              }
            }
          }
          if (wall_failure_reason.empty()) {
            wall_failure_reason = split_mid_source + " failed";
          }
          std::cerr << "    warning: skipped refinement on wall edge (" << va
                    << ", " << vb << ") because " << wall_failure_reason
                    << std::endl;
          ok = false;
          return -1;
        }
        mid = snapped_mid;
      }
    }
    auto createMidpointVertex = [&](const std::pair<int, int> &edge,
                                    const Vec2 &xmid) {
      const int vid = static_cast<int>(mesh.V.size());
      mesh.V.push_back(xmid);
      edge_midpoint[edge] = vid;
      rmap.new_vertex_edges.push_back({edge.first, edge.second});
      latest_created_midpoints.push_back({edge, vid});
      return vid;
    };
    const int vm = createMidpointVertex(key, mid);
    if (boundary_group >= 0) {
      old_boundary_edge[{std::min(va, vm), std::max(va, vm)}] = boundary_group;
      old_boundary_edge[{std::min(vb, vm), std::max(vb, vm)}] = boundary_group;
    }
    if (ppartner.count(key) && !edge_midpoint.count(ppartner[key])) {
      const auto partner = ppartner[key];
      const Vec2 partner_mid =
          (mesh.V[partner.first] + mesh.V[partner.second]) * 0.5;
      createMidpointVertex(partner, partner_mid);
    }
    ok = true;
    return vm;
  };

  auto getGeomNodeQ2 = [&](int va, int vb, const Vec2 &xq) {
    auto key = std::make_pair(std::min(va, vb), std::max(va, vb));
    if (edge_geom_midpoint_q2.count(key)) return edge_geom_midpoint_q2[key];
    int vm = static_cast<int>(mesh.V.size());
    mesh.V.push_back(xq);
    edge_geom_midpoint_q2[key] = vm;
    return vm;
  };

  auto rollbackCreatedMidpoints = [&](const std::vector<std::pair<std::pair<int, int>, int>> &created) {
    for (auto it = created.rbegin(); it != created.rend(); ++it) {
      const auto &edge = it->first;
      const int vid = it->second;
      auto map_it = edge_midpoint.find(edge);
      if (map_it != edge_midpoint.end() && map_it->second == vid) {
        edge_midpoint.erase(map_it);
      }
      if (vid == static_cast<int>(mesh.V.size()) - 1) {
        mesh.V.pop_back();
      }
      if (!rmap.new_vertex_edges.empty()) {
        const auto &back = rmap.new_vertex_edges.back();
        const auto back_key =
            std::make_pair(std::min(back.first, back.second),
                           std::max(back.first, back.second));
        if (back_key == edge) {
          rmap.new_vertex_edges.pop_back();
        }
      }
      for (auto be_it = old_boundary_edge.begin(); be_it != old_boundary_edge.end();) {
        if (be_it->first.first == vid || be_it->first.second == vid) {
          be_it = old_boundary_edge.erase(be_it);
        } else {
          ++be_it;
        }
      }
    }
  };
  auto noteLatestCreatedMidpoints =
      [&](std::vector<std::pair<std::pair<int, int>, int>> &created) {
        for (const auto &entry : latest_created_midpoints) {
          bool seen = false;
          for (const auto &existing : created) {
            if (existing.first == entry.first && existing.second == entry.second) {
              seen = true;
              break;
            }
          }
          if (!seen && edge_midpoint.count(entry.first) &&
              edge_midpoint[entry.first] == entry.second) {
            created.push_back(entry);
          }
        }
  };
  auto createTrackedAuxVertex =
      [&](const Vec2 &x, const std::pair<int, int> &ref_edge,
          std::vector<int> &created_aux_vertices) {
        const int vid = static_cast<int>(mesh.V.size());
        mesh.V.push_back(x);
        rmap.new_vertex_edges.push_back(ref_edge);
        created_aux_vertices.push_back(vid);
        return vid;
      };
  auto rollbackCreatedAuxVertices = [&](const std::vector<int> &created_aux_vertices) {
    for (auto it = created_aux_vertices.rbegin(); it != created_aux_vertices.rend(); ++it) {
      const int vid = *it;
      for (auto be_it = old_boundary_edge.begin(); be_it != old_boundary_edge.end();) {
        if (be_it->first.first == vid || be_it->first.second == vid) {
          be_it = old_boundary_edge.erase(be_it);
        } else {
          ++be_it;
        }
      }
      if (vid == static_cast<int>(mesh.V.size()) - 1) {
        mesh.V.pop_back();
      }
      if (!rmap.new_vertex_edges.empty()) {
        rmap.new_vertex_edges.pop_back();
      }
    }
  };

  auto elementTouchesWall = [&](int a, int b, int c) {
    const std::pair<int, int> edges[3] = {
      {std::min(a, b), std::max(a, b)},
      {std::min(b, c), std::max(b, c)},
      {std::min(c, a), std::max(c, a)}
    };
    for (const auto &edge : edges) {
      auto it = old_boundary_edge.find(edge);
      if (it == old_boundary_edge.end()) continue;
      int g = it->second;
      if (g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
          lowerCopy(mesh.Bname[g]) == "wall") {
        return true;
      }
    }
    return false;
  };

  std::vector<bool> old_elem_touches_wall(old_Ne, false);
  std::vector<std::array<int, 3>> old_elem_vertices(old_Ne);
  for (int e = 0; e < old_Ne; ++e) {
    old_elem_vertices[e] = {
        mesh.E[e].v[0],
        mesh.E[e].v[1],
        mesh.E[e].v[2],
    };
    old_elem_touches_wall[e] =
        elementTouchesWall(mesh.E[e].v[0], mesh.E[e].v[1], mesh.E[e].v[2]);
  };
  std::vector<int> old_elem_wall_distance(old_Ne, std::numeric_limits<int>::max());
  std::vector<std::vector<int>> old_adj(old_Ne);
  std::vector<int> old_wall_degree(old_Nv, 0);
  {
    std::map<std::pair<int, int>, std::vector<int>> old_edge_to_elem;
    for (int e = 0; e < old_Ne; ++e) {
      const auto &vv = old_elem_vertices[e];
      const std::pair<int, int> edges[3] = {
          {std::min(vv[0], vv[1]), std::max(vv[0], vv[1])},
          {std::min(vv[1], vv[2]), std::max(vv[1], vv[2])},
          {std::min(vv[2], vv[0]), std::max(vv[2], vv[0])},
      };
      for (const auto &edge : edges) old_edge_to_elem[edge].push_back(e);
    }
    for (const auto &[edge, elems] : old_edge_to_elem) {
      if (elems.size() != 2) continue;
      const int a = elems[0];
      const int b = elems[1];
      old_adj[a].push_back(b);
      old_adj[b].push_back(a);
    }
    for (const auto &[edge, bgroup] : old_boundary_edge) {
      if (bgroup < 0 || bgroup >= static_cast<int>(mesh.Bname.size()) ||
          lowerCopy(mesh.Bname[bgroup]) != "wall") {
        continue;
      }
      old_wall_degree[edge.first]++;
      old_wall_degree[edge.second]++;
    }
    std::deque<int> q;
    for (int e = 0; e < old_Ne; ++e) {
      if (!old_elem_touches_wall[e]) continue;
      old_elem_wall_distance[e] = 0;
      q.push_back(e);
    }
    while (!q.empty()) {
      const int e = q.front();
      q.pop_front();
      for (int nb : old_adj[e]) {
        if (old_elem_wall_distance[nb] <= old_elem_wall_distance[e] + 1) continue;
        old_elem_wall_distance[nb] = old_elem_wall_distance[e] + 1;
        q.push_back(nb);
      }
    }
  }
  std::set<std::pair<int, int>> rejected_edges;
  std::set<std::pair<int, int>> cavity_requested_edges;
  std::set<std::pair<int, int>> endpoint_fan_requested_edges;
  std::set<std::pair<int, int>> endpoint_fan_failed_edges;
  std::set<std::pair<int, int>> gradation_requested_edges;
  std::set<int> extension_refined_parents;
  std::set<int> special_cleanup_parents;
  // Track which original cells are "included" (initially marked + any fallback cells
  // added later). Used to count adjoint-driven splits for the fallback stopping criterion.
  std::vector<bool> included(old_Ne, false);
  for (int e = 0; e < old_Ne; e++) if (marked[e]) included[e] = true;
  int fb_cursor = 0;

  // Outer loop: run the refinement, then add fallback cells if the target wasn't met.
  for (;;) {
    changed = true;
    while(changed) {
      changed = false;
      int cur_Ne = mesh.E.size();
      const std::array<Vec2, 3> ref_corner = {{
          {0.0, 0.0},
          {1.0, 0.0},
          {0.0, 1.0},
      }};
      auto elementHasWallCandidate = [&](int e) {
        if (e < 0 || e >= static_cast<int>(mesh.E.size())) return false;
        int vloc[3] = {mesh.E[e].v[0], mesh.E[e].v[1], mesh.E[e].v[2]};
        std::pair<int, int> eloc[3] = {
            {std::min(vloc[0], vloc[1]), std::max(vloc[0], vloc[1])},
            {std::min(vloc[1], vloc[2]), std::max(vloc[1], vloc[2])},
            {std::min(vloc[2], vloc[0]), std::max(vloc[2], vloc[0])},
        };
        bool has_requested_candidate = false;
        for (int i = 0; i < 3; ++i) {
          if (!e2b.count(eloc[i])) continue;
          has_requested_candidate = true;
          auto it = old_boundary_edge.find(eloc[i]);
          if (it == old_boundary_edge.end()) continue;
          int g = it->second;
          if (g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
              lowerCopy(mesh.Bname[g]) == "wall") {
            return true;
          }
        }
        if (has_requested_candidate) return false;
        for (int i = 0; i < 3; ++i) {
          if (!edge_midpoint.count(eloc[i])) continue;
          auto it = old_boundary_edge.find(eloc[i]);
          if (it == old_boundary_edge.end()) continue;
          int g = it->second;
          if (g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
              lowerCopy(mesh.Bname[g]) == "wall") {
            return true;
          }
        }
        return false;
      };
      std::vector<int> elem_order(cur_Ne);
      std::iota(elem_order.begin(), elem_order.end(), 0);
      if (!use_minimal_q1_rules) {
        std::stable_sort(elem_order.begin(), elem_order.end(),
                         [&](int a, int b) {
                           const bool aw = elementHasWallCandidate(a);
                           const bool bw = elementHasWallCandidate(b);
                           if (aw != bw) return aw > bw;
                           return a < b;
                         });
      }
      auto currentEdgeOppositeVertices = [&](int a, int b) {
        std::vector<int> opp;
        for (int ee = 0; ee < static_cast<int>(mesh.E.size()); ++ee) {
          const auto &elem = mesh.E[ee];
          bool has_a = false, has_b = false;
          int vopp = -1;
          for (int k = 0; k < 3; ++k) {
            if (elem.v[k] == a) has_a = true;
            else if (elem.v[k] == b) has_b = true;
            else vopp = elem.v[k];
          }
          if (has_a && has_b && vopp >= 0) opp.push_back(vopp);
        }
        std::sort(opp.begin(), opp.end());
        opp.erase(std::unique(opp.begin(), opp.end()), opp.end());
        return opp;
      };
      auto interiorSplitEdgeEncroached = [&](int a, int b) {
        if (use_minimal_q1_rules) return false;
        const auto opp = currentEdgeOppositeVertices(a, b);
        for (int p : opp) {
          if (p == a || p == b) continue;
          if (diametralLensEncroached(mesh.V[a], mesh.V[b], mesh.V[p])) {
            return true;
          }
        }
        return false;
      };
      struct Q1WallCavityRequest {
        bool valid = false;
        int wall_elem = -1;
        std::pair<int, int> wall_edge{-1, -1};
        std::pair<int, int> support_edge{-1, -1};
      };
      auto edgeMidpoint = [&](const std::pair<int, int> &edge) {
        return 0.5 * (mesh.V[edge.first] + mesh.V[edge.second]);
      };
      auto edgesShareVertex = [&](const std::pair<int, int> &a,
                                  const std::pair<int, int> &b) {
        return a.first == b.first || a.first == b.second ||
               a.second == b.first || a.second == b.second;
      };
      auto edgeContainsVertex = [&](const std::pair<int, int> &edge, int vtx) {
        return edge.first == vtx || edge.second == vtx;
      };
      const int endpoint_override_hops = q1EndpointOverrideHopRadius();
      std::vector<int> endpoint_override_distance(mesh.V.size(),
                                                  std::numeric_limits<int>::max());
      if (endpoint_override_hops > 0 && !mesh.V.empty()) {
        std::vector<std::vector<int>> node_adj(mesh.V.size());
        auto append_adj = [&](int a, int b) {
          if (a < 0 || b < 0 || a >= static_cast<int>(node_adj.size()) ||
              b >= static_cast<int>(node_adj.size()) || a == b) {
            return;
          }
          node_adj[a].push_back(b);
          node_adj[b].push_back(a);
        };
        for (const auto &elem : mesh.E) {
          append_adj(elem.v[0], elem.v[1]);
          append_adj(elem.v[1], elem.v[2]);
          append_adj(elem.v[2], elem.v[0]);
        }
        std::queue<int> frontier;
        const int endpoint_scan_max =
            std::min(old_Nv, static_cast<int>(mesh.V.size()));
        for (int vid = 0; vid < endpoint_scan_max; ++vid) {
          if (old_wall_degree[vid] != 1) continue;
          endpoint_override_distance[vid] = 0;
          frontier.push(vid);
        }
        while (!frontier.empty()) {
          const int cur = frontier.front();
          frontier.pop();
          if (endpoint_override_distance[cur] >= endpoint_override_hops) continue;
          for (int nbr : node_adj[cur]) {
            if (nbr < 0 ||
                nbr >= static_cast<int>(endpoint_override_distance.size())) {
              continue;
            }
            if (endpoint_override_distance[nbr] <=
                endpoint_override_distance[cur] + 1) {
              continue;
            }
            endpoint_override_distance[nbr] =
                endpoint_override_distance[cur] + 1;
            frontier.push(nbr);
          }
        }
      }
      auto edgeInEndpointOverrideBand =
          [&](const std::pair<int, int> &edge) {
            if (endpoint_override_hops <= 0 || endpoint_override_distance.empty()) {
              return false;
            }
            auto endpoint_band_distance = [&](int vid) {
              if (vid < 0 ||
                  vid >= static_cast<int>(endpoint_override_distance.size())) {
                return std::numeric_limits<int>::max();
              }
              return endpoint_override_distance[vid];
            };
            return endpoint_band_distance(edge.first) <= endpoint_override_hops ||
                   endpoint_band_distance(edge.second) <= endpoint_override_hops;
          };
      auto wallEndpointNodeOnEdge = [&](const std::pair<int, int> &edge) {
        const bool first_is_endpoint =
            edge.first >= 0 && edge.first < old_Nv && old_wall_degree[edge.first] == 1;
        const bool second_is_endpoint =
            edge.second >= 0 && edge.second < old_Nv && old_wall_degree[edge.second] == 1;
        if (first_is_endpoint && !second_is_endpoint) return edge.first;
        if (second_is_endpoint && !first_is_endpoint) return edge.second;
        if (first_is_endpoint) return edge.first;
        if (second_is_endpoint) return edge.second;
        return -1;
      };
      auto isEndpointFanCandidateEdge = [&](const std::pair<int, int> &edge) {
        return endpoint_fan_requested_edges.count(edge);
      };
      auto noteEndpointFanFailure = [&](const std::pair<int, int> &edge) {
        if (!endpoint_fan_requested_edges.count(edge) &&
            !endpoint_fan_failed_edges.count(edge)) {
          return;
        }
        endpoint_fan_requested_edges.erase(edge);
        endpoint_fan_failed_edges.insert(edge);
      };
      auto endpointStarOutlierNodeOnEdge = [&](const std::pair<int, int> &edge) {
        std::map<std::pair<int, int>, std::vector<int>> current_edge_to_elem;
        for (int ecur = 0; ecur < static_cast<int>(mesh.E.size()); ++ecur) {
          const auto &elem = mesh.E[ecur];
          const std::pair<int, int> edges[3] = {
              {std::min(elem.v[0], elem.v[1]), std::max(elem.v[0], elem.v[1])},
              {std::min(elem.v[1], elem.v[2]), std::max(elem.v[1], elem.v[2])},
              {std::min(elem.v[2], elem.v[0]), std::max(elem.v[2], elem.v[0])},
          };
          for (const auto &elem_edge : edges) {
            current_edge_to_elem[elem_edge].push_back(ecur);
          }
        }
        auto maxNeighborArea = [&](int elem_idx) {
          double max_area = 0.0;
          const auto &elem = mesh.E[elem_idx];
          const std::pair<int, int> edges[3] = {
              {std::min(elem.v[0], elem.v[1]), std::max(elem.v[0], elem.v[1])},
              {std::min(elem.v[1], elem.v[2]), std::max(elem.v[1], elem.v[2])},
              {std::min(elem.v[2], elem.v[0]), std::max(elem.v[2], elem.v[0])},
          };
          for (const auto &elem_edge : edges) {
            auto eit = current_edge_to_elem.find(elem_edge);
            if (eit == current_edge_to_elem.end()) continue;
            for (int nb : eit->second) {
              if (nb == elem_idx) continue;
              max_area = std::max(
                  max_area, std::abs(triangleSignedArea(mesh.E[nb], mesh.V)));
            }
          }
          return max_area;
        };
        auto eit = current_edge_to_elem.find(edge);
        if (eit == current_edge_to_elem.end()) return -1;
        for (int elem_idx : eit->second) {
          if (elem_idx < 0 || elem_idx >= static_cast<int>(mesh.E.size())) continue;
          const auto &elem = mesh.E[elem_idx];
          if (elem.q_order != 1) continue;
          int endpoint_node = -1;
          for (int k = 0; k < 3; ++k) {
            const int vid = elem.v[k];
            if (vid >= 0 && vid < old_Nv && old_wall_degree[vid] == 1) {
              endpoint_node = vid;
              break;
            }
          }
          if (endpoint_node < 0) continue;
          const double elem_area = std::abs(triangleSignedArea(elem, mesh.V));
          const double nb_max_area = maxNeighborArea(elem_idx);
          if (nb_max_area <= 1e-14) continue;
          if (elem_area / nb_max_area >= q1EndpointFanAreaRatioThreshold()) {
            return endpoint_node;
          }
        }
        return -1;
      };
      auto minEndpointDistance2 = [&](const std::pair<int, int> &a,
                                      const std::pair<int, int> &b) {
        return std::min({
            edgeLength2(a.first, b.first),
            edgeLength2(a.first, b.second),
            edgeLength2(a.second, b.first),
            edgeLength2(a.second, b.second),
        });
      };
      auto chooseEndpointFanSupportEdgeForVertices =
          [&](const int vv[3], const std::pair<int, int> &wall_edge) {
            const int endpoint_node = wallEndpointNodeOnEdge(wall_edge);
            if (endpoint_node < 0) return std::make_pair(-1, -1);
            const std::pair<int, int> edges[3] = {
                {std::min(vv[0], vv[1]), std::max(vv[0], vv[1])},
                {std::min(vv[1], vv[2]), std::max(vv[1], vv[2])},
                {std::min(vv[2], vv[0]), std::max(vv[2], vv[0])},
            };
            std::pair<int, int> opposite_edge{-1, -1};
            std::pair<int, int> endpoint_edge{-1, -1};
            for (const auto &candidate : edges) {
              if (candidate == wall_edge || isWallEdge(candidate) ||
                  rejected_edges.count(candidate)) {
                continue;
              }
              auto be_it = old_boundary_edge.find(candidate);
              if (be_it != old_boundary_edge.end()) continue;
              if (edgeContainsVertex(candidate, endpoint_node)) {
                if (endpoint_edge.first < 0) endpoint_edge = candidate;
              } else {
                opposite_edge = candidate;
              }
            }
            if (opposite_edge.first >= 0) return opposite_edge;
            if (endpoint_edge.first >= 0) return endpoint_edge;
            return std::make_pair(-1, -1);
          };
      auto collectEndpointFanSupportEdges =
          [&](int endpoint_node, const std::pair<int, int> &wall_edge) {
            std::vector<std::pair<int, int>> supports;
            if (endpoint_node < 0 || endpoint_node >= old_Nv) return supports;
            auto current_node_to_elem = buildCurrentNodeToElem();
            std::map<std::pair<int, int>, std::vector<int>> current_edge_to_elem;
            for (int ecur = 0; ecur < static_cast<int>(mesh.E.size()); ++ecur) {
              const auto &elem = mesh.E[ecur];
              const std::pair<int, int> edges[3] = {
                  {std::min(elem.v[0], elem.v[1]), std::max(elem.v[0], elem.v[1])},
                  {std::min(elem.v[1], elem.v[2]), std::max(elem.v[1], elem.v[2])},
                  {std::min(elem.v[2], elem.v[0]), std::max(elem.v[2], elem.v[0])},
              };
              for (const auto &edge : edges) current_edge_to_elem[edge].push_back(ecur);
            }
            auto maxNeighborArea = [&](int elem_idx) {
              double max_area = 0.0;
              const auto &elem = mesh.E[elem_idx];
              const std::pair<int, int> edges[3] = {
                  {std::min(elem.v[0], elem.v[1]), std::max(elem.v[0], elem.v[1])},
                  {std::min(elem.v[1], elem.v[2]), std::max(elem.v[1], elem.v[2])},
                  {std::min(elem.v[2], elem.v[0]), std::max(elem.v[2], elem.v[0])},
              };
              for (const auto &edge : edges) {
                auto eit = current_edge_to_elem.find(edge);
                if (eit == current_edge_to_elem.end()) continue;
                for (int nb : eit->second) {
                  if (nb == elem_idx) continue;
                  max_area = std::max(
                      max_area, std::abs(triangleSignedArea(mesh.E[nb], mesh.V)));
                }
              }
              return max_area;
            };
            auto bestSupportAssessment =
                [&](const std::pair<int, int> &candidate) {
                  Q1SplitPatchAssessment out;
                  auto eit = current_edge_to_elem.find(candidate);
                  if (eit == current_edge_to_elem.end()) return out;
                  const Vec2 mid =
                      0.5 * (mesh.V[candidate.first] + mesh.V[candidate.second]);
                  out.feasible = true;
                  out.min_angle_deg = std::numeric_limits<double>::infinity();
                  out.min_quality = std::numeric_limits<double>::infinity();
                  for (int elem_idx : eit->second) {
                    if (elem_idx < 0 ||
                        elem_idx >= static_cast<int>(mesh.E.size())) {
                      continue;
                    }
                    const auto &elem = mesh.E[elem_idx];
                    if (elem.q_order != 1) continue;
                    int opp = -1;
                    for (int k = 0; k < 3; ++k) {
                      if (elem.v[k] != candidate.first &&
                          elem.v[k] != candidate.second) {
                        opp = elem.v[k];
                        break;
                      }
                    }
                    if (opp < 0) continue;
                    ++out.adjacent_count;
                    const double a1 = std::abs(
                        triangleSignedAreaPts(mesh.V[opp], mesh.V[candidate.first], mid));
                    const double a2 = std::abs(
                        triangleSignedAreaPts(mesh.V[opp], mid, mesh.V[candidate.second]));
                    if (!(a1 > 1e-12) || !(a2 > 1e-12)) {
                      out.feasible = false;
                      out.min_angle_deg = 0.0;
                      out.min_quality = 0.0;
                      return out;
                    }
                    out.min_angle_deg = std::min(
                        out.min_angle_deg,
                        std::min(triangleMinAngleDegPts(
                                     mesh.V[opp], mesh.V[candidate.first], mid),
                                 triangleMinAngleDegPts(
                                     mesh.V[opp], mid, mesh.V[candidate.second])));
                    out.min_quality = std::min(
                        out.min_quality,
                        std::min(triangleQualityPts(
                                     mesh.V[opp], mesh.V[candidate.first], mid),
                                 triangleQualityPts(
                                     mesh.V[opp], mid, mesh.V[candidate.second])));
                  }
                  if (out.adjacent_count == 0) {
                    out.feasible = false;
                    out.min_angle_deg = 0.0;
                    out.min_quality = 0.0;
                  }
                  if (!std::isfinite(out.min_angle_deg)) out.min_angle_deg = 0.0;
                  if (!std::isfinite(out.min_quality)) out.min_quality = 0.0;
                  return out;
                };
            std::set<std::pair<int, int>> seen;
            for (int elem_idx : current_node_to_elem[endpoint_node]) {
              if (elem_idx < 0 || elem_idx >= static_cast<int>(mesh.E.size())) continue;
              const auto &elem = mesh.E[elem_idx];
              if (elem.q_order != 1) continue;
              const double elem_area = std::abs(triangleSignedArea(elem, mesh.V));
              const double nb_max_area = maxNeighborArea(elem_idx);
              if (nb_max_area <= 1e-14) continue;
              const double area_ratio = elem_area / nb_max_area;
              if (area_ratio < q1EndpointFanAreaRatioThreshold()) continue;
              const std::pair<int, int> edges[3] = {
                  {std::min(elem.v[0], elem.v[1]), std::max(elem.v[0], elem.v[1])},
                  {std::min(elem.v[1], elem.v[2]), std::max(elem.v[1], elem.v[2])},
                  {std::min(elem.v[2], elem.v[0]), std::max(elem.v[2], elem.v[0])},
              };
              std::pair<int, int> best{-1, -1};
              int best_priority = std::numeric_limits<int>::max();
              double best_len2 = std::numeric_limits<double>::infinity();
              double best_score = -std::numeric_limits<double>::infinity();
              for (const auto &candidate : edges) {
                if (candidate == wall_edge || isWallEdge(candidate)) {
                  continue;
                }
                auto be_it = old_boundary_edge.find(candidate);
                if (be_it != old_boundary_edge.end()) continue;
                const Q1SplitPatchAssessment assessment =
                    bestSupportAssessment(candidate);
                if (!assessment.feasible ||
                    assessment.min_angle_deg < q1ClosureSplitMinAngleDeg() ||
                    assessment.min_quality < q1ClosureSplitMinQuality()) {
                  continue;
                }
                const bool contains_endpoint =
                    edgeContainsVertex(candidate, endpoint_node);
                const int priority = contains_endpoint ? 1 : 0;
                const double len2 = edgeLength2(candidate.first, candidate.second);
                const double score =
                    30.0 * assessment.min_angle_deg +
                    1200.0 * assessment.min_quality;
                if (best.first < 0 || priority < best_priority ||
                    (priority == best_priority &&
                     (score > best_score + 1e-8 ||
                      (std::abs(score - best_score) <= 1e-8 &&
                       len2 < best_len2 - 1e-12)))) {
                  best = candidate;
                  best_priority = priority;
                  best_len2 = len2;
                  best_score = score;
                }
              }
              if (best.first >= 0 && seen.insert(best).second) {
                supports.push_back(best);
              }
            }
            return supports;
          };
      auto enqueueEndpointFanSupports =
          [&](int endpoint_node, const std::pair<int, int> &wall_edge,
              const char *context_label) {
            if (endpoint_node < 0) return false;
            const auto endpoint_supports =
                collectEndpointFanSupportEdges(endpoint_node, wall_edge);
            bool inserted_any = false;
            for (const auto &endpoint_support : endpoint_supports) {
              if (endpoint_fan_failed_edges.count(endpoint_support)) continue;
              if (edge_midpoint.count(endpoint_support)) continue;
              rejected_edges.erase(endpoint_support);
              const bool inserted = e2b.insert(endpoint_support).second;
              endpoint_fan_requested_edges.insert(endpoint_support);
              if (ppartner.count(endpoint_support)) {
                e2b.insert(ppartner[endpoint_support]);
              }
              inserted_any = inserted_any || inserted;
            }
            if (inserted_any) {
              changed = true;
              std::cerr << "    q1 endpoint fan closure: requested "
                        << endpoint_supports.size()
                        << " support edge(s) for " << context_label << " ("
                        << wall_edge.first << ", " << wall_edge.second << ")"
                        << std::endl;
            }
            return inserted_any;
          };
      auto enqueueLocalGradationSupports =
          [&](const std::set<int> &seed_nodes, const char *context_label) {
            const int max_supports = q1LocalGradingMaxSupports();
            if (max_supports <= 0 || seed_nodes.empty()) return false;

            const auto current_node_to_elem = buildCurrentNodeToElem();
            const auto current_edge_to_elem = buildCurrentEdgeToElem();
            auto elemArea = [&](int elem_idx) {
              return std::abs(triangleSignedArea(mesh.E[elem_idx], mesh.V));
            };
            auto maxNeighborArea = [&](int elem_idx) {
              double max_area = 0.0;
              const auto &elem = mesh.E[elem_idx];
              const std::pair<int, int> edges[3] = {
                  {std::min(elem.v[0], elem.v[1]), std::max(elem.v[0], elem.v[1])},
                  {std::min(elem.v[1], elem.v[2]), std::max(elem.v[1], elem.v[2])},
                  {std::min(elem.v[2], elem.v[0]), std::max(elem.v[2], elem.v[0])},
              };
              for (const auto &edge : edges) {
                auto eit = current_edge_to_elem.find(edge);
                if (eit == current_edge_to_elem.end()) continue;
                for (int nb : eit->second) {
                  if (nb == elem_idx) continue;
                  max_area = std::max(max_area, elemArea(nb));
                }
              }
              return max_area;
            };

            std::set<int> candidate_elems;
            for (int vid : seed_nodes) {
              if (vid < 0 || vid >= static_cast<int>(current_node_to_elem.size())) {
                continue;
              }
              for (int elem_idx : current_node_to_elem[vid]) {
                candidate_elems.insert(elem_idx);
                const auto &elem = mesh.E[elem_idx];
                for (int k = 0; k < 3; ++k) {
                  const int nbr_vid = elem.v[k];
                  if (nbr_vid < 0 ||
                      nbr_vid >= static_cast<int>(current_node_to_elem.size())) {
                    continue;
                  }
                  for (int nbr_elem : current_node_to_elem[nbr_vid]) {
                    candidate_elems.insert(nbr_elem);
                  }
                }
              }
            }

            struct GradingSupportRequest {
              double area_ratio = 0.0;
              double score = 0.0;
              int elem = -1;
              std::pair<int, int> edge{-1, -1};
            };
            std::vector<GradingSupportRequest> requests;
            const double area_ratio_floor = q1LocalGradingAreaRatio();
            for (int elem_idx : candidate_elems) {
              if (elem_idx < 0 || elem_idx >= static_cast<int>(mesh.E.size())) {
                continue;
              }
              const auto &elem = mesh.E[elem_idx];
              if (elem.q_order != 1) continue;

              const double elem_area = elemArea(elem_idx);
              const double nb_max_area = maxNeighborArea(elem_idx);
              if (nb_max_area <= 1e-14) continue;
              const double area_ratio = elem_area / nb_max_area;
              if (area_ratio < area_ratio_floor) continue;

              const std::pair<int, int> elem_edges[3] = {
                  {std::min(elem.v[0], elem.v[1]), std::max(elem.v[0], elem.v[1])},
                  {std::min(elem.v[1], elem.v[2]), std::max(elem.v[1], elem.v[2])},
                  {std::min(elem.v[2], elem.v[0]), std::max(elem.v[2], elem.v[0])},
              };

              std::pair<int, int> best_edge{-1, -1};
              double best_score = -std::numeric_limits<double>::infinity();
              for (const auto &candidate : elem_edges) {
                if (isWallEdge(candidate) || edge_midpoint.count(candidate) ||
                    rejected_edges.count(candidate)) {
                  continue;
                }
                auto be_it = old_boundary_edge.find(candidate);
                if (be_it != old_boundary_edge.end() && !ppartner.count(candidate)) {
                  continue;
                }
                auto eit = current_edge_to_elem.find(candidate);
                if (eit == current_edge_to_elem.end()) continue;

                double shared_fine_ratio = 0.0;
                bool borders_finer_neighbor = false;
                for (int nb : eit->second) {
                  if (nb == elem_idx) continue;
                  const double nb_area = elemArea(nb);
                  if (nb_area <= 1e-14) continue;
                  if (elem_area / nb_area >= area_ratio_floor) {
                    borders_finer_neighbor = true;
                    shared_fine_ratio =
                        std::max(shared_fine_ratio, elem_area / nb_area);
                  }
                }

                const Q1SplitPatchAssessment assessment =
                    assessQ1SplitPatch(mesh, candidate.first, candidate.second,
                                       current_edge_to_elem);
                if (!assessment.feasible ||
                    assessment.min_angle_deg < q1ClosureSplitMinAngleDeg() ||
                    assessment.min_quality < q1ClosureSplitMinQuality()) {
                  continue;
                }

                const double score =
                    (borders_finer_neighbor ? 1000.0 : 0.0) +
                    80.0 * shared_fine_ratio +
                    25.0 * assessment.min_angle_deg +
                    900.0 * assessment.min_quality +
                    1e-6 * edgeLength2(candidate.first, candidate.second);
                if (score > best_score + 1e-8) {
                  best_score = score;
                  best_edge = candidate;
                }
              }

              if (best_edge.first >= 0) {
                requests.push_back(
                    {area_ratio, best_score, elem_idx, best_edge});
              }
            }

            std::sort(requests.begin(), requests.end(),
                      [](const GradingSupportRequest &a,
                         const GradingSupportRequest &b) {
                        if (a.area_ratio != b.area_ratio) {
                          return a.area_ratio > b.area_ratio;
                        }
                        if (a.score != b.score) return a.score > b.score;
                        if (a.elem != b.elem) return a.elem < b.elem;
                        return a.edge < b.edge;
                      });

            bool inserted_any = false;
            int inserted_count = 0;
            std::set<std::pair<int, int>> seen_edges;
            for (const auto &req : requests) {
              if (inserted_count >= max_supports) break;
              if (!seen_edges.insert(req.edge).second) continue;
              const bool inserted = e2b.insert(req.edge).second;
              gradation_requested_edges.insert(req.edge);
              if (ppartner.count(req.edge)) {
                const auto &partner = ppartner[req.edge];
                e2b.insert(partner);
                gradation_requested_edges.insert(partner);
              }
              inserted_any = inserted_any || inserted;
              if (inserted) ++inserted_count;
            }

            if (inserted_any) {
              changed = true;
              std::cerr << "    q1 local gradation closure: requested "
                        << inserted_count << " support edge(s) for "
                        << context_label << std::endl;
            }
            return inserted_any;
          };
      auto isEndpointFanSupportEdge =
          [&](const std::pair<int, int> &wall_edge,
              const std::pair<int, int> &support_edge) {
            const int endpoint_node = wallEndpointNodeOnEdge(wall_edge);
            return endpoint_node >= 0 &&
                   support_edge.first >= 0;
          };
      auto getWallEdgesForOldElement = [&](int e) {
        std::vector<std::pair<int, int>> wall_edges;
        if (e < 0 || e >= old_Ne) return wall_edges;
        const auto &vv = old_elem_vertices[e];
        const std::pair<int, int> edges[3] = {
            {std::min(vv[0], vv[1]), std::max(vv[0], vv[1])},
            {std::min(vv[1], vv[2]), std::max(vv[1], vv[2])},
            {std::min(vv[2], vv[0]), std::max(vv[2], vv[0])},
        };
        for (const auto &edge : edges) {
          if (isWallEdge(edge)) wall_edges.push_back(edge);
        }
        return wall_edges;
      };
      auto chooseWallCavitySupportEdge =
          [&](int wall_elem, const std::pair<int, int> &wall_edge,
              const std::pair<int, int> &blocked_edge) {
            const int endpoint_vv[3] = {
                old_elem_vertices[wall_elem][0],
                old_elem_vertices[wall_elem][1],
                old_elem_vertices[wall_elem][2],
            };
            const auto endpoint_support =
                chooseEndpointFanSupportEdgeForVertices(endpoint_vv, wall_edge);
            if (endpoint_support.first >= 0) return endpoint_support;
            std::pair<int, int> best{-1, -1};
            const Vec2 blocked_mid = edgeMidpoint(blocked_edge);
            double best_mid_gap2 = std::numeric_limits<double>::infinity();
            double best_len2 = -1.0;
            bool best_shares_blocked = false;
            const auto &vv = old_elem_vertices[wall_elem];
            const std::pair<int, int> edges[3] = {
                {std::min(vv[0], vv[1]), std::max(vv[0], vv[1])},
                {std::min(vv[1], vv[2]), std::max(vv[1], vv[2])},
                {std::min(vv[2], vv[0]), std::max(vv[2], vv[0])},
            };
            for (const auto &candidate : edges) {
              if (candidate == wall_edge || isWallEdge(candidate) ||
                  rejected_edges.count(candidate)) {
                continue;
              }
              const bool shares_blocked = edgesShareVertex(candidate, blocked_edge);
              const double mid_gap2 = (edgeMidpoint(candidate) - blocked_mid).normSq();
              const double len2 = edgeLength2(candidate.first, candidate.second);
              if (best.first < 0 ||
                  shares_blocked > best_shares_blocked ||
                  (shares_blocked == best_shares_blocked &&
                   (mid_gap2 < best_mid_gap2 - 1e-12 ||
                    (std::abs(mid_gap2 - best_mid_gap2) <= 1e-12 &&
                     len2 > best_len2)))) {
                best = candidate;
                best_shares_blocked = shares_blocked;
                best_mid_gap2 = mid_gap2;
                best_len2 = len2;
              }
            }
            if (best.first >= 0) return best;
            if (blocked_edge != wall_edge && !isWallEdge(blocked_edge) &&
                !rejected_edges.count(blocked_edge)) {
              return blocked_edge;
            }
            return best;
          };
      auto buildQ1WallCavityRequest = [&](const std::pair<int, int> &edge) {
        Q1WallCavityRequest best;
        int best_rank = std::numeric_limits<int>::max();
        bool best_shares_blocked = false;
        double best_endpoint_gap2 = std::numeric_limits<double>::infinity();
        double best_mid_gap2 = std::numeric_limits<double>::infinity();
        double best_len2 = -1.0;
        const Vec2 blocked_mid = edgeMidpoint(edge);
        std::set<int> candidate_wall_elems;
        auto eit = edge_to_elem.find(edge);
        if (eit == edge_to_elem.end()) return best;
        for (int eadj : eit->second) {
          if (eadj < 0 || eadj >= old_Ne) continue;
          if (old_elem_touches_wall[eadj]) candidate_wall_elems.insert(eadj);
          for (int nb : old_adj[eadj]) {
            if (nb >= 0 && nb < old_Ne && old_elem_touches_wall[nb]) {
              candidate_wall_elems.insert(nb);
            }
          }
        }
        for (int wall_elem : candidate_wall_elems) {
          const int rank = old_elem_wall_distance[wall_elem];
          for (const auto &wall_edge : getWallEdgesForOldElement(wall_elem)) {
            if (wall_edge == edge || rejected_edges.count(wall_edge)) continue;
            const bool shares_blocked = edgesShareVertex(wall_edge, edge);
            const double endpoint_gap2 = minEndpointDistance2(wall_edge, edge);
            const double mid_gap2 = (edgeMidpoint(wall_edge) - blocked_mid).normSq();
            const double len2 = edgeLength2(wall_edge.first, wall_edge.second);
            const auto support_edge =
                chooseWallCavitySupportEdge(wall_elem, wall_edge, edge);
            if (!best.valid || rank < best_rank ||
                (rank == best_rank &&
                 (shares_blocked > best_shares_blocked ||
                  (shares_blocked == best_shares_blocked &&
                   (endpoint_gap2 < best_endpoint_gap2 - 1e-12 ||
                    (std::abs(endpoint_gap2 - best_endpoint_gap2) <= 1e-12 &&
                     (mid_gap2 < best_mid_gap2 - 1e-12 ||
                      (std::abs(mid_gap2 - best_mid_gap2) <= 1e-12 &&
                       len2 > best_len2)))))))) {
              best.valid = true;
              best_rank = rank;
              best_shares_blocked = shares_blocked;
              best_endpoint_gap2 = endpoint_gap2;
              best_mid_gap2 = mid_gap2;
              best_len2 = len2;
              best.wall_elem = wall_elem;
              best.wall_edge = wall_edge;
              best.support_edge = support_edge;
            }
          }
        }
        return best;
      };
      auto enqueueForcedPeriodicPartnerSplits =
          [&](const std::vector<std::pair<std::pair<int, int>, int>> &created_midpoints,
              const std::pair<int, int> current_edges[3]) {
            for (const auto &entry : created_midpoints) {
              const auto &edge = entry.first;
              if (!ppartner.count(edge) || !edge_midpoint.count(edge)) continue;
              bool belongs_to_current_elem = false;
              for (int i = 0; i < 3; ++i) {
                if (current_edges[i] == edge) {
                  belongs_to_current_elem = true;
                  break;
                }
              }
              if (belongs_to_current_elem) continue;
              rejected_edges.erase(edge);
              e2b.insert(edge);
            }
          };
      for (int e : elem_order) {
        int v[3] = {mesh.E[e].v[0], mesh.E[e].v[1], mesh.E[e].v[2]};
        std::pair<int,int> ee[3] = {{std::min(v[0],v[1]), std::max(v[0],v[1])}, {std::min(v[1],v[2]), std::max(v[1],v[2])}, {std::min(v[2],v[0]), std::max(v[2],v[0])}};
        const bool touches_wall = elementTouchesWall(v[0], v[1], v[2]);
        const int parent = rmap.child_to_parent[e];
        struct EdgeCandidate {
          bool requested = false;
          double len2 = 0.0;
          int local_edge = -1;
        };
        std::vector<EdgeCandidate> candidates;
        bool has_requested_candidate = false;
        bool requested_edge_flags[3] = {false, false, false};
        for (int i = 0; i < 3; i++) {
          if (e2b.count(ee[i])) {
            has_requested_candidate = true;
            requested_edge_flags[i] = true;
            candidates.push_back({true, edgeLength2(v[i], v[(i + 1) % 3]), i});
          }
        }
        if (!has_requested_candidate) {
          bool has_conformity_candidate = false;
          for (int i = 0; i < 3; i++) {
            if (edge_midpoint.count(ee[i])) {
              has_conformity_candidate = true;
              candidates.push_back({true, edgeLength2(v[i], v[(i + 1) % 3]), i});
            }
          }
          if (!has_conformity_candidate) {
            candidates.clear();
          }
        }
        if(candidates.empty()) continue;
        std::sort(candidates.begin(), candidates.end(),
                  [](const EdgeCandidate &a, const EdgeCandidate &b) {
                    if (a.requested != b.requested) return a.requested > b.requested;
                    return a.len2 > b.len2;
                  });
        const bool conformity_only_candidates = !has_requested_candidate;

        bool split_done = false;
        bool accepted_q1_simple_split = false;
        std::set<int> accepted_q1_seed_nodes;
        for (const auto &candidate : candidates) {
          const int t = candidate.local_edge;
          const bool explicit_endpoint_fan_candidate =
              isEndpointFanCandidateEdge(ee[t]);
          const int endpoint_node_on_edge = wallEndpointNodeOnEdge(ee[t]);
          const int endpoint_star_node = endpointStarOutlierNodeOnEdge(ee[t]);
          const bool endpoint_influence_edge =
              edgeInEndpointOverrideBand(ee[t]);
          const bool endpoint_star_outlier_edge = endpoint_star_node >= 0;
          const bool wall_influence_edge =
              touches_wall ||
              (parent >= 0 && parent < old_Ne &&
               old_elem_wall_distance[parent] >= 0 &&
               old_elem_wall_distance[parent] <= 1);
          const bool endpoint_related_edge =
              endpoint_node_on_edge >= 0 || explicit_endpoint_fan_candidate ||
              endpoint_star_outlier_edge || endpoint_influence_edge;
          if (rejected_edges.count(ee[t]) && !explicit_endpoint_fan_candidate) continue;
          if (explicit_endpoint_fan_candidate) {
            rejected_edges.erase(ee[t]);
          }
          const bool selected_edge_is_wall = isWallEdge(ee[t]);
          const bool edge_midpoint_preexisting = edge_midpoint.count(ee[t]) > 0;
          const int va = v[t];
          const int vb = v[(t+1)%3];
          const int vc = v[(t+2)%3];
          bool used_endpoint_polygon_override = false;
          bool used_near_wall_polygon_override = false;
          Q1SplitPatchAssessment endpoint_override_assessment;
          if (use_minimal_q1_rules && selected_edge_is_wall) {
            const int endpoint_node = wallEndpointNodeOnEdge(ee[t]);
            enqueueEndpointFanSupports(endpoint_node, ee[t], "wall edge");
          }
          if (use_minimal_q1_rules && q1UniqueShortestEdgeVetoEnabled() &&
              !touches_wall && !selected_edge_is_wall &&
              !edge_midpoint_preexisting && !ppartner.count(ee[t]) &&
              !explicit_endpoint_fan_candidate &&
              !endpoint_star_outlier_edge &&
              !endpoint_influence_edge &&
              edgeIsUniqueShortestInAllAdjacentQ1Elements(ee[t])) {
            e2b.erase(ee[t]);
            rejected_edges.insert(ee[t]);
            std::cerr << "    warning: skipped refinement on edge ("
                      << ee[t].first << ", " << ee[t].second
                      << ") because it is the unique shortest edge of an interior cell"
                      << std::endl;
            continue;
          }
          if (use_minimal_q1_rules && !selected_edge_is_wall &&
              !ppartner.count(ee[t])) {
            const std::string guard_reason =
                q1InteriorSplitBladeGuardFailureReason(
                    mesh, va, vb, edge_to_elem, Q1FluidSideMode::env_selected);
            if (!guard_reason.empty()) {
              std::string endpoint_override_reject_reason;
              const bool allow_polygon_override =
                  q1EndpointPolygonOverrideEnabled() &&
                  (endpoint_related_edge || wall_influence_edge);
              if (allow_polygon_override) {
                const auto current_edge_to_elem = buildCurrentEdgeToElem();
                const std::string polygon_guard_reason =
                    q1InteriorSplitBladeGuardFailureReason(
                        mesh, va, vb, current_edge_to_elem,
                        Q1FluidSideMode::polygon_solid);
                if (polygon_guard_reason.empty()) {
                  endpoint_override_assessment =
                      assessQ1SplitPatch(mesh, va, vb, current_edge_to_elem);
                  if (endpoint_override_assessment.feasible &&
                      endpoint_override_assessment.min_angle_deg >=
                          q1EndpointOverrideMinAngleDeg() &&
                      endpoint_override_assessment.min_quality >=
                          q1EndpointOverrideMinQuality()) {
                    used_endpoint_polygon_override = endpoint_related_edge;
                    used_near_wall_polygon_override =
                        !endpoint_related_edge && wall_influence_edge;
                    rejected_edges.erase(ee[t]);
                    if (ppartner.count(ee[t])) {
                      rejected_edges.erase(ppartner[ee[t]]);
                    }
                    std::cerr << "    "
                              << (used_endpoint_polygon_override
                                      ? "endpoint override"
                                      : "near-wall polygon override")
                              << ": accepted edge ("
                              << ee[t].first << ", " << ee[t].second
                              << ") after polygon blade check"
                              << " | legacy_reason=" << guard_reason
                              << " | min_angle="
                              << endpoint_override_assessment.min_angle_deg
                              << " | min_quality="
                              << endpoint_override_assessment.min_quality
                              << std::endl;
                  } else {
                    std::ostringstream oss;
                    oss << "local patch quality too weak"
                        << " (min_angle="
                        << endpoint_override_assessment.min_angle_deg
                        << ", min_quality="
                        << endpoint_override_assessment.min_quality << ")";
                    endpoint_override_reject_reason = oss.str();
                  }
                } else {
                  endpoint_override_reject_reason =
                      "polygon guard also rejected the split";
                }
              }
              if (used_endpoint_polygon_override ||
                  used_near_wall_polygon_override) {
                // Accept this endpoint-related split despite the legacy guard.
              } else {
              e2b.erase(ee[t]);
              rejected_edges.insert(ee[t]);
              if (ppartner.count(ee[t])) {
                e2b.erase(ppartner[ee[t]]);
                rejected_edges.insert(ppartner[ee[t]]);
              }
              if (endpoint_related_edge) noteEndpointFanFailure(ee[t]);
              Q1WallCavityRequest cavity_request;
              const bool allow_nested_cavity_promotion =
                  !cavity_requested_edges.count(ee[t]);
              if (!edge_midpoint_preexisting && allow_nested_cavity_promotion) {
                cavity_request = buildQ1WallCavityRequest(ee[t]);
                if (cavity_request.valid) {
                  if (!rejected_edges.count(cavity_request.wall_edge)) {
                    e2b.insert(cavity_request.wall_edge);
                    cavity_requested_edges.insert(cavity_request.wall_edge);
                    if (ppartner.count(cavity_request.wall_edge)) {
                      e2b.insert(ppartner[cavity_request.wall_edge]);
                    }
                  }
                  if (cavity_request.support_edge.first >= 0 &&
                      !(isEndpointFanSupportEdge(cavity_request.wall_edge,
                                                 cavity_request.support_edge) &&
                        endpoint_fan_failed_edges.count(
                            cavity_request.support_edge)) &&
                      (!rejected_edges.count(cavity_request.support_edge) ||
                       isEndpointFanSupportEdge(cavity_request.wall_edge,
                                                cavity_request.support_edge))) {
                    if (isEndpointFanSupportEdge(cavity_request.wall_edge,
                                                 cavity_request.support_edge)) {
                      rejected_edges.erase(cavity_request.support_edge);
                    }
                    e2b.insert(cavity_request.support_edge);
                    cavity_requested_edges.insert(cavity_request.support_edge);
                    if (isEndpointFanSupportEdge(cavity_request.wall_edge,
                                                 cavity_request.support_edge)) {
                      endpoint_fan_requested_edges.insert(
                          cavity_request.support_edge);
                    }
                    if (ppartner.count(cavity_request.support_edge)) {
                      e2b.insert(ppartner[cavity_request.support_edge]);
                    }
                  }
                  const int endpoint_node =
                      wallEndpointNodeOnEdge(cavity_request.wall_edge);
                  enqueueEndpointFanSupports(endpoint_node,
                                             cavity_request.wall_edge,
                                             "wall edge");
                }
                if (endpoint_star_node >= 0) {
                  enqueueEndpointFanSupports(endpoint_star_node,
                                             cavity_request.valid
                                                 ? cavity_request.wall_edge
                                                 : std::make_pair(-1, -1),
                      "endpoint star");
                }
              }
              if (cavity_request.valid) changed = true;
              std::cerr << "    warning: skipped refinement on edge ("
                        << ee[t].first << ", " << ee[t].second << ") because "
                        << guard_reason;
              if (!edge_midpoint_preexisting) {
                if (cavity_request.valid) {
                  std::cerr << "; promoted q1 wall cavity with wall edge ("
                            << cavity_request.wall_edge.first << ", "
                            << cavity_request.wall_edge.second << ")";
                  if (cavity_request.support_edge.first >= 0) {
                    std::cerr << " and support edge ("
                              << cavity_request.support_edge.first << ", "
                              << cavity_request.support_edge.second << ")";
                  }
                }
                if (!allow_nested_cavity_promotion &&
                    cavity_requested_edges.count(ee[t])) {
                  std::cerr << "; recursive wall-cavity promotion suppressed";
                }
              } else {
                std::cerr << "; midpoint already existed, so no cavity promotion was applied";
              }
              if (!endpoint_override_reject_reason.empty()) {
                std::cerr << "; endpoint override not used because "
                          << endpoint_override_reject_reason;
              }
              std::cerr << std::endl;
              if (touches_wall) break;
              continue;
              }
            }
          }
          // Heuristic blocked-wall-patch deferral is disabled. If a wall split
          // fails, nearby interior edges are still allowed to compete normally.
          const bool parent_marked =
              parent >= 0 && parent < old_Ne && marked[parent];
          const auto current_edge = ee[t];
          auto classifyAcceptedSplitReason = [&](bool used_wall_fan) {
            if (used_wall_fan) return static_cast<unsigned char>(kSplitWallFan);
            if (used_endpoint_polygon_override ||
                used_near_wall_polygon_override) {
              return static_cast<unsigned char>(kSplitEndpointOverride);
            }
            if (endpoint_fan_requested_edges.count(current_edge)) {
              return static_cast<unsigned char>(kSplitEndpointFan);
            }
            if (cavity_requested_edges.count(current_edge)) {
              return static_cast<unsigned char>(kSplitWallCavity);
            }
            if (gradation_requested_edges.count(current_edge)) {
              return static_cast<unsigned char>(kSplitGrading);
            }
            if (ppartner.count(current_edge)) {
              return static_cast<unsigned char>(kSplitPeriodic);
            }
            if (conformity_only_candidates && edge_midpoint_preexisting) {
              return static_cast<unsigned char>(kSplitConformity);
            }
            if (parent_marked) return static_cast<unsigned char>(kSplitMarked);
            return static_cast<unsigned char>(kSplitFallback);
          };

          if (ppartner.count(ee[t]) && !edge_midpoint.count(ppartner[ee[t]])) {
            const auto partner = ppartner[ee[t]];
            if (!interiorMidpointAllowed(ee[t].first, ee[t].second) ||
                !interiorMidpointAllowed(partner.first, partner.second)) {
              e2b.erase(ee[t]);
              e2b.erase(partner);
              rejected_edges.insert(ee[t]);
              rejected_edges.insert(partner);
              if (endpoint_related_edge) noteEndpointFanFailure(ee[t]);
              if (touches_wall) break;
              continue;
            }
          }

          bool mid_ok = false;
          std::vector<std::pair<std::pair<int, int>, int>> created_midpoints;
          int vm = getMid(va, vb, mid_ok);
          if (mid_ok) noteLatestCreatedMidpoints(created_midpoints);
          if (!mid_ok) {
            bool fan_trisect_accepted = false;
            if (use_minimal_q1_rules && selected_edge_is_wall &&
                mesh.E[e].q_order == 1 && q1WallFanTrisectionEnabled()) {
              Vec2 wall_p1 = (2.0 * mesh.V[va] + mesh.V[vb]) / 3.0;
              Vec2 wall_p2 = (mesh.V[va] + 2.0 * mesh.V[vb]) / 3.0;
              if (!bladeEdgeArcFractionPoint(mesh.V[va], mesh.V[vb], 1.0 / 3.0, wall_p1)) {
                wall_p1 = projectToBladeSpline(wall_p1);
              }
              if (!bladeEdgeArcFractionPoint(mesh.V[va], mesh.V[vb], 2.0 / 3.0, wall_p2)) {
                wall_p2 = projectToBladeSpline(wall_p2);
              }
              const auto cur_node_to_elem = buildCurrentNodeToElem();
              const std::string fan_reason =
                  q1WallFanTrisectionFailureReason(mesh, va, vb, vc, wall_p1, wall_p2,
                                                   cur_node_to_elem);
              if (fan_reason.empty()) {
                std::vector<int> created_aux_vertices;
                const auto ref_edge = std::make_pair(std::min(va, vb), std::max(va, vb));
                const int vm1 = createTrackedAuxVertex(wall_p1, ref_edge, created_aux_vertices);
                const int vm2 = createTrackedAuxVertex(wall_p2, ref_edge, created_aux_vertices);
                auto fan_children =
                    buildQ1WallFanChildren(mesh.V, va, vb, vc, vm1, vm2);
                if (fan_children.local_children_valid) {
                  mesh.E[e] = std::move(fan_children.updated_parent);
                  mesh.E.push_back(std::move(fan_children.new_child_1));
                  mesh.E.push_back(std::move(fan_children.new_child_2));
                  const int g = old_boundary_edge[ee[t]];
                  old_boundary_edge[{std::min(va, vm1), std::max(va, vm1)}] = g;
                  old_boundary_edge[{std::min(vm1, vm2), std::max(vm1, vm2)}] = g;
                  old_boundary_edge[{std::min(vm2, vb), std::max(vm2, vb)}] = g;
                  rmap.child_to_parent.push_back(parent);
                  rmap.child_to_parent.push_back(parent);
                  rmap.parent_split_reason[parent] =
                      classifyAcceptedSplitReason(true);
                  special_cleanup_parents.insert(parent);
                  std::cerr << "    accepted wall fan trisect: e=" << e
                            << " parent=" << parent
                            << " edge=(" << ee[t].first << ", " << ee[t].second << ")"
                            << " source=" << (parent_marked ? "marked" : "fallback")
                            << " reason="
                            << parentSplitReasonName(rmap.parent_split_reason[parent])
                            << std::endl;
                  e2b.erase(ee[t]);
                  changed = true;
                  split_done = true;
                  fan_trisect_accepted = true;
                } else {
                  rollbackCreatedAuxVertices(created_aux_vertices);
                }
              } else {
                std::cerr << "    warning: wall fan trisect on edge ("
                          << ee[t].first << ", " << ee[t].second
                          << ") rejected because " << fan_reason << std::endl;
              }
            }
            if (fan_trisect_accepted) {
              break;
            }
            e2b.erase(ee[t]);
            rejected_edges.insert(ee[t]);
            if (ppartner.count(ee[t])) {
              e2b.erase(ppartner[ee[t]]);
              rejected_edges.insert(ppartner[ee[t]]);
            }
            if (endpoint_related_edge) noteEndpointFanFailure(ee[t]);
            if (touches_wall) break;
            continue;
          }
          const int parent_q_order = mesh.E[e].q_order;
          const Vec2 ra = ref_corner[t];
          const Vec2 rb = ref_corner[(t + 1) % 3];
          const Vec2 rc = ref_corner[(t + 2) % 3];
          const Vec2 rm = 0.5 * (ra + rb);
          const auto parent_map = [&](const Vec2 &r) {
            return mesh.evaluateElementGeometry(e, r.x, r.y).x;
          };
          if (!selected_edge_is_wall && interiorSplitEdgeEncroached(va, vb)) {
            rollbackCreatedMidpoints(created_midpoints);
            std::cerr << "    warning: skipped refinement on edge ("
                      << ee[t].first << ", " << ee[t].second
                      << ") because the edge is encroached by an adjacent vertex"
                      << std::endl;
            e2b.erase(ee[t]);
            rejected_edges.insert(ee[t]);
            if (ppartner.count(ee[t])) {
              e2b.erase(ppartner[ee[t]]);
              rejected_edges.insert(ppartner[ee[t]]);
            }
            if (endpoint_related_edge) noteEndpointFanFailure(ee[t]);
            if (touches_wall) break;
            continue;
          }
          const bool use_true_blade_wall_subdivision =
              parent_q_order == 2 && selected_edge_is_wall &&
              trueBladeWallSubdivisionEnabled();
          if (parent_q_order == 2 && !use_true_blade_wall_subdivision) {
            // For curved-parent refinement, new child corner nodes should lie on
            // the restricted parent geometry map, not on a straight endpoint
            // average. This preserves the parent q=2 geometry under bisection.
            mesh.V[vm] = parent_map(rm);
          }
          Q2BuildCallbacks q2_build_callbacks{
              parent_map,
              [&](int p, int q, const Vec2 &xq) {
                return getGeomNodeQ2(p, q, xq);
              },
              [&](const RefinementEdgeKey &edge) { return isWallEdge(edge); },
              [&](int p, int q) {
                Vec2 xq = 0.5 * (mesh.V[p] + mesh.V[q]);
                if (!bladeEdgeArcMidpoint(mesh.V[p], mesh.V[q], xq)) {
                  xq = projectToBladeSpline(xq);
                }
                return xq;
              }};

          bool use_q2_quality_driven_full_subdivision = false;
          if (parent_q_order == 2 && !selected_edge_is_wall &&
              edge_midpoint_preexisting) {
            const int wall_distance_limit = q2ConformityWallDistanceLimit();
            const bool within_wall_cavity =
                wall_distance_limit < 0 ||
                (parent >= 0 && parent < old_Ne &&
                 old_elem_wall_distance[parent] <= wall_distance_limit);
            const int other_i = (t + 1) % 3;
            const int other_j = (t + 2) % 3;
            const auto edge_i = ee[other_i];
            const auto edge_j = ee[other_j];
            const bool can_insert_edge_i =
                isWallEdge(edge_i) || edge_midpoint.count(edge_i) ||
                !interiorSplitEdgeEncroached(edge_i.first, edge_i.second);
            const bool can_insert_edge_j =
                isWallEdge(edge_j) || edge_midpoint.count(edge_j) ||
                !interiorSplitEdgeEncroached(edge_j.first, edge_j.second);
            const Vec2 xAB = parent_map(0.5 * (ref_corner[0] + ref_corner[1]));
            const Vec2 xBC = parent_map(0.5 * (ref_corner[1] + ref_corner[2]));
            const Vec2 xCA = parent_map(0.5 * (ref_corner[2] + ref_corner[0]));
            const Vec2 xSelected = parent_map(rm);

            use_q2_quality_driven_full_subdivision =
                shouldPromoteQ2ConformitySplit({
                    within_wall_cavity,
                    can_insert_edge_i,
                    can_insert_edge_j,
                    mesh.V[v[0]],
                    mesh.V[v[1]],
                    mesh.V[v[2]],
                    xAB,
                    xBC,
                    xCA,
                    xSelected,
                });
          }
          const bool use_q2_wall_control_subdivision =
              parent_q_order == 2 && selected_edge_is_wall;
          const bool rebuild_wall_children_from_blade =
              use_q2_wall_control_subdivision &&
              use_true_blade_wall_subdivision;
          if (use_q2_wall_control_subdivision ||
              use_q2_quality_driven_full_subdivision) {
            // Cardoze/Moxey-style local control subdivision for wall-selected
            // q2 elements and red-green-style full subdivision for quality-poor
            // q2 conformity splits: split all three parent edges and create the
            // four child triangles induced by the reference-space midpoint
            // subdivision. This gives the wall split immediate first-layer
            // support and avoids skinny green splits on short shared edges.
            const int a = v[0];
            const int b = v[1];
            const int c = v[2];
            const Vec2 rA = ref_corner[0];
            const Vec2 rB = ref_corner[1];
            const Vec2 rC = ref_corner[2];
            const Vec2 rAB = 0.5 * (rA + rB);
            const Vec2 rBC = 0.5 * (rB + rC);
            const Vec2 rCA = 0.5 * (rC + rA);

            auto getParentMappedMidpoint = [&](int p, int q, const Vec2 &rpq,
                                               bool &ok_mid) -> int {
              const auto edge = std::make_pair(std::min(p, q), std::max(p, q));
              int mid_idx = getMid(p, q, ok_mid);
              if (!ok_mid) return -1;
              noteLatestCreatedMidpoints(created_midpoints);
              if (!(use_true_blade_wall_subdivision && isWallEdge(edge))) {
                mesh.V[mid_idx] = parent_map(rpq);
              }
              return mid_idx;
            };

            bool ab_ok = false, bc_ok = false, ca_ok = false;
            const int mAB = getParentMappedMidpoint(a, b, rAB, ab_ok);
            const int mBC = getParentMappedMidpoint(b, c, rBC, bc_ok);
            const int mCA = getParentMappedMidpoint(c, a, rCA, ca_ok);
            if (!ab_ok || !bc_ok || !ca_ok) {
              rollbackCreatedMidpoints(created_midpoints);
              if (touches_wall) break;
              continue;
            }

            const bool local_children_valid =
                q2FullSubdivisionChildrenHavePositiveArea(mesh.V, a, b, c,
                                                          mAB, mBC, mCA);
            if (!local_children_valid) {
              std::cerr << "    warning: skipped wall subdivision on edge ("
                        << ee[t].first << ", " << ee[t].second
                        << ") because a child became non-positive locally"
                        << std::endl;
              rollbackCreatedMidpoints(created_midpoints);
              e2b.erase(ee[t]);
              rejected_edges.insert(ee[t]);
              if (ppartner.count(ee[t])) {
                e2b.erase(ppartner[ee[t]]);
                rejected_edges.insert(ppartner[ee[t]]);
              }
              if (endpoint_related_edge) noteEndpointFanFailure(ee[t]);
              if (touches_wall) break;
              continue;
            }

            auto q2_children = buildQ2FullSubdivisionChildren(
                a, b, c, mAB, mBC, mCA, ref_corner,
                rebuild_wall_children_from_blade, q2_build_callbacks);

            mesh.E[e] = std::move(q2_children.c0);
            mesh.E.push_back(std::move(q2_children.c1));
            mesh.E.push_back(std::move(q2_children.c2));
            mesh.E.push_back(std::move(q2_children.c3));
            // Disabled for now: snapping only the new wall q2 midsides to the
            // true blade is geometrically cleaner, but on the current
            // coarse-start `test_q2.gri` it destabilizes the solve around the
            // first accepted wall subdivision because the untouched wall
            // corner nodes still lie on the older approximate wall shape.
            //
            // const int c1_idx = static_cast<int>(mesh.E.size()) - 3;
            // const int c2_idx = static_cast<int>(mesh.E.size()) - 2;
            // const int c3_idx = static_cast<int>(mesh.E.size()) - 1;
            // const std::set<int> patch_elems = {e, c1_idx, c2_idx, c3_idx};
            // retuneWallPatchQ2Geometry(mesh, patch_elems, old_boundary_edge);
            rmap.child_to_parent.push_back(parent);
            rmap.child_to_parent.push_back(parent);
            rmap.child_to_parent.push_back(parent);
            std::cerr << "    accepted "
                      << (use_q2_wall_control_subdivision
                              ? "wall subdivision"
                              : "full q2 conformity subdivision")
                      << ": e=" << e
                      << " parent=" << parent
                      << " edge=(" << ee[t].first << ", " << ee[t].second << ")"
                      << " source=" << (parent_marked ? "marked" : "fallback")
                      << " children=4 periodic=" << (ppartner.count(ee[t]) ? "yes" : "no")
                      << std::endl;
            enqueueForcedPeriodicPartnerSplits(created_midpoints, ee);
            e2b.erase(ee[0]);
            e2b.erase(ee[1]);
            e2b.erase(ee[2]);
            changed = true;
            split_done = true;
            break;
          } else if (parent_q_order == 2) {
            const bool local_children_valid =
                q2BinaryChildrenHavePositiveArea(mesh.V, va, vb, vc, vm);
            if (!local_children_valid) {
              std::cerr << "    warning: skipped refinement on edge ("
                        << ee[t].first << ", " << ee[t].second
                        << ") because a child became non-positive locally"
                        << std::endl;
              rollbackCreatedMidpoints(created_midpoints);
              e2b.erase(ee[t]);
              rejected_edges.insert(ee[t]);
              if (ppartner.count(ee[t])) {
                e2b.erase(ppartner[ee[t]]);
                rejected_edges.insert(ppartner[ee[t]]);
              }
              if (endpoint_related_edge) noteEndpointFanFailure(ee[t]);
              if (touches_wall) break;
              continue;
            }
            auto q2_children = buildQ2BinarySplitChildren(
                va, vb, vc, vm, ra, rb, rc, rm, q2_build_callbacks);
            mesh.E[e] = std::move(q2_children.updated_parent);
            mesh.E.push_back(std::move(q2_children.new_child));
          } else {
            auto q1_children = buildQ1SplitChildren(mesh.V, va, vb, vc, vm);
            if (!q1_children.local_children_valid) {
              std::cerr << "    warning: skipped refinement on edge ("
                        << ee[t].first << ", " << ee[t].second
                        << ") because a child became non-positive locally"
                        << std::endl;
              rollbackCreatedMidpoints(created_midpoints);
              e2b.erase(ee[t]);
              rejected_edges.insert(ee[t]);
              if (ppartner.count(ee[t])) {
                e2b.erase(ppartner[ee[t]]);
                rejected_edges.insert(ppartner[ee[t]]);
              }
              if (endpoint_related_edge) noteEndpointFanFailure(ee[t]);
              if (touches_wall) break;
              continue;
            }
            const unsigned char q1_reason_preview =
                use_minimal_q1_rules ? classifyAcceptedSplitReason(false) : 0;
            const bool enforce_q1_local_quality_gate =
                use_minimal_q1_rules &&
                (selected_edge_is_wall ||
                 endpoint_related_edge ||
                 q1_reason_preview == kSplitWallCavity ||
                 q1_reason_preview == kSplitEndpointFan ||
                 q1_reason_preview == kSplitPeriodic);
            if (enforce_q1_local_quality_gate) {
              const double min_angle_floor = q1ClosureSplitMinAngleDeg();
              const double min_quality_floor = q1ClosureSplitMinQuality();
              bool closure_quality_ok = true;
              std::ostringstream closure_reason;
              if (selected_edge_is_wall) {
                const auto current_node_to_elem = buildCurrentNodeToElem();
                const auto current_edge_to_elem = buildCurrentEdgeToElem();
                std::vector<int> edge_adj_elems;
                const auto key = std::make_pair(std::min(va, vb), std::max(va, vb));
                auto edge_it = current_edge_to_elem.find(key);
                if (edge_it != current_edge_to_elem.end()) {
                  edge_adj_elems = edge_it->second;
                }
                std::set<int> patch_elems;
                for (int eadj : current_node_to_elem[va]) patch_elems.insert(eadj);
                for (int eadj : current_node_to_elem[vb]) patch_elems.insert(eadj);
                const WallSplitAssessment wall_assessment = assessWallSplitPatch(
                    mesh, va, vb, mesh.V[vm], edge_adj_elems, patch_elems,
                    current_node_to_elem);
                closure_quality_ok =
                    wall_assessment.feasible &&
                    wall_assessment.min_child_angle >= min_angle_floor &&
                    wall_assessment.min_child_quality >= min_quality_floor;
                if (!closure_quality_ok) {
                  closure_reason << "closure patch quality too weak"
                                 << " (min_angle="
                                 << wall_assessment.min_child_angle
                                 << ", min_quality="
                                 << wall_assessment.min_child_quality << ")";
                }
              } else {
                const auto current_edge_to_elem = buildCurrentEdgeToElem();
                const Q1SplitPatchAssessment patch_assessment =
                    assessQ1SplitPatch(mesh, va, vb, current_edge_to_elem);
                closure_quality_ok =
                    patch_assessment.feasible &&
                    patch_assessment.min_angle_deg >= min_angle_floor &&
                    patch_assessment.min_quality >= min_quality_floor;
                if (!closure_quality_ok) {
                  closure_reason << "closure patch quality too weak"
                                 << " (min_angle="
                                 << patch_assessment.min_angle_deg
                                 << ", min_quality="
                                 << patch_assessment.min_quality << ")";
                }
              }
              if (!closure_quality_ok) {
                std::cerr << "    warning: skipped refinement on edge ("
                          << ee[t].first << ", " << ee[t].second
                          << ") because " << closure_reason.str() << std::endl;
                rollbackCreatedMidpoints(created_midpoints);
                e2b.erase(ee[t]);
                rejected_edges.insert(ee[t]);
                if (ppartner.count(ee[t])) {
                  e2b.erase(ppartner[ee[t]]);
                  rejected_edges.insert(ppartner[ee[t]]);
                }
                if (endpoint_related_edge) noteEndpointFanFailure(ee[t]);
                if (touches_wall) break;
                continue;
              }
            }
            mesh.E[e] = std::move(q1_children.updated_parent);
            mesh.E.push_back(std::move(q1_children.new_child));
            accepted_q1_simple_split = true;
            accepted_q1_seed_nodes = {va, vb, vc, vm};
          }
          rmap.child_to_parent.push_back(parent);
          if (use_minimal_q1_rules && conformity_only_candidates &&
              edge_midpoint_preexisting) {
            extension_refined_parents.insert(parent);
          }
          if (use_minimal_q1_rules) {
            rmap.parent_split_reason[parent] =
                classifyAcceptedSplitReason(false);
            const unsigned char reason = rmap.parent_split_reason[parent];
            if (reason == kSplitPeriodic || reason == kSplitWallCavity ||
                reason == kSplitEndpointFan ||
                reason == kSplitEndpointOverride ||
                reason == kSplitGrading) {
              special_cleanup_parents.insert(parent);
            }
            if (accepted_q1_simple_split) {
              enqueueLocalGradationSupports(accepted_q1_seed_nodes,
                                            parentSplitReasonName(reason));
            }
          }
          std::cerr << "    accepted split: e=" << e
                    << " parent=" << parent
                    << " edge=(" << ee[t].first << ", " << ee[t].second << ")"
                    << " source=" << (parent_marked ? "marked" : "fallback")
                    << " wall_adjacent=" << (touches_wall ? "yes" : "no")
                    << " periodic=" << (ppartner.count(ee[t]) ? "yes" : "no")
                    << " reason="
                    << parentSplitReasonName(rmap.parent_split_reason[parent])
                    << std::endl;
          enqueueForcedPeriodicPartnerSplits(created_midpoints, ee);
          e2b.erase(ee[t]);
          changed = true;
          split_done = true;
          break;
        }
        if (!split_done) continue;
      }
    }

    // e2b is exhausted. Check if fallback is needed and available.
    if (fallback_priority.empty() || target_adj_splits <= 0) break;
    int adj_splits = 0;
    for (int c = old_Ne; c < (int)mesh.E.size(); c++) {
      int p = rmap.child_to_parent[c];
      if (p < old_Ne && included[p]) ++adj_splits;
    }
    if (adj_splits >= target_adj_splits || fb_cursor >= (int)fallback_priority.size()) break;

    // Add the next batch of fallback candidates to e2b.
    int need = target_adj_splits - adj_splits;
    bool added = false;
    while (fb_cursor < (int)fallback_priority.size() && need > 0) {
      int fe = fallback_priority[fb_cursor++];
      if (fe >= old_Ne || included[fe]) continue;
      if (use_minimal_q1_rules && q1WallOnlyRefinementEnabled() &&
          !old_elem_touches_wall[fe]) {
        continue;
      }
      auto fb_edge = getPreferredRefinementEdge(fe);
      if (!use_minimal_q1_rules && old_elem_touches_wall[fe]) {
        auto preferred_wall = getEncroachedWallEdge(fe);
        if (preferred_wall.first < 0) preferred_wall = getLongestWallEdge(fe);
        if (preferred_wall.first >= 0 && rejected_edges.count(preferred_wall)) {
          continue;
        }
      }
      included[fe] = true;
      std::cerr << "    fallback candidate e=" << fe
                << " edge=(" << fb_edge.first << ", " << fb_edge.second << ")"
                << " wall_adjacent=" << (old_elem_touches_wall[fe] ? "yes" : "no")
                << std::endl;
      if (!edge_midpoint.count(fb_edge)) {
        e2b.insert(fb_edge);
        if (ppartner.count(fb_edge)) e2b.insert(ppartner[fb_edge]);
        added = true;
      }
      --need;
    }
    if (!added) break;
  }

  for (auto const& [edge, vm] : edge_midpoint) {
    if (ppartner.count(edge)) {
      auto p = ppartner[edge];
      if (edge_midpoint.count(p) && edge < p) {
        int v1 = vm, v2 = edge_midpoint[p];
        if (mesh.V[v1].y > mesh.V[v2].y) std::swap(v1, v2);
        int gidx = -1;
        auto git = periodic_edge_group.find(edge);
        if (git == periodic_edge_group.end()) git = periodic_edge_group.find(p);
        if (git != periodic_edge_group.end()) gidx = git->second;
        if (gidx < 0 || gidx >= static_cast<int>(mesh.periodicGroups.size())) {
          std::cerr << "    warning: unable to place periodic midpoint pair for edge ("
                    << edge.first << ", " << edge.second
                    << ") because no periodic group was found" << std::endl;
          continue;
        }
        mesh.periodicGroups[gidx].pairs.push_back({v1, v2});
        mesh.periodicGroups[gidx].nPairs =
            static_cast<int>(mesh.periodicGroups[gidx].pairs.size());
      }
    }
  }

  applyExtensionNeighborEdgeSwapCleanup(mesh, old_boundary_edge, rmap,
                                        extension_refined_parents);
  applySeedParentPatchEdgeSwapCleanup(
      mesh, old_boundary_edge, rmap, blocked_wall_cleanup_parents,
      "blocked-wall cleanup", q1PatchRepairMinAngleDeg(),
      q1PatchRepairMinQuality(), true);
  applySeedParentPatchEdgeSwapCleanup(
      mesh, old_boundary_edge, rmap, special_cleanup_parents,
      "targeted q1 cleanup", q1EndpointOverrideMinAngleDeg(),
      q1EndpointOverrideMinQuality(), true);
  smoothRefinedCurvedMesh(mesh, old_boundary_edge, old_Nv);
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
  return bisectMarkedElementsImpl(mesh, marked_in);
}

RefinementMap bisectMarkedElements(Mesh &mesh, const std::vector<bool> &marked_in,
                                   const std::vector<int> &fallback_priority,
                                   int target_adj_splits) {
  return bisectMarkedElementsImpl(mesh, marked_in, fallback_priority,
                                  target_adj_splits);
}

void printWallRefinementDiagnostics(const Mesh &mesh,
                                    const std::vector<bool> &marked_in) {
  std::map<std::pair<int,int>, int> boundary_edge_groups;
  for (const auto &be : mesh.BE) {
    boundary_edge_groups[{std::min(be.v[0], be.v[1]), std::max(be.v[0], be.v[1])}] = be.bIndex;
  }

  std::map<std::pair<int,int>, std::vector<int>> edge_to_elem;
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    const int *v = mesh.E[e].v;
    edge_to_elem[{std::min(v[0], v[1]), std::max(v[0], v[1])}].push_back(e);
    edge_to_elem[{std::min(v[1], v[2]), std::max(v[1], v[2])}].push_back(e);
    edge_to_elem[{std::min(v[2], v[0]), std::max(v[2], v[0])}].push_back(e);
  }

  auto buildNodeToElem = [&]() {
    std::vector<std::vector<int>> node_to_elem(mesh.V.size());
    for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
      for (int k = 0; k < 3; ++k) node_to_elem[mesh.E[e].v[k]].push_back(e);
    }
    return node_to_elem;
  };

  auto buildVertexNeighbors = [&]() {
    std::vector<std::set<int>> vertex_neighbors(mesh.V.size());
    for (const auto &elem : mesh.E) {
      vertex_neighbors[elem.v[0]].insert(elem.v[1]);
      vertex_neighbors[elem.v[0]].insert(elem.v[2]);
      vertex_neighbors[elem.v[1]].insert(elem.v[0]);
      vertex_neighbors[elem.v[1]].insert(elem.v[2]);
      vertex_neighbors[elem.v[2]].insert(elem.v[0]);
      vertex_neighbors[elem.v[2]].insert(elem.v[1]);
    }
    return vertex_neighbors;
  };

  const auto node_to_elem = buildNodeToElem();
  const auto vertex_neighbors = buildVertexNeighbors();

  auto edgeLength2 = [&](int a, int b) {
    return (mesh.V[a] - mesh.V[b]).normSq();
  };

  auto elementTouchesWall = [&](int e) {
    const int *v = mesh.E[e].v;
    const std::pair<int, int> edges[3] = {
      {std::min(v[0], v[1]), std::max(v[0], v[1])},
      {std::min(v[1], v[2]), std::max(v[1], v[2])},
      {std::min(v[2], v[0]), std::max(v[2], v[0])}
    };
    for (const auto &edge : edges) {
      auto it = boundary_edge_groups.find(edge);
      if (it == boundary_edge_groups.end()) continue;
      int g = it->second;
      if (g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
          lowerCopy(mesh.Bname[g]) == "wall") return true;
    }
    return false;
  };

  auto centroid = [&](int e) {
    const int *v = mesh.E[e].v;
    return (mesh.V[v[0]] + mesh.V[v[1]] + mesh.V[v[2]]) / 3.0;
  };

  std::cerr << "  Wall refinement diagnostics:" << std::endl;
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    if (!elementTouchesWall(e)) continue;

    const int *v = mesh.E[e].v;
    int best_t = 0;
    double best_len2 = edgeLength2(v[0], v[1]);
    for (int t = 1; t < 3; ++t) {
      const double len2 = edgeLength2(v[t], v[(t + 1) % 3]);
      if (len2 > best_len2) {
        best_len2 = len2;
        best_t = t;
      }
    }

    const int va = v[best_t];
    const int vb = v[(best_t + 1) % 3];
    const int vopp = v[(best_t + 2) % 3];
    const auto key = std::make_pair(std::min(va, vb), std::max(va, vb));
    const Vec2 c = centroid(e);

    std::string status;
    if (e >= static_cast<int>(marked_in.size()) || !marked_in[e]) {
      status = "not marked";
    } else if (boundary_edge_groups.count(key) &&
               boundary_edge_groups.at(key) >= 0 &&
               boundary_edge_groups.at(key) < static_cast<int>(mesh.Bname.size()) &&
               lowerCopy(mesh.Bname[boundary_edge_groups.at(key)]) == "wall") {
      const Vec2 straight_mid = 0.5 * (mesh.V[va] + mesh.V[vb]);
      const Vec2 snapped_mid = projectToBladeSpline(straight_mid);
      Vec2 avg{0.0, 0.0};
      int count = 0;
      for (int nbr : vertex_neighbors[vopp]) {
        avg = avg + mesh.V[nbr];
        count++;
      }
      if (count > 0) avg = avg / static_cast<double>(count);

      bool accepted = true;
      std::string wall_reason;
      auto eit = edge_to_elem.find(key);
      if (eit != edge_to_elem.end()) {
        for (int eadj : eit->second) {
          const int *ev = mesh.E[eadj].v;
          int local_opp = -1;
          for (int k = 0; k < 3; ++k) {
            if (ev[k] != va && ev[k] != vb) {
              local_opp = ev[k];
              break;
            }
          }
          if (local_opp < 0) continue;
          std::string base_reason = localWallSplitPatchFailureReason(
              mesh, va, vb, local_opp, snapped_mid, mesh.V[local_opp], node_to_elem);
          if (base_reason.empty()) {
            continue;
          }
          if (count == 0) {
            wall_reason = "snap failed: " + base_reason + "; no opposite-node neighbors";
            accepted = false;
            break;
          }
          std::string avg_reason = localWallSplitPatchFailureReason(
              mesh, va, vb, local_opp, snapped_mid, avg, node_to_elem);
          if (!avg_reason.empty()) {
            wall_reason = "snap failed: " + base_reason +
                          "; opposite-node average failed: " + avg_reason;
            accepted = false;
            break;
          }
        }
      }
      status = accepted ? "eligible: wall-edge split passes" :
                          "blocked: " + wall_reason;
    } else {
      const Vec2 straight_mid = 0.5 * (mesh.V[va] + mesh.V[vb]);
      BladeProjection proj = projectToBladeSplineDetailed(straight_mid);
      if (proj.valid && !pointIsOnFluidSide(straight_mid, proj)) {
        status = "blocked: longest-edge midpoint outside blade";
      } else {
        bool bisector_bad = false;
        auto eit = edge_to_elem.find(key);
        if (eit != edge_to_elem.end()) {
          for (int eadj : eit->second) {
            const int *ev = mesh.E[eadj].v;
            int local_opp = -1;
            for (int k = 0; k < 3; ++k) {
              if (ev[k] != va && ev[k] != vb) {
                local_opp = ev[k];
                break;
              }
            }
            if (local_opp < 0) continue;
            const Vec2 bisector_mid = 0.5 * (straight_mid + mesh.V[local_opp]);
            BladeProjection bproj = projectToBladeSplineDetailed(bisector_mid);
            if (bproj.valid && !pointIsOnFluidSide(bisector_mid, bproj)) {
              bisector_bad = true;
              break;
            }
          }
        }
        status = bisector_bad ? "blocked: bisector midpoint outside blade" :
                                "eligible: interior-edge split passes";
      }
    }

    std::cerr << "    e=" << e
              << " centroid=(" << c.x << ", " << c.y << ")"
              << " longest=(" << va << ", " << vb << ")"
              << " opp=" << vopp
              << " status=" << status << std::endl;
  }
}

struct PatchQualityAssessment {
  double min_patch_detJ = 0.0;
  double min_angle_deg = 0.0;
  double min_quality = 0.0;
  double score = -std::numeric_limits<double>::infinity();
};

PatchQualityAssessment assessPatchQuality(const Mesh &mesh,
                                          const std::set<int> &patch_elems) {
  PatchQualityAssessment out;
  out.min_patch_detJ = patchMinimumExactDetJ(mesh, patch_elems);
  out.min_angle_deg = std::numeric_limits<double>::infinity();
  out.min_quality = std::numeric_limits<double>::infinity();

  for (int e : patch_elems) {
    if (e < 0 || e >= static_cast<int>(mesh.E.size())) continue;
    out.min_angle_deg = std::min(out.min_angle_deg,
                                 triangleMinAngleDeg(mesh.E[e], mesh.V));
    out.min_quality = std::min(out.min_quality,
                               triangleQuality(mesh.E[e], mesh.V));
  }

  if (!std::isfinite(out.min_angle_deg)) out.min_angle_deg = 0.0;
  if (!std::isfinite(out.min_quality)) out.min_quality = 0.0;

  const double delta = (out.min_patch_detJ < 0.0)
                           ? std::sqrt(1e-8 + 0.04 * out.min_patch_detJ * out.min_patch_detJ)
                           : 1e-4;
  const double JR = 0.5 * (out.min_patch_detJ +
                           std::sqrt(out.min_patch_detJ * out.min_patch_detJ +
                                     4.0 * delta * delta));
  out.score = 30.0 * out.min_angle_deg
              + 1200.0 * out.min_quality
              + 10.0 * std::log(std::max(JR, 1e-12));
  return out;
}

int improvePatchByEdgeSwaps(
    Mesh &mesh,
    const std::set<int> &patch_elems,
    const std::map<std::pair<int, int>, int> &boundary_edge_groups,
    int max_swaps,
    const std::function<bool(int)> &is_wall_boundary_vertex,
    bool allow_periodic_vertices) {
  if (max_swaps <= 0 || patch_elems.empty()) return 0;

  const auto periodic_vertices = collectPeriodicVertices(mesh);
  const auto periodic_edges = collectPeriodicEdgeKeys(mesh);
  const auto boundary_vertices = collectBoundaryVertices(mesh);
  int swap_count = 0;

  for (int swap_iter = 0; swap_iter < max_swaps; ++swap_iter) {
    std::map<std::pair<int, int>, std::vector<int>> edge_to_elem;
    for (int e : patch_elems) {
      if (e < 0 || e >= static_cast<int>(mesh.E.size())) continue;
      const auto &elem = mesh.E[e];
      if (elem.q_order != 1) continue;
      edge_to_elem[{std::min(elem.v[0], elem.v[1]), std::max(elem.v[0], elem.v[1])}].push_back(e);
      edge_to_elem[{std::min(elem.v[1], elem.v[2]), std::max(elem.v[1], elem.v[2])}].push_back(e);
      edge_to_elem[{std::min(elem.v[2], elem.v[0]), std::max(elem.v[2], elem.v[0])}].push_back(e);
    }

    bool changed = false;
    for (const auto &[edge, elems] : edge_to_elem) {
      if (elems.size() != 2) continue;
      if (boundary_edge_groups.count(edge)) continue;
      if (periodic_edges.count(edge)) continue;

      const int e0 = elems[0];
      const int e1 = elems[1];
      if (patch_elems.count(e0) == 0 || patch_elems.count(e1) == 0) continue;
      const auto &elem0 = mesh.E[e0];
      const auto &elem1 = mesh.E[e1];
      if (elem0.q_order != 1 || elem1.q_order != 1) continue;

      const int a = edge.first;
      const int b = edge.second;
      int c = -1;
      int d = -1;
      for (int k = 0; k < 3; ++k) {
        if (elem0.v[k] != a && elem0.v[k] != b) c = elem0.v[k];
        if (elem1.v[k] != a && elem1.v[k] != b) d = elem1.v[k];
      }
      if (c < 0 || d < 0 || c == d) continue;

      if (!allow_periodic_vertices &&
          (periodic_vertices.count(a) || periodic_vertices.count(b) ||
           periodic_vertices.count(c) || periodic_vertices.count(d))) {
        continue;
      }

      bool has_nonwall_boundary_vertex = false;
      for (int v : {a, b, c, d}) {
        if (boundary_vertices.count(v) && !is_wall_boundary_vertex(v)) {
          has_nonwall_boundary_vertex = true;
          break;
        }
      }
      if (has_nonwall_boundary_vertex) continue;

      const auto new_edge = std::make_pair(std::min(c, d), std::max(c, d));
      if (boundary_edge_groups.count(new_edge)) continue;
      if (periodic_edges.count(new_edge)) continue;
      auto edge_it = edge_to_elem.find(new_edge);
      if (edge_it != edge_to_elem.end() && !edge_it->second.empty()) continue;
      if (!segmentsIntersect(mesh.V[a], mesh.V[b], mesh.V[c], mesh.V[d], 1e-12)) continue;

      Element swapped0 = makePositiveElement(c, d, a, mesh.V);
      Element swapped1 = makePositiveElement(d, c, b, mesh.V);
      const double area0 = triangleSignedArea(swapped0, mesh.V);
      const double area1 = triangleSignedArea(swapped1, mesh.V);
      if (!(area0 > 1e-12) || !(area1 > 1e-12)) continue;

      const double old_min_angle = std::min(triangleMinAngleDeg(elem0, mesh.V),
                                            triangleMinAngleDeg(elem1, mesh.V));
      const double new_min_angle = std::min(triangleMinAngleDeg(swapped0, mesh.V),
                                            triangleMinAngleDeg(swapped1, mesh.V));
      const double old_min_quality = std::min(triangleQuality(elem0, mesh.V),
                                              triangleQuality(elem1, mesh.V));
      const double new_min_quality = std::min(triangleQuality(swapped0, mesh.V),
                                              triangleQuality(swapped1, mesh.V));
      const double old_score = 30.0 * old_min_angle + 1200.0 * old_min_quality;
      const double new_score = 30.0 * new_min_angle + 1200.0 * new_min_quality;
      if (new_score <= old_score + 1e-8) continue;

      mesh.E[e0] = swapped0;
      mesh.E[e1] = swapped1;
      changed = true;
      ++swap_count;
      break;
    }

    if (!changed) break;
  }

  return swap_count;
}

namespace {

double simplePatchScore(const SimplePatchQuality &patch) {
  return 30.0 * patch.min_angle_deg + 1200.0 * patch.min_quality;
}

std::set<int> buildQ1RepairPatch(const Mesh &mesh, int elem_idx, int rings) {
  std::set<int> patch;
  if (elem_idx < 0 || elem_idx >= static_cast<int>(mesh.E.size())) return patch;

  std::vector<std::vector<int>> node_to_elem(mesh.V.size());
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    const auto &elem = mesh.E[e];
    if (elem.q_order != 1) continue;
    for (int k = 0; k < 3; ++k) {
      node_to_elem[elem.v[k]].push_back(e);
    }
  }

  std::set<int> frontier{elem_idx};
  patch.insert(elem_idx);
  for (int ring = 0; ring < std::max(1, rings); ++ring) {
    std::set<int> next_frontier;
    for (int e : frontier) {
      if (e < 0 || e >= static_cast<int>(mesh.E.size())) continue;
      const auto &elem = mesh.E[e];
      if (elem.q_order != 1) continue;
      for (int k = 0; k < 3; ++k) {
        for (int nb : node_to_elem[elem.v[k]]) {
          if (patch.insert(nb).second) next_frontier.insert(nb);
        }
      }
    }
    if (next_frontier.empty()) break;
    frontier.swap(next_frontier);
  }
  return patch;
}

} // namespace

bool repairLowQualityQ1Patch(Mesh &mesh, int elem_idx,
                             bool allow_topology_changes) {
  if (elem_idx < 0 || elem_idx >= static_cast<int>(mesh.E.size())) return false;
  if (mesh.q_order_global > 1) return false;
  if (mesh.E[elem_idx].q_order != 1) return false;

  const std::set<int> patch_elems =
      buildQ1RepairPatch(mesh, elem_idx, q1PatchRepairRings());
  if (patch_elems.size() < 2) return false;

  std::map<std::pair<int, int>, int> boundary_edge_groups;
  for (const auto &be : mesh.BE) {
    boundary_edge_groups[{std::min(be.v[0], be.v[1]),
                          std::max(be.v[0], be.v[1])}] = be.bIndex;
  }
  auto isWallBoundaryVertex = [&](int vid) {
    for (const auto &be : mesh.BE) {
      if (be.v[0] != vid && be.v[1] != vid) continue;
      const int g = be.bIndex;
      if (g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
          lowerCopy(mesh.Bname[g]) == "wall") {
        return true;
      }
    }
    return false;
  };

  const Mesh mesh_before = mesh;
  const SimplePatchQuality before = assessSimplePatchQuality(mesh, patch_elems);
  const double angle_floor = q1PatchRepairMinAngleDeg();
  const double quality_floor = q1PatchRepairMinQuality();

  int total_swaps = 0;
  int total_smooth = 0;
  for (int pass = 0; pass < 3; ++pass) {
    int swaps = 0;
    if (allow_topology_changes) {
      rebuildMeshEdgeConnectivity(mesh, boundary_edge_groups);
      mesh.appendPeriodicToIE();
      swaps = improvePatchByEdgeSwaps(
          mesh, patch_elems, boundary_edge_groups, q1PatchRepairMaxSwaps(),
          isWallBoundaryVertex, true);
      if (swaps > 0) {
        total_swaps += swaps;
        rebuildMeshEdgeConnectivity(mesh, boundary_edge_groups);
        mesh.appendPeriodicToIE();
      }
    }

    const int smooth_moves =
        smoothQ1PatchVertices(mesh, patch_elems, angle_floor, quality_floor);
    total_smooth += smooth_moves;

    if (swaps == 0 && smooth_moves == 0) break;
  }

  const SimplePatchQuality after = assessSimplePatchQuality(mesh, patch_elems);
  if (total_swaps == 0 && total_smooth == 0) {
    mesh = mesh_before;
    return false;
  }
  if (simplePatchScore(after) <= simplePatchScore(before) + 1e-8) {
    mesh = mesh_before;
    return false;
  }

  rebuildMeshEdgeConnectivity(mesh, boundary_edge_groups);
  mesh.appendPeriodicToIE();
  std::cerr << "    q1 patch repair: seed=" << elem_idx
            << " patch_elems=" << patch_elems.size()
            << " swaps=" << total_swaps
            << " smooth_moves=" << total_smooth
            << " | min_angle " << before.min_angle_deg << " -> "
            << after.min_angle_deg
            << " | min_quality " << before.min_quality << " -> "
            << after.min_quality << std::endl;
  return true;
}

bool repairLowQualityCurvedPatch(Mesh &mesh, int elem_idx) {
  if (elem_idx < 0 || elem_idx >= static_cast<int>(mesh.E.size())) return false;

  std::map<std::pair<int, int>, int> boundary_edge_groups;
  for (const auto &be : mesh.BE) {
    boundary_edge_groups[{std::min(be.v[0], be.v[1]), std::max(be.v[0], be.v[1])}] = be.bIndex;
  }

  std::vector<std::vector<int>> node_to_elem(mesh.V.size());
  std::vector<std::set<int>> vertex_neighbors(mesh.V.size());
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    const auto &elem = mesh.E[e];
    for (int k = 0; k < 3; ++k) node_to_elem[elem.v[k]].push_back(e);
    vertex_neighbors[elem.v[0]].insert(elem.v[1]);
    vertex_neighbors[elem.v[0]].insert(elem.v[2]);
    vertex_neighbors[elem.v[1]].insert(elem.v[0]);
    vertex_neighbors[elem.v[1]].insert(elem.v[2]);
    vertex_neighbors[elem.v[2]].insert(elem.v[0]);
    vertex_neighbors[elem.v[2]].insert(elem.v[1]);
  }

  std::set<int> patch_nodes{
      mesh.E[elem_idx].v[0],
      mesh.E[elem_idx].v[1],
      mesh.E[elem_idx].v[2],
  };
  std::set<int> patch_elems{elem_idx};
  for (int vid : patch_nodes) {
    for (int e : node_to_elem[vid]) patch_elems.insert(e);
  }
  auto expandPatchOneRing = [&](std::set<int> &nodes, std::set<int> &elems) {
    std::vector<int> ring_nodes(nodes.begin(), nodes.end());
    for (int vid : ring_nodes) {
      for (int nbr : vertex_neighbors[vid]) nodes.insert(nbr);
    }
    for (int vid : nodes) {
      for (int e : node_to_elem[vid]) elems.insert(e);
    }
  };
  expandPatchOneRing(patch_nodes, patch_elems);

  const auto periodic_vertices = collectPeriodicVertices(mesh);
  const auto boundary_vertices = collectBoundaryVertices(mesh);
  auto isWallBoundaryVertex = [&](int vid) {
    for (int e : node_to_elem[vid]) {
      const auto &elem = mesh.E[e];
      const std::pair<int, int> edges[3] = {
          {std::min(elem.v[0], elem.v[1]), std::max(elem.v[0], elem.v[1])},
          {std::min(elem.v[1], elem.v[2]), std::max(elem.v[1], elem.v[2])},
          {std::min(elem.v[2], elem.v[0]), std::max(elem.v[2], elem.v[0])},
      };
      for (const auto &edge : edges) {
        if (edge.first != vid && edge.second != vid) continue;
        auto it = boundary_edge_groups.find(edge);
        if (it == boundary_edge_groups.end()) continue;
        int g = it->second;
        if (g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
            lowerCopy(mesh.Bname[g]) == "wall") {
          return true;
        }
      }
    }
    return false;
  };

  auto wallNeighborVertices = [&](int vid) {
    std::vector<int> nbrs;
    for (int nbr : vertex_neighbors[vid]) {
      const auto key = std::make_pair(std::min(vid, nbr), std::max(vid, nbr));
      auto it = boundary_edge_groups.find(key);
      if (it == boundary_edge_groups.end()) continue;
      int g = it->second;
      if (g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
          lowerCopy(mesh.Bname[g]) == "wall") {
        nbrs.push_back(nbr);
      }
    }
    return nbrs;
  };

  bool patch_touches_wall = false;
  for (int vid : patch_nodes) {
    if (isWallBoundaryVertex(vid)) {
      patch_touches_wall = true;
      break;
    }
  }
  if (patch_touches_wall) {
    expandPatchOneRing(patch_nodes, patch_elems);
  }

  std::vector<int> movable_vertices;
  for (int vid : patch_nodes) {
    if (periodic_vertices.count(vid)) continue;
    if (boundary_vertices.count(vid) && !isWallBoundaryVertex(vid)) continue;
    movable_vertices.push_back(vid);
  }
  if (movable_vertices.empty()) return false;

  retunePatchQ2Geometry(mesh, patch_elems, boundary_edge_groups);
  const PatchQualityAssessment initial = assessPatchQuality(mesh, patch_elems);
  PatchQualityAssessment best = initial;
  bool any_change = false;

  if (improvePatchByEdgeSwaps(mesh, patch_elems, boundary_edge_groups, 8,
                              isWallBoundaryVertex)) {
    retunePatchQ2Geometry(mesh, patch_elems, boundary_edge_groups);
    best = assessPatchQuality(mesh, patch_elems);
    any_change = best.score > initial.score + 1e-8;
  }

  for (int iter = 0; iter < 6; ++iter) {
    bool iter_changed = false;
    for (int vid : movable_vertices) {
      const Vec2 original = mesh.V[vid];
      std::vector<Vec2> candidates;

      if (!vertex_neighbors[vid].empty()) {
        Vec2 avg{0.0, 0.0};
        for (int nbr : vertex_neighbors[vid]) avg = avg + mesh.V[nbr];
        avg = avg / static_cast<double>(vertex_neighbors[vid].size());

        if (boundary_vertices.count(vid) && isWallBoundaryVertex(vid)) {
          auto wn = wallNeighborVertices(vid);
          if (wn.size() >= 2) {
            Vec2 arc_mid = original;
            if (bladeEdgeArcMidpoint(mesh.V[wn[0]], mesh.V[wn[1]], arc_mid)) {
              candidates.push_back(0.75 * original + 0.25 * arc_mid);
              candidates.push_back(0.5 * original + 0.5 * arc_mid);
              candidates.push_back(arc_mid);
            }
          }
          BladeProjection old_proj = projectToBladeSplineDetailed(original);
          BladeProjection avg_proj = projectToBladeSplineDetailed(avg);
          if (avg_proj.valid &&
              (!old_proj.valid || old_proj.branch == avg_proj.branch)) {
            candidates.push_back(0.75 * original + 0.25 * avg_proj.pt);
            candidates.push_back(0.5 * original + 0.5 * avg_proj.pt);
            candidates.push_back(avg_proj.pt);
          }
        } else {
          candidates.push_back(0.75 * original + 0.25 * avg);
          candidates.push_back(0.5 * original + 0.5 * avg);
          candidates.push_back(avg);
        }
      }

      PatchQualityAssessment best_local = best;
      Vec2 best_pos = original;
      for (const Vec2 &candidate : candidates) {
        std::vector<Vec2> verts_trial = mesh.V;
        if (!localMovePreservesMesh(mesh, vid, node_to_elem, mesh.V, verts_trial,
                                    candidate)) {
          continue;
        }

        mesh.V[vid] = candidate;
        retunePatchQ2Geometry(mesh, patch_elems, boundary_edge_groups);
        const PatchQualityAssessment trial = assessPatchQuality(mesh, patch_elems);
        if (trial.min_patch_detJ > 0.0 && trial.score > best_local.score + 1e-8) {
          best_local = trial;
          best_pos = candidate;
        }
        mesh.V[vid] = original;
        retunePatchQ2Geometry(mesh, patch_elems, boundary_edge_groups);
      }

      if ((best_pos - original).norm() > 1e-12) {
        mesh.V[vid] = best_pos;
        retunePatchQ2Geometry(mesh, patch_elems, boundary_edge_groups);
        best = best_local;
        iter_changed = true;
        any_change = true;
      }
    }
    if (!iter_changed) break;
  }

  mesh.computeGeometry();
  if (any_change) {
    std::cerr << "    local quality repair: seed=" << elem_idx
              << " patch_elems=" << patch_elems.size()
              << " min_angle " << initial.min_angle_deg << " -> "
              << best.min_angle_deg
              << " | min_quality " << initial.min_quality << " -> "
              << best.min_quality
              << " | min_detJ " << initial.min_patch_detJ << " -> "
              << best.min_patch_detJ << std::endl;
  }
  return any_change && best.score > initial.score + 1e-8 &&
         best.min_patch_detJ > 0.0;
}

bool repairInvalidCurvedPatch(Mesh &mesh, int elem_idx) {
  if (elem_idx < 0 || elem_idx >= static_cast<int>(mesh.E.size())) return false;
  if (mesh.E[elem_idx].q_order != 2) return false;

  std::map<std::pair<int, int>, int> boundary_edge_groups;
  for (const auto &be : mesh.BE) {
    boundary_edge_groups[{std::min(be.v[0], be.v[1]), std::max(be.v[0], be.v[1])}] = be.bIndex;
  }

  std::vector<std::vector<int>> node_to_elem(mesh.V.size());
  std::vector<std::set<int>> vertex_neighbors(mesh.V.size());
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    const auto &elem = mesh.E[e];
    for (int k = 0; k < 3; ++k) node_to_elem[elem.v[k]].push_back(e);
    vertex_neighbors[elem.v[0]].insert(elem.v[1]);
    vertex_neighbors[elem.v[0]].insert(elem.v[2]);
    vertex_neighbors[elem.v[1]].insert(elem.v[0]);
    vertex_neighbors[elem.v[1]].insert(elem.v[2]);
    vertex_neighbors[elem.v[2]].insert(elem.v[0]);
    vertex_neighbors[elem.v[2]].insert(elem.v[1]);
  }

  std::set<int> patch_nodes{
      mesh.E[elem_idx].v[0],
      mesh.E[elem_idx].v[1],
      mesh.E[elem_idx].v[2],
  };
  std::set<int> patch_elems{elem_idx};
  for (int vid : patch_nodes) {
    for (int e : node_to_elem[vid]) patch_elems.insert(e);
  }
  std::vector<int> ring_nodes(patch_nodes.begin(), patch_nodes.end());
  for (int vid : ring_nodes) {
    for (int nbr : vertex_neighbors[vid]) patch_nodes.insert(nbr);
  }
  for (int vid : patch_nodes) {
    for (int e : node_to_elem[vid]) patch_elems.insert(e);
  }

  const auto periodic_vertices = collectPeriodicVertices(mesh);
  const auto boundary_vertices = collectBoundaryVertices(mesh);
  auto isWallBoundaryVertex = [&](int vid) {
    for (int e : node_to_elem[vid]) {
      const auto &elem = mesh.E[e];
      const std::pair<int, int> edges[3] = {
          {std::min(elem.v[0], elem.v[1]), std::max(elem.v[0], elem.v[1])},
          {std::min(elem.v[1], elem.v[2]), std::max(elem.v[1], elem.v[2])},
          {std::min(elem.v[2], elem.v[0]), std::max(elem.v[2], elem.v[0])},
      };
      for (const auto &edge : edges) {
        if (edge.first != vid && edge.second != vid) continue;
        auto it = boundary_edge_groups.find(edge);
        if (it == boundary_edge_groups.end()) continue;
        int g = it->second;
        if (g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
            lowerCopy(mesh.Bname[g]) == "wall") {
          return true;
        }
      }
    }
    return false;
  };

  auto wallNeighborVertices = [&](int vid) {
    std::vector<int> nbrs;
    for (int nbr : vertex_neighbors[vid]) {
      const auto key = std::make_pair(std::min(vid, nbr), std::max(vid, nbr));
      auto it = boundary_edge_groups.find(key);
      if (it == boundary_edge_groups.end()) continue;
      int g = it->second;
      if (g >= 0 && g < static_cast<int>(mesh.Bname.size()) &&
          lowerCopy(mesh.Bname[g]) == "wall") {
        nbrs.push_back(nbr);
      }
    }
    return nbrs;
  };

  std::vector<int> movable_vertices;
  for (int vid : patch_nodes) {
    if (periodic_vertices.count(vid)) continue;
    if (boundary_vertices.count(vid) && !isWallBoundaryVertex(vid)) continue;
    movable_vertices.push_back(vid);
  }
  if (movable_vertices.empty()) return false;

  auto patchMetric = [&]() { return patchMinimumExactDetJ(mesh, patch_elems); };

  retunePatchQ2Geometry(mesh, patch_elems, boundary_edge_groups);
  double best_patch_detJ = patchMetric();
  const double initial_patch_detJ = best_patch_detJ;
  bool any_change = false;

  for (int iter = 0; iter < 6; ++iter) {
    bool iter_changed = false;
    for (int vid : movable_vertices) {
      const Vec2 original = mesh.V[vid];
      std::vector<Vec2> candidates;

      if (!vertex_neighbors[vid].empty()) {
        Vec2 avg{0.0, 0.0};
        for (int nbr : vertex_neighbors[vid]) avg = avg + mesh.V[nbr];
        avg = avg / static_cast<double>(vertex_neighbors[vid].size());

        if (boundary_vertices.count(vid) && isWallBoundaryVertex(vid)) {
          auto wn = wallNeighborVertices(vid);
          if (wn.size() >= 2) {
            Vec2 arc_mid = original;
            if (bladeEdgeArcMidpoint(mesh.V[wn[0]], mesh.V[wn[1]], arc_mid)) {
              candidates.push_back(0.75 * original + 0.25 * arc_mid);
              candidates.push_back(0.5 * original + 0.5 * arc_mid);
            }
          }
          BladeProjection old_proj = projectToBladeSplineDetailed(original);
          BladeProjection avg_proj = projectToBladeSplineDetailed(avg);
          if (avg_proj.valid &&
              (!old_proj.valid || old_proj.branch == avg_proj.branch)) {
            candidates.push_back(0.75 * original + 0.25 * avg_proj.pt);
            candidates.push_back(0.5 * original + 0.5 * avg_proj.pt);
            candidates.push_back(avg_proj.pt);
          }
        } else {
          candidates.push_back(0.75 * original + 0.25 * avg);
          candidates.push_back(0.5 * original + 0.5 * avg);
        }
      }

      double local_best = best_patch_detJ;
      Vec2 local_best_pos = original;
      for (const Vec2 &candidate : candidates) {
        std::vector<Vec2> verts_trial = mesh.V;
        if (!localMovePreservesMesh(mesh, vid, node_to_elem, mesh.V, verts_trial, candidate)) {
          continue;
        }

        mesh.V[vid] = candidate;
        retunePatchQ2Geometry(mesh, patch_elems, boundary_edge_groups);
        const double trial_detJ = patchMetric();
        if (trial_detJ > local_best + 1e-8) {
          local_best = trial_detJ;
          local_best_pos = candidate;
        }
        mesh.V[vid] = original;
        retunePatchQ2Geometry(mesh, patch_elems, boundary_edge_groups);
      }

      if ((local_best_pos - original).norm() > 1e-12) {
        mesh.V[vid] = local_best_pos;
        retunePatchQ2Geometry(mesh, patch_elems, boundary_edge_groups);
        best_patch_detJ = local_best;
        iter_changed = true;
        any_change = true;
      }
    }
    if (!iter_changed) break;
    if (best_patch_detJ > 0.0) break;
  }

  mesh.computeGeometry();
  if (any_change) {
    std::cerr << "    local curved repair: seed=" << elem_idx
              << " patch_elems=" << patch_elems.size()
              << " min_detJ " << initial_patch_detJ
              << " -> " << best_patch_detJ << std::endl;
  }
  return any_change && best_patch_detJ > std::max(initial_patch_detJ, 0.0);
}

std::vector<std::vector<Vec4>> interpolateSolution(const std::vector<std::vector<Vec4>> &U_old, const RefinementMap &rmap, int ndof) {
  std::vector<std::vector<Vec4>> U_new(rmap.child_to_parent.size(), std::vector<Vec4>(ndof));
  for (int i=0; i<(int)U_new.size(); i++) U_new[i] = U_old[rmap.child_to_parent[i]];
  return U_new;
}

#include "Solver.hpp"
#include "Adjoint.hpp"
#include "MeshRefinement.hpp"
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <numeric>
#include <cstdint>
#include <string>
#include <vector>

namespace {

struct MeshValidationReport {
  double min_signed_area = std::numeric_limits<double>::max();
  double max_signed_area = -std::numeric_limits<double>::max();
  int nonpositive_area_count = 0;
  int tiny_area_count = 0;
  double min_sampled_detJ = std::numeric_limits<double>::max();
  int nonpositive_sampled_detJ_count = 0;
  int tiny_sampled_detJ_count = 0;
  int duplicate_boundary_edge_count = 0;
  int invalid_boundary_owner_count = 0;
  int invalid_interior_owner_count = 0;
  int bad_edge_owner_count = 0;
  int nonmanifold_edge_count = 0;
  int orphan_edge_count = 0;
  double min_angle_deg = 180.0;
  double worst_quality = std::numeric_limits<double>::max();
  int worst_quality_elem = -1;
  Vec2 worst_quality_centroid{0.0, 0.0};
  int min_angle_elem = -1;
  Vec2 min_angle_centroid{0.0, 0.0};
  int min_sampled_detJ_elem = -1;
  bool min_sampled_detJ_exact = false;
  Vec2 min_sampled_detJ_centroid{0.0, 0.0};
  Vec2 min_sampled_detJ_ref{0.0, 0.0};
  Vec2 min_sampled_detJ_phys{0.0, 0.0};
  std::vector<std::pair<int, Vec2>> nonpositive_area_samples;
  std::vector<std::pair<int, Vec2>> tiny_area_samples;
  std::vector<std::string> nonpositive_detJ_samples;
  std::vector<std::string> bad_edge_samples;
};

constexpr std::uint32_t kCompanionMeshHoMagic = 0x484f3031u; // "HO01"
constexpr std::uint32_t kCompanionMeshPeriodicMagic = 0x50473031u; // "PG01"

std::pair<int, int> sortedEdgeKey(int a, int b);
double triangleSignedArea(const Vec2 &a, const Vec2 &b, const Vec2 &c);
double angleDegBetween(const Vec2 &u, const Vec2 &v);

void writeCompanionMeshSnapshot(const Mesh &mesh, const std::string &mesh_file) {
  std::ofstream m_out(mesh_file, std::ios::binary);
  int Nn = static_cast<int>(mesh.V.size());
  m_out.write((char*)&Nn, sizeof(int));
  for (const auto &v : mesh.V) {
    m_out.write((char*)&v.x, sizeof(double));
    m_out.write((char*)&v.y, sizeof(double));
  }

  int Ne_m = static_cast<int>(mesh.E.size());
  m_out.write((char*)&Ne_m, sizeof(int));
  for (const auto &e : mesh.E) {
    int q = e.q_order;
    m_out.write((char*)&q,      sizeof(int));
    m_out.write((char*)&e.v[0], sizeof(int));
    m_out.write((char*)&e.v[1], sizeof(int));
    m_out.write((char*)&e.v[2], sizeof(int));
  }

  int Nb = static_cast<int>(mesh.BE.size());
  m_out.write((char*)&Nb, sizeof(int));
  for (const auto &be : mesh.BE) {
    int v0 = be.v[0], v1 = be.v[1], bidx = be.bIndex;
    m_out.write((char*)&v0, sizeof(int));
    m_out.write((char*)&v1, sizeof(int));
    m_out.write((char*)&bidx, sizeof(int));
  }

  int Nnames = static_cast<int>(mesh.Bname.size());
  m_out.write((char*)&Nnames, sizeof(int));
  for (const auto &name : mesh.Bname) {
    int len = static_cast<int>(name.length());
    m_out.write((char*)&len, sizeof(int));
    m_out.write(name.c_str(), len);
  }

  m_out.write((char*)&kCompanionMeshHoMagic, sizeof(kCompanionMeshHoMagic));
  for (const auto &e : mesh.E) {
    int nrow = (e.q_order > 1 && !e.ho_nodes.empty()) ? static_cast<int>(e.ho_nodes.size()) : 3;
    m_out.write((char*)&nrow, sizeof(int));
    if (nrow == 3) {
      m_out.write((char*)&e.v[0], sizeof(int));
      m_out.write((char*)&e.v[1], sizeof(int));
      m_out.write((char*)&e.v[2], sizeof(int));
    } else {
      m_out.write((char*)e.ho_nodes.data(), sizeof(int) * nrow);
    }
  }

  m_out.write((char*)&kCompanionMeshPeriodicMagic, sizeof(kCompanionMeshPeriodicMagic));
  int ngroups = static_cast<int>(mesh.periodicGroups.size());
  m_out.write((char*)&ngroups, sizeof(int));
  for (const auto &pg : mesh.periodicGroups) {
    int type_len = static_cast<int>(pg.type.size());
    int npairs = static_cast<int>(pg.pairs.size());
    m_out.write((char*)&type_len, sizeof(int));
    m_out.write(pg.type.c_str(), type_len);
    m_out.write((char*)&npairs, sizeof(int));
    for (const auto &pair : pg.pairs) {
      int n0 = pair.first;
      int n1 = pair.second;
      m_out.write((char*)&n0, sizeof(int));
      m_out.write((char*)&n1, sizeof(int));
    }
  }
}

std::vector<int> topResidualRepairSeeds(const Mesh &mesh,
                                        const std::vector<double> &cell_residuals,
                                        int max_seeds = 6) {
  std::vector<int> order(cell_residuals.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return cell_residuals[a] > cell_residuals[b]; });

  std::vector<int> seeds;
  seeds.reserve(max_seeds);
  for (int e : order) {
    if (e < 0 || e >= static_cast<int>(mesh.E.size())) continue;
    if (!std::isfinite(cell_residuals[e]) || cell_residuals[e] <= 0.0) continue;
    seeds.push_back(e);
    if (static_cast<int>(seeds.size()) >= max_seeds) break;
  }
  return seeds;
}

bool q1ProactiveRepairEnabled() {
  const char *env = std::getenv("AMR_Q1_PROACTIVE_REPAIR");
  return env != nullptr && std::string(env) != "0";
}

bool q1LimitedLinearProlongationEnabled() {
  const char *env = std::getenv("AMR_Q1_LINEAR_PROLONGATION");
  return env != nullptr && std::string(env) != "0";
}

double q1ProactiveRepairMedianAreaRatio() {
  const char *env = std::getenv("AMR_Q1_PROACTIVE_REPAIR_AREA_RATIO");
  return env != nullptr ? std::atof(env) : 3.0;
}

double q1ProactiveRepairMinAngleDeg() {
  const char *env = std::getenv("AMR_Q1_PROACTIVE_REPAIR_MIN_ANGLE");
  return env != nullptr ? std::atof(env) : 6.0;
}

double q1ProactiveRepairMinQuality() {
  const char *env = std::getenv("AMR_Q1_PROACTIVE_REPAIR_MIN_QUALITY");
  return env != nullptr ? std::atof(env) : 0.14;
}

int q1ProactiveRepairMaxSeeds() {
  const char *env = std::getenv("AMR_Q1_PROACTIVE_REPAIR_MAX_SEEDS");
  if (env == nullptr) return 10;
  try {
    return std::stoi(env);
  } catch (...) {
    return 10;
  }
}

int q1ProactiveRepairMinElems() {
  const char *env = std::getenv("AMR_Q1_PROACTIVE_REPAIR_MIN_ELEMS");
  if (env == nullptr) return 1500;
  try {
    return std::stoi(env);
  } catch (...) {
    return 1500;
  }
}

double q1PreSolveSliverRepairMinAngleDeg() {
  const char *env = std::getenv("AMR_Q1_SLIVER_REPAIR_MIN_ANGLE");
  return env != nullptr ? std::atof(env) : 2.5;
}

double q1PreSolveSliverRepairMinQuality() {
  const char *env = std::getenv("AMR_Q1_SLIVER_REPAIR_MIN_QUALITY");
  return env != nullptr ? std::atof(env) : 0.06;
}

int q1PreSolveTransitionRepairMinElems() {
  const char *env = std::getenv("AMR_Q1_TRANSITION_REPAIR_MIN_ELEMS");
  if (env == nullptr) return 2000;
  try {
    return std::stoi(env);
  } catch (...) {
    return 2000;
  }
}

int q1PreSolveTransitionRepairMaxSeeds() {
  const char *env = std::getenv("AMR_Q1_TRANSITION_REPAIR_MAX_SEEDS");
  if (env == nullptr) return 6;
  try {
    return std::stoi(env);
  } catch (...) {
    return 6;
  }
}

struct Q1PreSolveRepairCandidate {
  int elem = -1;
  double median_area_ratio = 1.0;
  double max_area_ratio = 1.0;
  double min_angle_deg = 180.0;
  double quality = 1.0;
  double score = 0.0;
};

std::vector<int> topQ1PreSolveRepairSeeds(const Mesh &mesh,
                                          const RefinementMap &rmap,
                                          int max_seeds = 10) {
  if (mesh.q_order_global > 1 || mesh.E.empty() || max_seeds <= 0) return {};
  if (static_cast<int>(rmap.child_to_parent.size()) != static_cast<int>(mesh.E.size())) {
    return {};
  }

  auto lower_copy = [](std::string s) {
    for (char &c : s) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
  };
  std::vector<char> wall_vertex(mesh.V.size(), 0);
  for (const auto &be : mesh.BE) {
    const int g = be.bIndex;
    if (g < 0 || g >= static_cast<int>(mesh.Bname.size())) continue;
    if (lower_copy(mesh.Bname[g]) != "wall") continue;
    if (be.v[0] >= 0 && be.v[0] < static_cast<int>(wall_vertex.size()))
      wall_vertex[be.v[0]] = 1;
    if (be.v[1] >= 0 && be.v[1] < static_cast<int>(wall_vertex.size()))
      wall_vertex[be.v[1]] = 1;
  }

  std::vector<int> parent_refine_count;
  int max_parent = -1;
  for (int parent : rmap.child_to_parent) max_parent = std::max(max_parent, parent);
  if (max_parent < 0) return {};
  parent_refine_count.assign(max_parent + 1, 0);
  for (int parent : rmap.child_to_parent) {
    if (parent >= 0 && parent < static_cast<int>(parent_refine_count.size())) {
      parent_refine_count[parent]++;
    }
  }

  std::set<int> refined_parents;
  for (int parent = 0; parent < static_cast<int>(parent_refine_count.size()); ++parent) {
    if (parent_refine_count[parent] > 1) refined_parents.insert(parent);
  }
  if (refined_parents.empty()) return {};

  std::vector<std::vector<int>> node_to_elem(mesh.V.size());
  std::map<std::pair<int, int>, std::vector<int>> edge_to_elem;
  std::vector<double> areas(mesh.E.size(), 0.0);
  std::vector<double> min_angles(mesh.E.size(), 180.0);
  std::vector<double> qualities(mesh.E.size(), 1.0);
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    const auto &elem = mesh.E[e];
    if (elem.q_order != 1) continue;
    const Vec2 &a = mesh.V[elem.v[0]];
    const Vec2 &b = mesh.V[elem.v[1]];
    const Vec2 &c = mesh.V[elem.v[2]];
    areas[e] = std::abs(triangleSignedArea(a, b, c));
    min_angles[e] = std::min({angleDegBetween(b - a, c - a),
                              angleDegBetween(a - b, c - b),
                              angleDegBetween(a - c, b - c)});
    const Vec2 ab = b - a, bc = c - b, ca = a - c;
    const double l2sum = ab.normSq() + bc.normSq() + ca.normSq();
    qualities[e] =
        l2sum > 0.0 ? 4.0 * std::sqrt(3.0) * areas[e] / l2sum : 0.0;
    for (int k = 0; k < 3; ++k) node_to_elem[elem.v[k]].push_back(e);
    edge_to_elem[sortedEdgeKey(elem.v[0], elem.v[1])].push_back(e);
    edge_to_elem[sortedEdgeKey(elem.v[1], elem.v[2])].push_back(e);
    edge_to_elem[sortedEdgeKey(elem.v[2], elem.v[0])].push_back(e);
  }

  std::set<int> seed_elems;
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    if (mesh.E[e].q_order != 1) continue;
    const int parent = rmap.child_to_parent[e];
    if (refined_parents.count(parent)) seed_elems.insert(e);
  }
  if (seed_elems.empty()) return {};

  std::set<int> candidate_elems = seed_elems;
  for (int e : seed_elems) {
    const auto &elem = mesh.E[e];
    for (int k = 0; k < 3; ++k) {
      for (int adj : node_to_elem[elem.v[k]]) candidate_elems.insert(adj);
    }
  }

  const double ratio_floor = q1ProactiveRepairMedianAreaRatio();
  const double angle_floor = q1ProactiveRepairMinAngleDeg();
  const double quality_floor = q1ProactiveRepairMinQuality();
  std::vector<Q1PreSolveRepairCandidate> candidates;
  for (int e : candidate_elems) {
    if (e < 0 || e >= static_cast<int>(mesh.E.size())) continue;
    const auto &elem = mesh.E[e];
    if (elem.q_order != 1) continue;
    bool touches_wall_vertex = false;
    for (int k = 0; k < 3; ++k) {
      if (elem.v[k] >= 0 && elem.v[k] < static_cast<int>(wall_vertex.size()) &&
          wall_vertex[elem.v[k]]) {
        touches_wall_vertex = true;
        break;
      }
    }
    if (!touches_wall_vertex) continue;

    std::vector<double> neighbor_areas;
    const std::pair<int, int> edges[3] = {
        sortedEdgeKey(elem.v[0], elem.v[1]),
        sortedEdgeKey(elem.v[1], elem.v[2]),
        sortedEdgeKey(elem.v[2], elem.v[0]),
    };
    for (const auto &edge : edges) {
      auto it = edge_to_elem.find(edge);
      if (it == edge_to_elem.end()) continue;
      for (int nb : it->second) {
        if (nb == e || nb < 0 || nb >= static_cast<int>(mesh.E.size())) continue;
        if (mesh.E[nb].q_order != 1) continue;
        neighbor_areas.push_back(areas[nb]);
      }
    }
    if (neighbor_areas.empty()) continue;
    std::sort(neighbor_areas.begin(), neighbor_areas.end());
    const double nb_median =
        neighbor_areas[neighbor_areas.size() / 2];
    const double nb_max = neighbor_areas.back();
    if (!(nb_median > 1e-14) || !(nb_max > 1e-14)) continue;

    const double median_ratio = std::max(areas[e], nb_median) / std::min(areas[e], nb_median);
    const double max_ratio = std::max(areas[e], nb_max) / std::min(areas[e], nb_max);
    const bool flagged =
        median_ratio >= ratio_floor ||
        min_angles[e] < angle_floor ||
        qualities[e] < quality_floor;
    if (!flagged) continue;

    Q1PreSolveRepairCandidate cand;
    cand.elem = e;
    cand.median_area_ratio = median_ratio;
    cand.max_area_ratio = max_ratio;
    cand.min_angle_deg = min_angles[e];
    cand.quality = qualities[e];
    cand.score =
        8.0 * std::max(0.0, median_ratio - ratio_floor) +
        1.5 * std::max(0.0, angle_floor - min_angles[e]) +
        40.0 * std::max(0.0, quality_floor - qualities[e]);
    candidates.push_back(cand);
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Q1PreSolveRepairCandidate &a,
               const Q1PreSolveRepairCandidate &b) {
              if (a.score != b.score) return a.score > b.score;
              if (a.median_area_ratio != b.median_area_ratio)
                return a.median_area_ratio > b.median_area_ratio;
              if (a.min_angle_deg != b.min_angle_deg)
                return a.min_angle_deg < b.min_angle_deg;
              return a.elem < b.elem;
            });

  std::vector<int> seeds;
  seeds.reserve(max_seeds);
  for (const auto &cand : candidates) {
    seeds.push_back(cand.elem);
    if (static_cast<int>(seeds.size()) >= max_seeds) break;
  }
  return seeds;
}

std::vector<std::vector<Vec4>> interpolateQ1P0Solution(
    const Mesh &old_mesh,
    const Mesh &new_mesh,
    const std::vector<Vec4> &U_old_avg,
    const RefinementMap &rmap,
    double gamma) {
  std::vector<std::vector<Vec4>> U_new(
      rmap.child_to_parent.size(), std::vector<Vec4>(1));
  if (U_old_avg.empty()) return U_new;

  int max_parent = -1;
  for (int parent : rmap.child_to_parent) max_parent = std::max(max_parent, parent);
  std::vector<int> parent_refine_count(std::max(0, max_parent + 1), 0);
  std::vector<std::vector<int>> parent_to_children(std::max(0, max_parent + 1));
  for (int e = 0; e < static_cast<int>(rmap.child_to_parent.size()); ++e) {
    const int parent = rmap.child_to_parent[e];
    if (parent < 0 || parent >= static_cast<int>(parent_refine_count.size())) continue;
    parent_refine_count[parent]++;
    parent_to_children[parent].push_back(e);
  }

  std::map<std::pair<int, int>, std::vector<int>> edge_to_elem;
  std::vector<std::set<int>> neighbors(old_mesh.E.size());
  for (int e = 0; e < static_cast<int>(old_mesh.E.size()); ++e) {
    const auto &elem = old_mesh.E[e];
    if (elem.q_order != 1) continue;
    edge_to_elem[sortedEdgeKey(elem.v[0], elem.v[1])].push_back(e);
    edge_to_elem[sortedEdgeKey(elem.v[1], elem.v[2])].push_back(e);
    edge_to_elem[sortedEdgeKey(elem.v[2], elem.v[0])].push_back(e);
  }
  for (const auto &[edge, elems] : edge_to_elem) {
    (void)edge;
    if (elems.size() != 2) continue;
    neighbors[elems[0]].insert(elems[1]);
    neighbors[elems[1]].insert(elems[0]);
  }

  struct GradientFit {
    Vec4 gx{0.0, 0.0, 0.0, 0.0};
    Vec4 gy{0.0, 0.0, 0.0, 0.0};
    Vec4 umin{0.0, 0.0, 0.0, 0.0};
    Vec4 umax{0.0, 0.0, 0.0, 0.0};
    bool valid = false;
  };
  std::vector<GradientFit> grads(old_mesh.E.size());
  for (int e = 0; e < static_cast<int>(old_mesh.E.size()); ++e) {
    if (e >= static_cast<int>(U_old_avg.size()) || old_mesh.E[e].q_order != 1) continue;
    GradientFit fit;
    fit.umin = U_old_avg[e];
    fit.umax = U_old_avg[e];
    double m_xx = 0.0, m_xy = 0.0, m_yy = 0.0;
    std::array<double, 4> rhs_x{0.0, 0.0, 0.0, 0.0};
    std::array<double, 4> rhs_y{0.0, 0.0, 0.0, 0.0};
    for (int nb : neighbors[e]) {
      if (nb < 0 || nb >= static_cast<int>(U_old_avg.size())) continue;
      const Vec2 d = old_mesh.centroids[nb] - old_mesh.centroids[e];
      const double dist2 = std::max(d.normSq(), 1e-12);
      const double w = 1.0 / dist2;
      m_xx += w * d.x * d.x;
      m_xy += w * d.x * d.y;
      m_yy += w * d.y * d.y;
      const Vec4 du = U_old_avg[nb] - U_old_avg[e];
      for (int k = 0; k < 4; ++k) {
        rhs_x[k] += w * d.x * du[k];
        rhs_y[k] += w * d.y * du[k];
        fit.umin[k] = std::min(fit.umin[k], U_old_avg[nb][k]);
        fit.umax[k] = std::max(fit.umax[k], U_old_avg[nb][k]);
      }
    }
    const double det = m_xx * m_yy - m_xy * m_xy;
    if (neighbors[e].size() >= 2 && std::abs(det) > 1e-14) {
      for (int k = 0; k < 4; ++k) {
        fit.gx[k] = ( rhs_x[k] * m_yy - rhs_y[k] * m_xy) / det;
        fit.gy[k] = (-rhs_x[k] * m_xy + rhs_y[k] * m_xx) / det;
      }
      fit.valid = true;
    }
    grads[e] = fit;
  }

  auto makePhysical = [&](const Vec4 &parent, Vec4 child) {
    State s(child, gamma);
    if (std::isfinite(s.rho()) && std::isfinite(s.p()) &&
        s.rho() > 1e-8 && s.p() > 1e-8) {
      return child;
    }
    double theta = 0.5;
    while (theta >= 1e-6) {
      Vec4 blended = theta * child + (1.0 - theta) * parent;
      State sb(blended, gamma);
      if (std::isfinite(sb.rho()) && std::isfinite(sb.p()) &&
          sb.rho() > 1e-8 && sb.p() > 1e-8) {
        return blended;
      }
      theta *= 0.5;
    }
    return parent;
  };

  for (int e = 0; e < static_cast<int>(U_new.size()); ++e) {
    const int parent = rmap.child_to_parent[e];
    if (parent < 0 || parent >= static_cast<int>(U_old_avg.size())) continue;
    Vec4 value = U_old_avg[parent];
    if (parent < static_cast<int>(parent_refine_count.size()) &&
        parent_refine_count[parent] > 1 &&
        parent < static_cast<int>(grads.size()) &&
        grads[parent].valid) {
      const Vec2 d = new_mesh.centroids[e] - old_mesh.centroids[parent];
      value += grads[parent].gx * d.x + grads[parent].gy * d.y;
      for (int k = 0; k < 4; ++k) {
        value[k] = std::max(grads[parent].umin[k],
                            std::min(grads[parent].umax[k], value[k]));
      }
    }
    U_new[e][0] = value;
  }

  for (int parent = 0; parent < static_cast<int>(parent_to_children.size()); ++parent) {
    const auto &children = parent_to_children[parent];
    if (children.empty()) continue;
    double total_area = 0.0;
    Vec4 avg{0.0, 0.0, 0.0, 0.0};
    for (int child : children) {
      if (child < 0 || child >= static_cast<int>(new_mesh.areas.size())) continue;
      total_area += new_mesh.areas[child];
      avg += U_new[child][0] * new_mesh.areas[child];
    }
    if (!(total_area > 0.0) || parent < 0 || parent >= static_cast<int>(U_old_avg.size())) {
      continue;
    }
    avg = avg / total_area;
    const Vec4 correction = U_old_avg[parent] - avg;
    for (int child : children) {
      U_new[child][0] = makePhysical(U_old_avg[parent], U_new[child][0] + correction);
    }
  }

  return U_new;
}

bool transferRefinedSolutionToCurrentMesh(
    FiniteVolumeSolver &solver,
    const Mesh &mesh_before_refine,
    const std::vector<Vec4> &U_before_refine,
    const std::vector<std::vector<Vec4>> &U_dg_before_refine,
    const RefinementMap &rmap,
    const char *context = nullptr) {
  const bool used_linear_q1 =
      solver.ndof_per_elem == 1 && solver.mesh.q_order_global <= 1 &&
      q1LimitedLinearProlongationEnabled();
  if (used_linear_q1) {
    solver.U_dg = interpolateQ1P0Solution(
        mesh_before_refine, solver.mesh, U_before_refine, rmap, solver.gamma);
    if (context != nullptr) {
      std::cerr << "  " << context
                << ": using limited linear p0 prolongation on the refined q1 mesh."
                << std::endl;
    }
  } else {
    solver.U_dg =
        interpolateSolution(U_dg_before_refine, rmap, solver.ndof_per_elem);
  }
  solver.U.resize(solver.mesh.E.size());
  for (size_t e = 0; e < solver.mesh.E.size(); ++e) {
    solver.U[e] = solver.cellAverage(solver.U_dg[e]);
  }
  return used_linear_q1;
}

std::pair<int, int> sortedEdgeKey(int a, int b) {
  return {std::min(a, b), std::max(a, b)};
}

void printElementGeometryDebug(const Mesh &mesh, int elem_idx,
                               const std::string &label) {
  if (elem_idx < 0 || elem_idx >= static_cast<int>(mesh.E.size())) return;
  const auto &elem = mesh.E[elem_idx];
  std::cerr << "    " << label << ": e=" << elem_idx
            << " q=" << elem.q_order
            << " corners=(" << elem.v[0] << ", " << elem.v[1] << ", " << elem.v[2]
            << ")";
  if (!elem.ho_nodes.empty()) {
    std::cerr << " ho_nodes=(";
    for (size_t i = 0; i < elem.ho_nodes.size(); ++i) {
      if (i) std::cerr << ", ";
      std::cerr << elem.ho_nodes[i];
    }
    std::cerr << ")";
  }
  std::cerr << std::endl;

  const auto print_node = [&](int vid, const std::string &node_label) {
    if (vid < 0 || vid >= static_cast<int>(mesh.V.size())) return;
    const auto &v = mesh.V[vid];
    std::cerr << "      " << node_label << " " << vid << " = (" << v.x
              << ", " << v.y << ")" << std::endl;
  };
  print_node(elem.v[0], "corner");
  print_node(elem.v[1], "corner");
  print_node(elem.v[2], "corner");
  for (int vid : elem.ho_nodes) {
    if (vid == elem.v[0] || vid == elem.v[1] || vid == elem.v[2]) continue;
    print_node(vid, "ho");
  }
}

void printElementOneRingDebug(const Mesh &mesh, int elem_idx,
                              const std::string &label) {
  if (elem_idx < 0 || elem_idx >= static_cast<int>(mesh.E.size())) return;
  std::set<int> ring_elems{elem_idx};
  const auto &elem = mesh.E[elem_idx];
  std::set<int> ring_nodes{elem.v[0], elem.v[1], elem.v[2]};
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    const auto &other = mesh.E[e];
    if (ring_nodes.count(other.v[0]) || ring_nodes.count(other.v[1]) ||
        ring_nodes.count(other.v[2])) {
      ring_elems.insert(e);
    }
  }
  std::cerr << "    " << label << " one-ring elements:";
  for (int e : ring_elems) std::cerr << " " << e;
  std::cerr << std::endl;
  for (int e : ring_elems) {
    printElementGeometryDebug(mesh, e, "one-ring");
  }
}

double triangleSignedArea(const Vec2 &a, const Vec2 &b, const Vec2 &c) {
  return 0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
}

double angleDegBetween(const Vec2 &u, const Vec2 &v) {
  double nu = u.norm();
  double nv = v.norm();
  if (nu <= 0.0 || nv <= 0.0) return 0.0;
  double c = (u.x * v.x + u.y * v.y) / (nu * nv);
  c = std::max(-1.0, std::min(1.0, c));
  return std::acos(c) * 180.0 / M_PI;
}

MeshValidationReport validateRefinedMesh(const Mesh &mesh) {
  MeshValidationReport report;
  constexpr int kMaxValidationSamples = 5;
  constexpr int kCurvedDetJSampleDenom = 4;

  std::map<std::pair<int, int>, int> elem_edge_counts;
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    const auto &elem = mesh.E[e];
    const Vec2 &a = mesh.V[elem.v[0]];
    const Vec2 &b = mesh.V[elem.v[1]];
    const Vec2 &c = mesh.V[elem.v[2]];
    const Vec2 centroid = (a + b + c) / 3.0;

    double sarea = triangleSignedArea(a, b, c);
    report.min_signed_area = std::min(report.min_signed_area, sarea);
    report.max_signed_area = std::max(report.max_signed_area, sarea);
    if (!(sarea > 0.0)) {
      report.nonpositive_area_count++;
      if (static_cast<int>(report.nonpositive_area_samples.size()) < kMaxValidationSamples) {
        report.nonpositive_area_samples.push_back({e, centroid});
      }
    }
    if (std::abs(sarea) < 1e-12) {
      report.tiny_area_count++;
      if (static_cast<int>(report.tiny_area_samples.size()) < kMaxValidationSamples) {
        report.tiny_area_samples.push_back({e, centroid});
      }
    }

    if (elem.q_order == 2) {
      const double detJ_scale = std::max(std::abs(2.0 * sarea), 1e-14);
      CurvedElementDetJMinimum detJ_info;
      if (exactQ2DetJMinimum(mesh, e, detJ_info)) {
        if (detJ_info.detJ < report.min_sampled_detJ) {
          report.min_sampled_detJ = detJ_info.detJ;
          report.min_sampled_detJ_elem = e;
          report.min_sampled_detJ_exact = detJ_info.exact;
          report.min_sampled_detJ_centroid = centroid;
          report.min_sampled_detJ_ref = detJ_info.ref;
          report.min_sampled_detJ_phys = detJ_info.x;
        }
        if (!(detJ_info.detJ > 0.0)) {
          report.nonpositive_sampled_detJ_count++;
          if (static_cast<int>(report.nonpositive_detJ_samples.size()) <
              kMaxValidationSamples) {
            report.nonpositive_detJ_samples.push_back(
                "e=" + std::to_string(e) +
                " centroid=(" + std::to_string(centroid.x) + ", " +
                std::to_string(centroid.y) + ")" +
                " ref=(" + std::to_string(detJ_info.ref.x) + ", " +
                std::to_string(detJ_info.ref.y) + ")" +
                " x=(" + std::to_string(detJ_info.x.x) + ", " +
                std::to_string(detJ_info.x.y) + ")" +
                " detJ=" + std::to_string(detJ_info.detJ));
          }
        } else if (detJ_info.detJ < 1e-3 * detJ_scale) {
          report.tiny_sampled_detJ_count++;
        }
      }
    } else if (elem.q_order > 1) {
      const double detJ_scale = std::max(std::abs(2.0 * sarea), 1e-14);
      for (int i = 0; i <= kCurvedDetJSampleDenom; ++i) {
        for (int j = 0; j <= kCurvedDetJSampleDenom - i; ++j) {
          const double xi = static_cast<double>(i) / kCurvedDetJSampleDenom;
          const double eta = static_cast<double>(j) / kCurvedDetJSampleDenom;
          const ElementGeomEval geom = mesh.evaluateElementGeometry(e, xi, eta);
          if (geom.detJ < report.min_sampled_detJ) {
            report.min_sampled_detJ = geom.detJ;
            report.min_sampled_detJ_elem = e;
            report.min_sampled_detJ_centroid = centroid;
            report.min_sampled_detJ_ref = {xi, eta};
            report.min_sampled_detJ_phys = geom.x;
          }
          if (!(geom.detJ > 0.0)) {
            report.nonpositive_sampled_detJ_count++;
            if (static_cast<int>(report.nonpositive_detJ_samples.size()) <
                kMaxValidationSamples) {
              report.nonpositive_detJ_samples.push_back(
                  "e=" + std::to_string(e) +
                  " centroid=(" + std::to_string(centroid.x) + ", " +
                  std::to_string(centroid.y) + ")" +
                  " ref=(" + std::to_string(xi) + ", " + std::to_string(eta) + ")" +
                  " x=(" + std::to_string(geom.x.x) + ", " + std::to_string(geom.x.y) + ")" +
                  " detJ=" + std::to_string(geom.detJ));
            }
          } else if (geom.detJ < 1e-3 * detJ_scale) {
            report.tiny_sampled_detJ_count++;
          }
        }
      }
    }

    Vec2 ab = b - a, bc = c - b, ca = a - c;
    double lab = ab.norm(), lbc = bc.norm(), lca = ca.norm();
    double l2sum = lab * lab + lbc * lbc + lca * lca;
    if (l2sum > 0.0) {
      double quality = 4.0 * std::sqrt(3.0) * std::abs(sarea) / l2sum;
      if (quality < report.worst_quality) {
        report.worst_quality = quality;
        report.worst_quality_elem = e;
        report.worst_quality_centroid = centroid;
      }
    }

    double ang_a = angleDegBetween(b - a, c - a);
    double ang_b = angleDegBetween(a - b, c - b);
    double ang_c = angleDegBetween(a - c, b - c);
    double min_elem_angle = std::min(ang_a, std::min(ang_b, ang_c));
    if (min_elem_angle < report.min_angle_deg) {
      report.min_angle_deg = min_elem_angle;
      report.min_angle_elem = e;
      report.min_angle_centroid = centroid;
    }

    elem_edge_counts[sortedEdgeKey(elem.v[0], elem.v[1])]++;
    elem_edge_counts[sortedEdgeKey(elem.v[1], elem.v[2])]++;
    elem_edge_counts[sortedEdgeKey(elem.v[2], elem.v[0])]++;
  }

  std::map<std::pair<int, int>, int> boundary_edge_counts;
  std::set<std::pair<int, int>> boundary_edge_keys;
  for (const auto &be : mesh.BE) {
    auto key = sortedEdgeKey(be.v[0], be.v[1]);
    boundary_edge_counts[key]++;
    boundary_edge_keys.insert(key);
    if (be.elemL < 0 || be.elemL >= static_cast<int>(mesh.E.size()))
      report.invalid_boundary_owner_count++;
  }
  for (const auto &[edge, count] : boundary_edge_counts) {
    if (count > 1) report.duplicate_boundary_edge_count++;
  }

  std::set<std::pair<int, int>> interior_edge_keys;
  for (const auto &ie : mesh.IE) {
    interior_edge_keys.insert(sortedEdgeKey(ie.v[0], ie.v[1]));
    if (ie.elemL < 0 || ie.elemL >= static_cast<int>(mesh.E.size()) ||
        ie.elemR < 0 || ie.elemR >= static_cast<int>(mesh.E.size()) ||
        ie.elemL == ie.elemR) {
      report.invalid_interior_owner_count++;
    }
  }

  for (const auto &[edge, count] : elem_edge_counts) {
    const bool in_be = boundary_edge_keys.count(edge) > 0;
    const bool in_ie = interior_edge_keys.count(edge) > 0;

    if (count == 1 && !in_be && !in_ie) {
      report.bad_edge_owner_count++;
      if (static_cast<int>(report.bad_edge_samples.size()) < kMaxValidationSamples) {
        report.bad_edge_samples.push_back("edge=(" + std::to_string(edge.first) + ", " +
                                          std::to_string(edge.second) +
                                          ") elem_count=1 missing_boundary_owner");
      }
    }
    if (count == 2 && !in_ie) {
      report.bad_edge_owner_count++;
      if (static_cast<int>(report.bad_edge_samples.size()) < kMaxValidationSamples) {
        report.bad_edge_samples.push_back("edge=(" + std::to_string(edge.first) + ", " +
                                          std::to_string(edge.second) +
                                          ") elem_count=2 missing_interior_owner");
      }
    }
    if (count > 2) {
      report.bad_edge_owner_count++;
      if (static_cast<int>(report.bad_edge_samples.size()) < kMaxValidationSamples) {
        report.bad_edge_samples.push_back("edge=(" + std::to_string(edge.first) + ", " +
                                          std::to_string(edge.second) + ") elem_count=" +
                                          std::to_string(count));
      }
    }
  }

  for (const auto &[edge, count] : elem_edge_counts) {
    const bool in_be = boundary_edge_keys.count(edge) > 0;
    const bool in_ie = interior_edge_keys.count(edge) > 0;
    if (count == 1 && !in_be && !in_ie) report.orphan_edge_count++;
    else if (count > 2) report.nonmanifold_edge_count++;
  }

  if (mesh.E.empty()) {
    report.min_signed_area = 0.0;
    report.max_signed_area = 0.0;
    report.min_angle_deg = 0.0;
    report.worst_quality = 0.0;
    report.min_sampled_detJ = 0.0;
  } else if (report.worst_quality == std::numeric_limits<double>::max()) {
    report.worst_quality = 0.0;
  }
  if (report.min_sampled_detJ == std::numeric_limits<double>::max()) {
    report.min_sampled_detJ = 0.0;
  }

  return report;
}

void printMeshValidationReport(const MeshValidationReport &r) {
  std::cerr << "  Mesh validation:" << std::endl;
  std::cerr << "    signed area range = [" << r.min_signed_area << ", "
            << r.max_signed_area << "]" << std::endl;
  std::cerr << "    non-positive elements = " << r.nonpositive_area_count
            << " | tiny-area elements = " << r.tiny_area_count << std::endl;
  std::cerr << "    min angle = " << r.min_angle_deg
            << " deg | worst quality = " << r.worst_quality << std::endl;
  if (r.min_sampled_detJ_elem >= 0) {
    std::cerr << "    min "
              << (r.min_sampled_detJ_exact ? "exact q2 detJ" : "sampled detJ")
              << " = " << r.min_sampled_detJ
              << " on curved elem " << r.min_sampled_detJ_elem
              << " @ centroid=(" << r.min_sampled_detJ_centroid.x << ", "
              << r.min_sampled_detJ_centroid.y << ")"
              << " @ ref=(" << r.min_sampled_detJ_ref.x << ", "
              << r.min_sampled_detJ_ref.y << ")"
              << " x=(" << r.min_sampled_detJ_phys.x << ", "
              << r.min_sampled_detJ_phys.y << ")"
              << " | non-positive sampled detJ = "
              << r.nonpositive_sampled_detJ_count
              << " | tiny sampled detJ = " << r.tiny_sampled_detJ_count
              << std::endl;
  }
  if (r.min_angle_elem >= 0) {
    std::cerr << "    min-angle element = " << r.min_angle_elem
              << " @ centroid=(" << r.min_angle_centroid.x << ", "
              << r.min_angle_centroid.y << ")" << std::endl;
  }
  if (r.worst_quality_elem >= 0) {
    std::cerr << "    worst-quality element = " << r.worst_quality_elem
              << " @ centroid=(" << r.worst_quality_centroid.x << ", "
              << r.worst_quality_centroid.y << ")" << std::endl;
  }
  if (!r.nonpositive_area_samples.empty()) {
    std::cerr << "    non-positive area samples:";
    for (const auto &[e, c] : r.nonpositive_area_samples) {
      std::cerr << " e=" << e << "@(" << c.x << ", " << c.y << ")";
    }
    std::cerr << std::endl;
  }
  if (!r.tiny_area_samples.empty()) {
    std::cerr << "    tiny-area samples:";
    for (const auto &[e, c] : r.tiny_area_samples) {
      std::cerr << " e=" << e << "@(" << c.x << ", " << c.y << ")";
    }
    std::cerr << std::endl;
  }
  if (!r.nonpositive_detJ_samples.empty()) {
    std::cerr << "    non-positive detJ samples:";
    for (const auto &sample : r.nonpositive_detJ_samples) {
      std::cerr << " [" << sample << "]";
    }
    std::cerr << std::endl;
  }
  std::cerr << "    bad edge ownership = " << r.bad_edge_owner_count
            << " | nonmanifold edges = " << r.nonmanifold_edge_count
            << " | orphan edges = " << r.orphan_edge_count << std::endl;
  if (!r.bad_edge_samples.empty()) {
    std::cerr << "    bad edge samples:";
    for (const auto &sample : r.bad_edge_samples) std::cerr << " [" << sample << "]";
    std::cerr << std::endl;
  }
  std::cerr << "    duplicate boundary edges = "
            << r.duplicate_boundary_edge_count
            << " | invalid boundary owners = "
            << r.invalid_boundary_owner_count
            << " | invalid interior owners = "
            << r.invalid_interior_owner_count << std::endl;
}

} // namespace

int main(int argc, char **argv) {
  std::cout << std::unitbuf; // Enable unbuffered output for immediate feedback
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <meshfile> [order] [CFL] [fluxname] [itercap] "
              << "[steady/unsteady/freestream] [ic_file] [t_end] "
              << "[--p0-itercap <iters>] "
              << "[--save-after <time>] "
              << "[--save-interval-time <dt_save>] "
              << "[--checkpoint-interval-time <dt_chk>] "
              << "[--map-ic <coarse_meshfile> <coarse_statefile>]"
              << "\n\nArguments:"
              << "\n  meshfile      : Grid file (e.g., grids/2k.gri)"
              << "\n  order         : DG polynomial order (0, 1, 2, or 3) [default: 0]"
              << "\n  CFL           : CFL number [default: 1.0]"
              << "\n  fluxname      : Flux scheme (roe or hlle) [default: roe]"
              << "\n  itercap       : Maximum iterations [default: 1e6]"
              << "\n  steady/unsteady/freestream : Solution mode [default: steady]"
              << "\n  ic_file       : Initial condition file (optional)"
              << "\n  t_end         : End time for unsteady (optional, requires ic_file)"
              << "\n  --p0-itercap  : Separate iteration cap for auto-chained p=0 seed solve"
              << "\n  --save-after  : For unsteady runs, only save snapshots once time >= this value"
              << "\n  --save-interval-time : For unsteady runs, save checkpoints every dt in physical time"
              << "\n  --checkpoint-interval-time : For unsteady runs, overwrite one rolling checkpoint every dt in physical time"
              << "\n  --map-ic      : Load IC from coarser mesh (optional)"
              << "\n  --final-ar-cleanup : After adjoint adaptation, do one extra refinement pass for cells above a max aspect ratio and re-solve"
              << "\n  --smooth-iters : Set post-refinement mesh smoothing iterations"
              << "\n  --wall-geom-tol : Force wall-edge refinement when blade chord error exceeds this tolerance"
              << "\n  --write-adjoint-indicators : Solve adjoint on the current fixed mesh and write element indicators to the given file"
              << "\n\nNotes:"
              << "\n  - For p>0 steady runs without ic_file: automatically converges p=0 first"
              << "\n  - Output files tagged by order: steady_<mesh>_p<order>_results.bin"
              << "\n  - Full DG coefficients are also written to *_results_dg.bin"
              << "\n  - freestream mode bypasses physical BCs for preservation testing"
              << std::endl;
    return 1;
  }

  std::string meshfile = argv[1];
  int p_order = 0;  // DG polynomial order (0, 1, 2, 3)
  double cfl = 1.0;
  std::string fluxname = "roe";
  int itercap = 1e6;
  bool unsteady = false;
  bool freestream_mode = false;
  std::string ic_file = "";
  double t_end = -1.0;  // negative means run until itercap
  bool use_mapped_ic = false;
  int p0_itercap = -1;
  double save_after = 0.0;
  double save_interval_time = -1.0;
  double checkpoint_interval_time = -1.0;
  std::string coarse_meshfile = "";
  std::string coarse_statefile = "";
  bool adjoint_adapt = false;
  double adjoint_tol = 1e-4;
  int adapt_max_cycles = 10;
  double adapt_fraction = 0.25;
  double final_ar_cleanup = 0.0;
  int smooth_iters = 0;
  double wall_geom_tol = 0.15;
  bool write_adjoint_indicators = false;
  std::string adjoint_indicators_file = "";

  auto is_flag_token = [](const char *s) {
    return std::string(s).rfind("--", 0) == 0;
  };

  auto infer_unsteady_restart_time = [](const std::string &filename) -> double {
    std::string basename = std::filesystem::path(filename).filename().string();
    const std::string prefix = "results_";
    if (basename.rfind(prefix, 0) != 0) {
      return -1.0;
    }
    std::size_t start = prefix.size();
    std::size_t end = basename.find('_', start);
    if (end == std::string::npos) {
      return -1.0;
    }
    std::string time_token = basename.substr(start, end - start);
    char *parse_end = nullptr;
    double t = std::strtod(time_token.c_str(), &parse_end);
    if (parse_end == time_token.c_str() || *parse_end != '\0') {
      return -1.0;
    }
    return t;
  };

  if (argc >= 3 && !is_flag_token(argv[2])) {
    p_order = std::stoi(argv[2]);  // Parse as integer for DG order
    if (p_order < 0 || p_order > 3) {
      std::cerr << "Error: order must be 0, 1, 2, or 3" << std::endl;
      return 1;
    }
  }
  if (argc >= 4 && !is_flag_token(argv[3])) {
    cfl = std::stod(argv[3]);
  }
  if (argc >= 5 && !is_flag_token(argv[4])) {
    fluxname = argv[4];
  }
  if (argc >= 6 && !is_flag_token(argv[5])) {
    itercap = std::stoi(argv[5]);
  }
  if (argc >= 7 && !is_flag_token(argv[6])) {
    std::string mode = argv[6];
    unsteady = (mode == "unsteady");
    freestream_mode = (mode == "freestream");
  }
  if (argc >= 8 && !is_flag_token(argv[7])) {
    ic_file = argv[7];
  }
  if (argc >= 9 && !is_flag_token(argv[7]) && !is_flag_token(argv[8])) {
    t_end = std::stod(argv[8]);
  }

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--map-ic") {
      if (i + 2 >= argc) {
        std::cerr << "Error: --map-ic requires <coarse_meshfile> <coarse_statefile>"
                  << std::endl;
        return 1;
      }
      use_mapped_ic = true;
      coarse_meshfile = argv[i + 1];
      coarse_statefile = argv[i + 2];
      i += 2;
    } else if (arg == "--p0-itercap") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --p0-itercap requires <iters>" << std::endl;
        return 1;
      }
      p0_itercap = std::stoi(argv[i + 1]);
      i += 1;
    } else if (arg == "--save-after") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --save-after requires <time>" << std::endl;
        return 1;
      }
      save_after = std::stod(argv[i + 1]);
      i += 1;
    } else if (arg == "--save-interval-time") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --save-interval-time requires <dt_save>" << std::endl;
        return 1;
      }
      save_interval_time = std::stod(argv[i + 1]);
      i += 1;
    } else if (arg == "--checkpoint-interval-time") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --checkpoint-interval-time requires <dt_chk>" << std::endl;
        return 1;
      }
      checkpoint_interval_time = std::stod(argv[i + 1]);
      i += 1;
    } else if (arg == "--adjoint-adapt") {
      adjoint_adapt = true;
      // Optional: --adjoint-adapt <tol> <max_cycles> <fraction>
      if (i + 1 < argc && !is_flag_token(argv[i + 1])) {
        adjoint_tol = std::stod(argv[i + 1]); i += 1;
      }
      if (i + 1 < argc && !is_flag_token(argv[i + 1])) {
        adapt_max_cycles = std::stoi(argv[i + 1]); i += 1;
      }
      if (i + 1 < argc && !is_flag_token(argv[i + 1])) {
        adapt_fraction = std::stod(argv[i + 1]); i += 1;
      }
    } else if (arg == "--final-ar-cleanup") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --final-ar-cleanup requires <max_aspect_ratio>" << std::endl;
        return 1;
      }
      final_ar_cleanup = std::stod(argv[i + 1]);
      i += 1;
    } else if (arg == "--smooth-iters") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --smooth-iters requires <iterations>" << std::endl;
        return 1;
      }
      smooth_iters = std::stoi(argv[i + 1]);
      i += 1;
    } else if (arg == "--wall-geom-tol") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --wall-geom-tol requires <tolerance>" << std::endl;
        return 1;
      }
      wall_geom_tol = std::stod(argv[i + 1]);
      i += 1;
    } else if (arg == "--write-adjoint-indicators") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --write-adjoint-indicators requires <output_file>" << std::endl;
        return 1;
      }
      write_adjoint_indicators = true;
      adjoint_indicators_file = argv[i + 1];
      i += 1;
    } else if (is_flag_token(argv[i])) {
      std::cerr << "Error: Unknown option " << arg << std::endl;
      return 1;
    }
  }

  try {
    setMeshSmoothingIterations(smooth_iters);
    setWallGeometryTolerance(wall_geom_tol);
    FiniteVolumeSolver solver(meshfile);

    // Extract grid name from mesh file path (needed for auto-chaining filenames)
    std::string grid_name = meshfile;
    size_t last_slash = grid_name.find_last_of("/\\");
    if (last_slash != std::string::npos) {
      grid_name = grid_name.substr(last_slash + 1);
    }
    size_t dot_pos = grid_name.find_last_of(".");
    if (dot_pos != std::string::npos) {
      grid_name = grid_name.substr(0, dot_pos);
    }

    solver.p_order = p_order;  // Set DG polynomial order
    solver.initializeDG();     // Initialize DG structures based on p_order
    solver.CFL = cfl;
    solver.fluxname = fluxname;
    solver.freestream_test_mode = freestream_mode;

    std::string order_tag = "p" + std::to_string(p_order);
    std::string output_dir =
        unsteady ? ("unsteady_data/" + grid_name + "_" + order_tag) : "data_steady";
    std::string file_prefix = unsteady ? "" : "steady_" + grid_name + "_" + order_tag + "_";
    std::filesystem::create_directories(output_dir);
    if (unsteady) {
      solver.unsteady_output_dir = output_dir;
      solver.unsteady_save_after = save_after;
      solver.unsteady_save_interval_time = save_interval_time;
      solver.unsteady_checkpoint_interval_time = checkpoint_interval_time;
      if (!ic_file.empty()) {
        solver.unsteady_restart_time_override = infer_unsteady_restart_time(ic_file);
      }
    } else {
      solver.steady_output_dir = output_dir;
      solver.steady_output_prefix = file_prefix;
    }
    
    if (use_mapped_ic) {
      if (!ic_file.empty()) {
        std::cerr << "Warning: both ic_file and --map-ic provided; using --map-ic"
                  << std::endl;
      }
      solver.loadMappedInitialCondition(coarse_meshfile, coarse_statefile);
    } else if (!ic_file.empty()) {
      // Load same-mesh initial condition if provided. Full DG restart files
      // use the *_dg.bin suffix and restore every DG coefficient.
      if (ic_file.size() >= 7 &&
          ic_file.substr(ic_file.size() - 7) == "_dg.bin") {
        solver.loadDGInitialCondition(ic_file);
      } else {
        solver.loadInitialCondition(ic_file);
      }
    } else if (p_order > 0 && !unsteady && !freestream_mode) {
      // Automatically converge p=0 first, then use as IC for p>0
      int p0_seed_itercap = (p0_itercap > 0) ? p0_itercap : itercap;
      std::cerr << "Auto-chaining: converging p=0 first to seed p=" 
                << p_order << " IC"
                << " (p0 itercap=" << p0_seed_itercap << ")..."
                << std::endl;
      
      // Save current DG order and initialize as p=0
      int saved_order = solver.p_order;
      solver.p_order = 0;
      solver.initializeDG();
      solver.CFL = 1.0;  // p=0 can use larger CFL
      solver.solveSteady(p0_seed_itercap);

      if (!solver.last_steady_converged) {
        std::cerr << "Error: auto-chained p=0 seed solve did not converge";
        if (solver.last_steady_failed_nonphysical) {
          std::cerr << " (non-physical state encountered)";
        } else if (solver.last_steady_hit_itercap) {
          std::cerr << " (hit iteration cap before convergence)";
        }
        std::cerr << ". Aborting auto-chain." << std::endl;
        return 1;
      }
      
      // Save p=0 result to temporary file
      system("mkdir -p data_steady");
      std::string p0_file = "data_steady/steady_" + grid_name + "_p0_results.bin";
      {
        std::ofstream p0out(p0_file, std::ios::binary);
        int Ne = solver.U.size();
        p0out.write((char *)&Ne, sizeof(int));
        for (const auto &u : solver.U)
          p0out.write((char *)u.v, sizeof(double) * 4);
        p0out.close();
      }
      std::cerr << "p=0 converged. Saved to " << p0_file << std::endl;
      
      // Now re-initialize at the target order and load p=0 IC
      solver.p_order = saved_order;
      solver.initializeDG();
      solver.CFL = cfl;
      solver.res_history.clear();
      solver.loadInitialCondition(p0_file);
      std::cerr << "Loaded p=0 solution as IC for p=" << saved_order << std::endl;
    }

    std::cerr << "Starting solver for " << meshfile
              << " (DG Order p=" << p_order
              << ", CFL: " << cfl << ", Flux: " << fluxname
              << ", IterCap: " << itercap 
              << ", Mode: "
              << (unsteady ? "Unsteady" : (freestream_mode ? "Freestream" : "Steady"))
              << (t_end > 0.0 ? ", t_end: " + std::to_string(t_end) : "")
              << (unsteady && save_after > 0.0 ? ", save_after: " + std::to_string(save_after) : "")
              << (unsteady && save_interval_time > 0.0 ? ", save_interval_time: " + std::to_string(save_interval_time) : "")
              << (unsteady && checkpoint_interval_time > 0.0 ? ", checkpoint_interval_time: " + std::to_string(checkpoint_interval_time) : "")
              << (use_mapped_ic ? ", Mapped IC: " + coarse_meshfile + " + " +
                                      coarse_statefile
                                : "")
              << ")" << std::endl;

    if (unsteady) {
      solver.solveUnsteady(itercap, t_end);
    } else if (adjoint_adapt) {
      // ── ADJOINT-BASED h-ADAPTATION LOOP ──────────────────────────
      Mesh pending_rollback_mesh;
      std::vector<Vec4> pending_rollback_U;
      std::vector<std::vector<Vec4>> pending_rollback_U_dg;
      RefinementMap pending_rollback_rmap;
      bool have_pending_rollback = false;
      bool saved_failed_refined_mesh_this_cycle = false;

      for (int cycle = 0; cycle < adapt_max_cycles; ++cycle) {
        std::cerr << "=== Adaptation cycle " << cycle << " ===" << std::endl;
        std::cerr << "  Mesh: " << solver.mesh.E.size() << " elements" << std::endl;

        // Step 1: Converge the primal at current p_order
        if (cycle == 0) {
          solver.initializeDG();
          solver.setInitialCondition();
        } else {
          // Restored interpolation mapping for all p_orders
          auto U_dg_save = solver.U_dg;
          solver.initializeDG();  // re-build DG data and reset to freestream IC

          int Ne_cur = (int)solver.mesh.E.size();
          bool ic_ok = ((int)U_dg_save.size() == Ne_cur);
          if (ic_ok) {
            for (int e = 0; e < Ne_cur && ic_ok; ++e) {
              Vec4 avg = solver.cellAverage(U_dg_save[e]);
              State s(avg, solver.gamma);
              if (!std::isfinite(s.rho()) || !std::isfinite(s.p()) ||
                  s.rho() < 1e-6 || s.p() < 1e-6)
                ic_ok = false;
            }
          }

          if (ic_ok) {
            solver.U_dg = U_dg_save;
            std::cerr << "  Using interpolated IC from previous cycle." << std::endl;
          } else {
            std::cerr << "  Interpolated IC non-physical — falling back to freestream." << std::endl;
            // U_dg already set to freestream by initializeDG()
          }

          solver.U.resize(Ne_cur);
          for (int e = 0; e < Ne_cur; ++e)
            solver.U[e] = solver.cellAverage(solver.U_dg[e]);
        }
        solver.solveSteady(itercap);

        // Accept stagnated solutions (limit-cycle oscillations) as valid primal.
        // last_steady_converged is true for both actual convergence and stagnation,
        // including cases where the RK4 hit a non-physical state and restored the
        // last valid solution.
        if (!solver.last_steady_converged) {
          bool repaired_and_recovered = false;
          if (have_pending_rollback) {
            if (!saved_failed_refined_mesh_this_cycle) {
              std::string failed_mesh_file = output_dir + "/" + file_prefix
                  + "adjoint_mesh_cycle" + std::to_string(cycle)
                  + "_failed.bin";
              writeCompanionMeshSnapshot(solver.mesh, failed_mesh_file);
              std::cerr << "  Failed refined mesh snapshot saved to "
                        << failed_mesh_file << std::endl;
              saved_failed_refined_mesh_this_cycle = true;
            }
            const Mesh failed_mesh_base = solver.mesh;
            const auto U_failed_base = solver.U;
            const auto U_dg_failed_base = solver.U_dg;
            const double cfl_failed_base = solver.CFL;

            bool cfl_retry_recovered = false;
            for (double factor : {0.5, 0.25, 0.125, 0.05, 0.025, 0.01}) {
              const double trial_cfl = cfl_failed_base * factor;
              if (!(trial_cfl > 0.0) || trial_cfl >= cfl_failed_base - 1e-14) {
                continue;
              }
              solver.mesh = failed_mesh_base;
              solver.U = U_failed_base;
              solver.U_dg = U_dg_failed_base;
              solver.CFL = trial_cfl;
              std::cerr << "  Retrying primal on refined mesh with reduced CFL="
                        << trial_cfl << "." << std::endl;
              solver.solveSteady(itercap);
              if (solver.last_steady_converged) {
                cfl = trial_cfl;
                std::cerr << "  Reduced-CFL retry recovered the refined primal"
                          << " solve." << std::endl;
                repaired_and_recovered = true;
                cfl_retry_recovered = true;
                break;
              }
            }
            if (cfl_retry_recovered) {
              continue;
            }

            solver.mesh = failed_mesh_base;
            solver.U = U_failed_base;
            solver.U_dg = U_dg_failed_base;
            solver.CFL = cfl_failed_base;
            auto failure_report = validateRefinedMesh(solver.mesh);
            std::vector<int> repair_seeds;
            auto residual_seeds =
                topResidualRepairSeeds(solver.mesh, solver.cell_residuals, 6);
            for (int e : residual_seeds) repair_seeds.push_back(e);
            if (failure_report.worst_quality_elem >= 0)
              repair_seeds.push_back(failure_report.worst_quality_elem);
            if (failure_report.min_angle_elem >= 0 &&
                failure_report.min_angle_elem != failure_report.worst_quality_elem)
              repair_seeds.push_back(failure_report.min_angle_elem);
            if (failure_report.min_sampled_detJ_elem >= 0 &&
                failure_report.min_sampled_detJ_elem != failure_report.worst_quality_elem &&
                failure_report.min_sampled_detJ_elem != failure_report.min_angle_elem)
              repair_seeds.push_back(failure_report.min_sampled_detJ_elem);

            std::vector<int> unique_repair_seeds;
            for (int e : repair_seeds) {
              if (e < 0) continue;
              if (std::find(unique_repair_seeds.begin(), unique_repair_seeds.end(), e) !=
                  unique_repair_seeds.end())
                continue;
              unique_repair_seeds.push_back(e);
            }

            const Mesh mesh_retry_base = solver.mesh;
            const auto U_retry_base = solver.U;
            const auto U_dg_retry_base = solver.U_dg;
            bool any_repaired = false;

            for (int seed_elem : unique_repair_seeds) {
              if (seed_elem < 0) continue;
              std::cerr << "  Primal failed on refined mesh — attempting local"
                        << " curved-patch repair around elem " << seed_elem
                        << "." << std::endl;
              const Mesh mesh_before_seed = solver.mesh;
              bool repaired = repairLowQualityCurvedPatch(solver.mesh, seed_elem);
              if (!repaired) {
                repaired = repairInvalidCurvedPatch(solver.mesh, seed_elem);
              }
              if (!repaired) continue;

              auto repaired_report = validateRefinedMesh(solver.mesh);
              std::cerr << "  Revalidated mesh after local repair:"
                        << std::endl;
              printMeshValidationReport(repaired_report);
              bool repaired_invalid =
                  repaired_report.nonpositive_area_count > 0 ||
                  repaired_report.nonpositive_sampled_detJ_count > 0 ||
                  repaired_report.nonmanifold_edge_count > 0 ||
                  repaired_report.duplicate_boundary_edge_count > 0 ||
                  repaired_report.invalid_boundary_owner_count > 0 ||
                  repaired_report.invalid_interior_owner_count > 0;
              if (repaired_invalid) {
                std::cerr << "  Local repair produced an invalid mesh —"
                          << " discarding that patch move." << std::endl;
                solver.mesh = mesh_before_seed;
                continue;
              }

              any_repaired = true;
            }

            if (any_repaired) {
              solver.initializeDG();
              if (have_pending_rollback) {
                transferRefinedSolutionToCurrentMesh(
                    solver, pending_rollback_mesh, pending_rollback_U,
                    pending_rollback_U_dg, pending_rollback_rmap,
                    "Retry after cumulative local curved-patch repairs");
              } else if (static_cast<int>(U_dg_retry_base.size()) ==
                         static_cast<int>(solver.mesh.E.size())) {
                solver.U_dg = U_dg_retry_base;
                solver.U.resize(solver.mesh.E.size());
                for (size_t e = 0; e < solver.mesh.E.size(); ++e) {
                  solver.U[e] = solver.cellAverage(solver.U_dg[e]);
                }
              }

              const Mesh repaired_mesh_base = solver.mesh;
              const auto U_repaired_base = solver.U;
              const auto U_dg_repaired_base = solver.U_dg;
              bool repaired_retry_recovered = false;
              for (double factor : {1.0, 0.5, 0.25, 0.125, 0.05, 0.025, 0.01}) {
                const double trial_cfl = cfl_failed_base * factor;
                if (!(trial_cfl > 0.0)) continue;
                solver.mesh = repaired_mesh_base;
                solver.U = U_repaired_base;
                solver.U_dg = U_dg_repaired_base;
                solver.CFL = trial_cfl;
                std::cerr << "  Retrying primal after cumulative local"
                          << " curved-patch repairs with CFL="
                          << trial_cfl << "." << std::endl;
                solver.solveSteady(itercap);
                if (solver.last_steady_converged) {
                  if (trial_cfl < cfl) cfl = trial_cfl;
                  std::cerr << "  Cumulative local curved-patch repairs recovered"
                            << " the refined primal solve." << std::endl;
                  repaired_and_recovered = true;
                  repaired_retry_recovered = true;
                  break;
                }
              }
              if (!repaired_retry_recovered) {
                std::cerr << "  Retry after cumulative local curved-patch"
                          << " repairs still failed." << std::endl;
                solver.mesh = mesh_retry_base;
                solver.U = U_retry_base;
                solver.U_dg = U_dg_retry_base;
                solver.CFL = cfl_failed_base;
              }
            }
          }
          if (!repaired_and_recovered) {
            if (have_pending_rollback) {
              solver.mesh = pending_rollback_mesh;
              solver.U = pending_rollback_U;
              solver.U_dg = pending_rollback_U_dg;
              have_pending_rollback = false;
              std::cerr << "  Primal failed on the newly refined mesh — restored"
                        << " the previous accepted mesh and stopping adaptation."
                        << std::endl;
            } else {
              std::cerr << "  Primal failed (itercap exceeded without stagnation) —"
                        << " stopping adaptation." << std::endl;
            }
            break;
          }
        }
        have_pending_rollback = false;
        saved_failed_refined_mesh_this_cycle = false;

        double Cl_cycle = solver.integrateCl();
        std::cerr << "  Cl = " << std::setprecision(6) << Cl_cycle
                  << "  (" << solver.mesh.E.size() << " elements)" << std::endl;

        // Save the converged/stagnated primal state for this adaptation cycle
        // so post-processing can render solution fields on the adapted mesh.
        {
          std::string primal_file = output_dir + "/" + file_prefix
              + "adjoint_primal_cycle" + std::to_string(cycle) + "_dg.bin";
          std::ofstream primal_out(primal_file, std::ios::binary);
          int Ne_p = (int)solver.U_dg.size();
          int p_p  = solver.p_order;
          int nd_p = solver.ndof_per_elem;
          primal_out.write((char*)&Ne_p, sizeof(int));
          primal_out.write((char*)&p_p,  sizeof(int));
          primal_out.write((char*)&nd_p, sizeof(int));
          for (const auto &elem_u : solver.U_dg)
            for (const auto &d : elem_u)
              primal_out.write((char*)d.v, sizeof(double) * 4);
          primal_out.close();
          std::cerr << "  Primal DG state saved to " << primal_file << std::endl;
        }

        // Step 2: Solve the adjoint
        AdjointSolver adj(solver);
        adj.solve();

        // Cycle 0 adjoint validation: compare adjoint sensitivity with FD
        if (cycle == 0) {
          double sens_adj = adj.sensitivityAlpha(1e-5);
          std::cerr << "  dCl/dalpha (adjoint) = " << sens_adj << " rad^-1" << std::endl;
        }

        // Step 3: Compute error indicators via fine-space residual
        FiniteVolumeSolver fine_solver(solver.mesh);
        fine_solver.fluxname = solver.fluxname;
        fine_solver.CFL = solver.CFL;
        auto indicators = adj.errorIndicators(fine_solver);

        // Save adjoint solution (psi) — same binary format as results_dg.bin
        {
          std::string psi_file = output_dir + "/" + file_prefix
              + "adjoint_psi_cycle" + std::to_string(cycle) + "_dg.bin";
          std::ofstream psi_out(psi_file, std::ios::binary);
          int Ne_a = (int)adj.psi().size();
          int p_a  = solver.p_order;
          int nd_a = solver.ndof_per_elem;
          psi_out.write((char*)&Ne_a, sizeof(int));
          psi_out.write((char*)&p_a,  sizeof(int));
          psi_out.write((char*)&nd_a, sizeof(int));
          for (const auto &elem_psi : adj.psi())
            for (const auto &d : elem_psi)
              psi_out.write((char*)d.v, sizeof(double) * 4);
          psi_out.close();
          std::cerr << "  Adjoint psi saved to " << psi_file << std::endl;
        }

        // Save companion mesh for this cycle: Nn, x/y per node, Ne, q/v0/v1/v2 per elem
        {
          std::string mesh_file = output_dir + "/" + file_prefix
              + "adjoint_mesh_cycle" + std::to_string(cycle) + ".bin";
          writeCompanionMeshSnapshot(solver.mesh, mesh_file);
          std::cerr << "  Mesh saved to " << mesh_file << std::endl;
        }

        // Save error indicators — header: Ne (int32), then Ne doubles
        {
          std::string ind_file = output_dir + "/" + file_prefix
              + "adjoint_indicators_cycle" + std::to_string(cycle) + ".bin";
          std::ofstream ind_out(ind_file, std::ios::binary);
          int Ne_i = (int)indicators.size();
          ind_out.write((char*)&Ne_i, sizeof(int));
          ind_out.write((char*)indicators.data(), sizeof(double) * Ne_i);
          ind_out.close();
          std::cerr << "  Error indicators saved to " << ind_file << std::endl;
        }

        // Step 4: Aspect-ratio cleanup marking on the current mesh
        std::vector<bool> ar_marked;
        int n_ar_marked = 0;
        if (final_ar_cleanup > 0.0) {
          ar_marked = markByAspectRatio(solver.mesh, final_ar_cleanup);
          for (bool m : ar_marked) if (m) n_ar_marked++;
          std::string ar_marked_file = output_dir + "/" + file_prefix
              + "adjoint_ar_cleanup_marked_cycle" + std::to_string(cycle) + ".bin";
          std::ofstream ar_out(ar_marked_file, std::ios::binary);
          int Ne_ar = static_cast<int>(ar_marked.size());
          ar_out.write((char*)&Ne_ar, sizeof(int));
          for (bool m : ar_marked) {
            unsigned char flag = m ? 1 : 0;
            ar_out.write((char*)&flag, sizeof(unsigned char));
          }
          ar_out.close();
          std::cerr << "  AR cleanup threshold " << final_ar_cleanup
                    << " marked " << n_ar_marked << " elements" << std::endl;
        }

        // Step 5: Check convergence
        double total_error = 0.0;
        for (double e : indicators) total_error += e;
        std::cerr << "  Estimated |delta_Cl| = " << total_error << std::endl;
        if (total_error < adjoint_tol && n_ar_marked == 0) {
          std::cerr << "  Converged! Error below tolerance " << adjoint_tol << std::endl;
          break;
        }

        // Step 6: Mark and refine with in-loop fallback to lower-priority cells.
        // Fallback logic runs inside a single bisectMarkedElements call so that
        // rebuildMeshEdgeConnectivity and appendPeriodicToIE fire exactly once,
        // avoiding periodic boundary corruption from multiple calls.
        const int Ne_orig = static_cast<int>(indicators.size());
        std::vector<int> sorted_cells(Ne_orig);
        std::iota(sorted_cells.begin(), sorted_cells.end(), 0);
        if (total_error >= adjoint_tol)
          std::sort(sorted_cells.begin(), sorted_cells.end(),
                    [&](int a, int b){ return indicators[a] > indicators[b]; });

        const int target_splits = std::max(1, (int)(adapt_fraction * Ne_orig));

        // Initial top-N% marks.
        std::vector<bool> marked(Ne_orig, false);
        int n_marked = 0;
        if (total_error >= adjoint_tol) {
          marked = markByIndicator(indicators, adapt_fraction);
          for (bool m : marked) if (m) n_marked++;
        }
        if (!ar_marked.empty())
          for (size_t e = 0; e < marked.size(); ++e) marked[e] = marked[e] || ar_marked[e];
        int n_total_marked = 0;
        for (bool m : marked) if (m) n_total_marked++;
        std::cerr << "  Marking " << n_marked << " adjoint elements";
        if (final_ar_cleanup > 0.0)
          std::cerr << " + " << n_ar_marked << " AR-cleanup elements";
        std::cerr << " => " << n_total_marked << " total" << std::endl;

        {
          std::string marked_file = output_dir + "/" + file_prefix
              + "adjoint_marked_cycle" + std::to_string(cycle) + ".bin";
          std::ofstream marked_out(marked_file, std::ios::binary);
          int Ne_marked = static_cast<int>(marked.size());
          marked_out.write((char*)&Ne_marked, sizeof(int));
          for (bool m : marked) {
            unsigned char flag = m ? 1 : 0;
            marked_out.write((char*)&flag, sizeof(unsigned char));
          }
          marked_out.close();
          std::cerr << "  Final refinement marks saved to " << marked_file << std::endl;
        }

        if (n_total_marked == 0) {
          std::cerr << "  No elements marked for refinement." << std::endl;
          break;
        }

        if (const char *env = std::getenv("AMR_DEBUG_WALL_DIAGNOSTICS")) {
          if (std::string(env) != "0") {
            printWallRefinementDiagnostics(solver.mesh, marked);
          }
        }

        // Build fallback priority: all non-marked cells in descending indicator
        // order. Passed to bisectMarkedElements so that if top-N% cells are
        // rejected by geometry constraints the next-best cells are tried
        // automatically — all within one connectivity-rebuild call.
        std::vector<int> fallback_priority;
        if (total_error >= adjoint_tol) {
          fallback_priority.reserve(Ne_orig);
          for (int i = 0; i < Ne_orig; ++i) {
            int e = sorted_cells[i];
            if (!marked[e]) fallback_priority.push_back(e);
          }
        }

        const Mesh mesh_before_refine = solver.mesh;
        const auto U_before_refine = solver.U;
        const auto U_dg_before_refine = solver.U_dg;
        auto rmap = bisectMarkedElements(solver.mesh, marked, fallback_priority,
                                         target_splits);
        {
          std::vector<int> parent_refine_count(Ne_orig, 0);
          for (int parent : rmap.child_to_parent) {
            if (parent >= 0 && parent < Ne_orig) {
              parent_refine_count[parent]++;
            }
          }
          std::vector<bool> refined_parent_mask(Ne_orig, false);
          for (int e = 0; e < Ne_orig; ++e) {
            refined_parent_mask[e] = parent_refine_count[e] > 1;
          }
          std::string refined_file = output_dir + "/" + file_prefix
              + "adjoint_refined_cycle" + std::to_string(cycle) + ".bin";
          std::ofstream refined_out(refined_file, std::ios::binary);
          int Ne_refined = static_cast<int>(refined_parent_mask.size());
          refined_out.write((char*)&Ne_refined, sizeof(int));
          for (bool m : refined_parent_mask) {
            unsigned char flag = m ? 1 : 0;
            refined_out.write((char*)&flag, sizeof(unsigned char));
          }
          refined_out.close();
          std::cerr << "  Actual refined-parent cells saved to "
                    << refined_file << std::endl;

          std::string reason_file = output_dir + "/" + file_prefix
              + "adjoint_refined_reason_cycle" + std::to_string(cycle) + ".bin";
          std::ofstream reason_out(reason_file, std::ios::binary);
          int Ne_reason = Ne_orig;
          reason_out.write((char*)&Ne_reason, sizeof(int));
          for (int e = 0; e < Ne_orig; ++e) {
            unsigned char reason = 0;
            if (e < static_cast<int>(rmap.parent_split_reason.size())) {
              reason = rmap.parent_split_reason[e];
            }
            reason_out.write((char*)&reason, sizeof(unsigned char));
          }
          reason_out.close();
          std::cerr << "  Refined-parent reason codes saved to "
                    << reason_file << std::endl;
        }
        auto mesh_report = validateRefinedMesh(solver.mesh);
        printMeshValidationReport(mesh_report);

        bool invalid_refined_pass =
            mesh_report.nonpositive_area_count > 0 ||
            mesh_report.nonpositive_sampled_detJ_count > 0 ||
            mesh_report.nonmanifold_edge_count > 0 ||
            mesh_report.duplicate_boundary_edge_count > 0 ||
            mesh_report.invalid_boundary_owner_count > 0 ||
            mesh_report.invalid_interior_owner_count > 0;
        if (invalid_refined_pass &&
            mesh_report.nonpositive_sampled_detJ_count > 0 &&
            mesh_report.min_sampled_detJ_elem >= 0) {
          std::cerr << "  Attempting local curved-patch repair on elem "
                    << mesh_report.min_sampled_detJ_elem << "..."
                    << std::endl;
          if (repairInvalidCurvedPatch(solver.mesh,
                                       mesh_report.min_sampled_detJ_elem)) {
            mesh_report = validateRefinedMesh(solver.mesh);
            std::cerr << "  Revalidated mesh after local curved repair:"
                      << std::endl;
            printMeshValidationReport(mesh_report);
            invalid_refined_pass =
                mesh_report.nonpositive_area_count > 0 ||
                mesh_report.nonpositive_sampled_detJ_count > 0 ||
                mesh_report.nonmanifold_edge_count > 0 ||
                mesh_report.duplicate_boundary_edge_count > 0 ||
                mesh_report.invalid_boundary_owner_count > 0 ||
                mesh_report.invalid_interior_owner_count > 0;
          }
        }
        if (invalid_refined_pass) {
          {
            std::string rejected_mesh_file = output_dir + "/" + file_prefix
                + "adjoint_mesh_cycle" + std::to_string(cycle + 1)
                + "_rejected.bin";
            writeCompanionMeshSnapshot(solver.mesh, rejected_mesh_file);
            std::cerr << "  Rejected refined mesh snapshot saved to "
                      << rejected_mesh_file << std::endl;
          }
          if (mesh_report.nonpositive_sampled_detJ_count > 0 &&
              mesh_report.min_sampled_detJ_elem >= 0) {
            std::cerr << "  Curved-element debug for invalid refinement:"
                      << std::endl;
            printElementGeometryDebug(
                solver.mesh, mesh_report.min_sampled_detJ_elem,
                "min-detJ element");
            printElementOneRingDebug(
                solver.mesh, mesh_report.min_sampled_detJ_elem,
                "min-detJ");
          }
          std::cerr << "  Rejecting refinement pass: refined mesh became invalid."
                    << std::endl;
          solver.mesh = mesh_before_refine;
          solver.U = U_before_refine;
          solver.U_dg = U_dg_before_refine;
          std::cerr << "  Restored pre-refinement mesh/state and stopping adaptation."
                    << std::endl;
          break;
        }

        std::vector<int> transition_repair_seeds;
        if (solver.mesh.q_order_global <= 1 &&
            static_cast<int>(solver.mesh.E.size()) >=
                q1PreSolveTransitionRepairMinElems()) {
          transition_repair_seeds = topQ1PreSolveRepairSeeds(
              solver.mesh, rmap, q1PreSolveTransitionRepairMaxSeeds());
        }
        if (solver.mesh.q_order_global <= 1 &&
            (mesh_report.min_angle_deg < q1PreSolveSliverRepairMinAngleDeg() ||
             mesh_report.worst_quality < q1PreSolveSliverRepairMinQuality() ||
             !transition_repair_seeds.empty())) {
          bool repaired_low_quality = false;
          for (int pass = 0; pass < 3; ++pass) {
            std::vector<int> sliver_seeds;
            auto append_unique_seed = [&](int seed_elem) {
              if (seed_elem < 0) return;
              if (std::find(sliver_seeds.begin(), sliver_seeds.end(), seed_elem) !=
                  sliver_seeds.end()) {
                return;
              }
              sliver_seeds.push_back(seed_elem);
            };
            if (mesh_report.worst_quality_elem >= 0)
              append_unique_seed(mesh_report.worst_quality_elem);
            if (mesh_report.min_angle_elem >= 0 &&
                mesh_report.min_angle_elem != mesh_report.worst_quality_elem)
              append_unique_seed(mesh_report.min_angle_elem);
            for (int seed_elem : transition_repair_seeds) {
              append_unique_seed(seed_elem);
            }
            if (sliver_seeds.empty()) break;

            bool pass_repaired = false;
            for (int seed_elem : sliver_seeds) {
              const Mesh mesh_before_seed = solver.mesh;
              const bool repaired =
                  (solver.mesh.q_order_global <= 1)
                      ? repairLowQualityQ1Patch(solver.mesh, seed_elem, false)
                      : repairLowQualityCurvedPatch(solver.mesh, seed_elem);
              if (!repaired) continue;
              auto repaired_report = validateRefinedMesh(solver.mesh);
              const bool repaired_invalid =
                  repaired_report.nonpositive_area_count > 0 ||
                  repaired_report.nonpositive_sampled_detJ_count > 0 ||
                  repaired_report.nonmanifold_edge_count > 0 ||
                  repaired_report.duplicate_boundary_edge_count > 0 ||
                  repaired_report.invalid_boundary_owner_count > 0 ||
                  repaired_report.invalid_interior_owner_count > 0;
              if (repaired_invalid) {
                std::cerr << "  Pre-solve q1 sliver repair around elem "
                          << seed_elem
                          << " produced an invalid mesh — discarding that move."
                          << std::endl;
                solver.mesh = mesh_before_seed;
                continue;
              }
              mesh_report = repaired_report;
              pass_repaired = true;
              repaired_low_quality = true;
            }
            if (static_cast<int>(solver.mesh.E.size()) >=
                q1PreSolveTransitionRepairMinElems()) {
              transition_repair_seeds = topQ1PreSolveRepairSeeds(
                  solver.mesh, rmap, q1PreSolveTransitionRepairMaxSeeds());
            } else {
              transition_repair_seeds.clear();
            }
            if (!pass_repaired ||
                ((mesh_report.min_angle_deg >=
                  q1PreSolveSliverRepairMinAngleDeg()) &&
                 (mesh_report.worst_quality >=
                  q1PreSolveSliverRepairMinQuality()) &&
                 transition_repair_seeds.empty())) {
              break;
            }
          }
          if (repaired_low_quality) {
            std::cerr << "  Revalidated mesh after pre-solve q1 sliver repair:"
                      << std::endl;
            printMeshValidationReport(mesh_report);
            invalid_refined_pass =
                mesh_report.nonpositive_area_count > 0 ||
                mesh_report.nonpositive_sampled_detJ_count > 0 ||
                mesh_report.nonmanifold_edge_count > 0 ||
                mesh_report.duplicate_boundary_edge_count > 0 ||
                mesh_report.invalid_boundary_owner_count > 0 ||
                mesh_report.invalid_interior_owner_count > 0;
            if (invalid_refined_pass) {
              std::cerr << "  Pre-solve q1 sliver repair invalidated the refined"
                        << " mesh — restoring the pre-refinement mesh/state."
                        << std::endl;
              solver.mesh = mesh_before_refine;
              solver.U = U_before_refine;
              solver.U_dg = U_dg_before_refine;
              break;
            }
          }
        }

        if (q1ProactiveRepairEnabled() && solver.mesh.q_order_global <= 1 &&
            static_cast<int>(solver.mesh.E.size()) >= q1ProactiveRepairMinElems()) {
          auto proactive_seeds = topQ1PreSolveRepairSeeds(
              solver.mesh, rmap, q1ProactiveRepairMaxSeeds());
          bool any_proactive_repaired = false;
          if (!proactive_seeds.empty()) {
            std::cerr << "  Proactive q1 transition repair seeds:";
            for (int e : proactive_seeds) std::cerr << " " << e;
            std::cerr << std::endl;
          }
          for (int seed_elem : proactive_seeds) {
            if (seed_elem < 0 || seed_elem >= static_cast<int>(solver.mesh.E.size())) {
              continue;
            }
            const Mesh mesh_before_seed = solver.mesh;
            const bool repaired =
                (solver.mesh.q_order_global <= 1)
                    ? repairLowQualityQ1Patch(solver.mesh, seed_elem, false)
                    : repairLowQualityCurvedPatch(solver.mesh, seed_elem);
            if (!repaired) continue;

            auto repaired_report = validateRefinedMesh(solver.mesh);
            bool repaired_invalid =
                repaired_report.nonpositive_area_count > 0 ||
                repaired_report.nonpositive_sampled_detJ_count > 0 ||
                repaired_report.nonmanifold_edge_count > 0 ||
                repaired_report.duplicate_boundary_edge_count > 0 ||
                repaired_report.invalid_boundary_owner_count > 0 ||
                repaired_report.invalid_interior_owner_count > 0;
            if (repaired_invalid) {
              std::cerr << "  Proactive q1 repair around elem " << seed_elem
                        << " produced an invalid mesh — discarding that move."
                        << std::endl;
              solver.mesh = mesh_before_seed;
              continue;
            }
            any_proactive_repaired = true;
          }
          if (any_proactive_repaired) {
            mesh_report = validateRefinedMesh(solver.mesh);
            std::cerr << "  Revalidated mesh after proactive q1 transition repairs:"
                      << std::endl;
            printMeshValidationReport(mesh_report);
            invalid_refined_pass =
                mesh_report.nonpositive_area_count > 0 ||
                mesh_report.nonpositive_sampled_detJ_count > 0 ||
                mesh_report.nonmanifold_edge_count > 0 ||
                mesh_report.duplicate_boundary_edge_count > 0 ||
                mesh_report.invalid_boundary_owner_count > 0 ||
                mesh_report.invalid_interior_owner_count > 0;
            if (invalid_refined_pass) {
              std::cerr << "  Proactive q1 transition repair invalidated the"
                        << " refined mesh — restoring the pre-refinement mesh/state."
                        << std::endl;
              solver.mesh = mesh_before_refine;
              solver.U = U_before_refine;
              solver.U_dg = U_dg_before_refine;
              break;
            }
          }
        }

        if (solver.mesh.E.size() == mesh_before_refine.E.size()) {
          std::cerr << "  No cells added this cycle — stopping adaptation."
                    << std::endl;
          break;
        }
        pending_rollback_mesh = mesh_before_refine;
        pending_rollback_U = U_before_refine;
        pending_rollback_U_dg = U_dg_before_refine;
        pending_rollback_rmap = rmap;
        have_pending_rollback = true;

        // Save the newly refined mesh immediately so it can be inspected even
        // if the next primal solve stalls or is interrupted before completion.
        {
          std::string refined_mesh_file = output_dir + "/" + file_prefix
              + "adjoint_mesh_cycle" + std::to_string(cycle + 1) + ".bin";
          writeCompanionMeshSnapshot(solver.mesh, refined_mesh_file);
          std::cerr << "  Refined mesh preview saved to " << refined_mesh_file << std::endl;
        }

        // Step 6: Transfer solution to refined mesh
        transferRefinedSolutionToCurrentMesh(
            solver, mesh_before_refine, U_before_refine, U_dg_before_refine,
            rmap, "Initial refined transfer");
      }
    } else {
      solver.solveSteady(itercap);
      std::cerr << "  Cl = " << std::setprecision(6) << solver.integrateCl()
                << "  (" << solver.mesh.E.size() << " elements)" << std::endl;
      if (write_adjoint_indicators) {
        std::cerr << "  Solving adjoint on fixed mesh for indicators..." << std::endl;
        AdjointSolver adj(solver);
        adj.solve();
        FiniteVolumeSolver fine_solver(solver.mesh);
        fine_solver.fluxname = solver.fluxname;
        fine_solver.CFL = solver.CFL;
        auto indicators = adj.errorIndicators(fine_solver);

        std::filesystem::path out_path(adjoint_indicators_file);
        if (out_path.has_parent_path()) {
          std::filesystem::create_directories(out_path.parent_path());
        }
        std::ofstream ind_out(adjoint_indicators_file, std::ios::binary);
        int Ne_i = static_cast<int>(indicators.size());
        ind_out.write((char *)&Ne_i, sizeof(int));
        ind_out.write((char *)indicators.data(), sizeof(double) * Ne_i);
        ind_out.close();
        std::cerr << "  Fixed-mesh adjoint indicators saved to "
                  << adjoint_indicators_file << std::endl;
      }
    }

    if (adjoint_adapt) {
      std::string accepted_mesh_file = output_dir + "/" + file_prefix
          + "adjoint_mesh_latest_accepted.bin";
      writeCompanionMeshSnapshot(solver.mesh, accepted_mesh_file);
      std::cerr << "Latest accepted adapted mesh saved to "
                << accepted_mesh_file << std::endl;
    }

    // Save results to a simple binary or text format for Python to read
    std::string results_file = output_dir + "/" + file_prefix + "results.bin";
    std::ofstream out(results_file, std::ios::binary);
    int Ne = solver.U.size();
    out.write((char *)&Ne, sizeof(int));
    for (const auto &u : solver.U) {
      out.write((char *)u.v, sizeof(double) * 4);
    }
    out.close();
    std::cout << "Results saved to " << results_file << std::endl;

    std::string dg_results_file = output_dir + "/" + file_prefix + "results_dg.bin";
    std::ofstream dg_out(dg_results_file, std::ios::binary);
    int Ne_dg = solver.U_dg.size();
    int p_out = solver.p_order;
    int ndof = solver.ndof_per_elem;
    dg_out.write((char *)&Ne_dg, sizeof(int));
    dg_out.write((char *)&p_out, sizeof(int));
    dg_out.write((char *)&ndof, sizeof(int));
    for (const auto &elem_dofs : solver.U_dg) {
      for (const auto &u : elem_dofs) {
        dg_out.write((char *)u.v, sizeof(double) * 4);
      }
    }
    dg_out.close();
    std::cout << "DG coefficients saved to " << dg_results_file << std::endl;

    std::string residual_file = output_dir + "/" + file_prefix + "residual.bin";
    std::ofstream res_out(residual_file, std::ios::binary);
    int Nit = solver.res_history.size();
    res_out.write((char *)&Nit, sizeof(int));
    res_out.write((char *)solver.res_history.data(), sizeof(double) * Nit);
    res_out.close();
    std::cout << "Residual history saved to " << residual_file << std::endl;

    std::string cell_res_file = output_dir + "/" + file_prefix + "cell_res.bin";
    std::ofstream cell_res_out(cell_res_file, std::ios::binary);
    int Ne_res = solver.cell_residuals.size();
    cell_res_out.write((char *)&Ne_res, sizeof(int));
    cell_res_out.write((char *)solver.cell_residuals.data(),
                       sizeof(double) * Ne_res);
    cell_res_out.close();
    std::cout << "Spatial residuals saved to " << cell_res_file << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

#include "Solver.hpp"
#include "Adjoint.hpp"
#include "MeshRefinement.hpp"
#include <cstdlib>
#include <algorithm>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <cstdint>
#include <string>
#include <vector>

namespace {

struct MeshValidationReport {
  double min_signed_area = std::numeric_limits<double>::max();
  double max_signed_area = -std::numeric_limits<double>::max();
  int nonpositive_area_count = 0;
  int tiny_area_count = 0;
  int duplicate_boundary_edge_count = 0;
  int invalid_boundary_owner_count = 0;
  int invalid_interior_owner_count = 0;
  int bad_edge_owner_count = 0;
  int nonmanifold_edge_count = 0;
  int orphan_edge_count = 0;
  double min_angle_deg = 180.0;
  double worst_quality = std::numeric_limits<double>::max();
};

constexpr std::uint32_t kCompanionMeshHoMagic = 0x484f3031u; // "HO01"
constexpr std::uint32_t kCompanionMeshPeriodicMagic = 0x50473031u; // "PG01"

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

std::pair<int, int> sortedEdgeKey(int a, int b) {
  return {std::min(a, b), std::max(a, b)};
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

  std::map<std::pair<int, int>, int> elem_edge_counts;
  for (int e = 0; e < static_cast<int>(mesh.E.size()); ++e) {
    const auto &elem = mesh.E[e];
    const Vec2 &a = mesh.V[elem.v[0]];
    const Vec2 &b = mesh.V[elem.v[1]];
    const Vec2 &c = mesh.V[elem.v[2]];

    double sarea = triangleSignedArea(a, b, c);
    report.min_signed_area = std::min(report.min_signed_area, sarea);
    report.max_signed_area = std::max(report.max_signed_area, sarea);
    if (!(sarea > 0.0)) report.nonpositive_area_count++;
    if (std::abs(sarea) < 1e-12) report.tiny_area_count++;

    Vec2 ab = b - a, bc = c - b, ca = a - c;
    double lab = ab.norm(), lbc = bc.norm(), lca = ca.norm();
    double l2sum = lab * lab + lbc * lbc + lca * lca;
    if (l2sum > 0.0) {
      double quality = 4.0 * std::sqrt(3.0) * std::abs(sarea) / l2sum;
      report.worst_quality = std::min(report.worst_quality, quality);
    }

    double ang_a = angleDegBetween(b - a, c - a);
    double ang_b = angleDegBetween(a - b, c - b);
    double ang_c = angleDegBetween(a - c, b - c);
    report.min_angle_deg = std::min(report.min_angle_deg,
                                    std::min(ang_a, std::min(ang_b, ang_c)));

    elem_edge_counts[sortedEdgeKey(elem.v[0], elem.v[1])]++;
    elem_edge_counts[sortedEdgeKey(elem.v[1], elem.v[2])]++;
    elem_edge_counts[sortedEdgeKey(elem.v[2], elem.v[0])]++;
  }

  for (const auto &[edge, count] : elem_edge_counts) {
    if (count == 1) report.orphan_edge_count++;
    else if (count > 2) report.nonmanifold_edge_count++;
  }

  std::map<std::pair<int, int>, int> boundary_edge_counts;
  for (const auto &be : mesh.BE) {
    auto key = sortedEdgeKey(be.v[0], be.v[1]);
    boundary_edge_counts[key]++;
    if (be.elemL < 0 || be.elemL >= static_cast<int>(mesh.E.size()))
      report.invalid_boundary_owner_count++;
  }
  for (const auto &[edge, count] : boundary_edge_counts) {
    if (count > 1) report.duplicate_boundary_edge_count++;
  }

  for (const auto &ie : mesh.IE) {
    if (ie.elemL < 0 || ie.elemL >= static_cast<int>(mesh.E.size()) ||
        ie.elemR < 0 || ie.elemR >= static_cast<int>(mesh.E.size()) ||
        ie.elemL == ie.elemR) {
      report.invalid_interior_owner_count++;
    }
  }

  for (const auto &[edge, count] : elem_edge_counts) {
    bool in_be = boundary_edge_counts.count(edge) > 0;
    bool in_ie = false;
    for (const auto &ie : mesh.IE) {
      if (sortedEdgeKey(ie.v[0], ie.v[1]) == edge) {
        in_ie = true;
        break;
      }
    }

    if (count == 1 && !in_be) report.bad_edge_owner_count++;
    if (count == 2 && !in_ie) report.bad_edge_owner_count++;
    if (count > 2) report.bad_edge_owner_count++;
  }

  if (mesh.E.empty()) {
    report.min_signed_area = 0.0;
    report.max_signed_area = 0.0;
    report.min_angle_deg = 0.0;
    report.worst_quality = 0.0;
  } else if (report.worst_quality == std::numeric_limits<double>::max()) {
    report.worst_quality = 0.0;
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
  std::cerr << "    bad edge ownership = " << r.bad_edge_owner_count
            << " | nonmanifold edges = " << r.nonmanifold_edge_count
            << " | orphan edges = " << r.orphan_edge_count << std::endl;
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
  int smooth_iters = 120;
  double wall_geom_tol = 0.15;

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
          std::cerr << "  Primal failed (itercap exceeded without stagnation) — stopping adaptation." << std::endl;
          break;
        }

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
        if (!std::isfinite(total_error)) {
          std::cerr << "  Adjoint produced invalid error indicators — stopping adaptation." << std::endl;
          break;
        }
        if (total_error < adjoint_tol && n_ar_marked == 0) {
          std::cerr << "  Converged! Error below tolerance " << adjoint_tol << std::endl;
          break;
        }

        // Step 6: Mark and refine
        std::vector<bool> marked(indicators.size(), false);
        if (total_error >= adjoint_tol) {
          marked = markByIndicator(indicators, adapt_fraction);
        }
        int n_marked = 0;
        for (bool m : marked) if (m) n_marked++;
        if (!ar_marked.empty()) {
          for (size_t e = 0; e < marked.size(); ++e) marked[e] = marked[e] || ar_marked[e];
        }
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

        auto rmap = bisectMarkedElements(solver.mesh, marked);
        auto mesh_report = validateRefinedMesh(solver.mesh);
        printMeshValidationReport(mesh_report);

        // Save the newly refined mesh immediately so it can be inspected even
        // if the next primal solve stalls or is interrupted before completion.
        {
          std::string refined_mesh_file = output_dir + "/" + file_prefix
              + "adjoint_mesh_cycle" + std::to_string(cycle + 1) + ".bin";
          writeCompanionMeshSnapshot(solver.mesh, refined_mesh_file);
          std::cerr << "  Refined mesh preview saved to " << refined_mesh_file << std::endl;
        }

        // Step 6: Transfer solution to refined mesh
        solver.U_dg = interpolateSolution(solver.U_dg, rmap, solver.ndof_per_elem);
        solver.U.resize(solver.mesh.E.size());
        for (size_t e = 0; e < solver.mesh.E.size(); ++e)
          solver.U[e] = solver.cellAverage(solver.U_dg[e]);
      }
    } else {
      solver.solveSteady(itercap);
      std::cerr << "  Cl = " << std::setprecision(6) << solver.integrateCl()
                << "  (" << solver.mesh.E.size() << " elements)" << std::endl;
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

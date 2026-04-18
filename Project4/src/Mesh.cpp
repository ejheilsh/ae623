#include "Mesh.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

namespace {
void getElementGeometryNodes(const Mesh &mesh, const Element &el,
                             std::vector<Vec2> &xnode) {
  static const int perm_q2[] = {0, 2, 5, 1, 4, 3};
  static const int perm_q3[] = {0, 3, 9, 6, 8, 7, 4, 1, 2, 5};

  if (el.q_order == 1) {
    xnode = {mesh.V[el.v[0]], mesh.V[el.v[1]], mesh.V[el.v[2]]};
  } else if (el.q_order == 2) {
    xnode.resize(6);
    for (int k = 0; k < 6; ++k)
      xnode[k] = mesh.V[el.ho_nodes[perm_q2[k]]];
  } else {
    xnode.resize(10);
    for (int k = 0; k < 10; ++k)
      xnode[k] = mesh.V[el.ho_nodes[perm_q3[k]]];
  }
}

void evalGeomBasisTri(int q, double xi_q, double eta_q,
                      std::vector<double> &N,
                      std::vector<double> &dNdxi,
                      std::vector<double> &dNdeta) {
  int nnode = (q + 1) * (q + 2) / 2;
  N.assign(nnode, 0.0);
  dNdxi.assign(nnode, 0.0);
  dNdeta.assign(nnode, 0.0);

  if (q == 1) {
    N[0] = 1.0 - xi_q - eta_q;
    N[1] = xi_q;
    N[2] = eta_q;
    dNdxi[0] = -1.0; dNdxi[1] = 1.0; dNdxi[2] = 0.0;
    dNdeta[0] = -1.0; dNdeta[1] = 0.0; dNdeta[2] = 1.0;
  } else if (q == 2) {
    double l1 = 1.0 - xi_q - eta_q, l2 = xi_q, l3 = eta_q;
    N[0] = l1 * (2 * l1 - 1);
    N[1] = l2 * (2 * l2 - 1);
    N[2] = l3 * (2 * l3 - 1);
    N[3] = 4 * l1 * l2;
    N[4] = 4 * l2 * l3;
    N[5] = 4 * l3 * l1;

    dNdxi[0] = -3.0 + 4.0 * xi_q + 4.0 * eta_q;
    dNdxi[1] = -1.0 + 4.0 * xi_q;
    dNdxi[2] = 0.0;
    dNdxi[3] = 4.0 - 8.0 * xi_q - 4.0 * eta_q;
    dNdxi[4] = 4.0 * eta_q;
    dNdxi[5] = -4.0 * eta_q;

    dNdeta[0] = -3.0 + 4.0 * xi_q + 4.0 * eta_q;
    dNdeta[1] = 0.0;
    dNdeta[2] = -1.0 + 4.0 * eta_q;
    dNdeta[3] = -4.0 * xi_q;
    dNdeta[4] = 4.0 * xi_q;
    dNdeta[5] = 4.0 - 4.0 * xi_q - 8.0 * eta_q;
  } else {
    double xi2 = xi_q * xi_q, xi3 = xi_q * xi2;
    double et2 = eta_q * eta_q, et3 = eta_q * et2;
    N[0] = 1.0 - 11.0 / 2 * xi_q - 11.0 / 2 * eta_q + 9 * xi2 + 18 * xi_q * eta_q +
           9 * et2 - 9.0 / 2 * xi3 - 27.0 / 2 * xi2 * eta_q -
           27.0 / 2 * xi_q * et2 - 9.0 / 2 * et3;
    N[1] = xi_q - 9.0 / 2 * xi2 + 9.0 / 2 * xi3;
    N[2] = eta_q - 9.0 / 2 * et2 + 9.0 / 2 * et3;
    N[3] = -9.0 / 2 * xi_q * eta_q + 27.0 / 2 * xi2 * eta_q;
    N[4] = -9.0 / 2 * xi_q * eta_q + 27.0 / 2 * xi_q * et2;
    N[5] = -9.0 / 2 * eta_q + 9.0 / 2 * xi_q * eta_q + 18 * et2 -
           27.0 / 2 * xi_q * et2 - 27.0 / 2 * et3;
    N[6] = 9 * eta_q - 45.0 / 2 * xi_q * eta_q - 45.0 / 2 * et2 +
           27.0 / 2 * xi2 * eta_q + 27 * xi_q * et2 + 27.0 / 2 * et3;
    N[7] = 9 * xi_q - 45.0 / 2 * xi2 - 45.0 / 2 * xi_q * eta_q +
           27.0 / 2 * xi3 + 27 * xi2 * eta_q + 27.0 / 2 * xi_q * et2;
    N[8] = -9.0 / 2 * xi_q + 18 * xi2 + 9.0 / 2 * xi_q * eta_q -
           27.0 / 2 * xi3 - 27.0 / 2 * xi2 * eta_q;
    N[9] = 27 * xi_q * eta_q - 27 * xi2 * eta_q - 27 * xi_q * et2;

    dNdxi[0] = -11.0 / 2 + 18 * xi_q + 18 * eta_q - 27.0 / 2 * xi2 -
               27 * xi_q * eta_q - 27.0 / 2 * et2;
    dNdxi[1] = 1 - 9 * xi_q + 27.0 / 2 * xi2;
    dNdxi[2] = 0.0;
    dNdxi[3] = -9.0 / 2 * eta_q + 27 * xi_q * eta_q;
    dNdxi[4] = -9.0 / 2 * eta_q + 27.0 / 2 * et2;
    dNdxi[5] = 9.0 / 2 * eta_q - 27.0 / 2 * et2;
    dNdxi[6] = -45.0 / 2 * eta_q + 27 * xi_q * eta_q + 27 * et2;
    dNdxi[7] = 9 - 45 * xi_q - 45.0 / 2 * eta_q + 81.0 / 2 * xi2 +
               54 * xi_q * eta_q + 27.0 / 2 * et2;
    dNdxi[8] = -9.0 / 2 + 36 * xi_q + 9.0 / 2 * eta_q - 81.0 / 2 * xi2 -
               27 * xi_q * eta_q;
    dNdxi[9] = 27 * eta_q - 54 * xi_q * eta_q - 27 * et2;

    dNdeta[0] = -11.0 / 2 + 18 * xi_q + 18 * eta_q - 27.0 / 2 * xi2 -
                27 * xi_q * eta_q - 27.0 / 2 * et2;
    dNdeta[1] = 0.0;
    dNdeta[2] = 1 - 9 * eta_q + 27.0 / 2 * et2;
    dNdeta[3] = -9.0 / 2 * xi_q + 27.0 / 2 * xi2;
    dNdeta[4] = -9.0 / 2 * xi_q + 27 * xi_q * eta_q;
    dNdeta[5] = -9.0 / 2 + 9.0 / 2 * xi_q + 36 * eta_q - 27 * xi_q * eta_q -
                81.0 / 2 * et2;
    dNdeta[6] = 9 - 45.0 / 2 * xi_q - 45 * eta_q + 27.0 / 2 * xi2 +
                54 * xi_q * eta_q + 81.0 / 2 * et2;
    dNdeta[7] = -45.0 / 2 * xi_q + 27 * xi2 + 27 * xi_q * eta_q;
    dNdeta[8] = 9.0 / 2 * xi_q - 27.0 / 2 * xi2;
    dNdeta[9] = 27 * xi_q - 27 * xi2 - 54 * xi_q * eta_q;
  }
}

bool edgeRefParamLocal(const int ev[3], int va, int vb,
                       double &xi0, double &eta0,
                       double &xi1, double &eta1) {
  const double rx[3] = {0.0, 1.0, 0.0};
  const double ry[3] = {0.0, 0.0, 1.0};

  int ia = -1, ib = -1;
  for (int k = 0; k < 3; ++k) {
    if (ev[k] == va) ia = k;
    if (ev[k] == vb) ib = k;
  }
  if (ia < 0 || ib < 0) return false;
  xi0 = rx[ia]; eta0 = ry[ia];
  xi1 = rx[ib]; eta1 = ry[ib];
  return true;
}

struct TriQuadPoint {
  double xi;
  double eta;
  double w;
};

const std::vector<TriQuadPoint> &geometryTriangleQuadrature() {
  // 12-point rule, exact through degree 6 on the reference triangle.
  // This is exact for q=2 geometry moments and sufficiently accurate for the
  // q=3 metric reconstruction needed by computeGeometry().
  static const std::vector<TriQuadPoint> qpts = {
      {0.063089014491502, 0.063089014491502, 0.025422453185104},
      {0.063089014491502, 0.873821971016996, 0.025422453185104},
      {0.873821971016996, 0.063089014491502, 0.025422453185104},
      {0.249286745170910, 0.249286745170910, 0.058393137863189},
      {0.249286745170910, 0.501426509658179, 0.058393137863189},
      {0.501426509658179, 0.249286745170910, 0.058393137863189},
      {0.053145049844816, 0.310352451033785, 0.041425537809187},
      {0.053145049844816, 0.636502499121399, 0.041425537809187},
      {0.310352451033785, 0.053145049844816, 0.041425537809187},
      {0.310352451033785, 0.636502499121399, 0.041425537809187},
      {0.636502499121399, 0.053145049844816, 0.041425537809187},
      {0.636502499121399, 0.310352451033785, 0.041425537809187},
  };
  return qpts;
}

struct EdgeQuadPoint {
  double t;
  double w;
};

const std::vector<EdgeQuadPoint> &geometryEdgeQuadrature() {
  // 4-point Gauss-Legendre rule on [0,1].
  static const std::vector<EdgeQuadPoint> qpts = {
      {0.069431844202974, 0.173927422568727},
      {0.330009478207572, 0.326072577431273},
      {0.669990521792428, 0.326072577431273},
      {0.930568155797026, 0.173927422568727},
  };
  return qpts;
}
} // namespace

bool Mesh::readGRI(const std::string &filename) {
  std::ifstream f(filename);
  if (!f.is_open()) {
    std::cerr << "Error: Could not open mesh file " << filename << std::endl;
    return false;
  }

  int Nn, Ne, dim;
  f >> Nn >> Ne >> dim;

  V.resize(Nn);
  for (int i = 0; i < Nn; ++i) {
    f >> V[i].x >> V[i].y;
  }

  int NB;
  f >> NB;
  Bname.resize(NB);
  std::vector<std::vector<std::vector<int>>> Braw(NB);
  std::vector<int> Bnnode(NB);

  for (int i = 0; i < NB; ++i) {
    int Nb;
    f >> Nb >> Bnnode[i] >> Bname[i];
    Braw[i].resize(Nb, std::vector<int>(Bnnode[i]));
    for (int j = 0; j < Nb; ++j) {
      for (int k = 0; k < Bnnode[i]; ++k) {
        f >> Braw[i][j][k];
      }
    }
  }

  E.clear();
  int Ne_read = 0;
  q_order_global = 1;
  has_curved_elements = false;

  while (Ne_read < Ne && f.good()) {
    int ne, degree;
    std::string type;
    if (!(f >> ne >> degree >> type))
      break;

    // Standard GRI format: nElements degree TriLagrange
    // degree = geometry order: 1 → 3 nodes, 2 → 6 nodes, 3 → 10 nodes
    // nodes_per_elem = (degree+1)*(degree+2)/2
    if (type != "TriLagrange") {
      std::cerr << "Warning: skipping unexpected non-element block: "
                << ne << " " << degree << " " << type << std::endl;
      continue;
    }

    if (ne <= 0) {
      std::cerr << "Warning: skipping empty TriLagrange block with degree "
                << degree << std::endl;
      continue;
    }

    int nodes_per_elem = (degree + 1) * (degree + 2) / 2;
    int q = degree;

    // Sanity check: degree should be 1, 2, or 3 for TriLagrange
    if (nodes_per_elem < 3 || nodes_per_elem > 10) {
      std::cerr << "Warning: unexpected degree " << degree
                << " in element block (nodes_per_elem=" << nodes_per_elem
                << "). Falling back to 3 nodes (q=1)." << std::endl;
      nodes_per_elem = 3;
      q = 1;
    }

    if (q > q_order_global) q_order_global = q;
    if (q > 1) has_curved_elements = true;

    for (int i = 0; i < ne; ++i) {
      Element el;
      el.q_order = q;
      std::vector<int> allnodes(nodes_per_elem);
      for (int k = 0; k < nodes_per_elem; ++k) {
        if (!(f >> allnodes[k])) break;
        allnodes[k]--;  // 1-based to 0-based
      }

      if (q == 1) {
        // q=1: all 3 nodes are corners
        el.v[0] = allnodes[0];
        el.v[1] = allnodes[1];
        el.v[2] = allnodes[2];
      } else {
        // q>1: GRI uses row-by-row ordering.  Corners are at positions:
        //   v0 = index 0,  v1 = index q,  v2 = index npe-1
        // (q=2: 0,2,5;  q=3: 0,3,9)
        // Store ALL geometry nodes in GRI order in ho_nodes for the
        // isoparametric mapping; extract corners into v[0..2] for
        // edge-hashing and connectivity.
        el.v[0] = allnodes[0];
        el.v[1] = allnodes[q];
        el.v[2] = allnodes[nodes_per_elem - 1];
        for (int k = 0; k < nodes_per_elem; ++k)
          el.ho_nodes.push_back(allnodes[k]);
      }
      E.push_back(el);
    }
    Ne_read += ne;
  }

  q_order_global = std::max(q_order_global, 1);

  if (Ne_read < Ne) {
    std::cerr << "Error: Only read " << Ne_read << " elements, expected " << Ne
              << std::endl;
    return false;
  }

  // Convert boundaries to node-pair format (0-based)
  std::vector<std::vector<std::vector<int>>> B(NB);
  for (int i = 0; i < NB; ++i) {
    if (Braw[i].empty())
      continue;
    bool is_elem_face = (Braw[i][0].size() == 2 && Bnnode[i] == 2);
    if (is_elem_face) {
      // Robust check: if any second value > 3, it's node-pair format.
      // Or if any first value > Ne, it's node-pair format.
      for (const auto &row : Braw[i]) {
        if (row[1] > 3 || row[0] > (int)E.size()) {
          is_elem_face = false;
          break;
        }
      }
    }

    if (is_elem_face) {
      for (const auto &bi : Braw[i]) {
        int elem_idx = bi[0] - 1;
        int face_idx = bi[1] - 1;
        int n1 = E[elem_idx].v[(face_idx + 1) % 3];
        int n2 = E[elem_idx].v[(face_idx + 2) % 3];
        B[i].push_back({n1, n2});
      }
    } else {
      for (const auto &bi : Braw[i]) {
        std::vector<int> nodes;
        for (int node : bi)
          nodes.push_back(node - 1);
        B[i].push_back(nodes);
      }
    }
  }

  // Periodic Groups
  periodicGroups.clear();
  std::string token;
  while (f >> token) {
    if (isdigit(token[0])) {
      int nGroups = std::stoi(token);
      std::string pgLabel;
      f >> pgLabel;
      if (pgLabel == "PeriodicGroup") {
        for (int i = 0; i < nGroups; ++i) {
          PeriodicGroup pg;
          f >> pg.nPairs >> pg.type;
          for (int j = 0; j < pg.nPairs; ++j) {
            int n1, n2;
            f >> n1 >> n2;
            int pair_n1 = n1 - 1;
            int pair_n2 = n2 - 1;
            pg.pairs.push_back({pair_n1, pair_n2});
          }
          periodicGroups.push_back(pg);
        }
      }
    }
  }
  int totalPeriodicPairs = 0;
  for (const auto &pg : periodicGroups) {
    totalPeriodicPairs += static_cast<int>(pg.pairs.size());
  }
  if (!periodicGroups.empty()) {
    std::cout << "PeriodicGroup read: total_pairs=" << totalPeriodicPairs
              << ", groups=" << periodicGroups.size() << ", per_group=[";
    for (size_t i = 0; i < periodicGroups.size(); ++i) {
      if (i > 0)
        std::cout << ", ";
      std::cout << periodicGroups[i].pairs.size();
    }
    std::cout << "]" << std::endl;
  } else {
    std::cout << "PeriodicGroup read: total_pairs=0, groups=0" << std::endl;
  }

  edgeHash(B);
  appendPeriodicToIE();
  buildInteriorFaceColors();
  computeGeometry();
  std::cout << "Mesh loaded: Ni=" << IE.size() << ", Nb=" << BE.size()
            << ", Ne=" << E.size() << std::endl;
  std::cout << "Mesh geometry order: q=" << q_order_global;
  if (has_curved_elements)
    std::cout << " (curved elements present)" << std::endl;
  else
    std::cout << " (straight triangles)" << std::endl;
  return true;
}

void Mesh::appendPeriodicToIE() {
  if (periodicGroups.empty())
    return;

  std::cout << "pairing periodic edges from ordered node pairs..." << std::endl;

  auto sortedEdge = [](int a, int b) {
    return std::make_pair(std::min(a, b), std::max(a, b));
  };

  // edge -> adjacent element ids from connectivity
  std::map<std::pair<int, int>, std::vector<int>> edgeToElems;
  for (int elem = 0; elem < static_cast<int>(E.size()); ++elem) {
    const int n0 = E[elem].v[0];
    const int n1 = E[elem].v[1];
    const int n2 = E[elem].v[2];
    edgeToElems[sortedEdge(n0, n1)].push_back(elem);
    edgeToElems[sortedEdge(n1, n2)].push_back(elem);
    edgeToElems[sortedEdge(n2, n0)].push_back(elem);
  }

  // Existing IE set to avoid duplicates.
  std::set<std::pair<int, int>> ieSet;
  for (const auto &ie : IE) {
    ieSet.insert(sortedEdge(ie.v[0], ie.v[1]));
  }

  // BE lookup so periodic sides can be removed if present.
  std::map<std::pair<int, int>, std::vector<int>> beKeyToIdx;
  for (int i = 0; i < static_cast<int>(BE.size()); ++i) {
    beKeyToIdx[sortedEdge(BE[i].v[0], BE[i].v[1])].push_back(i);
  }

  std::set<int> removeBEIdx;
  int addedCount = 0;
  int expectedCount = 0;
  std::vector<std::string> unresolved;

  for (int gidx = 0; gidx < static_cast<int>(periodicGroups.size()); ++gidx) {
    const auto &pg = periodicGroups[gidx];
    if (pg.pairs.size() < 2)
      continue;

    std::vector<std::pair<int, int>> pairs = pg.pairs;

    // Ensure each pair is (lower-y node, upper-y node).
    for (auto &p : pairs) {
      if (V[p.first].y > V[p.second].y) {
        std::swap(p.first, p.second);
      }
    }

    // Sort by x of lower node, stable for ties.
    std::stable_sort(
        pairs.begin(), pairs.end(),
        [&](const std::pair<int, int> &a, const std::pair<int, int> &b) {
          return V[a.first].x < V[b.first].x;
        });

    for (int i = 0; i < static_cast<int>(pairs.size()) - 1; ++i) {
      const int b1 = pairs[i].first;
      const int t1 = pairs[i].second;
      const int b2 = pairs[i + 1].first;
      const int t2 = pairs[i + 1].second;

      const auto keyBottom = sortedEdge(b1, b2);
      const auto keyTop = sortedEdge(t1, t2);
      expectedCount += 1;

      const auto itBottom = edgeToElems.find(keyBottom);
      const auto itTop = edgeToElems.find(keyTop);
      if (itBottom == edgeToElems.end() || itTop == edgeToElems.end()) {
        if (unresolved.size() < 5) {
          std::ostringstream oss;
          oss << "(g=" << gidx << ", i=" << i << ", bottom=(" << keyBottom.first
              << "," << keyBottom.second << "), top=(" << keyTop.first << ","
              << keyTop.second << "), missing_edge)";
          unresolved.push_back(oss.str());
        }
        continue;
      }
      if (itBottom->second.size() != 1 || itTop->second.size() != 1) {
        if (unresolved.size() < 5) {
          std::ostringstream oss;
          oss << "(g=" << gidx << ", i=" << i << ", bottom=(" << keyBottom.first
              << "," << keyBottom.second << "), top=(" << keyTop.first << ","
              << keyTop.second << "), adj=" << itBottom->second.size() << "/"
              << itTop->second.size() << ")";
          unresolved.push_back(oss.str());
        }
        continue;
      }
      if (ieSet.count(keyBottom)) {
        continue;
      }

      int n1b = b1;
      int n2b = b2;
      const double x1 = V[n1b].x, x2 = V[n2b].x;
      const double y1 = V[n1b].y, y2 = V[n2b].y;
      if ((x1 > x2) || (std::abs(x1 - x2) <= 1e-12 && y1 > y2)) {
        std::swap(n1b, n2b);
      }

      const int elemL = itBottom->second[0];
      const int elemR = itTop->second[0];
      // v[] = bottom nodes (elemL's side), vR[] = top nodes (elemR's side).
      // Match orientation: t1 corresponds to b1 (same x-position), t2 to b2.
      // n1b,n2b may be swapped for left-to-right ordering; apply same swap to top.
      int n1t = t1, n2t = t2;
      if (n1b == b2) { std::swap(n1t, n2t); }  // b1<->b2 were swapped above, mirror for top
      IE.push_back({n1b, n2b, n1t, n2t, elemL, elemR});
      ieSet.insert(keyBottom);
      addedCount += 1;

      auto itB = beKeyToIdx.find(keyBottom);
      if (itB != beKeyToIdx.end()) {
        for (int idx : itB->second) {
          removeBEIdx.insert(idx);
        }
      }
      auto itT = beKeyToIdx.find(keyTop);
      if (itT != beKeyToIdx.end()) {
        for (int idx : itT->second) {
          removeBEIdx.insert(idx);
        }
      }
    }
  }

  if (!removeBEIdx.empty()) {
    std::vector<BoundaryEdge> beFiltered;
    beFiltered.reserve(BE.size() - removeBEIdx.size());
    for (int i = 0; i < static_cast<int>(BE.size()); ++i) {
      if (!removeBEIdx.count(i)) {
        beFiltered.push_back(BE[i]);
      }
    }
    BE.swap(beFiltered);
  }

  std::cout << "Added " << addedCount << " periodic IEs to mesh" << std::endl;
  if (!unresolved.empty()) {
    std::cout << "WARNING: periodic ordered-pair candidates not added. count="
              << (expectedCount - addedCount) << ", samples=[";
    for (size_t i = 0; i < unresolved.size(); ++i) {
      if (i > 0)
        std::cout << ", ";
      std::cout << unresolved[i];
    }
    std::cout << "]" << std::endl;
  }
}

void Mesh::edgeHash(const std::vector<std::vector<std::vector<int>>> &B) {
  int Ne_count = E.size();
  std::map<std::pair<int, int>, int> H; // Directional hash: (n1, n2) -> elem+1
  IE.clear();

  for (int e = 0; e < Ne_count; ++e) {
    for (int i = 0; i < 3; ++i) {
      int n1 = E[e].v[i];
      int n2 = E[e].v[(i + 1) % 3];

      // If the reverse edge existed, it means eL was already found.
      // eL had edge n2 -> n1 (CCW), so its normal points OUT.
      // n1 -> n2 is CCW for the CURRENT element eR.
      // So if we store {n2, n1, eL, eR}, the normal computed from n2 -> n1
      // will point OUT of eL and INTO eR.
      if (H.count({n2, n1})) {
        int eL = H[{n2, n1}] - 1;
        int eR = e;
        IE.push_back({n2, n1, n2, n1, eL, eR});
        H.erase({n2, n1});
      } else {
        H[{n1, n2}] = e + 1;
      }
    }
  }

  BE.clear();
  for (int g = 0; g < (int)B.size(); ++g) {
    for (const auto &bi : B[g]) {
      int n1 = bi[0];
      int n2 = bi[1];
      // Boundary edges: H only contains keys that aren't shared.
      if (H.count({n1, n2})) {
        BE.push_back({n1, n2, H[{n1, n2}] - 1, g});
      } else if (H.count({n2, n1})) {
        BE.push_back({n2, n1, H[{n2, n1}] - 1, g});
      }
    }
  }
}

void Mesh::buildInteriorFaceColors() {
  ie_face_colors.assign(IE.size(), -1);
  ie_faces_by_color.clear();
  if (IE.empty() || E.empty())
    return;

  std::vector<std::vector<int>> elem_to_faces(E.size());
  for (int i = 0; i < static_cast<int>(IE.size()); ++i) {
    elem_to_faces[IE[i].elemL].push_back(i);
    elem_to_faces[IE[i].elemR].push_back(i);
  }

  for (int i = 0; i < static_cast<int>(IE.size()); ++i) {
    std::set<int> used_colors;
    int eL = IE[i].elemL;
    int eR = IE[i].elemR;
    for (int fidx : elem_to_faces[eL]) {
      if (fidx != i && ie_face_colors[fidx] >= 0)
        used_colors.insert(ie_face_colors[fidx]);
    }
    for (int fidx : elem_to_faces[eR]) {
      if (fidx != i && ie_face_colors[fidx] >= 0)
        used_colors.insert(ie_face_colors[fidx]);
    }

    int color = 0;
    while (used_colors.count(color))
      ++color;
    ie_face_colors[i] = color;
    if (color >= static_cast<int>(ie_faces_by_color.size()))
      ie_faces_by_color.resize(color + 1);
    ie_faces_by_color[color].push_back(i);
  }
}

void Mesh::computeGeometry() {
  centroids.resize(E.size());
  areas.resize(E.size());
  for (int i = 0; i < (int)E.size(); ++i) {
    if (E[i].q_order == 1) {
      Vec2 v1 = V[E[i].v[0]];
      Vec2 v2 = V[E[i].v[1]];
      Vec2 v3 = V[E[i].v[2]];
      centroids[i] = (v1 + v2 + v3) / 3.0;
      areas[i] = 0.5 * std::abs((v2.x - v1.x) * (v3.y - v1.y) -
                                (v3.x - v1.x) * (v2.y - v1.y));
      continue;
    }

    double signed_area = 0.0;
    Vec2 first_moment = {0.0, 0.0};
    for (const auto &qp : geometryTriangleQuadrature()) {
      ElementGeomEval geom = evaluateElementGeometry(i, qp.xi, qp.eta);
      double w = qp.w * geom.detJ;
      signed_area += w;
      first_moment += geom.x * w;
    }

    if (std::abs(signed_area) < 1e-14) {
      Vec2 v1 = V[E[i].v[0]];
      Vec2 v2 = V[E[i].v[1]];
      Vec2 v3 = V[E[i].v[2]];
      centroids[i] = (v1 + v2 + v3) / 3.0;
      areas[i] = 0.5 * std::abs((v2.x - v1.x) * (v3.y - v1.y) -
                                (v3.x - v1.x) * (v2.y - v1.y));
      continue;
    }

    double area = std::abs(signed_area);
    if (signed_area < 0.0)
      first_moment = first_moment * -1.0;
    areas[i] = area;
    centroids[i] = first_moment / area;
  }

  inormals.resize(IE.size());
  ilengths.resize(IE.size());
  for (int i = 0; i < (int)IE.size(); ++i) {
    Vec2 v1 = V[IE[i].v[0]];
    Vec2 v2 = V[IE[i].v[1]];
    double dx = v2.x - v1.x;
    double dy = v2.y - v1.y;
    double len = std::sqrt(dx * dx + dy * dy);
    ilengths[i] = len;
    inormals[i] = {dy / len, -dx / len};

    // Ensure normal points from elemL to elemR
    Vec2 dLR = centroids[IE[i].elemR] - centroids[IE[i].elemL];
    // Periodic shift correction for dLR
    bool isPeriodic = false;
    if (std::abs(dLR.y) > 9.0) { // Standard shift for expected meshes
      isPeriodic = true;
      if (dLR.y > 0)
        dLR.y -= 18.0;
      else
        dLR.y += 18.0;
    }

    double dot = inormals[i].dot(dLR);
    if (dot < 0) {
      inormals[i] = inormals[i] * -1.0;
    }
  }

  bnormals.resize(BE.size());
  blengths.resize(BE.size());
  for (int i = 0; i < (int)BE.size(); ++i) {
    if (E[BE[i].elemL].q_order == 1) {
      Vec2 v1 = V[BE[i].v[0]];
      Vec2 v2 = V[BE[i].v[1]];
      double dx = v2.x - v1.x;
      double dy = v2.y - v1.y;
      double len = std::sqrt(dx * dx + dy * dy);
      blengths[i] = len;
      bnormals[i] = {dy / len, -dx / len};

      // Ensure normal points outward
      Vec2 dCm = (0.5 * (v1 + v2)) - centroids[BE[i].elemL];
      if (bnormals[i].dot(dCm) < 0) {
        bnormals[i] = bnormals[i] * -1.0;
      }
      continue;
    }

    double len = 0.0;
    for (const auto &qp : geometryEdgeQuadrature()) {
      EdgeGeomEval geom = evaluateEdgeGeometry(BE[i].elemL, BE[i].v[0], BE[i].v[1], qp.t);
      len += qp.w * geom.ds_dt;
    }
    blengths[i] = len;

    // Store a representative outward normal at the edge midpoint. The DG
    // residual uses pointwise curved normals directly; this cached normal is
    // for FV paths, diagnostics, and postprocessing that only need one normal.
    EdgeGeomEval mid_geom = evaluateEdgeGeometry(BE[i].elemL, BE[i].v[0], BE[i].v[1], 0.5);
    bnormals[i] = mid_geom.normal;
  }
}

ElementGeomEval Mesh::evaluateElementGeometry(int e, double xi, double eta) const {
  const Element &el = E[e];
  std::vector<Vec2> xnode;
  std::vector<double> N, dNdxi, dNdeta;
  getElementGeometryNodes(*this, el, xnode);
  evalGeomBasisTri(el.q_order, xi, eta, N, dNdxi, dNdeta);

  ElementGeomEval geom{};
  geom.x = {0.0, 0.0};
  geom.dx_dxi = geom.dx_deta = 0.0;
  geom.dy_dxi = geom.dy_deta = 0.0;
  for (int k = 0; k < static_cast<int>(xnode.size()); ++k) {
    geom.x.x += N[k] * xnode[k].x;
    geom.x.y += N[k] * xnode[k].y;
    geom.dx_dxi += dNdxi[k] * xnode[k].x;
    geom.dx_deta += dNdeta[k] * xnode[k].x;
    geom.dy_dxi += dNdxi[k] * xnode[k].y;
    geom.dy_deta += dNdeta[k] * xnode[k].y;
  }
  geom.detJ = geom.dx_dxi * geom.dy_deta - geom.dx_deta * geom.dy_dxi;
  return geom;
}

EdgeGeomEval Mesh::evaluateEdgeGeometry(int e, int va, int vb, double t) const {
  double xi0 = 0.0, eta0 = 0.0, xi1 = 0.0, eta1 = 0.0;
  if (!edgeRefParamLocal(E[e].v, va, vb, xi0, eta0, xi1, eta1)) {
    throw std::runtime_error("evaluateEdgeGeometry: edge vertices not found in element");
  }

  double xi = xi0 + t * (xi1 - xi0);
  double eta = eta0 + t * (eta1 - eta0);
  ElementGeomEval geom = evaluateElementGeometry(e, xi, eta);

  double dxi_dt = xi1 - xi0;
  double deta_dt = eta1 - eta0;
  Vec2 tangent = {
      geom.dx_dxi * dxi_dt + geom.dx_deta * deta_dt,
      geom.dy_dxi * dxi_dt + geom.dy_deta * deta_dt,
  };
  double ds_dt = tangent.norm();
  Vec2 normal = {tangent.y / ds_dt, -tangent.x / ds_dt};

  Vec2 dCm = geom.x - centroids[e];
  if (normal.dot(dCm) < 0.0) {
    normal = normal * -1.0;
  }

  EdgeGeomEval edge_geom{};
  edge_geom.x = geom.x;
  edge_geom.tangent = tangent;
  edge_geom.normal = normal;
  edge_geom.ds_dt = ds_dt;
  return edge_geom;
}

// ─────────────────────────────────────────────────────────────────────────────
// Mesh::globalToReference — Newton-Raphson inverse coordinate mapping
//   global (x,y)  →  reference (xi, eta)   for element e.
//
// Implements the algorithm from Listing 4.4.1 of the course notes.
// For straight (q=1) elements the mapping is affine so this converges in one
// Newton step; for curved (q>1) elements it iterates to the requested tolerance.
//
// The reference triangle has corners:  v0→(0,0),  v1→(1,0),  v2→(0,1).
// The isoparametric mapping for a degree-q triangle is:
//   x(ξ,η) = Σ_k  N_k(ξ,η) * x_k
// where N_k are the nodal Lagrange shape functions and x_k are the physical
// positions of all geometry nodes (corners + high-order nodes).
//
// For q=1: x = x0*(1-ξ-η) + x1*ξ + x2*η   (exactly 3 nodes, affine)
// For q=2: 6 nodes, quadratic mapping
// For q=3: 10 nodes, cubic mapping
//
// The Jacobian J = ∂x/∂ξ  (2×2) is computed analytically from the shape
// function gradients and inverted analytically (2×2 case).
// ─────────────────────────────────────────────────────────────────────────────
bool Mesh::globalToReference(int e, const Vec2 &xglob,
                             double &xi, double &eta,
                             double tol, int maxIter) const {
  // Initialise Newton at the centroid of the reference triangle
  xi  = 1.0/3.0;
  eta = 1.0/3.0;
  const double dmax = 1.0;  // maximum update magnitude (per Listing 4.4.1 line 30)
  for (int iter = 0; iter < maxIter; ++iter) {
    ElementGeomEval geom = evaluateElementGeometry(e, xi, eta);
    Vec2 R = geom.x - xglob;
    if (std::sqrt(R.x*R.x + R.y*R.y) < tol) return true;

    // Analytical 2×2 inverse:  J^{-1} = 1/det * [[J11,-J01],[-J10,J00]]
    double J00 = geom.dx_dxi;
    double J01 = geom.dx_deta;
    double J10 = geom.dy_dxi;
    double J11 = geom.dy_deta;
    double det = geom.detJ;
    if (std::abs(det) < 1e-14) return false;  // singular — not inside this element
    double inv_det = 1.0 / det;

    // Newton update: dxref = -J^{-1} * R
    double dxi  = inv_det * (-J11*R.x + J01*R.y);
    double deta = inv_det * ( J10*R.x - J00*R.y);

    // Limit update magnitude to dmax (Listing 4.4.1 lines 29-30)
    double d = std::max(std::abs(dxi), std::abs(deta));
    if (d > dmax) { dxi *= dmax/d; deta *= dmax/d; }

    xi  += dxi;
    eta += deta;
  }
  return false;  // did not converge within maxIter
}

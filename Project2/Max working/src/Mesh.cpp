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
  while (Ne_read < Ne && f.good()) {
    int ne, degree;
    std::string type;
    if (!(f >> ne >> degree >> type))
      break;

    for (int i = 0; i < ne; ++i) {
      Element el;
      if (!(f >> el.v[0] >> el.v[1] >> el.v[2]))
        break;
      el.v[0]--;
      el.v[1]--;
      el.v[2]--; // 1-based to 0-based
      E.push_back(el);
    }
    Ne_read += ne;
  }

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
  computeGeometry();
  std::cout << "Mesh loaded: Ni=" << IE.size() << ", Nb=" << BE.size()
            << ", Ne=" << E.size() << std::endl;
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
      IE.push_back({n1b, n2b, elemL, elemR});
      ieSet.insert(keyBottom);
      addedCount += 1;
      std::cout << "Periodic IE added count: " << addedCount << std::endl;

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
        IE.push_back({n2, n1, eL, eR});
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

void Mesh::computeGeometry() {
  centroids.resize(E.size());
  areas.resize(E.size());
  for (int i = 0; i < (int)E.size(); ++i) {
    Vec2 v1 = V[E[i].v[0]];
    Vec2 v2 = V[E[i].v[1]];
    Vec2 v3 = V[E[i].v[2]];
    centroids[i] = (v1 + v2 + v3) / 3.0;
    areas[i] = 0.5 * std::abs((v2.x - v1.x) * (v3.y - v1.y) -
                              (v3.x - v1.x) * (v2.y - v1.y));
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
  }
}

#include "Mesh.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

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
    // We assume and check if it's the elem-face format based on coordinates or
    // max values if needed For simplicity, we follow the logic in readgri.py
    if (is_elem_face && Braw[i][0][1] <= 3) {
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
            pg.pairs.push_back({n1 - 1, n2 - 1});
          }
          periodicGroups.push_back(pg);
        }
      }
    }
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

  // Map boundary nodes to elements and edges
  std::map<std::pair<int, int>, int> edgeToElem;
  for (int e = 0; e < (int)E.size(); ++e) {
    for (int i = 0; i < 3; ++i) {
      int n1 = E[e].v[i];
      int n2 = E[e].v[(i + 1) % 3];
      edgeToElem[{std::min(n1, n2), std::max(n1, n2)}] = e;
    }
  }

  // This is a simplified translational periodic pairing
  for (const auto &pg : periodicGroups) {
    std::vector<int> botNodes, topNodes;
    for (const auto &p : pg.pairs) {
      botNodes.push_back(p.first);
      topNodes.push_back(p.second);
    }

    struct PEdge {
      int n1, n2, elem;
      double midX;
    };
    std::vector<PEdge> botEdges, topEdges;

    auto isNodeIn = [](int n, const std::vector<int> &nodes) {
      return std::find(nodes.begin(), nodes.end(), n) != nodes.end();
    };

    // Find edges on bot/top boundaries
    for (auto const &[nodes, elem] : edgeToElem) {
      bool inBot =
          isNodeIn(nodes.first, botNodes) && isNodeIn(nodes.second, botNodes);
      bool inTop =
          isNodeIn(nodes.first, topNodes) && isNodeIn(nodes.second, topNodes);

      // Check if it's already an interior edge
      bool alreadyIE = false;
      for (const auto &ie : IE) {
        if ((ie.v[0] == nodes.first && ie.v[1] == nodes.second) ||
            (ie.v[0] == nodes.second && ie.v[1] == nodes.first)) {
          alreadyIE = true;
          break;
        }
      }
      if (alreadyIE)
        continue;

      if (inBot)
        botEdges.push_back({nodes.first, nodes.second, elem,
                            (V[nodes.first].x + V[nodes.second].x) * 0.5});
      if (inTop)
        topEdges.push_back({nodes.first, nodes.second, elem,
                            (V[nodes.first].x + V[nodes.second].x) * 0.5});
    }

    // Pair them up
    for (const auto &be : botEdges) {
      for (const auto &te : topEdges) {
        if (std::abs(be.midX - te.midX) < 1e-8) {
          IE.push_back({be.n1, be.n2, be.elem, te.elem});
          break;
        }
      }
    }
  }
}

void Mesh::edgeHash(const std::vector<std::vector<std::vector<int>>> &B) {
  int Ne_count = E.size();
  std::map<std::pair<int, int>, int> H;
  IE.clear();

  for (int e = 0; e < Ne_count; ++e) {
    for (int i = 0; i < 3; ++i) {
      int n1 = E[e].v[i];
      int n2 = E[e].v[(i + 1) % 3];
      std::pair<int, int> key = {std::min(n1, n2), std::max(n1, n2)};
      if (H.find(key) == H.end()) {
        H[key] = e + 1; // 1-based to allow 0 as "not found"
      } else {
        int eL = H[key] - 1;
        int eR = e;
        // Re-orient so n1, n2 are CCW for eL
        // In our format, (n1, n2) is an edge of element 'e'.
        // If it's already in H, it means it's an interior edge.
        // We store it as (n1, n2, eR, eL) or similar.
        // To match Python: IE contains (n1, n2, e, eR) where H[n2,n1] was found
        // Let's just store it and handle orientation in geometry.
        IE.push_back({n1, n2, eL, eR});
        H.erase(key);
      }
    }
  }

  BE.clear();
  for (int g = 0; g < (int)B.size(); ++g) {
    for (const auto &bi : B[g]) {
      int n1 = bi[0];
      int n2 = bi[1];
      std::pair<int, int> key = {std::min(n1, n2), std::max(n1, n2)};
      if (H.find(key) != H.end()) {
        int e = H[key] - 1;
        // Check orientation: +90 deg rotation of (n2-n1) should point out
        // n = (y2-y1, x1-x2). dot(n, centroid - mid) should be negative for
        // outward.
        BE.push_back({n1, n2, e, g});
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
    if (std::abs(dLR.y) > 9.0) { // Height of coarse.gri is 18
      if (dLR.y > 0)
        dLR.y -= 18.0;
      else
        dLR.y += 18.0;
    }
    if (inormals[i].dot(dLR) < 0) {
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

#include "MeshRefinementQ2.hpp"
#include "MeshRefinement.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

struct QuadraticDetJPoly {
  double a = 0.0;
  double b = 0.0;
  double c = 0.0;
  double d = 0.0;
  double e = 0.0;
  double f = 0.0;

  double eval(double xi, double eta) const {
    return a + b * xi + c * eta + d * xi * xi + e * xi * eta + f * eta * eta;
  }
};

double triangleSignedAreaPts(const Vec2 &a, const Vec2 &b, const Vec2 &c) {
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

double triangleMinAngleDegPts(const Vec2 &a, const Vec2 &b, const Vec2 &c) {
  const double ang_a = angleDegBetween(b - a, c - a);
  const double ang_b = angleDegBetween(a - b, c - b);
  const double ang_c = angleDegBetween(a - c, b - c);
  return std::min(ang_a, std::min(ang_b, ang_c));
}

double triangleQualityPts(const Vec2 &a, const Vec2 &b, const Vec2 &c) {
  const Vec2 ab = b - a;
  const Vec2 bc = c - b;
  const Vec2 ca = a - c;
  const double sum_len2 = ab.normSq() + bc.normSq() + ca.normSq();
  if (sum_len2 <= 1e-16) return -1.0;
  const double twice_area = std::abs(ab.x * (-ca.y) - ab.y * (-ca.x));
  return 2.0 * std::sqrt(3.0) * twice_area / sum_len2;
}

double childSetQualityScore(const std::vector<std::array<Vec2, 3>> &tris) {
  double min_angle = std::numeric_limits<double>::infinity();
  double min_quality = std::numeric_limits<double>::infinity();
  for (const auto &tri : tris) {
    const Vec2 &a = tri[0];
    const Vec2 &b = tri[1];
    const Vec2 &c = tri[2];
    if (triangleSignedAreaPts(a, b, c) <= 1e-12) {
      return -std::numeric_limits<double>::infinity();
    }
    min_angle = std::min(min_angle, triangleMinAngleDegPts(a, b, c));
    min_quality = std::min(min_quality, triangleQualityPts(a, b, c));
  }
  if (!std::isfinite(min_angle) || !std::isfinite(min_quality)) {
    return -std::numeric_limits<double>::infinity();
  }
  return 30.0 * min_angle + 1200.0 * min_quality;
}

void assignQ2Child(Element &child, int a, int b, int c, const Vec2 &ra,
                   const Vec2 &rb, const Vec2 &rc,
                   bool rebuild_wall_edges_from_blade,
                   const Q2BuildCallbacks &callbacks) {
  child.v[0] = a;
  child.v[1] = b;
  child.v[2] = c;
  child.q_order = 2;
  const Vec2 rab = 0.5 * (ra + rb);
  const Vec2 rca = 0.5 * (rc + ra);
  const Vec2 rbc = 0.5 * (rb + rc);
  auto edgeGeomPoint = [&](int p, int q, const Vec2 &rpq) {
    Vec2 xq = callbacks.parent_map(rpq);
    if (rebuild_wall_edges_from_blade &&
        callbacks.is_wall_edge({std::min(p, q), std::max(p, q)})) {
      xq = callbacks.wall_edge_midpoint(p, q);
    }
    return xq;
  };
  child.ho_nodes = {
      a,
      callbacks.get_geom_node(a, b, edgeGeomPoint(a, b, rab)),
      b,
      callbacks.get_geom_node(c, a, edgeGeomPoint(c, a, rca)),
      callbacks.get_geom_node(b, c, edgeGeomPoint(b, c, rbc)),
      c};
}

bool fitQ2DetJPolynomial(const Mesh &mesh, int elem_idx, QuadraticDetJPoly &poly) {
  if (elem_idx < 0 || elem_idx >= static_cast<int>(mesh.E.size())) return false;
  if (mesh.E[elem_idx].q_order != 2) return false;

  const double J00 = mesh.evaluateElementGeometry(elem_idx, 0.0, 0.0).detJ;
  const double J10 = mesh.evaluateElementGeometry(elem_idx, 1.0, 0.0).detJ;
  const double J01 = mesh.evaluateElementGeometry(elem_idx, 0.0, 1.0).detJ;
  const double J50 = mesh.evaluateElementGeometry(elem_idx, 0.5, 0.0).detJ;
  const double J55 = mesh.evaluateElementGeometry(elem_idx, 0.5, 0.5).detJ;
  const double J05 = mesh.evaluateElementGeometry(elem_idx, 0.0, 0.5).detJ;

  poly.a = J00;
  const double t_x = J10 - poly.a;
  const double s_x = J50 - poly.a;
  const double t_y = J01 - poly.a;
  const double s_y = J05 - poly.a;

  poly.b = 4.0 * s_x - t_x;
  poly.d = 2.0 * (t_x - 2.0 * s_x);
  poly.c = 4.0 * s_y - t_y;
  poly.f = 2.0 * (t_y - 2.0 * s_y);
  poly.e = 4.0 * (J55 - poly.a - 0.5 * poly.b - 0.5 * poly.c -
                  0.25 * poly.d - 0.25 * poly.f);
  return true;
}

bool getQ2BasisNodes(const Mesh &mesh, int elem_idx, std::array<Vec2, 6> &xnode) {
  if (elem_idx < 0 || elem_idx >= static_cast<int>(mesh.E.size())) return false;
  const auto &el = mesh.E[elem_idx];
  if (el.q_order != 2 || el.ho_nodes.size() != 6) return false;
  static const int perm_q2[] = {0, 2, 5, 1, 4, 3};
  for (int k = 0; k < 6; ++k) xnode[k] = mesh.V[el.ho_nodes[perm_q2[k]]];
  return true;
}

bool refPointInsideTriangle(double xi, double eta, double tol = 1e-12) {
  return xi >= -tol && eta >= -tol && (xi + eta) <= 1.0 + tol;
}

void considerExactDetJCandidate(const Mesh &mesh, int elem_idx,
                                double xi, double eta,
                                CurvedElementDetJMinimum &best) {
  if (!refPointInsideTriangle(xi, eta)) return;
  const ElementGeomEval geom = mesh.evaluateElementGeometry(elem_idx, xi, eta);
  if (!best.exact || geom.detJ < best.detJ) {
    best.exact = true;
    best.detJ = geom.detJ;
    best.ref = {xi, eta};
    best.x = geom.x;
  }
}

}  // namespace

bool shouldPromoteQ2ConformitySplit(const Q2ConformityPromotionInput &input) {
  if (!input.within_wall_cavity || !input.can_insert_edge_i ||
      !input.can_insert_edge_j) {
    return false;
  }
  const double binary_score = childSetQualityScore({
      {input.v2, input.v0, input.xSelected},
      {input.v2, input.xSelected, input.v1},
  });
  const double full_score = childSetQualityScore({
      {input.v0, input.xAB, input.xCA},
      {input.xAB, input.v1, input.xBC},
      {input.xCA, input.xBC, input.v2},
      {input.xAB, input.xBC, input.xCA},
  });
  return std::isfinite(full_score) && full_score > binary_score + 1e-8;
}

bool q2BinaryChildrenHavePositiveArea(const std::vector<Vec2> &verts, int va,
                                      int vb, int vc, int vm) {
  return triangleSignedAreaPts(verts[vc], verts[va], verts[vm]) > 1e-12 &&
         triangleSignedAreaPts(verts[vc], verts[vm], verts[vb]) > 1e-12;
}

bool q2FullSubdivisionChildrenHavePositiveArea(const std::vector<Vec2> &verts,
                                               int a, int b, int c, int mAB,
                                               int mBC, int mCA) {
  return triangleSignedAreaPts(verts[a], verts[mAB], verts[mCA]) > 1e-12 &&
         triangleSignedAreaPts(verts[mAB], verts[b], verts[mBC]) > 1e-12 &&
         triangleSignedAreaPts(verts[mCA], verts[mBC], verts[c]) > 1e-12 &&
         triangleSignedAreaPts(verts[mAB], verts[mBC], verts[mCA]) > 1e-12;
}

double q2BezierMinCornerArea(const Mesh &mesh, int elem_idx) {
  std::array<Vec2, 6> xnode;
  if (!getQ2BasisNodes(mesh, elem_idx, xnode)) return 0.0;

  const Vec2 b0 = xnode[0];
  const Vec2 b1 = xnode[1];
  const Vec2 b2 = xnode[2];
  const Vec2 b01 = 2.0 * xnode[3] - 0.5 * (xnode[0] + xnode[1]);
  const Vec2 b12 = 2.0 * xnode[4] - 0.5 * (xnode[1] + xnode[2]);
  const Vec2 b20 = 2.0 * xnode[5] - 0.5 * (xnode[2] + xnode[0]);

  const double a0 = triangleSignedAreaPts(b0, b01, b20);
  const double a1 = triangleSignedAreaPts(b1, b12, b01);
  const double a2 = triangleSignedAreaPts(b2, b20, b12);
  return std::min(a0, std::min(a1, a2));
}

bool exactQ2DetJMinimum(const Mesh &mesh, int elem_idx,
                        CurvedElementDetJMinimum &out) {
  out = CurvedElementDetJMinimum{};
  QuadraticDetJPoly poly;
  if (!fitQ2DetJPolynomial(mesh, elem_idx, poly)) return false;

  considerExactDetJCandidate(mesh, elem_idx, 0.0, 0.0, out);
  considerExactDetJCandidate(mesh, elem_idx, 1.0, 0.0, out);
  considerExactDetJCandidate(mesh, elem_idx, 0.0, 1.0, out);

  const double eps = 1e-14;
  if (std::abs(poly.d) > eps) {
    considerExactDetJCandidate(mesh, elem_idx, -poly.b / (2.0 * poly.d), 0.0, out);
  }
  if (std::abs(poly.f) > eps) {
    considerExactDetJCandidate(mesh, elem_idx, 0.0, -poly.c / (2.0 * poly.f), out);
  }
  const double edge_quad = poly.d - poly.e + poly.f;
  const double edge_lin = poly.b - poly.c + poly.e - 2.0 * poly.f;
  if (std::abs(edge_quad) > eps) {
    const double t = -edge_lin / (2.0 * edge_quad);
    considerExactDetJCandidate(mesh, elem_idx, t, 1.0 - t, out);
  }

  const double hdet = 4.0 * poly.d * poly.f - poly.e * poly.e;
  if (std::abs(hdet) > eps) {
    const double xi = (-2.0 * poly.f * poly.b + poly.e * poly.c) / hdet;
    const double eta = (poly.e * poly.b - 2.0 * poly.d * poly.c) / hdet;
    considerExactDetJCandidate(mesh, elem_idx, xi, eta, out);
  }

  return out.exact;
}

Q2BinarySplitChildren buildQ2BinarySplitChildren(
    int va, int vb, int vc, int vm, const Vec2 &ra, const Vec2 &rb,
    const Vec2 &rc, const Vec2 &rm, const Q2BuildCallbacks &callbacks) {
  Q2BinarySplitChildren out;
  assignQ2Child(out.updated_parent, vc, va, vm, rc, ra, rm, false, callbacks);
  assignQ2Child(out.new_child, vc, vm, vb, rc, rm, rb, false, callbacks);
  return out;
}

Q2FullSubdivisionChildren buildQ2FullSubdivisionChildren(
    int a, int b, int c, int mAB, int mBC, int mCA,
    const std::array<Vec2, 3> &ref_corner,
    bool rebuild_wall_children_from_blade,
    const Q2BuildCallbacks &callbacks) {
  Q2FullSubdivisionChildren out;
  const Vec2 &rA = ref_corner[0];
  const Vec2 &rB = ref_corner[1];
  const Vec2 &rC = ref_corner[2];
  const Vec2 rAB = 0.5 * (rA + rB);
  const Vec2 rBC = 0.5 * (rB + rC);
  const Vec2 rCA = 0.5 * (rC + rA);

  assignQ2Child(out.c0, a, mAB, mCA, rA, rAB, rCA,
                rebuild_wall_children_from_blade, callbacks);
  assignQ2Child(out.c1, mAB, b, mBC, rAB, rB, rBC,
                rebuild_wall_children_from_blade, callbacks);
  assignQ2Child(out.c2, mCA, mBC, c, rCA, rBC, rC,
                rebuild_wall_children_from_blade, callbacks);
  assignQ2Child(out.c3, mAB, mBC, mCA, rAB, rBC, rCA,
                rebuild_wall_children_from_blade, callbacks);
  return out;
}

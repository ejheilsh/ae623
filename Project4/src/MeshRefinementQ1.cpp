#include "MeshRefinementQ1.hpp"

namespace {

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
    elem.v[0] = a;
    elem.v[1] = b;
    elem.v[2] = c;
  } else {
    elem.v[0] = a;
    elem.v[1] = c;
    elem.v[2] = b;
  }
  return elem;
}

}  // namespace

Q1SplitChildren buildQ1SplitChildren(const std::vector<Vec2> &verts,
                                     int va, int vb, int vc, int vm) {
  Q1SplitChildren out;
  out.local_children_valid =
      triangleSignedAreaPts(verts[vc], verts[va], verts[vm]) > 1e-12 &&
      triangleSignedAreaPts(verts[vc], verts[vm], verts[vb]) > 1e-12;
  if (!out.local_children_valid) return out;

  out.updated_parent = makePositiveElement(vc, va, vm, verts);
  out.new_child = makePositiveElement(vc, vm, vb, verts);
  return out;
}

Q1WallFanChildren buildQ1WallFanChildren(const std::vector<Vec2> &verts,
                                         int va, int vb, int vc,
                                         int vm1, int vm2) {
  Q1WallFanChildren out;
  out.local_children_valid =
      triangleSignedAreaPts(verts[vc], verts[va], verts[vm1]) > 1e-12 &&
      triangleSignedAreaPts(verts[vc], verts[vm1], verts[vm2]) > 1e-12 &&
      triangleSignedAreaPts(verts[vc], verts[vm2], verts[vb]) > 1e-12;
  if (!out.local_children_valid) return out;

  out.updated_parent = makePositiveElement(vc, va, vm1, verts);
  out.new_child_1 = makePositiveElement(vc, vm1, vm2, verts);
  out.new_child_2 = makePositiveElement(vc, vm2, vb, verts);
  return out;
}

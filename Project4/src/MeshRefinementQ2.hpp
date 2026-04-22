#ifndef MESHREFINEMENTQ2_HPP
#define MESHREFINEMENTQ2_HPP

#include "Mesh.hpp"
#include <array>
#include <functional>
#include <utility>
#include <vector>

struct CurvedElementDetJMinimum;

using RefinementEdgeKey = std::pair<int, int>;

struct Q2ConformityPromotionInput {
  bool within_wall_cavity = false;
  bool can_insert_edge_i = false;
  bool can_insert_edge_j = false;
  Vec2 v0;
  Vec2 v1;
  Vec2 v2;
  Vec2 xAB;
  Vec2 xBC;
  Vec2 xCA;
  Vec2 xSelected;
};

bool shouldPromoteQ2ConformitySplit(const Q2ConformityPromotionInput &input);

bool q2BinaryChildrenHavePositiveArea(const std::vector<Vec2> &verts,
                                      int va, int vb, int vc, int vm);
bool q2FullSubdivisionChildrenHavePositiveArea(const std::vector<Vec2> &verts,
                                               int a, int b, int c,
                                               int mAB, int mBC, int mCA);
double q2BezierMinCornerArea(const Mesh &mesh, int elem_idx);
bool exactQ2DetJMinimum(const Mesh &mesh, int elem_idx,
                        CurvedElementDetJMinimum &out);

struct Q2BuildCallbacks {
  std::function<Vec2(const Vec2 &)> parent_map;
  std::function<int(int, int, const Vec2 &)> get_geom_node;
  std::function<bool(const RefinementEdgeKey &)> is_wall_edge;
  std::function<Vec2(int, int)> wall_edge_midpoint;
};

struct Q2BinarySplitChildren {
  Element updated_parent;
  Element new_child;
};

Q2BinarySplitChildren buildQ2BinarySplitChildren(
    int va, int vb, int vc, int vm, const Vec2 &ra, const Vec2 &rb,
    const Vec2 &rc, const Vec2 &rm, const Q2BuildCallbacks &callbacks);

struct Q2FullSubdivisionChildren {
  Element c0;
  Element c1;
  Element c2;
  Element c3;
};

Q2FullSubdivisionChildren buildQ2FullSubdivisionChildren(
    int a, int b, int c, int mAB, int mBC, int mCA,
    const std::array<Vec2, 3> &ref_corner,
    bool rebuild_wall_children_from_blade,
    const Q2BuildCallbacks &callbacks);

#endif

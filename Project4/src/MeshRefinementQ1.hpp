#ifndef MESHREFINEMENTQ1_HPP
#define MESHREFINEMENTQ1_HPP

#include "Mesh.hpp"
#include <vector>

struct Q1SplitChildren {
  bool local_children_valid = false;
  Element updated_parent;
  Element new_child;
};

struct Q1WallFanChildren {
  bool local_children_valid = false;
  Element updated_parent;
  Element new_child_1;
  Element new_child_2;
};

Q1SplitChildren buildQ1SplitChildren(const std::vector<Vec2> &verts,
                                     int va, int vb, int vc, int vm);

Q1WallFanChildren buildQ1WallFanChildren(const std::vector<Vec2> &verts,
                                         int va, int vb, int vc,
                                         int vm1, int vm2);

#endif

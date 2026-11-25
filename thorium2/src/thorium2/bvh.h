// Thorium 2 — bounding volume hierarchy for ray casting on the CPU.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "thorium2/vec.h"

#include <vector>

namespace th {

class Bvh {
public:
  struct Tri {
    Vec3 a, b, c;
    int face = -1;     ///< Model face (for shells: the front face).
    int element = -1;  ///< Model element.
  };
  struct Hit {
    int tri = -1;      ///< -1: no hit.
    double t = 0;      ///< Distance along the (unit) direction.
  };

  Bvh() = default;
  explicit Bvh(std::vector<Tri> tris);

  /// Nearest intersection with t in (0, t_max), ignoring triangles of element `skip_element`.
  Hit intersect(const Vec3& origin, const Vec3& dir, double t_max, int skip_element) const;
  /// Reference implementation: tests every triangle (for tests).
  Hit intersect_brute_force(const Vec3& origin, const Vec3& dir, double t_max, int skip_element) const;

  const Tri& tri(int i) const { return tris_[static_cast<size_t>(i)]; }
  size_t size() const { return tris_.size(); }

private:
  struct Box {
    Vec3 lo, hi;
  };
  struct Node {
    Box box;
    int left = -1;   ///< Inner node: index of the left child (right = left + 1). Leaf: -1.
    int first = 0;   ///< Leaf: first triangle index.
    int count = 0;   ///< Leaf: number of triangles.
  };
  void build(int index, int first, int count, int depth);
  static bool hit_triangle(const Tri& t, const Vec3& o, const Vec3& d, double t_max, double& t_hit);

  std::vector<Tri> tris_;
  std::vector<Node> nodes_;
};

}  // namespace th

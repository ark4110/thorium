// SPDX-License-Identifier: GPL-3.0-or-later
#include "thorium2/bvh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace th {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Vec3 vmin(const Vec3& a, const Vec3& b) { return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
Vec3 vmax(const Vec3& a, const Vec3& b) { return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }

double surface(const Vec3& lo, const Vec3& hi) {
  const Vec3 d = hi - lo;
  if (d.x < 0) return 0;
  return 2 * (d.x * d.y + d.y * d.z + d.z * d.x);
}

/// Slab test; returns entry distance or +inf.
double hit_box(const Vec3& lo, const Vec3& hi, const Vec3& o, const Vec3& inv, double t_max) {
  double t0 = 0, t1 = t_max;
  for (int a = 0; a < 3; ++a) {
    double tn = (lo[a] - o[a]) * inv[a];
    double tf = (hi[a] - o[a]) * inv[a];
    if (tn > tf) std::swap(tn, tf);
    // NaN (0 * inf) means the ray lies in the slab plane: no constraint from this axis.
    if (!std::isnan(tn)) t0 = std::max(t0, tn);
    if (!std::isnan(tf)) t1 = std::min(t1, tf + std::abs(tf) * 4 * std::numeric_limits<double>::epsilon());
    if (t0 > t1) return kInf;
  }
  return t0;
}

}  // namespace

Bvh::Bvh(std::vector<Tri> tris) : tris_(std::move(tris)) {
  if (tris_.empty()) return;
  nodes_.reserve(2 * tris_.size());
  nodes_.push_back({});
  build(0, 0, static_cast<int>(tris_.size()), 0);
}

void Bvh::build(int index, int first, int count, int depth) {
  Box box{{kInf, kInf, kInf}, {-kInf, -kInf, -kInf}};
  Box cbox = box;  // box of triangle centroids
  for (int i = first; i < first + count; ++i) {
    const Tri& t = tris_[static_cast<size_t>(i)];
    box.lo = vmin(box.lo, vmin(t.a, vmin(t.b, t.c)));
    box.hi = vmax(box.hi, vmax(t.a, vmax(t.b, t.c)));
    const Vec3 c = (t.a + t.b + t.c) / 3.0;
    cbox.lo = vmin(cbox.lo, c);
    cbox.hi = vmax(cbox.hi, c);
  }
  nodes_[static_cast<size_t>(index)].box = box;
  auto make_leaf = [&] {
    Node& n = nodes_[static_cast<size_t>(index)];
    n.left = -1;
    n.first = first;
    n.count = count;
  };
  if (count <= 4 || depth > 60) return make_leaf();

  // Binned surface area heuristic on the longest centroid axis.
  const Vec3 ext = cbox.hi - cbox.lo;
  const int axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z ? 1 : 2);
  if (!(ext[axis] > 0)) return make_leaf();
  constexpr int kBins = 16;
  struct Bin {
    Vec3 lo{kInf, kInf, kInf}, hi{-kInf, -kInf, -kInf};
    int n = 0;
  };
  std::array<Bin, kBins> bins;
  auto bin_of = [&](const Tri& t) {
    const double c = ((t.a + t.b + t.c) / 3.0)[axis];
    int b = static_cast<int>(kBins * (c - cbox.lo[axis]) / ext[axis]);
    return std::clamp(b, 0, kBins - 1);
  };
  for (int i = first; i < first + count; ++i) {
    const Tri& t = tris_[static_cast<size_t>(i)];
    Bin& b = bins[static_cast<size_t>(bin_of(t))];
    b.lo = vmin(b.lo, vmin(t.a, vmin(t.b, t.c)));
    b.hi = vmax(b.hi, vmax(t.a, vmax(t.b, t.c)));
    ++b.n;
  }
  double best_cost = kInf;
  int best_split = -1;
  for (int s = 1; s < kBins; ++s) {
    Vec3 llo{kInf, kInf, kInf}, lhi{-kInf, -kInf, -kInf}, rlo = llo, rhi = lhi;
    int nl = 0, nr = 0;
    for (int b = 0; b < s; ++b) llo = vmin(llo, bins[static_cast<size_t>(b)].lo), lhi = vmax(lhi, bins[static_cast<size_t>(b)].hi), nl += bins[static_cast<size_t>(b)].n;
    for (int b = s; b < kBins; ++b) rlo = vmin(rlo, bins[static_cast<size_t>(b)].lo), rhi = vmax(rhi, bins[static_cast<size_t>(b)].hi), nr += bins[static_cast<size_t>(b)].n;
    if (nl == 0 || nr == 0) continue;
    const double cost = surface(llo, lhi) * nl + surface(rlo, rhi) * nr;
    if (cost < best_cost) best_cost = cost, best_split = s;
  }
  if (best_split < 0 || best_cost >= surface(box.lo, box.hi) * count) return make_leaf();

  auto mid = std::partition(tris_.begin() + first, tris_.begin() + first + count,
                            [&](const Tri& t) { return bin_of(t) < best_split; });
  const int left_count = static_cast<int>(mid - (tris_.begin() + first));
  if (left_count == 0 || left_count == count) return make_leaf();

  const int left = static_cast<int>(nodes_.size());
  nodes_.push_back({});
  nodes_.push_back({});
  nodes_[static_cast<size_t>(index)].left = left;
  // Children are built into the two reserved slots (nodes_ may grow: no references kept).
  build(left, first, left_count, depth + 1);
  build(left + 1, first + left_count, count - left_count, depth + 1);
}

bool Bvh::hit_triangle(const Tri& t, const Vec3& o, const Vec3& d, double t_max, double& t_hit) {
  // Moller-Trumbore, double sided.
  const Vec3 e1 = t.b - t.a, e2 = t.c - t.a;
  const Vec3 p = cross(d, e2);
  const double det = dot(e1, p);
  if (det == 0) return false;
  const double inv = 1.0 / det;
  const Vec3 s = o - t.a;
  const double u = dot(s, p) * inv;
  if (u < 0 || u > 1) return false;
  const Vec3 q = cross(s, e1);
  const double v = dot(d, q) * inv;
  if (v < 0 || u + v > 1) return false;
  const double tt = dot(e2, q) * inv;
  if (!(tt > 0) || !(tt < t_max)) return false;
  t_hit = tt;
  return true;
}

Bvh::Hit Bvh::intersect(const Vec3& o, const Vec3& d, double t_max, int skip_element) const {
  Hit best;
  best.t = t_max;
  if (nodes_.empty()) return {};
  const Vec3 inv{1.0 / d.x, 1.0 / d.y, 1.0 / d.z};
  int stack[128];
  int sp = 0;
  stack[sp++] = 0;
  while (sp > 0) {
    const Node& n = nodes_[static_cast<size_t>(stack[--sp])];
    if (hit_box(n.box.lo, n.box.hi, o, inv, best.t) == kInf) continue;
    if (n.left < 0) {
      for (int i = n.first; i < n.first + n.count; ++i) {
        const Tri& t = tris_[static_cast<size_t>(i)];
        if (t.element == skip_element) continue;
        double th = 0;
        if (hit_triangle(t, o, d, best.t, th)) best = {i, th};
      }
      continue;
    }
    // Visit the nearer child first.
    const Node& l = nodes_[static_cast<size_t>(n.left)];
    const Node& r = nodes_[static_cast<size_t>(n.left + 1)];
    const double tl = hit_box(l.box.lo, l.box.hi, o, inv, best.t);
    const double tr = hit_box(r.box.lo, r.box.hi, o, inv, best.t);
    if (tl <= tr) {
      if (tr != kInf) stack[sp++] = n.left + 1;
      if (tl != kInf) stack[sp++] = n.left;
    } else {
      if (tl != kInf) stack[sp++] = n.left;
      if (tr != kInf) stack[sp++] = n.left + 1;
    }
  }
  if (best.tri < 0) return {};
  return best;
}

Bvh::Hit Bvh::intersect_brute_force(const Vec3& o, const Vec3& d, double t_max, int skip_element) const {
  Hit best;
  best.t = t_max;
  for (size_t i = 0; i < tris_.size(); ++i) {
    if (tris_[i].element == skip_element) continue;
    double th = 0;
    if (hit_triangle(tris_[i], o, d, best.t, th)) best = {static_cast<int>(i), th};
  }
  if (best.tri < 0) return {};
  return best;
}

}  // namespace th

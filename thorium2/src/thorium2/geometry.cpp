// SPDX-License-Identifier: GPL-3.0-or-later
#include "thorium2/geometry.h"

#include <cmath>

#include <algorithm>
#include <stdexcept>

namespace th {

namespace {
Vec3 vertex_mean(const std::vector<Vec3>& p) {
  Vec3 c;
  for (const Vec3& v : p) c += v;
  return c / static_cast<double>(p.size());
}
}  // namespace

std::vector<Triangle> triangulate(const std::vector<Vec3>& p) {
  if (p.size() == 3) return {{p[0], p[1], p[2]}};
  const Vec3 c = vertex_mean(p);
  std::vector<Triangle> out;
  for (size_t i = 0; i < p.size(); ++i) out.push_back({p[i], p[(i + 1) % p.size()], c});
  return out;
}

Vec3 area_vector(const std::vector<Vec3>& p) {
  Vec3 s;
  for (const Triangle& t : triangulate(p)) s += 0.5 * cross(t.b - t.a, t.c - t.a);
  return s;
}

double area(const std::vector<Vec3>& p) {
  double s = 0;
  for (const Triangle& t : triangulate(p)) s += 0.5 * norm(cross(t.b - t.a, t.c - t.a));
  return s;
}

Vec3 centroid(const std::vector<Vec3>& p) {
  Vec3 s;
  double a = 0;
  for (const Triangle& t : triangulate(p)) {
    const double ta = 0.5 * norm(cross(t.b - t.a, t.c - t.a));
    s += ta * (t.a + t.b + t.c) / 3.0;
    a += ta;
  }
  return a > 0 ? s / a : vertex_mean(p);
}

const std::vector<std::vector<int>>& element_faces(ElementType type) {
  // T.H.O.R.I.U.M. manual, table 4.2 (converted to 0-based node positions).
  static const std::vector<std::vector<int>> hexa = {{3, 2, 1, 0}, {4, 5, 6, 7}, {0, 1, 5, 4},
                                                     {3, 7, 6, 2}, {0, 4, 7, 3}, {1, 2, 6, 5}};
  static const std::vector<std::vector<int>> penta = {{2, 1, 0}, {3, 4, 5}, {1, 2, 5, 4}, {0, 1, 4, 3}, {0, 3, 5, 2}};
  static const std::vector<std::vector<int>> tetra = {{0, 1, 3}, {1, 2, 3}, {2, 0, 3}, {0, 1, 2}};
  static const std::vector<std::vector<int>> quad = {{0, 1, 2, 3}, {3, 2, 1, 0}};
  static const std::vector<std::vector<int>> tria = {{0, 1, 2}, {2, 1, 0}};
  switch (type) {
    case ElementType::CHEXA: return hexa;
    case ElementType::CPENTA: return penta;
    case ElementType::CTETRA: return tetra;
    case ElementType::CQUAD4: return quad;
    case ElementType::CTRIA3: return tria;
  }
  throw std::invalid_argument("unknown element type");
}

SolidProperties solid_properties(const std::vector<std::vector<Vec3>>& faces) {
  SolidProperties r;
  Vec3 ref;
  size_t n = 0;
  for (const auto& f : faces)
    for (const Vec3& v : f) ref += v, ++n;
  ref = ref / static_cast<double>(n);

  // Orientation of each face: outward if its area vector points away from the reference point.
  r.flipped.resize(faces.size());
  double scale = 0;
  for (size_t i = 0; i < faces.size(); ++i) {
    const Vec3 a = area_vector(faces[i]);
    r.flipped[i] = dot(a, centroid(faces[i]) - ref) < 0;
    scale = std::max(scale, norm(a));
  }
  // Divergence theorem with tetrahedra from the reference point.
  Vec3 moment;
  for (size_t i = 0; i < faces.size(); ++i) {
    for (const Triangle& t : triangulate(faces[i])) {
      Vec3 a = t.a - ref, b = t.b - ref, c = t.c - ref;
      if (r.flipped[i]) std::swap(a, b);
      const double v = dot(a, cross(b, c)) / 6.0;
      r.volume += v;
      moment += v * (a + b + c) / 4.0;
    }
  }
  if (!(r.volume > 1e-9 * std::pow(scale, 1.5)))
    throw std::invalid_argument("degenerate or inverted solid element (volume " + std::to_string(r.volume) + ")");
  r.centroid = ref + moment / r.volume;
  return r;
}

double distance_to_line(const Vec3& p, const Vec3& a, const Vec3& b) {
  const Vec3 d = b - a;
  return norm(cross(p - a, d)) / norm(d);
}

double warping(const std::vector<Vec3>& p) {
  const Vec3 n = normalized(area_vector(p));
  const Vec3 c = vertex_mean(p);
  double dmax = 0;
  for (const Vec3& v : p) dmax = std::max(dmax, std::abs(dot(v - c, n)));
  return dmax / std::sqrt(area(p));
}

}  // namespace th

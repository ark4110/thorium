// Thorium 2 — geometry of finite elements.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "thorium2/bdf.h"
#include "thorium2/vec.h"

#include <array>
#include <vector>

namespace th {

struct Triangle {
  Vec3 a, b, c;
};

/// Planar or slightly warped polygon given by its vertices in order.
/// A quadrilateral is split into 4 triangles around the vertex centroid: unlike a split by a
/// diagonal this is unique, so two elements sharing a warped face see the same surface.
std::vector<Triangle> triangulate(const std::vector<Vec3>& polygon);

/// Area vector (direction by the right-hand rule of the vertex order, length = area).
Vec3 area_vector(const std::vector<Vec3>& polygon);
double area(const std::vector<Vec3>& polygon);
/// Area-weighted centroid of the triangulated polygon.
Vec3 centroid(const std::vector<Vec3>& polygon);

/// Local node lists (0-based) of the faces of an element in the order of the T.H.O.R.I.U.M.
/// manual, table 4.2 (face numbers 1..n). For shells face 1 is the front side (normal by the
/// right-hand rule of G1, G2, G3) and face 2 is the back side.
const std::vector<std::vector<int>>& element_faces(ElementType type);

/// Volume and centroid of a closed polyhedron given by its faces. Faces may be listed with
/// any orientation: each one is turned to point away from the vertex centroid. Throws
/// std::invalid_argument if the polyhedron is degenerate or inverted.
struct SolidProperties {
  double volume = 0;
  Vec3 centroid;
  std::vector<bool> flipped;  ///< Per face: listed order was reversed to make it outward.
};
SolidProperties solid_properties(const std::vector<std::vector<Vec3>>& faces);

/// Distance from point p to the straight line through a and b.
double distance_to_line(const Vec3& p, const Vec3& a, const Vec3& b);

/// Largest distance of a polygon vertex from the best plane through the polygon, divided by
/// the square root of its area (0 for a flat polygon).
double warping(const std::vector<Vec3>& polygon);

}  // namespace th

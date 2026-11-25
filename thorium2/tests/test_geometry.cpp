#include "thorium2/geometry.h"

#include <gtest/gtest.h>

using namespace th;

namespace {
std::vector<std::vector<Vec3>> faces_of(ElementType type, const std::vector<Vec3>& v) {
  std::vector<std::vector<Vec3>> out;
  for (const auto& f : element_faces(type)) {
    std::vector<Vec3> poly;
    for (int i : f) poly.push_back(v[static_cast<size_t>(i)]);
    out.push_back(poly);
  }
  return out;
}

// NASTRAN corner order: bottom G1..G4 counter-clockwise, top G5..G8 above them.
std::vector<Vec3> box(double a, double b, double c) {
  return {{0, 0, 0}, {a, 0, 0}, {a, b, 0}, {0, b, 0}, {0, 0, c}, {a, 0, c}, {a, b, c}, {0, b, c}};
}
}  // namespace

TEST(Geometry, PolygonAreaNormalCentroid) {
  std::vector<Vec3> quad = {{0, 0, 0}, {2, 0, 0}, {2, 1, 0}, {0, 1, 0}};
  EXPECT_DOUBLE_EQ(area(quad), 2.0);
  Vec3 n = area_vector(quad);
  EXPECT_DOUBLE_EQ(n.z, 2.0);
  Vec3 c = centroid(quad);
  EXPECT_DOUBLE_EQ(c.x, 1.0);
  EXPECT_DOUBLE_EQ(c.y, 0.5);
  EXPECT_EQ(triangulate(quad).size(), 4u);
  std::vector<Vec3> tri = {{0, 0, 0}, {3, 0, 0}, {0, 3, 0}};
  EXPECT_DOUBLE_EQ(area(tri), 4.5);
  EXPECT_DOUBLE_EQ(centroid(tri).x, 1.0);
  // Trapezoid: centroid is not the vertex average.
  std::vector<Vec3> trap = {{0, 0, 0}, {4, 0, 0}, {3, 1, 0}, {1, 1, 0}};
  EXPECT_NEAR(centroid(trap).y, (1.0 / 3.0) * (4 + 2 * 2) / (4 + 2), 1e-12);
}

TEST(Geometry, FaceTablesMatchManual) {
  // Table 4.2 of the T.H.O.R.I.U.M. manual, 1-based there.
  EXPECT_EQ(element_faces(ElementType::CHEXA)[0], (std::vector<int>{3, 2, 1, 0}));
  EXPECT_EQ(element_faces(ElementType::CHEXA)[5], (std::vector<int>{1, 2, 6, 5}));
  EXPECT_EQ(element_faces(ElementType::CPENTA)[4], (std::vector<int>{0, 3, 5, 2}));
  EXPECT_EQ(element_faces(ElementType::CTETRA)[3], (std::vector<int>{0, 1, 2}));
  EXPECT_EQ(element_faces(ElementType::CQUAD4)[1], (std::vector<int>{3, 2, 1, 0}));
  EXPECT_EQ(element_faces(ElementType::CTRIA3).size(), 2u);
}

TEST(Geometry, BoxVolumeAndCentroid) {
  auto p = solid_properties(faces_of(ElementType::CHEXA, box(1, 2, 3)));
  EXPECT_NEAR(p.volume, 6.0, 1e-12);
  EXPECT_NEAR(p.centroid.x, 0.5, 1e-12);
  EXPECT_NEAR(p.centroid.y, 1.0, 1e-12);
  EXPECT_NEAR(p.centroid.z, 1.5, 1e-12);
}

TEST(Geometry, TetraAndPenta) {
  std::vector<Vec3> t = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  auto pt = solid_properties(faces_of(ElementType::CTETRA, t));
  EXPECT_NEAR(pt.volume, 1.0 / 6.0, 1e-12);
  EXPECT_NEAR(pt.centroid.x, 0.25, 1e-12);
  std::vector<Vec3> w = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 2}, {1, 0, 2}, {0, 1, 2}};
  auto pw = solid_properties(faces_of(ElementType::CPENTA, w));
  EXPECT_NEAR(pw.volume, 1.0, 1e-12);
  EXPECT_NEAR(pw.centroid.z, 1.0, 1e-12);
  EXPECT_NEAR(pw.centroid.x, 1.0 / 3.0, 1e-12);
}

TEST(Geometry, DistortedHexMatchesTrilinearVolume) {
  // Top face lifted at one corner: a non-planar face. Volume of the trilinear hexahedron
  // with heights 1,1,1,2 over the unit square = mean height = 1.25; the 4-triangle fan of the
  // warped face reproduces this exactly for a bilinear surface over a square.
  auto v = box(1, 1, 1);
  v[6].z = 2.0;
  auto p = solid_properties(faces_of(ElementType::CHEXA, v));
  EXPECT_NEAR(p.volume, 1.25, 1e-12);
}

TEST(Geometry, OrientationOfListingDoesNotMatter) {
  // Mirror-ordered element (G1..G4 clockwise): faces are turned outward automatically.
  auto v = box(1, 2, 3);
  std::swap(v[1], v[3]);
  std::swap(v[5], v[7]);
  auto p = solid_properties(faces_of(ElementType::CHEXA, v));
  EXPECT_NEAR(p.volume, 6.0, 1e-12);
  for (bool f : p.flipped) EXPECT_TRUE(f);
}

TEST(Geometry, DegenerateSolidIsRejected) {
  auto v = box(1, 1, 0);
  EXPECT_THROW(solid_properties(faces_of(ElementType::CHEXA, v)), std::invalid_argument);
}

TEST(Geometry, DistanceToLineAndWarping) {
  EXPECT_DOUBLE_EQ(distance_to_line({0, 2, 0}, {-1, 0, 0}, {5, 0, 0}), 2.0);
  std::vector<Vec3> flat = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
  EXPECT_NEAR(warping(flat), 0.0, 1e-15);
  std::vector<Vec3> warped = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0.2}, {0, 1, 0}};
  EXPECT_GT(warping(warped), 0.04);
}

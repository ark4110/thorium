// Copyright 2015 Stepan Tezyunichev (stepan.tezyunichev@gmail.com).
// This file is part of form_factors.
//
// thorium is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// thorium is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with thorium.  If not, see <http://www.gnu.org/licenses/>.

#include "gtest/gtest.h"

#include <math/operations.h>
#include <math/triangle.h>
#include <math/tetrahedron.h>
#include <math/rod.h>
#include <subject/objects.h>

#include <algorithm>
#include <float.h>

using namespace testing;
using namespace math;

const vec3 Epsilon = { FLT_EPSILON, FLT_EPSILON, FLT_EPSILON };
const vec3 A = { 0.f, 0.f, 0.f };
const vec3 B = { 1.f, 0.f, 0.f };
const vec3 C = { 0.f, .5f, 0.f };
const vec3 D = { 1.f, .5f, 0.f };
const vec3 E = (D + B) / 2.f;
const vec3 F = (D + C) / 2.f;

TEST(FindLeastSide, ScanAllEdges)
{
  triangle_t t = make_face(A, B, C);
  float least_side = triangle_least_side_square(t);
  EXPECT_NEAR(0.25f, least_side, 0.01);

  t = make_face(C, A, B);
  least_side = triangle_least_side_square(t);
  EXPECT_NEAR(0.25f, least_side, 0.01);

  t = make_face(B, C, A);
  least_side = triangle_least_side_square(t);
  EXPECT_NEAR(0.25f, least_side, 0.01);
};

TEST(FindAdjacentVertex, ScanAllVertex)
{
  triangle_t t = make_face(A, B, C);
  EXPECT_EQ(0, triangle_find_adjacent_vertex(t, A + Epsilon));
  EXPECT_EQ(1, triangle_find_adjacent_vertex(t, B + Epsilon));
  EXPECT_EQ(2, triangle_find_adjacent_vertex(t, C + Epsilon));
};

TEST(FindAdjacentVertex, ConsiderTriangleScale)
{
  triangle_t t = make_face(A * FLT_EPSILON, B * FLT_EPSILON, C * FLT_EPSILON);
  EXPECT_EQ(-1, triangle_find_adjacent_vertex(t, A + Epsilon));
  EXPECT_EQ(-1, triangle_find_adjacent_vertex(t, B + Epsilon));
  EXPECT_EQ(-1, triangle_find_adjacent_vertex(t, C + Epsilon));
};

TEST(FindAdjacentVertex, MapTriangles)
{
  triangle_t t1 = make_face(A, B, C);
  triangle_t t2 = make_face(C, A, B);
  int actual = triangle_find_adjacent_vertices(t1, t2);

  /*
    l[2] ? r[2], l[2] ? r[1], l[2] ? r[0],
    l[1] ? r[2], l[1] ? r[1], l[1] ? r[0],
    l[0] ? r[2], l[0] ? r[1], l[0] ? r[0]

    0 0 1 // C
    1 0 0 // B
    0 1 0 // A
  */

  int expected = 1 << 6 | 1 << 5 | 1 << 1;
  EXPECT_EQ(expected, actual);
};

TEST(NormalsDirectivity, UnidirectionalForSameTriangle)
{
  int mapping = triangle_find_adjacent_vertices(make_face(A, D, B), make_face(A, D, B));
  EXPECT_TRUE(math::triangle_has_unidirectrional_normals(mapping));
}

TEST(NormalsDirectivity, UnidirectionalForADB_ACD)
{
  int mapping = triangle_find_adjacent_vertices(make_face(A, D, B), make_face(A, C, D));
  EXPECT_TRUE(math::triangle_has_unidirectrional_normals(mapping));
}

TEST(NormalsDirectivity, ContradictionalForADB_ADC)
{
  int mapping = triangle_find_adjacent_vertices(make_face(A, D, B), make_face(A, D, C));
  EXPECT_FALSE(math::triangle_has_unidirectrional_normals(mapping));
}

TEST(HasAdjacentEdge, IgnoreUnrelatedTriangles)
{
  triangle_t t1 = make_face(A, B, C);
  triangle_t t2 = make_face(D, E, F);
  EXPECT_FALSE(triangle_has_adjacent_edge(t1, t2));
}

TEST(HasAdjacentEdge, IgnoreTriangleWithSingleCommonVertex)
{
  triangle_t t1 = make_face(A, B, C);
  triangle_t t2 = make_face(A, E, F);
  EXPECT_FALSE(triangle_has_adjacent_edge(t1, t2));
}

TEST(HasAdjacentEdge, AcceptCommonEdge)
{
  triangle_t t1 = make_face(A, B, C);
  triangle_t t2 = make_face(B, C, D);
  EXPECT_TRUE(triangle_has_adjacent_edge(t1, t2));
}

TEST(HasAdjacentEdge, AcceptMultipleCommonEdges)
{
  triangle_t t1 = make_face(A, B, C);
  triangle_t t2 = make_face(C, A, B);
  EXPECT_TRUE(triangle_has_adjacent_edge(t1, t2));
}

TEST(TetrahedronVolume, SimpleVolume1)
{
  math::tetrahedron_t test_tetra = {{{1, 0, 0},
                                            {1, 0, 1},
                                            {1, 1, 1},
                                            {0, 0, 1}}};
  float volume = tetrahedron_volume(test_tetra.points);
  EXPECT_NEAR(volume, 1.f / 6.f, 0.00001);
}

TEST(TetrahedronVolume, SimpleVolume2)
{
  math::tetrahedron_t test_tetra = {{{0, 0, 0},
                                            {1, 0, 0},
                                            {0, 1, 0},
                                            {0, 0, 1}}};
  float volume = tetrahedron_volume(test_tetra.points);
  EXPECT_NEAR(volume, 1.f / 6.f, 0.00001);
}

TEST(TetrahedronVolume, FromFaces)
{
  math::tetrahedron_t test_tetra = {{{0, 0, 0},
                                            {1, 0, 0},
                                            {0, 1, 0},
                                            {0, 0, 1}}};
  math::face_t faces[4];
  tetrahedron_build_faces(test_tetra, faces);
  float volume = tetrahedron_volume(faces);
  EXPECT_NEAR(volume, 1.f / 6.f, 0.00001);
}

TEST(TetrahedronVolume, FacesPermutation)
{
  math::tetrahedron_t test_tetra = {{{0, 0, 0},
                                            {1, 0, 0},
                                            {0, 1, 0},
                                            {0, 0, 1}}};
  math::face_t faces[4];
  tetrahedron_build_faces(test_tetra, faces);

  for (int fi0 = 0; fi0 != 6; ++fi0)
  {
    std::next_permutation(faces[0].points, faces[0].points + 3);
    for (int fi1 = 0; fi1 != 6; ++fi1)
    {
      std::next_permutation(faces[1].points, faces[1].points + 3);
      for (int fi2 = 0; fi2 != 6; ++fi2)
      {
        std::next_permutation(faces[2].points, faces[2].points + 3);
        for (int fi3 = 0; fi3 != 6; ++fi3)
        {
          std::next_permutation(faces[3].points, faces[3].points + 3);

          float volume = tetrahedron_volume(faces);
          EXPECT_NEAR(volume, 1.f / 6.f, 0.00001);
        }
      }
    }
  }
}

TEST(RodToSixTetrahedronSplit, PreserveVolume)
{
  math::tetrahedron_t tetras[6];
  math::rod_to_six_tetrahedrons(subject::rod_cube(), tetras);

  float volume = 0;
  for (int ti = 0; ti != 6; ++ti)
    volume += math::tetrahedron_volume(tetras[ti]);

  EXPECT_NEAR(volume, 1.f, 0.000001);
}

TEST(RodSixTetrahedronSplit, HasTwoAdjacentFaces)
{
  math::tetrahedron_t tetras[6];
  math::rod_to_six_tetrahedrons(subject::rod_cube(), tetras);

  for (int i = 0; i != 6; ++i)
  {
    int adjacent = 0;
    for (int j = 0; j != 6; ++j)
    {
      if (i == j)
        continue;

      if (tetrahedron_has_adjacent_face(tetras[i], tetras[j]))
        ++adjacent;
    }

      EXPECT_EQ(adjacent, 2);
  }
}

TEST(TetrahedronNormals, Outside)
{
  auto tetra = math::make_simple_tetrahedron();
  int faces[12];
  math::tetrahedron_build_faces_indices(faces);
  int basis[4];
  ASSERT_TRUE(math::is_tetra(faces, basis));

  for (int i = 0; i != 4; ++i)
    ASSERT_EQ(i, basis[i]);

  int* f = faces;
  for (int i = 0; i != 4; ++i, f += 3)
  {
    ASSERT_TRUE(math::tetrahedron_is_outside(f, basis, tetra.points));
    swap_normal(f);
    ASSERT_FALSE(math::tetrahedron_is_outside(f, basis, tetra.points));
  }

  normalize_tetra(faces, basis, tetra.points);

  f = faces;
  for (int i = 0; i != 4; ++i, f += 3)
    ASSERT_TRUE(math::tetrahedron_is_outside(f, basis, tetra.points));
}

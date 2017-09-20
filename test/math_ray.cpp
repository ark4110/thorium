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
// along with form_factors.  If not, see <http://www.gnu.org/licenses/>.

#include "gtest/gtest.h"

#include "math/ray.h"
#include "math/operations.h"

using namespace math;

TEST(RaySegment, Inersect)
{
  ray_t segment = make_ray(make_vec3(1, 1, 0), make_vec3(1, 0, 0));
  ray_t ray = make_ray(make_vec3(.5f, 2, 0), make_vec3(0.5f, -0.5f, 0));
  float s;
  bool hit = ray_intersect_segment(ray, segment, s);
  ASSERT_TRUE(hit);
  ASSERT_NEAR(2, s, 0.0001f);
}

TEST(RaySegment, NotInersect)
{
  ray_t segment = make_ray(make_vec3(1, 1, 0), make_vec3(1, 0, 0));
  ray_t ray = make_ray(make_vec3(0, 2, 0), make_vec3(1, 1.1f, 0));
  float s;
  bool hit = ray_intersect_segment(ray, segment, s);
  ASSERT_FALSE(hit);
}

TEST(RayReflection, DownToUp)
{
  ray_t down = make_ray(make_vec3(0, 1, 0), make_vec3(0, -1, 0));
  triangle_t face_top_normal = make_face(make_vec3(0, 0, 1), make_vec3(1, 0, -1), make_vec3(-1, 0, 1));
  triangle_t face_bot_normal = make_face(make_vec3(0, 0, 1), make_vec3(-1, 0, 1), make_vec3(1, 0, -1));
  ray_t top_reflection = ray_reflect(down, face_top_normal, make_vec3(0, 0, 0));
  ray_t bot_reflection = ray_reflect(down, face_bot_normal, make_vec3(0, 0, 0));

  ray_t expected = make_ray(make_vec3(0, 0, 0), make_vec3(0, 1, 0));

  ASSERT_NEAR(norm(top_reflection.origin - expected.origin), 0, 0.0001f);
  ASSERT_NEAR(norm(bot_reflection.direction - expected.direction), 0, 0.0001f);
}

TEST(RayReflection, DownToRight)
{
  ray_t down = make_ray(make_vec3(0, 1, 0), make_vec3(0, -1, 0));
  triangle_t face_top_normal = make_face(make_vec3(-1, 1, 0), make_vec3(1, -1, -1), make_vec3(1, -1, 1));
  triangle_t face_bot_normal = make_face(make_vec3(0, 0, 1), make_vec3(1, -1, 1), make_vec3(1, -1, -1));
  ray_t top_reflection = ray_reflect(down, face_top_normal, make_vec3(0, 0, 0));
  ray_t bot_reflection = ray_reflect(down, face_bot_normal, make_vec3(0, 0, 0));

  ray_t expected = make_ray(make_vec3(0, 0, 0), make_vec3(1, 0, 0));

  ASSERT_NEAR(norm(top_reflection.origin - expected.origin), 0, 0.0001f);
  ASSERT_NEAR(norm(bot_reflection.direction - expected.direction), 0, 0.0001f);
}
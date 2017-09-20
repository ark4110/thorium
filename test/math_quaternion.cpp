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
// along with form_factors.  If not, see <http://www.gnu.org/licenses/>.

#include "gtest/gtest.h"

#include "math/mat.h"
#include "math/quaternion.h"

using namespace math;

namespace math {
  static bool operator==(const math::vec3& l, const math::vec3& r)
  {
    return near_enough(l, r);
  }
}

TEST(quaternion_rotate_towards, Collinear)
{
  vec3 from = make_vec3(0, 0, 1);
  vec3 to = make_vec3(0, 0, 1);
  quaternion rotation(from, to);
  vec3 result = rotation.mult_vec(from);
  ASSERT_EQ(to, result);
}

TEST(quaternion_rotate_towards, CollinearOpposite)
{
  vec3 from = make_vec3(0, 0, 1);
  vec3 to = make_vec3(0, 0, -1);
  quaternion rotation(from, to);
  vec3 result = rotation * from;
  ASSERT_EQ(to, result);
}

TEST(quaternion_rotate_towards, Orthogonal)
{
  vec3 from = make_vec3(0, 0, 1);
  vec3 to = make_vec3(1, 0, 0);
  quaternion rotation(from, to);
  vec3 result = rotation * from;
  ASSERT_EQ(to, result);
}

TEST(quaternion_rotate_towards, NotNormalizedTarget)
{  
  vec3 to = make_vec3(10.f, -1.f, 3.f);
  vec3 from = make_vec3(1.f, 0, 0) * norm(to);
  quaternion rotation(from, to);
  vec3 result = rotation * from;
  ASSERT_EQ(to, result);
}

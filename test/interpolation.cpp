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
#include "../math/interpolation.h"
#include "../math/operations.h"
#include <algorithm>
#include <random>

using namespace testing;

TEST(LagrangeInterpolation, Tests2)
{
  // y = x
  float x[2] = { 0.f, 1.f };
  float y[2] = { 0.f, 1.f };
  float r[2] = { 0.f, 1.f };
  auto lip = math::make_lip_5(2, x, y);
  for (int i = 0; i != 2; ++i)
    ASSERT_NEAR(r[i], lip.c[i], 0.00001);
}

TEST(LagrangeInterpolation, Tests3)
{
  // y = (x - 1)^2 = x^2 - 2x + 1
  float x[3] = { 0.f, 1.f, 2.f };
  float y[3] = { 1.f, 0.f, 1.f };
  float r[3] = { 1.f, -2.f, 1.f };
  auto lip = math::make_lip_5(3, x, y);
  for (int i = 0; i != 3; ++i)
    ASSERT_NEAR(r[i], lip.c[i], 0.00001);
}

TEST(LagrangeInterpolation, Tests4)
{
  // y = (x - 1)(x - 2)(x - 3) = x^3 - 6 x^2 + 11 x - 6
  float x[4] = { 0.f, 1.f, 2.f, 3.f };
  float y[4] = { -6.f, 0.f, 0.f, 0.f };
  float r[4] = { -6.f, 11.f, -6.f, 1.f };
  auto lip = math::make_lip_5(4, x, y);
  for (int i = 0; i != 4; ++i)
    ASSERT_NEAR(r[i], lip.c[i], 0.00001);
}

TEST(LagrangeInterpolation, Tests5)
{
  // y = (x - 1)(x - 2)(x - 3)(x - 4) = x^4 - 10 x^3 + 35 x^2 - 50 x + 24
  float x[5] = { 0.f, 1.f, 2.f, 3.f, 4.f };
  float y[5] = { 24.f, 0.f, 0.f, 0.f, 0.f };
  float r[5] = { 24.f, -50.f, 35.f, -10.f, 1.f };
  auto lip = math::make_lip_5(5, x, y);
  for (int i = 0; i != 5; ++i)
    ASSERT_NEAR(r[i], lip.c[i], 0.00001);
}

TEST(LagrangeInterpolation, Sin5)
{
  float x[5] = { -0.1f, -0.05f, 0, 0.05f, 0.1f };
  float y[5] = { sinf(x[0]), sinf(x[1]), sinf(x[2]), sinf(x[3]), sinf(x[4]) };
  auto lip = math::make_lip_5(5, x, y);
  for (float v = -0.05f; v < 0.05f; v += 0.001f)
  {
    float r = math::eval_lip_5(lip, v);
    ASSERT_NEAR(sinf(v), r, 0.0001);
  } 
}

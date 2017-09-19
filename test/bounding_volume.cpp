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

#include "../math/operations.h"
#include "../math/triangle.h"
#include "../subject/generator.h"

using namespace testing;

math::triangle_t* make_confetti(int n_triangles, float radius, subject::generator_t* generator)
{
  math::triangle_t* triangles = (math::triangle_t*)malloc(n_triangles * sizeof(math::triangle_t));
  for (int i = 0; i != n_triangles; ++i)
  {
    math::vec3 scale_factor;
    generator_volume_point(generator, 1, &scale_factor);
    float factor = math::norm(scale_factor);

    math::triangle_t& f = triangles[i];
    generator_surface_point(generator, 3, f.points);

    f.points[0] *= 1 + 2 * factor;
    f.points[1] *= 1 + 2 * factor;
    f.points[2] *= 1 + 2 * factor;

    math::vec3 position;
    generator_volume_point(generator, 1, &position);

    position *= radius;
    f.points[0] += position;
    f.points[1] += position;
    f.points[2] += position;
  }

  return triangles;
}

TEST(BoundingSphere, RitterEfficiency)
{
  subject::generator_t* generator = subject::generator_create_spherical();
  int n_triangles = 100;
  float radius = 10;
  math::triangle_t* triangles = make_confetti(n_triangles, 10, generator);
  math::sphere_t naive = math::triangles_bsphere(triangles, n_triangles);
  math::sphere_t ritter = math::triangles_ritter_bsphere(triangles, n_triangles);
  free(triangles);
  subject::generator_free(generator);
}

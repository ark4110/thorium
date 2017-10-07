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

#include "../math/operations.h"
#include "../math/triangle.h"
#include "../subject/generator.h"

using namespace testing;

namespace {
  math::triangle_t *make_confetti(int n_triangles, float radius, subject::generator_t *generator)
  {
    math::triangle_t *triangles = (math::triangle_t *) malloc(n_triangles * sizeof(math::triangle_t));
    for (int i = 0; i != n_triangles; ++i)
    {
      math::vec3 scale_factor;
      generator_volume_point(generator, 1, &scale_factor);
      float factor = math::norm(scale_factor);

      math::triangle_t &f = triangles[i];
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

// We know that center is at (0,0,0)
  float calculate_near_real_radius(const math::triangle_t *triangles, int n_triangles)
  {
    float m = 0;
    for (int i = 0; i != n_triangles; ++i)
    {
      const math::triangle_t &t = triangles[i];
      m = std::max(m, math::dot(t.points[0], t.points[0]));
      m = std::max(m, math::dot(t.points[1], t.points[1]));
      m = std::max(m, math::dot(t.points[2], t.points[2]));
    }

    return sqrtf(m);
  }

  float sqrf(float v)
  {
    return v * v;
  }
}

TEST(BoundingSphere, AlgorithmEfficiency)
{
  subject::generator_t* generator = subject::generator_create_spherical();
  int n_triangles = 100;
  math::triangle_t* triangles = make_confetti(n_triangles, 10, generator);
  float near_real = calculate_near_real_radius(triangles, n_triangles);
  math::sphere_t linear = math::triangles_bsphere(triangles, n_triangles);
  math::sphere_t ritter = math::triangles_ritter_bsphere(triangles, n_triangles);
  math::sphere_t gartner = math::triangles_gartner_bsphere(triangles, n_triangles);

  fprintf(stderr, "Radius near_real %f, linear %f, ritter %f, gartner %f\n", near_real, linear.radius, ritter.radius, gartner.radius);
  fprintf(stderr, "Radius ratio linear %f, ritter %f, gartner %f\n", linear.radius / near_real, ritter.radius / near_real, gartner.radius / near_real);
  fprintf(stderr, "Radius square ratio linear %f, ritter %f, gartner %f\n", sqrf(linear.radius / near_real), sqrf(ritter.radius / near_real), sqrf(gartner.radius / near_real));
  free(triangles);
  subject::generator_free(generator);
}

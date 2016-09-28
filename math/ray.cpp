// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
// This file is part of form_factors.
//
// form_factors is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// form_factors is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with form_factors.  If not, see <http://www.gnu.org/licenses/>.

#include "ray.h"
#include "operations.h"
#include "triangle.h"
#include "mat.h"
#include <random>

namespace math
{
  /// See for details http ://geomalgorithms.com/a05-_intersect-1.html
  bool ray_intersect_segment(ray_t p, ray_t q, float& s)
  {
    vec3 v = q.direction - q.origin;
    vec3 w = p.origin - q.origin;
    vec3 u = p.direction - p.origin;

    float d = (v.x * u.y - v.y * u.x);

    s = (v.y * w.x - v.x * w.y) / d;
    float t = (u.x * w.y - u.y * w.x) / -d;

    return t >= 0 && t <= 1 && s >= 0;
  }

  bool ray_check_hit_front(const ray_t& ray, const triangle_t& plane)
  {
    vec3 normal = triangle_normal(plane);
    vec3 v = ray.direction - ray.origin;
    float side = dot(normal, v);

    return side < 0;
  }

  ray_t ray_reflect(const ray_t& ray, const triangle_t& triangle, const vec3& hit_point)
  {
    math::vec3 normal = normalize(triangle_normal(triangle));
    math::vec3 v = normalize(ray.direction - ray.origin);
    float angle = dot(normal, v);
    math::vec3 reflection = v - 2 * angle * normalize(normal);
    return{ hit_point + reflection * 0.0001f, hit_point + reflection };
  }

  ray_t ray_transmit(const ray_t& ray, const vec3& point)
  {
    vec3 v = ray.direction - ray.origin;
    return { point + v * 0.0001f, point + v };
  }

  struct ray_malley_emission_impl : ray_malley_emission
  {
    ray_malley_emission_impl()
      : HSGenTheta(5)
      , HSGenR(6)
      , Distr_0_1(0, 1)
      , Distr_0_2PI(0, float(M_2PI))
    {
    }

    vec3 emmit()
    {
      float r = Distr_0_1(HSGenR);
      float rad = sqrtf(r);
      float phi = Distr_0_2PI(HSGenTheta);
      return { rad * cosf(phi), rad * sinf(phi), sqrtf(1 - r) };
    }

    std::mt19937 HSGenTheta;
    std::mt19937 HSGenR;   
    std::uniform_real_distribution<float> Distr_0_1;
    std::uniform_real_distribution<float> Distr_0_2PI;
  };

  ray_malley_emission* ray_malley_emission_create()
  {
    return new ray_malley_emission_impl();
  }

  void ray_malley_emission_free(ray_malley_emission* system)
  {
    delete static_cast<ray_malley_emission_impl*>(system);
  }

  vec3 ray_malley_emmit(ray_malley_emission* system)
  {
    return static_cast<ray_malley_emission_impl*>(system)->emmit();
  }

  ray_t ray_malley_emmit(ray_malley_emission* system, const triangle_t& face, const vec3& origin)
  {
    // Pick direction from cosine-weighted distribution
    vec3 malley = ray_malley_emmit(system);

    // Store rotation for a given face (from Z axis towards face's normal).
    math::mat33 rotation = pick_face_rotation(face, math::make_vec3(0, 0, 1));

    // Rotate ray towards face's normal
    math::vec3 relative_dist = rotation * malley;

    return { origin + relative_dist * 0.0001f, origin + relative_dist };
  }
}

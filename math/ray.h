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

#pragma once

#include "types.h"

namespace math
{
  /**
  @param p ray
  @param q segment
  @param s ray shift
  @retrun intersect or not
  */
  bool ray_intersect_segment(ray_t p, ray_t q, float& s);

  /// @brief Return true if ray hit front side of triangle plane.
  bool ray_check_hit_front(const ray_t& ray, const triangle_t& plane);

  /**
    @brief Reflect ray from triangle at hit point.
    @note There is not check that point belongs to ray or triangle.
    @note Start ray a little bit out of new origin point.
  */
  ray_t ray_reflect(const ray_t& ray, const triangle_t& triangle, const vec3& hit_point);
  
  /**
    @brief Move along ray to new origin
    @note There is not check that point belongs to ray.
    @note Start ray a little bit out of new origin point.
  */
  ray_t ray_transmit(const ray_t& ray, const vec3& point);

  /**
    @brief Malley ray emission algorithm.
    @detail Malley emission algorithm is an nice and efficient hack to emit ray from black-body.
    @detail It consider normal distribution of rays AND cosine-weighted distribution.
  */
  struct ray_malley_emission
  {};

  ray_malley_emission* ray_malley_emission_create();
  void ray_malley_emission_free(ray_malley_emission*);

  // @brief Pick direction from cosine-weighted distribution.
  vec3 ray_malley_emmit(ray_malley_emission* system);

  /// @brief Emmit ray from triangle point.
  /// @note Don't use for massive rays emission because face normal rotation matrix must be cached!
  ray_t ray_malley_emmit(ray_malley_emission* system, const triangle_t& face, const vec3& origin);
}
// Copyright (c) 2021 Contributors as noted in the AUTHORS file.
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

#pragma once

#include "triangle.h"

namespace math
{
  struct tetrahedron_t
  {
    vec3 points[4];
  };

  int tetrahedron_build_from_faces(const triangle_t *faces, vec3 *vertices);
  void tetrahedron_build_faces(const tetrahedron_t &object, triangle_t *faces);
  float tetrahedron_volume(const vec3 *vertices);
  float tetrahedron_volume(const tetrahedron_t& object);
  float tetrahedron_volume(const triangle_t *faces);

  /**
   @brief Check if two tetrahedrons has adjacent face
   @note Suboptimal, for tests only.
   */
  bool tetrahedron_has_adjacent_face(const tetrahedron_t& l, const tetrahedron_t& r);
}

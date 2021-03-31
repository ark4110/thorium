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

#include "types.h"
#include "tetrahedron.h"

namespace math
{
  struct rod_t
  {
    vec3 points[8];
  };

  /**
   @note Don't preserve correct normals.
  */
  void rod_to_six_tetrahedrons(const rod_t& rod, tetrahedron_t* result);
}

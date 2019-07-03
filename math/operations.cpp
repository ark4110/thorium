// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
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

#include "operations.h"
#include <float.h>

namespace math
{
  static const float precision = 0.00001f;

  bool near_enough(float a, float b)
  {
    float diff = fabs(a - b);
    if (diff < precision)
      return true;
    float scale = ::max(fabs(a), fabs(b));
    return diff < scale * precision;
  }

  bool near_enough(const vec3& a, const vec3& b)
  {
    return near_enough(a.x, b.x) &&
           near_enough(a.y, b.y) &&
           near_enough(a.z, b.z);
  }

  bool operator < (const vec3& l, const vec3& r)
  {
    return !near_enough(l.x, r.x) ? l.x < r.x :
      (
        !near_enough(l.y, r.y) ? l.y < r.y :
        (
          !near_enough(l.z, r.z) ? l.z < r.z : false
        )
      );
  }
}

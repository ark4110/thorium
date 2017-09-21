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
// along with form_factors.  If not, see <http://www.gnu.org/licenses/>.

#include "interpolation.h"

namespace math
{
  lip_5_t make_lip_5(int n, const float* x, const float* y)
  {
    lip_5_t r = {};
    for (int i = 0; i != n; ++i)
    {
      r.x[i] = x[i];
      r.yl[i] = y[i];
      for (int j = 0; j != n; ++j)
      {
        if (j != i)
          r.yl[i] /= x[i] - x[j];
      }
    }    
  }

  float lip_5_apply(const lip_5_t& lip, float x)
  {
    float r = 0;
    for (int i = 0; i != 5; ++i)
      r += (x - lip.x[i]) * lip.yl[i];
    return r;
  }
}

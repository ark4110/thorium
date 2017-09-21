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

#pragma once

namespace math
{

  /**
    @brief Lagrange interpolation polynomial for five or less points
  */
  struct lip_5_t
  {
    float x[5];
    float yl[5];
  };

  lip_5_t make_lip_5(int n, const float* x, const float* y);
  float lip_5_apply(const lip_5_t& lip, float x);
}

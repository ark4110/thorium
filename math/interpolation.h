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
    @brief Lagrange interpolation polynomial for 8 or less points
  */
  struct lip_8_t
  {
    float c[8];
  };
  
  lip_8_t make_lip_5(int n, const float* x, const float* y);
  float eval_lip_5(const lip_8_t& lip, float x);
  float eval_lip_6(const lip_8_t& lip, float x);
  float eval_lip_n(int n, const lip_8_t& lip, float x);

  lip_8_t make_lip_8(int n, const float* x, const float* y);
  
  
  float eval_polynomial(int n, const float* c, float x);
}

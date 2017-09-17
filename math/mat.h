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

/**
 * This module contains math operations between matrices and vectors.
 */

#pragma once

#include "types.h"
#include "operations.h"
#include <float.h>

namespace math
{
  /// @brief 3x3 matrix to vector multiplication.
  inline vec3 operator * (mat33 mat, vec3 vec)
  {
    return make_vec3(
      (mat._11 * vec.x + mat._12 * vec.y + mat._13 * vec.z),
      (mat._21 * vec.x + mat._22 * vec.y + mat._23 * vec.z),
      (mat._31 * vec.x + mat._32 * vec.y + mat._33 * vec.z)
      );
  }

  /// @brief 3x3 matrix to matrix multiplication.
  inline mat33 operator * (mat33 a, mat33 b)
  {
    mat33 r = {};

    for (int i = 0; i != 3; i++)
      for (int j = 0; j != 3; j++)
        for (int c = 0; c != 3; c++)
        {
          r.element(i, j) += a(i, c) * b(c, j);
        }

    return r;
  }

  /**
   * @brief SSC (skew-symmetric) 3x3 matrix to matrix multiplication.
   *
   * Contains less operations than common matrices multiplication.
   * Can be used in vector-towards vector rotation.
   * @see rotate_towards(vec3 subject, vec3 to)
   * @param[in] a 1st 3x3 SSC matrix
   * @param[in] b 2nd 3x3 SSC matrix
   */
  inline mat33 ssc_mul(mat33 a, mat33 b)
  {
    return a * b;
  }

  /// @brief 3x3 matrices sum.
  inline mat33 operator + (mat33 a, mat33 b)
  {
    mat33 r;
    for (int i = 0; i != 9; ++i)
      r.p[i] = a.p[i] + b.p[i];
    return r;
  }

  /// @brief Matrix to scalar multiplication.
  inline mat33 operator * (mat33 a, point_t b)
  {
    mat33 r;
    for (int i = 0; i != 9; ++i)
      r.p[i] = a.p[i] * b;
    return r;
  }

  /// @brief Identity diagonal 3x3 matrix.
  static const mat33 IDENTITY_33 = make_mat33(1, 0, 0, 0, 1, 0, 0, 0, 1);

  /// @brief Rotation matrix whiñh reverse vector direction.
  static const mat33 REVERSE_33 = make_mat33(1, 0, 0, 0, -1, 0, 0, 0, -1);

  /**
   * @brief Produces 3x3 matrix of rotation source 3d vector to target 3d vector.
   *
   * @param[in] subject source 3D vector. Must be normalized.
   * @param[in] to target 3D vector. May be not normalized(?).
   * @return 3x3 rotation matrix
   * @warning Parameters should be normalized
   */
  mat33 rotate_towards(vec3 subject, vec3 to);

  /**
   * @brief Produces 3x3 matrix of rotation around axes.
   * @param[in] x X-axis rotation angle
   * @param[in] y Y-axis rotation angle
   * @param[in] z Z-axis rotation angle
   * @return 3x3 rotation matrix
   */
  mat33 axis_rotation(float x, float y, float z);

  bool load_from_string(const char* str, mat44& m);
  vec3 translation(const mat44& m);
  mat33 rotation(const mat44& m);
}
// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
// This file is part of thorium.

/*
* Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

//
// Template math library for common 3D functionality
//
// nvQuaterion.h - quaternion template and utility functions
//
// This code is in part deriver from glh, a cross platform glut helper library.
// The copyright for glh follows this notice.
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

/*
Copyright (c) 2000 Cass Everitt
Copyright (c) 2000 NVIDIA Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or
without modification, are permitted provided that the following
conditions are met:

* Redistributions of source code must retain the above
copyright notice, this list of conditions and the following
disclaimer.

* Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials
provided with the distribution.

* The names of contributors to this software may not be used
to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


Cass Everitt - cass@r3.nu
*/

#include "quaternion.h"
#include "operations.h"
#include <math.h>

namespace math
{
  quaternion normalize(const quaternion &q)
  {
    quaternion r(q);
    float rnorm = 1.0f / sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);

    r.x *= rnorm;
    r.y *= rnorm;
    r.z *= rnorm;
    r.w *= rnorm;
    return r;
  }

  void quaternion::get_value(vec3& axis, float &radians) const
  {
    radians = float(acosf(_array[3]) * 2.0f);

    if (radians == 0.0f)
    {
      axis = make_vec3(0.0, 0.0, 1.0);
    }
    else
    {
      axis.x = _array[0];
      axis.y = _array[1];
      axis.z = _array[2];
      axis = normalize(axis);
    }
  }

  void quaternion::get_value(mat33 &m) const
  {
    float s, xs, ys, zs, wx, wy, wz, xx, xy, xz, yy, yz, zz;

    float norm = _array[0] * _array[0] + _array[1] * _array[1] + _array[2] * _array[2] + _array[3] * _array[3];

    s = (norm == float(0.0)) ? float(0.0) : (float(2.0) / norm);

    xs = _array[0] * s;
    ys = _array[1] * s;
    zs = _array[2] * s;

    wx = _array[3] * xs;
    wy = _array[3] * ys;
    wz = _array[3] * zs;

    xx = _array[0] * xs;
    xy = _array[0] * ys;
    xz = _array[0] * zs;

    yy = _array[1] * ys;
    yz = _array[1] * zs;
    zz = _array[2] * zs;

    m.element(0, 0) = 1.0f - (yy + zz);
    m.element(1, 0) = xy + wz;
    m.element(2, 0) = xz - wy;

    m.element(0, 1) = xy - wz;
    m.element(1, 1) = 1.0f - (xx + zz);
    m.element(2, 1) = yz + wx;

    m.element(0, 2) = xz + wy;
    m.element(1, 2) = yz - wx;
    m.element(2, 2) = 1.0f - (xx + yy);
  }

  quaternion& quaternion::set_value(const mat33 &m)
  {
    float tr, s;
    int i, j, k;
    const int nxt[3] = { 1, 2, 0 };

    tr = m(0, 0) + m(1, 1) + m(2, 2);

    if (tr > 0.f)
    {
      s = sqrtf(tr + m(3, 3));
      _array[3] = s * 0.5f;
      s = 0.5f / s;

      _array[0] = (m(1, 2) - m(2, 1)) * s;
      _array[1] = (m(2, 0) - m(0, 2)) * s;
      _array[2] = (m(0, 1) - m(1, 0)) * s;
    }
    else
    {
      i = 0;

      if (m(1, 1) > m(0, 0))
      {
        i = 1;
      }

      if (m(2, 2) > m(i, i))
      {
        i = 2;
      }

      j = nxt[i];
      k = nxt[j];

      s = sqrtf((m(i, j) - (m(j, j) + m(k, k))) + float(1.0));

      _array[i] = s * 0.5f;
      s = 0.5f / s;

      _array[3] = (m(j, k) - m(k, j)) * s;
      _array[j] = (m(i, j) + m(j, i)) * s;
      _array[k] = (m(i, k) + m(k, i)) * s;
    }

    return *this;
  }

  quaternion& quaternion::set_value(const vec3& axis, float theta)
  {
    float sqnorm = dot(axis, axis);

    if (sqnorm == 0.0f)
    {
      // axis too small.
      x = y = z = 0.0f;
      w = 1.0f;
    }
    else
    {
      theta *= 0.5f;
      float sin_theta = sinf(theta);

      if (sqnorm != 1.f)
      {
        sin_theta /= sqrtf(sqnorm);
      }

      x = sin_theta * axis.x;
      y = sin_theta * axis.y;
      z = sin_theta * axis.z;
      w = cosf(theta);
    }

    return *this;
  }

  quaternion& quaternion::set_value(const vec3& rotate_from, const vec3& rotate_to)
  {
    vec3 p1, p2;
    float alpha;

    p1 = normalize(rotate_from);
    p2 = normalize(rotate_to);

    alpha = dot(p1, p2);

    if (alpha == 1.0f)
    {
      *this = quaternion(rotate_from, 0);
      return *this;
    }

    // ensures that the anti-parallel case leads to a positive dot
    if (alpha == -1.0f)
    {
      vec3 v;

      if (p1.x != p1.y || p1.x != p1.z)
      {
        v = make_vec3(p1.y, p1.z, p1.x);
      }
      else
      {
        v = make_vec3(-p1.x, p1.y, p1.z);
      }

      v -= p1 * dot(p1, v);
      v = normalize(v);

      set_value(v, 3.1415926f);
      return *this;
    }

    p1 = normalize(cross(p1, p2));

    set_value(p1, acosf(alpha));

    return *this;
  }

  quaternion slerp(const quaternion &p, const quaternion &q, float alpha)
  {
    quaternion r;

    float cos_omega = p.x * q.x + p.y * q.y + p.z * q.z + p.w * q.w;
    // if B is on opposite hemisphere from A, use -B instead

    int bflip;

    if ((bflip = (cos_omega < float(0))))
    {
      cos_omega = -cos_omega;
    }

    // complementary interpolation parameter
    float beta = float(1) - alpha;

    if (cos_omega >= float(1))
    {
      return p;
    }

    float omega = float(acos(cos_omega));
    float one_over_sin_omega = float(1.0) / float(sin(omega));

    beta = float(sin(omega*beta)  * one_over_sin_omega);
    alpha = float(sin(omega*alpha) * one_over_sin_omega);

    if (bflip)
    {
      alpha = -alpha;
    }

    r.x = beta * p._array[0] + alpha * q._array[0];
    r.y = beta * p._array[1] + alpha * q._array[1];
    r.z = beta * p._array[2] + alpha * q._array[2];
    r.w = beta * p._array[3] + alpha * q._array[3];
    return r;
  }
}

#pragma once

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

#include "types.h"

namespace math
{
  class quaternion
  {
  public:
    quaternion() : x(0.0), y(0.0), z(0.0), w(0.0)
    {
    }

    quaternion(const float v[4])
    {
      set_value(v);
    }

    quaternion(float q0, float q1, float q2, float q3)
    {
      set_value(q0, q1, q2, q3);
    }

    quaternion(const vec3& axis, float radians)
    {
      set_value(axis, radians);
    }

    quaternion(const vec3& rotate_from, const vec3& rotate_to)
    {
      set_value(rotate_from, rotate_to);
    }

    const float *get_value() const
    {
      return  &_array[0];
    }

    void get_value(float &q0, float &q1, float &q2, float &q3) const
    {
      q0 = _array[0];
      q1 = _array[1];
      q2 = _array[2];
      q3 = _array[3];
    }

    quaternion &set_value(float q0, float q1, float q2, float q3)
    {
      _array[0] = q0;
      _array[1] = q1;
      _array[2] = q2;
      _array[3] = q3;
      return *this;
    }

    void get_value(vec3& axis, float &radians) const;
    void get_value(mat33 &m) const;

    quaternion &set_value(const float *qp)
    {
      for (int i = 0; i < 4; i++)
      {
        _array[i] = qp[i];
      }

      return *this;
    }

    quaternion &set_value(const mat33 &m);

    quaternion &set_value(const vec3& axis, float theta);

    quaternion &set_value(const vec3& rotateFrom, const vec3& rotateTo);    

    quaternion &operator *= (const quaternion &qr)
    {
      quaternion ql(*this);

      w = ql.w * qr.w - ql.x * qr.x - ql.y * qr.y - ql.z * qr.z;
      x = ql.w * qr.x + ql.x * qr.w + ql.y * qr.z - ql.z * qr.y;
      y = ql.w * qr.y + ql.y * qr.w + ql.z * qr.x - ql.x * qr.z;
      z = ql.w * qr.z + ql.z * qr.w + ql.x * qr.y - ql.y * qr.x;

      return *this;
    }

    friend quaternion normalize(const quaternion &q);

    friend quaternion conjugate(const quaternion &q)
    {
      quaternion r(q);
      r._array[0] *= -1.0f;
      r._array[1] *= -1.0f;
      r._array[2] *= -1.0f;
      return r;
    }

    friend quaternion inverse(const quaternion &q)
    {
      return conjugate(q);
    }

    ///  @brief Quaternion multiplication with cartesian vector v' = q*v*q(star)
    vec3 mult_vec(const vec3& src) const
    {
      float v_coef = w * w - x * x - y * y - z * z;
      float u_coef = 2.0f * (src.x * x + src.y * y + src.z * z);
      float c_coef = 2.0f * w;

      return make_vec3(v_coef * src.x + u_coef * x + c_coef * (y * src.z - z * src.y),
        v_coef * src.y + u_coef * y + c_coef * (z * src.x - x * src.z),
        v_coef * src.z + u_coef * z + c_coef * (x * src.y - y * src.x));
    }

    vec3 operator*(const vec3& src) const
    {
      return mult_vec(src);
    }

    void scale_angle(float scaleFactor)
    {
      vec3 axis;
      float radians;

      get_value(axis, radians);
      radians *= scaleFactor;
      set_value(axis, radians);
    }

    friend quaternion slerp(const quaternion &p, const quaternion &q, float alpha);

    float &operator [](int i)
    {
      return _array[i];
    }

    const float &operator [](int i) const
    {
      return _array[i];
    }

    friend bool operator == (const quaternion &lhs, const quaternion &rhs)
    {
      bool r = true;

      for (int i = 0; i < 4; i++)
      {
        r &= lhs._array[i] == rhs._array[i];
      }

      return r;
    }

    friend bool operator != (const quaternion &lhs, const quaternion &rhs)
    {
      bool r = true;

      for (int i = 0; i < 4; i++)
      {
        r &= lhs._array[i] == rhs._array[i];
      }

      return r;
    }

    friend quaternion operator * (const quaternion &lhs, const quaternion &rhs)
    {
      quaternion r(lhs);
      r *= rhs;
      return r;
    }


    union
    {
      struct
      {
        float x;
        float y;
        float z;
        float w;
      };
      float _array[4];
    };
  };
}

// Copyright (c) 2015-2016 Contributors as noted in the AUTHORS file.
// This file is part of Thorium.
//
// Thorium is free software: you can redistribute it and/or modify
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

#include <helper_math.h>
#include <float.h>

namespace cuda_math
{
  struct face_t
  {
    float3 points[3];
  };

  struct prep_face_t
  {
    float3 normalized_normal;
  };

  struct ray_t
  {
    float3 origin;
    float3 direction;
  };

  union mat33
  {
    struct
    {
      float _11, _12, _13;
      float _21, _22, _23;
      float _31, _32, _33;
    };

    __device__ float& element(int i, int j)
    {
      return p[i * 3 + j];
    }

    __device__ float operator()(int i, int j) const
    {
      return p[i * 3 + j];
    }

    float p[9];
  };

  inline __device__ void convert_face_to_uv(face_t& face)
  {
    float3 origin = face.points[0];
    face.points[1] -= origin;
    face.points[2] -= origin;
  }

  inline __device__ float face_area(const face_t& face)
  {
    float3 n = cross(face.points[1], face.points[2]);
    return sqrtf(dot(n, n)) / 2;
  }

  inline __device__ float3 face_normal(const face_t& face)
  {
    return cross(face.points[1], face.points[2]);
  }

  inline __device__ mat33 make_mat33(float a00, float a01, float a02, float a10, float a11, float a12, float a20, float a21, float a22)
  {
    mat33 result = { a00, a01, a02, a10, a11, a12, a20, a21, a22 };
    return result;
  }

  inline __device__ mat33 make_identity()
  {
    return make_mat33(1, 0, 0, 0, 1, 0, 0, 0, 1);
  }

  /// @brief Rotation matrix whiñh reverse vector direction.
  inline __device__ mat33 make_reverse()
  {
    return make_mat33(1, 0, 0, 0, -1, 0, 0, 0, -1);
  }

  /// @brief 3x3 matrix to matrix multiplication.
  inline __device__ mat33 operator * (mat33 a, mat33 b)
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

  inline __device__ float3 operator * (mat33 mat, float3 vec)
  {
    return make_float3(
      (mat._11 * vec.x + mat._12 * vec.y + mat._13 * vec.z),
      (mat._21 * vec.x + mat._22 * vec.y + mat._23 * vec.z),
      (mat._31 * vec.x + mat._32 * vec.y + mat._33 * vec.z)
    );
  }

  /// @brief 3x3 matrices sum.
  inline __device__ mat33 operator + (mat33 a, mat33 b)
  {
    mat33 r;
    for (int i = 0; i != 9; ++i)
      r.p[i] = a.p[i] + b.p[i];
    return r;
  }

  /// @brief Matrix to scalar multiplication.
  inline __device__ mat33 operator * (mat33 a, float b)
  {
    mat33 r;
    for (int i = 0; i != 9; ++i)
      r.p[i] = a.p[i] * b;
    return r;
  }

  inline __device__ mat33 ssc_mul(mat33 a, mat33 b)
  {
    return a * b;
  }

  inline __device__ mat33 rotate_towards(float3 subject, float3 to)
  {
    float3 v = cross(subject, to);
    float s2 = dot(v, v);
    float c = dot(subject, to); // TODO: Normalize?

    mat33 rot = make_identity();

    if (s2 > FLT_MIN)
    {
      mat33 ssc = make_mat33(0, -v.z, v.y, v.z, 0, -v.x, -v.y, v.x, 0);
      mat33 ssc2 = ssc_mul(ssc, ssc);
      ssc2 = ssc2 * ((1 - c) / s2);
      rot = rot + ssc + ssc2;
    }
    else
    {
      float3 f = subject + to;
      float l = dot(f, f);
      if (l < FLT_EPSILON)
        rot = make_reverse();
    }

    return rot;
  }

  inline __device__ mat33 pick_face_rotation(const face_t& face)
  {
    float3 norm = face_normal(face);
    return rotate_towards(make_float3(0, 0, 1), norm);
  }

  inline __device__ mat33 pick_face_rotation(const prep_face_t& face)
  {
    return rotate_towards(make_float3(0, 0, 1), face.normalized_normal);
  }

  inline __device__ float3 invert_z(float3 v)
  {
    v.z = -v.z;
    return v;
  }

  inline __device__ float2 ceilf(float2 v)
  {
    return make_float2(::ceilf(v.x), ::ceilf(v.y));
  }

  inline __device__ float3 triangle_normal(const face_t& t)
  {
    float3 v0 = t.points[1] - t.points[0];
    float3 v1 = t.points[2] - t.points[0];
    return cross(v0, v1);
  }

  inline __device__ bool ray_check_hit_front(const ray_t& ray, const face_t& plane)
  {
    float3 normal = triangle_normal(plane);
    float3 v = ray.direction;
    float side = dot(normal, v);

    return side < 0;
  }

  inline __device__ bool ray_check_hit_front(const ray_t& ray, const prep_face_t& plane)
  { 
    return dot(plane.normalized_normal, ray.direction) < 0;
  }

  inline __device__ ray_t ray_reflect(const ray_t& ray, const float3& normalized_normal, const float3& hit_point)
  {
    float3 v = normalize(ray.direction);
    float angle = dot(normalized_normal, v);
    float3 reflection = v - 2 * angle * normalized_normal;
    return{ hit_point + reflection * 0.0001f, reflection };
  }

  inline __device__ ray_t ray_reflect(const ray_t& ray, const face_t& triangle, const float3& hit_point)
  {
    float3 normal = normalize(triangle_normal(triangle));
    return ray_reflect(ray, normal, hit_point);
  }

  inline __device__ ray_t ray_reflect(const ray_t& ray, const prep_face_t& triangle, const float3& hit_point)
  {
    return ray_reflect(ray, triangle.normalized_normal, hit_point);
  }

  inline __device__ ray_t ray_transmit(const ray_t& ray, const float3& point)
  {
    float3 v = ray.direction;
    return{ point + v * 0.0001f, v };
  }
}

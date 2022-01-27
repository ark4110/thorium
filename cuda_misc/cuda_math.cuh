// Copyright (c) 2015-2016 Contributors as noted in the AUTHORS file.
// This file is part of Thorium.
//
// Thorium is free software: you can redistribute it and/or modify
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

//#include <helper_math.h>
#include <cmath>
#include <float.h>

namespace cuda_math
{
  typedef float3 vec3;

  __device__ inline float3 operator*(float3 v, float r)
  {
      return { v.x * r, v.y * r, v.z * r };
  }

  __device__ inline float3 operator*(float r, float3 v)
  {
      return { v.x * r, v.y * r, v.z * r };
  }

  __device__ inline float dot(float3 a, float3 b)
  {
      return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  __device__ inline float3 cross(float3 a, float3 b)
  {
      return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
  }

  __device__ inline float3 operator+(float3 a, float3 b)
  {
      return { a.x + b.x, a.y + b.y, a.z + b.z };
  }

  __device__ inline float3 operator+=(float3& a, float3 b)
  {
      a.x += b.x;
      a.y += b.y;
      a.z += b.z;  
  }

  __device__ inline float3 operator-(float3 a, float3 b)
  {
      return { a.x - b.x, a.y - b.y, a.z - b.z };
  }

  __device__ inline float3 operator-=(float3& a, float3 b)
  {
      a.x -= b.x;
      a.y -= b.y;
      a.z -= b.z;
  }

  __device__ inline float3 normalize(float3 v)
  {
      float rnorm = rsqrtf(dot(v, v));
      return v * rnorm;
  }

  struct face_t
  {
    float3 points[3];
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

  struct quat_t
  {
    float3 a;
    float w;
    float v;
  };

    // from (0, 0, 1) to target 'to' vector
  inline __device__ quat_t make_z_rotation_quat(float3 to)
  {
    float cos_alpha= to.z;

    //if (cos_alpha < 0.000001f && to.x * to.x < 0.000001f && to.y * to.y < 0.000001f)
    //  return {{}, 0, 0};

    if (1.f - cos_alpha < 0.000001f)
        return {{}, 0, 1.f};

    if (cos_alpha + 1.0f < 0.000001f)
        return {{1.f, 0, 0}, 0, -1.f};

    float theta = acosf(cos_alpha) / 2;
    float sin_theta, cos_theta;
    __sincosf(theta, &sin_theta, &cos_theta);
    float3 axis = normalize(make_float3(-to.y, to.x, 0)) * sin_theta;

    return {axis, cos_theta, cos_theta * cos_theta - sin_theta * sin_theta};
  };

// q * v * q(star)
inline __device__ float3 quat_mult_vec(const quat_t& q, float3 s)
{
  float v = q.v;
  float u = 2.f * dot(q.a, s);
  float c = 2.f * q.w;
  return v * s + u * q.a + c * cross(q.a, s);
}


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

  /// @brief Rotation matrix whi�h reverse vector direction.
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


  inline __device__ mat33 pick_face_rotation(const float3& face_normalized_normal)
  {
    return rotate_towards(make_float3(0, 0, 1), face_normalized_normal);
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
  
  inline __device__ bool ray_check_hit_front(const ray_t& ray, float3 normalized_normal)
  {
    return dot(normalized_normal, ray.direction) < 0;
  }

  inline __device__ ray_t ray_reflect(const ray_t& ray, const float3& normalized_normal, const float3& hit_point)
  {
    // @todo Provide rays with normalized direction?
    // @note May be it is better to normalize here because there are less reflected than emitted rays. Yet it may have zero-cost.
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

  inline __device__ ray_t ray_transmit(const ray_t& ray, const float3& point)
  {
    float3 v = ray.direction;
    return{ point + v * 0.0001f, v };
  }

  inline __device__ float3 ray_point(const ray_t& ray, float distance)
  {
    return ray.origin + ray.direction * distance;
  }
}

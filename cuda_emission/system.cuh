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

#include "../subject/system.h"
#include <curand_kernel.h>
#include <helper_math.h>
#include <float.h>

namespace cuda_math
{
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

  inline __device__ float3 invert_z(float3 v)
  {
    v.z = -v.z;
    return v;
  }

  inline __device__ float2 ceilf(float2 v)
  {
    return make_float2(::ceilf(v.x), ::ceilf(v.y));
  }
}

namespace cuda_malley_emission
{
  using cuda_math::face_t;
  using cuda_math::ray_t;

  struct face_data_t
  {
    int mesh_idx;
    float area;
    int2 n_rays;
    float2 weights;
    float2 emission;
  };

  struct context_t
  {
    int n_faces;
    face_t* faces;
    face_data_t* faces_data;
    int* face_ray_offsets;

    // Buffers for task
    int n_ray_buffer;
    float* mesh_emission;
    float* ray_power;
    ray_t* rays;

    float weight_factor;
    int n_total_rays;

    subject::material_t* materials;
    int* mesh_materials;

    curandStateMtgp32* TPGenA;
    curandStateMtgp32* TPGenB;
    curandStateMtgp32* HSGenR;
    curandStateMtgp32* HSGenTheta;
  };

  __global__ void prepare_context(context_t* ctx, const subject::mesh_t* meshes, int n_meshes);
  __global__ void calculate_weights(context_t* ctx, const float* temperatures);
  __global__ void calculate_n_rays(context_t* ctx, int n_target_rays);
  __global__ void emit_malley_rays(context_t* ctx, ray_t* rays, float* mesh_emissions, float* rays_power);
}

namespace cuda_parallel_rays_emission
{
  using cuda_math::mat33;
  using cuda_math::ray_t;

  struct source_params_t
  { 
    float3 direction;
    int n_rays;
  };

  struct context_t
  { 
    // Scene parameters
    float3 origin; // bounding sphere center
    float radius; // bounding sphere radius

    ray_t* rays;

    curandStateMtgp32* HSGenR;
    curandStateMtgp32* HSGenTheta;
  };

  __global__ void emit_parallel_rays(context_t* ctx, source_params_t params, ray_t* rays);
}

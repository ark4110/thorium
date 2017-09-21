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
// along with form_factors.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include "../cuda_misc/cuda_math.cuh"
#include <curand_kernel.h>
#include <helper_math.h>
#include <float.h>

namespace malley_cuda
{
  using cuda_math::ray_t;
  using cuda_math::face_t;

  struct context_t
  {
    curandStateMtgp32* HSGenR;
    curandStateMtgp32* HSGenTheta;
  };

  inline __device__ float3 pick_uniform_ray(context_t* ctx)
  {
    float r = curand_uniform(&ctx->HSGenR[blockIdx.x]);
    float theta = M_2PI * curand_uniform(&ctx->HSGenTheta[blockIdx.x]);

    float rad = sqrtf(r);
    float s, c;
    sincos(theta, &s, &c);
    return make_float3(rad * c, rad * s, sqrtf(1 - r));
  }

  inline __device__ ray_t ray_malley_align(float3 malley_ray, const cuda_math::mat33& rotation, float3 origin)
  {
    using namespace cuda_math;
    float3 relative_dist = rotation * malley_ray;
    return{ origin + relative_dist * 0.0001f, relative_dist };
  }

  inline __device__ ray_t ray_malley_emmit(float3 malley_ray, const face_t& face, float3 origin)
  {
    const cuda_math::mat33& rotation = cuda_math::pick_face_rotation(face);
    return ray_malley_align(malley_ray, rotation, origin);
  }

  inline __device__ ray_t ray_malley_emmit(float3 malley_ray, const float3& face_normalized_normal, float3 origin)
  {
    const cuda_math::mat33& rotation = cuda_math::pick_face_rotation(face_normalized_normal);
    return ray_malley_align(malley_ray, rotation, origin);
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

  struct context_t : malley_cuda::context_t
  {
    int n_faces;
    face_t* faces;
    face_data_t* faces_data;
    int* face_ray_offsets;

    // Buffers for task
    int n_ray_buffer;
    float* mesh_emission;

    float weight_factor;
    int n_total_rays;

    subject::material_t* materials;
    int* mesh_materials;

    curandStateMtgp32* TPGenA;
    curandStateMtgp32* TPGenB;
  };

  __global__ void prepare_context(context_t* ctx, const subject::mesh_t* meshes, int n_meshes);
  __global__ void calculate_weights(context_t* ctx, const float* temperatures);
  __global__ void calculate_n_rays(context_t* ctx, int n_target_rays);
  __global__ void generate_random(context_t* ctx, ray_t* rays);
  __global__ void emit_malley_rays(context_t* ctx, ray_t* rays, float* mesh_emissions, float* rays_power);
}

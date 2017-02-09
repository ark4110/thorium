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
#include "../cuda_misc/cuda_math.cuh"
#include "malley_cuda.cuh"
#include <curand_kernel.h>
#include <helper_math.h>
#include <float.h>

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
    float* ray_power;
    ray_t* rays;

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

  struct multi_source_params_t
  {
    float3 direction;
    int n_rays_offset;
    int n_rays;
  };

  struct context_t
  { 
    // Scene parameters
    float3 origin; // bounding sphere center
    float radius; // bounding sphere radius

    int n_allocated;
    ray_t* rays;

    curandStateMtgp32* HSGenR;
    curandStateMtgp32* HSGenTheta;
  };

  __global__ void emit_parallel_rays(context_t* ctx, source_params_t params, ray_t* rays);
  __global__ void emit_multiple_parallel_rays(context_t* ctx, const multi_source_params_t* params, ray_t* rays);
}

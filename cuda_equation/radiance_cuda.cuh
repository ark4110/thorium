// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
// This file is part of Thorium.
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

#pragma once

#include "../subject/system.h"
#include "../cuda_misc/cuda_math.cuh"
#include "../cuda_emission/malley_cuda.cuh"
#include <curand_kernel.h>
#include <helper_math.h>

namespace radiance_equation_cuda
{
  using cuda_math::prep_face_t;
  using cuda_math::ray_t;
  
  struct context_t : malley_cuda::context_t
  {
    prep_face_t* faces;
    int* face_to_mesh_index;
    
    subject::material_t* materials;

    int n_meshes;
    int* mesh_materials;
    float* mesh_power;

    curandStateMtgp32* HSGenDescision;

    // Per step vars
    int n_rays_alloc;
    char* hit_descision;
    float* ray_power; // owned by task
    float* postponed_ray_power;  // owned by task
    int* postponed_offsets;
  };
  
  __global__ void process_rays(context_t* ctx, int n_rays, const int* hit_face_idx, const float3* hit_point, ray_t* rays, char* hit_status);
}

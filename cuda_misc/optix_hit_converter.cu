// Copyright (c) 2015-2016 Contributors as noted in the AUTHORS file.
// This file is part of Thorium.
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

#include "optix_hit_converter.h"
#include "optix_hit_converter.cuh"
#include <stdlib.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include "../math/operations.h"
#include "../math/triangle.h"

namespace optix_hit_converter
{
  __device__ float3 triangle_uv_to_point(face_t face, float u, float v)
  {
    return u * face.points[0] + v * face.points[1] + (1.f - u - v) * face.points[2];
  }

  __global__ void convert_hits(const optix_hit_t* __restrict__ hits, const face_t* __restrict__ faces, int n_rays, float3* hit_point, int* hit_face_idx)
  {
    int ray_idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (; ray_idx < n_rays; ray_idx += gridDim.x * blockDim.x)
    {
      const optix_hit_t& hit = hits[ray_idx];
      if (hit.distance < 0)
        hit_face_idx[ray_idx] = -1;
      else
      {
        hit_face_idx[ray_idx] = hit.triangle_id;
        hit_point[ray_idx] = triangle_uv_to_point(faces[hit.triangle_id], hit.u, hit.v);
      }
    }
  }

  bool resolve_supported()
  {
    int count = 0;
    return cudaSuccess == cudaGetDeviceCount(&count) && count > 0;
  }  

  bool is_supported()
  {
    static const bool result = resolve_supported();
    return result;
  }

  bool is_supported(const state_t* state)
  {
    return state->optix_buffer_type == 0x201;
  }

  state_t* state_init(subject::scene_t* scene, bool force_cpu)
  {
    state_t* state = (state_t*)calloc(1, sizeof(state_t));

    bool supported = is_supported() && !force_cpu;

    state->optix_buffer_type = supported ? 0x201 : 0x200; // RTP_BUFFER_TYPE_CUDA_LINEAR : RTP_BUFFER_TYPE_HOST

    state->scene = scene;
    if (supported)
    {
      int minGridSize;
      cudaOccupancyMaxPotentialBlockSize(&minGridSize, &state->convert_block_size, convert_hits, 0, 0);
    }

    return state;
  }

  void state_free(state_t* state)
  {
    if (!state)
      return;
    
    if (is_supported(state))
      cudaFree(state->hits);
    else
      free(state->hits);

    free(state);
  }

  int state_prepare(state_t* state, ray_caster::task_t* task)
  {
    if (state->n_hits >= task->n_tasks)
      return RAY_CASTER_OK;

    bool cuda_enabled = is_supported(state);
    const bool allocation_valid = (cuda_enabled && task->allocation_device >= 0) ||
      (!cuda_enabled && task->allocation_device < 0);

    if (!allocation_valid)
      return -RAY_CASTER_INVALID_TASK_ALLOCATION;

    state->n_hits = task->n_tasks;

    if (cuda_enabled)
    {
      if (state->hits)
        cudaFree(state->hits);
      checkCudaErrors(cudaMalloc((void**)&state->hits, sizeof(optix_hit_t) * task->n_tasks));
    }
    else
    {
      state->hits = (optix_hit_t*)realloc(state->hits, task->n_tasks * sizeof(optix_hit_t));
    }

    return RAY_CASTER_OK;
  }

  int convert_cuda(state_t* state, ray_caster::task_t* task)
  {
    if (int r = scene_relocate(state->scene, 0))
    {
      if (r == -SCENE_ENGINE_NOT_SUPPORTED)
        fprintf(stderr, "Optix ray caster requires CUDA enabled scene!\n");
      return r;
    }
    cudaThreadSetCacheConfig(cudaFuncCachePreferL1);

    int n_grid_size = (task->n_tasks + state->convert_block_size - 1) / state->convert_block_size;

    convert_hits << <n_grid_size, state->convert_block_size >> > (state->hits
      , (const face_t*)state->scene->faces
      , task->n_tasks
      , (float3*)task->hit_point
      , task->hit_face_idx);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cudaThreadSetCacheConfig(cudaFuncCachePreferNone);

    // @todo Remove it
    if (int r = scene_relocate(state->scene, -1))
      return r;

    return RAY_CASTER_OK;
  }

  int convert_cpu(state_t* state, ray_caster::task_t* task)
  {
    if (int r = scene_relocate(state->scene, -1))
      return r;

    const subject::scene_t* scene = state->scene;
    for (int r = 0; r != task->n_tasks; ++r)
    {
      const optix_hit_t& hit = state->hits[r];
      if (hit.distance < 0)
        task->hit_face_idx[r] = -1;
      else
      {
        task->hit_face_idx[r] = hit.triangle_id;
        task->hit_point[r] = math::triangle_uv_to_point(scene->faces[hit.triangle_id], hit.u, hit.v);
      }
    }

    return RAY_CASTER_OK;
  }  

  int convert(state_t* state, ray_caster::task_t* task)
  {
    return is_supported(state) ? convert_cuda(state, task) : convert_cpu(state, task);
  }
}
